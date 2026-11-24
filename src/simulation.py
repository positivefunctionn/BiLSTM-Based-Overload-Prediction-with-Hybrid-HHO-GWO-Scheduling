"""Discrete-event execution simulation for the existing HHO-GWO scheduler.

This is a lightweight Python simulation (not CloudSim Plus). It takes the
scheduler's chosen task-to-VM assignment, models task arrivals and per-VM FIFO
execution, and exports task-, VM-, and horizon-level results.
"""
from __future__ import annotations
import random
import csv
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Iterable

from .fitness import evaluate_realized
from .models import Task, VM


def _write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def simulate_assignment(tasks: Iterable[Task], assignment: list[int], vms: list[VM],
                        cfg: dict, seed: int) -> tuple[list[dict], list[dict], dict]:
    """Simulate a fixed schedule with FIFO execution on each VM.

    Source timestamps are intentionally not converted because the project data
    does not establish their unit. Tasks arrive in source order over one
    scheduling interval. Actual CPU values are used only for post-schedule
    execution/evaluation, never to choose the assignment.
    """
    tasks = list(tasks)
    if len(tasks) != len(assignment):
        raise ValueError("assignment length must equal task count")
    if not vms:
        raise ValueError("at least one VM is required")
    if any(not isinstance(i, int) or i < 0 or i >= len(vms) for i in assignment):
        raise ValueError("assignment contains an invalid VM index")

    interval = max(float(cfg.get("scheduling_interval", 1.0)), 1e-9)
    reference_capacity = float(vms[0].cpu_capacity)
    arrivals = []
    for i, (task, vm_index) in enumerate(zip(tasks, assignment)):
        # Spread this horizon's tasks across one interval, preserving input order.
        arrival = (i / max(1, len(tasks))) * interval
        actual_cpu = max(0.0, float(task.actual_cpu))
        actual_demand = actual_cpu / 100.0 * reference_capacity
        duration = actual_demand / max(float(vms[vm_index].cpu_capacity), 1e-9)
        arrivals.append((arrival, i, task, vm_index, actual_cpu, actual_demand, duration))

    available_at = [0.0] * len(vms)
    task_rows = []
    for arrival, original_i, task, vm_index, actual_cpu, actual_demand, duration in arrivals:
        vm = vms[vm_index]
        start = max(arrival, available_at[vm_index])
        finish = start + duration
        available_at[vm_index] = finish
        wait = start - arrival
        task_rows.append({
            "horizon_id": int(task.horizon_id),
            "task_id": int(task.task_id),
            "source_order": original_i,
            "source_timestamp_unconverted": float(task.timestamp),
            "seed": int(seed),
            "selected_vm_id": vm.vm_id,
            "host_id": vm.host_id,
            "predicted_cpu": float(task.predicted_cpu),
            "actual_cpu_evaluation_only": actual_cpu,
            "predicted_demand": float(task.demand),
            "actual_demand_evaluation_only": actual_demand,
            "arrival_time_sim_units": arrival,
            "start_time_sim_units": start,
            "finish_time_sim_units": finish,
            "execution_time_sim_units": duration,
            "waiting_time_sim_units": wait,
            "turnaround_time_sim_units": finish - arrival,
            "predicted_overload": int(task.predicted_overload),
            "actual_overload_label": int(actual_cpu >= float(cfg.get("overload_threshold", 80.0))),
        })

    horizon_end = max((r["finish_time_sim_units"] for r in task_rows), default=0.0)
    vm_rows = []
    for i, vm in enumerate(vms):
        assigned = [r for r in task_rows if r["selected_vm_id"] == vm.vm_id]
        busy = sum(r["execution_time_sim_units"] for r in assigned)
        vm_rows.append({
            "horizon_id": int(tasks[0].horizon_id) if tasks else -1,
            "seed": int(seed),
            "vm_id": vm.vm_id,
            "host_id": vm.host_id,
            "cpu_capacity": float(vm.cpu_capacity),
            "initial_cpu_load": float(vm.current_cpu_load),
            "assigned_task_count": len(assigned),
            "busy_time_sim_units": busy,
            "makespan_sim_units": horizon_end,
            "execution_utilization": busy / horizon_end if horizon_end > 0 else 0.0,
            "mean_actual_cpu_of_assigned_tasks": (
                sum(r["actual_cpu_evaluation_only"] for r in assigned) / len(assigned) if assigned else 0.0
            ),
        })

    waiting = [r["waiting_time_sim_units"] for r in task_rows]
    execution = [r["execution_time_sim_units"] for r in task_rows]
    turnaround = [r["turnaround_time_sim_units"] for r in task_rows]
    horizon_metrics = {
        "horizon_id": int(tasks[0].horizon_id) if tasks else -1,
        "seed": int(seed),
        "task_count": len(tasks),
        "vm_count": len(vms),
        "simulated_makespan": horizon_end,
        "mean_execution_time": sum(execution) / len(execution) if execution else 0.0,
        "mean_waiting_time": sum(waiting) / len(waiting) if waiting else 0.0,
        "max_waiting_time": max(waiting, default=0.0),
        "mean_turnaround_time": sum(turnaround) / len(turnaround) if turnaround else 0.0,
        "tasks_with_wait": sum(w > 1e-12 for w in waiting),
        "overload_prediction_count": sum(int(t.predicted_overload) for t in tasks),
        "actual_overload_count_evaluation_only": sum(int(float(t.actual_cpu) >= float(cfg.get("overload_threshold", 80.0))) for t in tasks),
    }
    return task_rows, vm_rows, horizon_metrics


def run(config_path: str | Path | None = None, output_dir: str | Path | None = None,
        seed: int = 11, max_horizons: int | None = None) -> dict:
    """Run the actual HHO-GWO optimizer and simulate its assignment per horizon."""
    import yaml
    from .data_adapter import load_and_validate, horizons
    from .demand_adapter import adapt, write_repair_log
    from .hybrid import hho_gwo
    from .fitness import reset_evaluation_count, get_evaluation_count

    root = Path(__file__).resolve().parents[1]
    config_path = Path(config_path) if config_path else root / "config" / "config.yaml"
    output_dir = Path(output_dir) if output_dir else root / "results" / "simulation"
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg["objective_mode"] = cfg.get("objective_mode", "legacy_vm_proxy")
    cfg.setdefault("normalization_references", {"energy": 1000.0, "makespan": 10.0, "sla": 1.0, "imbalance": 1.0})
    cfg.setdefault("weights", {"energy": 0.25, "makespan": 0.25, "sla": 0.25, "imbalance": 0.25})
    input_path = root / cfg.get("input_file", "data/person2_scheduler_input.csv")
    rows, validation = load_and_validate(
        input_path, expected_rows=None,
        overload_threshold=cfg.get("overload_threshold", 80.0),
        forecast_mode=cfg.get("forecast_mode", "bilstm"),
    )
    if not validation.get("valid"):
        raise ValueError("Input CSV validation failed: " + "; ".join(validation.get("errors", [])))

    vms = [VM(**item) for item in cfg["vms"]]
    if not vms:
        raise ValueError("config must define at least one VM")
    repair_log: list[dict] = []
    all_task_rows, all_vm_rows, all_horizon_rows = [], [], []
    # Randomize all task records before creating horizons
    rows = list(rows)
    random.shuffle(rows)

# Divide the shuffled tasks into groups of 10
    horizon_batches = horizons(rows, int(cfg.get("horizon_length", 10)))
    if max_horizons is not None:
        horizon_batches = horizon_batches[:max(0, int(max_horizons))]

    for horizon_id, batch in enumerate(horizon_batches):
        tasks = adapt(batch, horizon_id, vms[0].cpu_capacity, repair_log)
        # Attach realized demand strictly for post-hoc evaluation; HHO-GWO only sees task.demand.
        tasks = [replace(t, actual_demand=max(0.0, float(t.actual_cpu)) / 100.0 * vms[0].cpu_capacity)
                 for t in tasks]
        if not tasks:
            continue
        reset_evaluation_count(limit=int(cfg.get("evaluation_budget", 300)))
        assignment, predicted_metrics, _history, _ranked = hho_gwo(
            tasks, vms, cfg, int(seed), max_evaluations=int(cfg.get("evaluation_budget", 300))
        )
        eval_calls = get_evaluation_count()
        realized_metrics = evaluate_realized(tasks, assignment, vms, cfg)
        task_rows, vm_rows, execution_metrics = simulate_assignment(tasks, assignment, vms, cfg, int(seed))
        for r in task_rows:
            r["predicted_schedule_fitness"] = float(predicted_metrics.fitness)
            r["predicted_schedule_energy_proxy"] = float(predicted_metrics.energy)
            r["predicted_schedule_sla_violation_count"] = int(predicted_metrics.sla_violation_count)
        for r in vm_rows:
            r["objective_evaluations"] = eval_calls
        all_task_rows.extend(task_rows)
        all_vm_rows.extend(vm_rows)
        all_horizon_rows.append({
            **execution_metrics,
            "forecast_mode": cfg.get("forecast_mode", "bilstm"),
            "objective_evaluations": eval_calls,
            "predicted_fitness": float(predicted_metrics.fitness),
            "predicted_energy_proxy": float(predicted_metrics.energy),
            "predicted_makespan_proxy": float(predicted_metrics.makespan),
            "predicted_sla_violation_count": int(predicted_metrics.sla_violation_count),
            "predicted_load_imbalance": float(predicted_metrics.load_imbalance),
            "realized_fitness_posthoc": float(realized_metrics.fitness),
            "realized_energy_proxy_posthoc": float(realized_metrics.energy),
            "realized_makespan_proxy_posthoc": float(realized_metrics.makespan),
            "realized_sla_violation_count_posthoc": int(realized_metrics.sla_violation_count),
            "realized_load_imbalance_posthoc": float(realized_metrics.load_imbalance),
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "task_execution.csv", all_task_rows)
    _write_csv(output_dir / "vm_metrics.csv", all_vm_rows)
    _write_csv(output_dir / "horizon_metrics.csv", all_horizon_rows)
    write_repair_log(output_dir / "prediction_repair_log.csv", repair_log)
    report = {
        "simulation_type": "lightweight Python discrete-event execution simulation; not CloudSim Plus",
        "input_file": str(input_path),
        "forecast_mode": cfg.get("forecast_mode", "bilstm"),
        "input_validation": validation,
        "seed": int(seed),
        "horizons_simulated": len(all_horizon_rows),
        "tasks_simulated": len(all_task_rows),
        "vm_count": len(vms),
        "evaluation_budget_per_horizon": int(cfg.get("evaluation_budget", 300)),
        "outputs": ["task_execution.csv", "vm_metrics.csv", "horizon_metrics.csv", "prediction_repair_log.csv"],
        "interpretation": [
            "The optimizer chooses assignments using predicted demand only.",
            "Actual CPU is used only for post-hoc realized metrics and execution-duration simulation.",
            "Execution times are simulation units, not measured wall-clock times.",
            "Energy is the project's configured proxy, not measured physical energy.",
            "This does not instantiate CloudSim Plus hosts, VMs, or Cloudlets."
        ],
    }
    (output_dir / "simulation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Simulation completed: {len(all_task_rows)} tasks across {len(all_horizon_rows)} horizons")
    print(f"Output folder: {output_dir.resolve()}")
    print("Files: task_execution.csv, vm_metrics.csv, horizon_metrics.csv, prediction_repair_log.csv, simulation_report.json")
    return report


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Run HHO-GWO scheduling and lightweight task execution simulation")
    parser.add_argument("--config", default=None, help="Optional path to config YAML")
    parser.add_argument("--output", default=None, help="Output directory (default: results/simulation)")
    parser.add_argument("--seed", type=int, default=11, help="Deterministic optimizer seed (default: 11)")
    parser.add_argument("--max-horizons", type=int, default=None, help="Optional small test run; omit to simulate all horizons")
    args = parser.parse_args()
    run(args.config, args.output, args.seed, args.max_horizons)


if __name__ == "__main__":
    main()
