"""Replay recorded task-to-VM assignments through the discrete-event simulator.

Reads results/schedule_results.csv and config/config.yaml, simulates every
(algorithm, trial) group, and writes new files under results/event_simulation/.
It never modifies existing project files.

Usage (from the project root):
    py experiments/run_event_simulation.py
    py experiments/run_event_simulation.py --horizon-spacing 1.0 --output results/event_simulation_fixed
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.event_simulator import SimTask, simulate, vm_energy_proxy  # noqa: E402
from src.models import VM  # noqa: E402


def load_vms(cfg):
    return [VM(v["vm_id"], v["host_id"], float(v["cpu_capacity"]),
               float(v["current_cpu_load"]), float(v["idle_power"]), float(v["max_power"]))
            for v in cfg["vms"]]


def read_schedule(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_tasks(rows, vms):
    """Actual demand mirrors the post-hoc evaluation in run_experiments.py."""
    ref = vms[0].cpu_capacity
    host_of = {v.vm_id: v.host_id for v in vms}
    tasks = []
    for r in sorted(rows, key=lambda x: (int(x["horizon_id"]), int(x["task_id"]))):
        if host_of.get(r["selected_vm_id"]) != r["host_id"]:
            raise ValueError(f"task {r['task_id']}: host {r['host_id']} does not match "
                             f"configured host of {r['selected_vm_id']}")
        actual = max(0.0, float(r["actual_cpu_evaluation_only"]))
        tasks.append(SimTask(int(r["task_id"]), int(r["horizon_id"]),
                             r["selected_vm_id"], actual / 100.0 * ref))
    return tasks


def analytical_total_makespan(tasks, vms):
    cap = {v.vm_id: v.cpu_capacity for v in vms}
    per_h = {}
    for t in tasks:
        d = per_h.setdefault(t.horizon_id, {})
        d[t.vm_id] = d.get(t.vm_id, 0.0) + t.demand / cap[t.vm_id]
    return sum(max(d.values()) for d in per_h.values())


def summarize(result, tasks, vms, execution_sla, label):
    waits = np.array([r.wait for r in result.records])
    resp = np.array([r.response for r in result.records])
    end = result.end_time
    vm_rows, utils, energy = [], [], 0.0
    for v in vms:
        s = result.vm_stats[v.vm_id]
        util = s.busy_time / end if end > 0 else 0.0
        utils.append(util)
        energy += vm_energy_proxy(v, s.busy_time, end)
        vm_rows.append({**label, "vm_id": v.vm_id, "host_id": v.host_id, "tasks": s.tasks,
                        "busy_time": round(s.busy_time, 6), "utilization": round(util, 6),
                        "max_queue_length": s.max_queue_length,
                        "mean_queue_length": round(s.queue_area / end, 6) if end > 0 else 0.0})
    run = {**label,
           "tasks": len(result.records),
           "sim_end_time": round(end, 6),
           "mean_wait": round(float(waits.mean()), 6),
           "p95_wait": round(float(np.percentile(waits, 95)), 6),
           "max_wait": round(float(waits.max()), 6),
           "mean_response": round(float(resp.mean()), 6),
           "p95_response": round(float(np.percentile(resp, 95)), 6),
           "max_response": round(float(resp.max()), 6),
           "throughput_tasks_per_interval": round(len(result.records) / end, 6) if end > 0 else 0.0,
           "response_sla_violation_rate": round(float((resp > execution_sla).mean()), 6),
           "tasks_that_waited": int((waits > 1e-12).sum()),
           "mean_vm_utilization": round(float(np.mean(utils)), 6),
           "max_queue_length": max(s.max_queue_length for s in result.vm_stats.values()),
           "energy_proxy": round(energy, 4)}
    return run, vm_rows


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def aggregate(run_rows):
    metrics = [k for k in run_rows[0] if k not in ("algorithm", "trial", "seed", "analytical_total_makespan",
                                                  "matches_analytical_makespan")]
    out = []
    for algo in sorted({r["algorithm"] for r in run_rows}):
        group = [r for r in run_rows if r["algorithm"] == algo]
        row = {"algorithm": algo, "runs": len(group)}
        for m in metrics:
            vals = np.array([float(r[m]) for r in group])
            row[f"{m}_mean"] = round(float(vals.mean()), 6)
            row[f"{m}_std"] = round(float(vals.std(ddof=1)) if len(vals) > 1 else 0.0, 6)
        out.append(row)
    return out


def plot_timeline(records, vms, path, first_horizon, n_horizons):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    wanted = set(range(first_horizon, first_horizon + n_horizons))
    recs = [r for r in records if r.horizon_id in wanted]
    t0 = min(r.arrival for r in recs)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    order = {v.vm_id: i for i, v in enumerate(vms)}
    for r in recs:
        y = order[r.vm_id]
        # Running time is the thick bar; time spent waiting in the VM queue is the thin strip above it.
        ax.broken_barh([(r.start, r.service)], (y - 0.40, 0.52), facecolors="#4c78a8",
                       edgecolors="white", linewidth=0.8)
        if r.wait > 1e-12:
            ax.broken_barh([(r.arrival, r.wait)], (y + 0.18, 0.2), facecolors="#e0a800",
                           edgecolors="white", linewidth=0.5)
    ax.set_yticks(list(order.values()))
    ax.set_yticklabels(list(order.keys()))
    ax.set_xlim(t0, max(r.finish for r in recs) * 1.01)
    ax.set_xlabel("Simulated time (scheduling intervals)")
    ax.set_title(f"VM queues and task execution, horizons {first_horizon}-{first_horizon + n_horizons - 1}"
                 "  (thick blue = running, thin amber strip = waiting in queue)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def cross_check(args, trace, out, stagger):
    """Compare simulations only when their task inputs and assignments match."""
    import json

    report_path = out / "cross_check_vs_src_simulation.json"
    ref = ROOT / "results" / "simulation" / "task_execution.csv"

    report = {
        "comparable": False,
        "compared_tasks": 0,
        "max_abs_difference_sim_units": None,
        "reference": str(ref.relative_to(ROOT)),
        "stagger_interval": stagger,
        "agrees": None,
        "reason": ""
    }

    if trace is None:
        report["reason"] = "No event-simulation trace was selected."
    elif args.horizon_spacing is not None:
        report["reason"] = (
            "Cross-check skipped because explicit horizon spacing "
            "changes the timing model."
        )
    elif not ref.exists():
        report["reason"] = "Reference simulation file was not found."
    else:
        result, label = trace

        with open(ref, newline="", encoding="utf-8-sig") as f:
            ref_rows = list(csv.DictReader(f))

        # Match tasks by both horizon and task ID.
        ref_by_key = {
            (int(r["horizon_id"]), int(r["task_id"])): r
            for r in ref_rows
            if int(float(r["seed"])) == label["seed"]
        }

        event_by_key = {
            (rec.horizon_id, rec.task_id): rec
            for rec in result.records
        }

        if set(ref_by_key) != set(event_by_key):
            report["reason"] = (
                "Not comparable: the two simulations contain different "
                "task/horizon sets. The event simulation uses "
                "results/schedule_results.csv, while the reference uses "
                "results/simulation/task_execution.csv."
            )
        else:
            # Verify that both simulations use the same VM assignments
            # and actual task demands before comparing execution times.
            same_inputs = True

            for key, rec in event_by_key.items():
                row = ref_by_key[key]

                if rec.vm_id != row["selected_vm_id"]:
                    same_inputs = False
                    break

                ref_demand = float(row["actual_demand_evaluation_only"])
                if abs(rec.demand - ref_demand) > 1e-6:
                    same_inputs = False
                    break

            if not same_inputs:
                report["reason"] = (
                    "Not comparable: task VM assignments or actual demands "
                    "differ between the two simulation inputs."
                )
            else:
                starts = {}
                for rec in result.records:
                    starts[rec.horizon_id] = min(
                        starts.get(rec.horizon_id, rec.arrival),
                        rec.arrival
                    )

                worst = 0.0

                for rec in result.records:
                    row = ref_by_key[(rec.horizon_id, rec.task_id)]
                    base = starts[rec.horizon_id]

                    differences = (
                        abs(
                            (rec.start - base)
                            - float(row["start_time_sim_units"])
                        ),
                        abs(
                            (rec.finish - base)
                            - float(row["finish_time_sim_units"])
                        ),
                        abs(
                            rec.wait
                            - float(row["waiting_time_sim_units"])
                        ),
                    )
                    worst = max(worst, *differences)

                report.update({
                    "comparable": True,
                    "compared_tasks": len(result.records),
                    "max_abs_difference_sim_units": worst,
                    "agrees": worst < 1e-6,
                    "reason": (
                        "Inputs match; execution times were compared."
                    )
                })

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    if report["comparable"]:
        print(
            "Cross-check:",
            report["compared_tasks"],
            "tasks, max difference",
            report["max_abs_difference_sim_units"],
            "agrees:",
            report["agrees"]
        )
    else:
        print("Cross-check skipped:", report["reason"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--schedule", default=str(ROOT / "results" / "schedule_results.csv"))
    ap.add_argument("--config", default=str(ROOT / "config" / "config.yaml"))
    ap.add_argument("--output", default=str(ROOT / "results" / "event_simulation"))
    ap.add_argument("--horizon-spacing", type=float, default=None,
                    help="omit for drain mode; a number S makes horizon h arrive at h*S")
    ap.add_argument("--stagger-interval", type=float, default=None,
                    help="spread each horizon's arrivals over this many time units; "
                         "default = scheduling_interval from the config (same as src/simulation.py); 0 = batch arrival")
    ap.add_argument("--trace-algorithm", default="HHO-GWO")
    ap.add_argument("--trace-trial", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    stagger = float(cfg.get("scheduling_interval", 1.0)) if args.stagger_interval is None else args.stagger_interval
    vms = load_vms(cfg)
    rows = read_schedule(args.schedule)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    groups = {}
    for r in rows:
        groups.setdefault((r["algorithm"], int(r["trial"]), int(r["seed"])), []).append(r)

    run_rows, vm_rows = [], []
    trace = None
    for (algo, trial, seed), grp in sorted(groups.items()):
        tasks = build_tasks(grp, vms)
        is_trace = algo == args.trace_algorithm and trial == args.trace_trial
        result = simulate(tasks, vms, args.horizon_spacing, record_events=is_trace, stagger_interval=stagger)
        label = {"algorithm": algo, "trial": trial, "seed": seed}
        run, vrows = summarize(result, tasks, vms, cfg["execution_sla"], label)
        ana = analytical_total_makespan(tasks, vms)
        run["analytical_total_makespan"] = round(ana, 6)
        if args.horizon_spacing is None and stagger == 0:
            run["matches_analytical_makespan"] = int(abs(ana - result.end_time) < 1e-6)
        else:
            run["matches_analytical_makespan"] = ""  # only defined for drain mode with batch arrivals
        run_rows.append(run)
        vm_rows.extend(vrows)
        if is_trace:
            trace = (result, label)

    write_csv(out / "run_summary.csv", run_rows)
    write_csv(out / "vm_summary.csv", vm_rows)
    write_csv(out / "algorithm_summary.csv", aggregate(
        [{k: v for k, v in r.items() if k != "matches_analytical_makespan"} for r in run_rows]))

    if trace is not None:
        result, label = trace
        write_csv(out / "event_log_trace.csv",
                  [{**label, **e, "time": round(e["time"], 6)} for e in result.events])
        write_csv(out / "task_trace.csv", [
            {**label, "task_id": r.task_id, "horizon_id": r.horizon_id, "vm_id": r.vm_id,
             "demand": round(r.demand, 6), "arrival": round(r.arrival, 6), "start": round(r.start, 6),
             "finish": round(r.finish, 6), "wait": round(r.wait, 6), "service": round(r.service, 6),
             "response": round(r.response, 6),
             "response_sla_violation": int(r.response > cfg["execution_sla"])}
            for r in result.records])
        # Show a window that contains both heavy and idle tasks when possible.
        mixed = [h for h in sorted({r.horizon_id for r in result.records})
                 if len({r.demand > 50 for r in result.records if r.horizon_id == h}) == 2]
        first = max(0, (mixed[0] if mixed else 0) - 1)
        plot_timeline(result.records, vms, out / "vm_queue_timeline.png", first, 6)

    cross_check(args, trace, out, stagger)
    mode = "drain" if args.horizon_spacing is None else f"fixed spacing {args.horizon_spacing}"
    print(f"Simulated {len(run_rows)} runs ({mode}); wrote files to {out}")
    if args.horizon_spacing is None and stagger == 0:
        bad = [r for r in run_rows if r["matches_analytical_makespan"] != 1]
        print("Runs matching the analytical makespan:", len(run_rows) - len(bad), "of", len(run_rows))


if __name__ == "__main__":
    main()
