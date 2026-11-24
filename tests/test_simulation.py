"""Tests for src/simulation.py and its agreement with src/event_simulator.py."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.event_simulator import SimTask, simulate  # noqa: E402
from src.models import Task, VM  # noqa: E402
from src.simulation import simulate_assignment  # noqa: E402

CFG = {"scheduling_interval": 1.0, "overload_threshold": 80.0}


def _vms():
    return [VM("vm_1", "host_1", 100.0, 0.0, 0.0, 0.0),
            VM("vm_2", "host_1", 50.0, 0.0, 0.0, 0.0)]


def _task(i, actual_cpu, horizon=0):
    return Task(i, horizon, actual_cpu, 0.0, int(actual_cpu >= 80), actual_cpu, actual_cpu=actual_cpu, timestamp=float(i))


class SimulationTests(unittest.TestCase):
    def test_fifo_waiting_on_one_vm(self):
        tasks = [_task(0, 100.0), _task(1, 50.0)]
        rows, vm_rows, horizon = simulate_assignment(tasks, [0, 0], _vms(), CFG, seed=1)
        self.assertEqual(rows[0]["waiting_time_sim_units"], 0.0)
        self.assertAlmostEqual(rows[1]["start_time_sim_units"], 1.0)          # waits for task 0
        self.assertAlmostEqual(rows[1]["waiting_time_sim_units"], 0.5)        # arrives at 0.5
        self.assertAlmostEqual(horizon["simulated_makespan"], 1.5)
        self.assertEqual(horizon["tasks_with_wait"], 1)

    def test_slower_vm_takes_longer(self):
        rows, _, _ = simulate_assignment([_task(0, 100.0)], [1], _vms(), CFG, seed=1)
        self.assertAlmostEqual(rows[0]["execution_time_sim_units"], 2.0)       # capacity 50

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            simulate_assignment([_task(0, 10.0)], [0, 0], _vms(), CFG, seed=1)
        with self.assertRaises(ValueError):
            simulate_assignment([_task(0, 10.0)], [5], _vms(), CFG, seed=1)
        with self.assertRaises(ValueError):
            simulate_assignment([_task(0, 10.0)], [0], [], CFG, seed=1)

    def test_actual_cpu_drives_execution_not_prediction(self):
        t = Task(0, 0, 10.0, 0.0, 0, 10.0, actual_cpu=100.0, timestamp=0.0)
        rows, _, _ = simulate_assignment([t], [0], _vms(), CFG, seed=1)
        self.assertAlmostEqual(rows[0]["execution_time_sim_units"], 1.0)

    def test_event_simulator_agrees_with_simulation_module(self):
        cpus = [104.0, 0.3, 103.0, 99.0, 0.3, 104.5, 60.0, 5.0, 103.2, 101.0]
        assignment = [0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
        vms = _vms()
        tasks = [_task(i, c) for i, c in enumerate(cpus)]
        rows, _, horizon = simulate_assignment(tasks, assignment, vms, CFG, seed=1)
        sim_tasks = [SimTask(t.task_id, 0, vms[a].vm_id, max(0.0, t.actual_cpu) / 100.0 * vms[0].cpu_capacity)
                     for t, a in zip(tasks, assignment)]
        result = simulate(sim_tasks, vms, stagger_interval=1.0)
        for rec, row in zip(result.records, rows):
            self.assertAlmostEqual(rec.start, row["start_time_sim_units"], places=9)
            self.assertAlmostEqual(rec.finish, row["finish_time_sim_units"], places=9)
            self.assertAlmostEqual(rec.wait, row["waiting_time_sim_units"], places=9)
        self.assertAlmostEqual(result.end_time, horizon["simulated_makespan"], places=9)


if __name__ == "__main__":
    unittest.main()
