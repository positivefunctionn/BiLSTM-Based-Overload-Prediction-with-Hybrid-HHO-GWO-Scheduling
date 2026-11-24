# Event-level queue simulation

`src/event_simulator.py` is a discrete-event simulator. It replays recorded task-to-VM assignments over simulated
time with explicit **ARRIVAL**, **START** and **FINISH** events and one **FIFO queue per VM**. It complements
`src/simulation.py` (Section 7 of the dashboard), which simulates a single HHO-GWO run horizon by horizon.

## Model

- Each VM is a single-server FIFO queue: it runs one task at a time, later arrivals wait.
- Service time = actual demand / VM capacity (the same unit as `src/fitness.py`). Actual CPU is used only here,
  after scheduling, never by the optimizers.
- Tasks are grouped in horizons of 10. The i-th of n tasks in a horizon arrives at `start + (i/n) * scheduling_interval`
  (same convention as `src/simulation.py`; `--stagger-interval 0` makes the whole horizon arrive together).
- **Drain mode (default):** horizon h+1 arrives when horizon h has finished, so queues never carry over, like the
  per-horizon batch evaluation in the analytical model.
- **Fixed-spacing mode (`--horizon-spacing S`):** horizon h arrives at `h*S` regardless of VM state, so work queues up.
- Events at the same time are processed FINISH first, then ARRIVAL, in insertion order, so runs are deterministic.
- Per-task output: arrival, start, finish, wait, service, response. Per-VM output: tasks, busy time, utilization,
  max and time-averaged queue length. Per-run output: total simulated time, mean/P95/max wait and response,
  throughput, response-time SLA violation rate (response > `execution_sla`), and a VM-level energy proxy.

## Run

```bash
py experiments/run_event_simulation.py                                   # drain mode -> results/event_simulation/
py experiments/run_event_simulation.py --horizon-spacing 1.0 --output results/event_simulation_fixed_spacing_1.0
py -m unittest discover -s tests -v
```

Outputs: `run_summary.csv` (60 runs), `algorithm_summary.csv`, `vm_summary.csv`, `task_trace.csv` and
`event_log_trace.csv` (HHO-GWO trial 0), `vm_queue_timeline.png`, and `cross_check_vs_src_simulation.json`.

## Validation

- **Cross-check:** for HHO-GWO trial 0 (seed 11) the event simulator reproduces every start, finish and wait time in
  `results/simulation/task_execution.csv` (produced by `src/simulation.py`); maximum difference 3.3e-14 over 1,094 tasks.
- **Unit tests** (`tests/test_event_simulator.py`, `tests/test_simulation.py`): FIFO order, capacity-scaled service,
  per-horizon makespan equal to the analytical value, backlog carry-over, no overlapping service, event order, and
  agreement between the two simulators.

## Results to quote carefully

- Drain mode: the BFD baseline is best on every replay metric; among the five metaheuristics HHO-GWO has the lowest
  P95 response, maximum response and maximum queue length. The optimizers do not minimise queue delay, differences are
  small and no significance test was run, so do not describe this as HHO-GWO outperforming BFD.
- Fixed spacing of 1.0 unit: the four VMs cannot keep up (mean wait about 50 units, queues in the hundreds), because
  an overload task needs about 1.05 time units on its own. This is a stress scenario showing how heavily loaded the
  configured workload is.

## Limitations

Lightweight Python simulation, **not CloudSim Plus**. VMs start empty (configured `current_cpu_load` is not used as a
backlog), service is non-preemptive and deterministic, hosts are not modelled beyond grouping, time is in simulation
units (the source timestamp unit is not established), and the energy value is a VM-level proxy. For a CloudSim Plus
comparison, `experiments/export_cloudsim_input.py` writes `results/cloudsim_export/` (hosts, VMs and cloudlets with
expected start/finish times); the Java CloudSim run itself is not included.
