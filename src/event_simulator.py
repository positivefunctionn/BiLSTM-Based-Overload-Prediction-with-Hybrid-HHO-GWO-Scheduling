"""Discrete-event simulation of task execution on VMs with FIFO queues.

This module is additive: it consumes task-to-VM assignments produced by the
scheduling algorithms and replays them over simulated time. It imports nothing
from the optimizers and does not change any existing project file.

Model (all times are in scheduling-interval units, the same unit used by
src/fitness.py, where service time = demand / vm.cpu_capacity):

* Each VM is a single-server FIFO queue: it runs one task at a time and tasks
  that arrive while it is busy wait in its queue.
* Tasks are grouped in horizons. All tasks of a horizon arrive together.
* Two horizon-arrival modes are supported.
    - "drain"  (horizon_spacing=None): horizon h+1 arrives when every task of
      horizon h has finished. Queues never carry over between horizons, which
      matches the per-horizon batch evaluation in src/fitness.py.
    - "fixed"  (horizon_spacing=S): horizon h arrives at time h*S regardless of
      the VM state, so unfinished work from earlier horizons queues up.
* Arrival stagger: with stagger_interval=0 all tasks of a horizon arrive at the
  horizon start. With stagger_interval=I, the i-th of n tasks arrives at
  start + (i/n)*I, the same convention as src/simulation.py (I = the
  scheduling interval).
* Events: ARRIVAL (task joins its VM queue), START (task begins service) and
  FINISH (task completes). Ties at the same time are processed FINISH first,
  then ARRIVAL, each in insertion order, so results are deterministic.

Limitations: VMs start empty (the configured current_cpu_load is not used as a
backlog, as in src/fitness.py makespan), service is non-preemptive and
deterministic, and the energy value is a simple VM-level proxy.
"""
from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass, field

ARRIVAL = "ARRIVAL"
START = "START"
FINISH = "FINISH"
_PRIORITY = {FINISH: 0, ARRIVAL: 1}


@dataclass(frozen=True)
class SimTask:
    task_id: int
    horizon_id: int
    vm_id: str
    demand: float  # CPU units of work; service time = demand / vm capacity


@dataclass
class TaskRecord:
    task_id: int
    horizon_id: int
    vm_id: str
    demand: float
    arrival: float = 0.0
    start: float = 0.0
    finish: float = 0.0

    @property
    def service(self):
        return self.finish - self.start

    @property
    def wait(self):
        return self.start - self.arrival

    @property
    def response(self):
        return self.finish - self.arrival


@dataclass
class VMStats:
    vm_id: str
    tasks: int = 0
    busy_time: float = 0.0
    max_queue_length: int = 0
    queue_area: float = 0.0  # integral of waiting-queue length over time


@dataclass
class SimulationResult:
    records: list
    events: list = field(default_factory=list)
    vm_stats: dict = field(default_factory=dict)
    end_time: float = 0.0
    horizon_end_times: dict = field(default_factory=dict)


def simulate(tasks, vms, horizon_spacing=None, record_events=True, stagger_interval=0.0):
    """Run the simulation.

    tasks: iterable of SimTask (order within a horizon is the arrival order).
    vms:   iterable of objects with vm_id and cpu_capacity (src.models.VM works).
    horizon_spacing: None for "drain" mode, or a positive number for "fixed".
    stagger_interval: spread each horizon's arrivals over this many time units.
    """
    if stagger_interval < 0:
        raise ValueError("stagger_interval must be non-negative")
    if horizon_spacing is not None and horizon_spacing <= 0:
        raise ValueError("horizon_spacing must be positive or None")
    capacity = {v.vm_id: float(v.cpu_capacity) for v in vms}
    if not capacity or any(c <= 0 for c in capacity.values()):
        raise ValueError("every VM needs a positive cpu_capacity")

    by_horizon = {}
    for t in tasks:
        if t.vm_id not in capacity:
            raise ValueError(f"task {t.task_id} assigned to unknown VM {t.vm_id}")
        if t.demand < 0:
            raise ValueError(f"task {t.task_id} has negative demand")
        by_horizon.setdefault(t.horizon_id, []).append(t)
    horizon_order = sorted(by_horizon)
    records = {}
    for h in horizon_order:
        for t in by_horizon[h]:
            records[t.task_id] = TaskRecord(t.task_id, t.horizon_id, t.vm_id, t.demand)
    if len(records) != sum(len(v) for v in by_horizon.values()):
        raise ValueError("task_id values must be unique")

    stats = {vm_id: VMStats(vm_id) for vm_id in capacity}
    queues = {vm_id: deque() for vm_id in capacity}
    busy = {vm_id: False for vm_id in capacity}
    last_change = {vm_id: 0.0 for vm_id in capacity}
    remaining = {h: len(by_horizon[h]) for h in horizon_order}
    horizon_end = {}
    events = []
    heap = []
    seq = 0

    def push(time, kind, task_id):
        nonlocal seq
        heapq.heappush(heap, (time, _PRIORITY[kind], seq, kind, task_id))
        seq += 1

    def schedule_horizon(index, time):
        batch = by_horizon[horizon_order[index]]
        for position, t in enumerate(batch):
            push(time + (position / len(batch)) * stagger_interval, ARRIVAL, t.task_id)

    def log(time, kind, rec):
        if record_events:
            events.append({"time": time, "event": kind, "task_id": rec.task_id,
                           "horizon_id": rec.horizon_id, "vm_id": rec.vm_id,
                           "queue_length": len(queues[rec.vm_id]),
                           "vm_busy": int(busy[rec.vm_id])})

    def account_queue(vm_id, now):
        stats[vm_id].queue_area += len(queues[vm_id]) * (now - last_change[vm_id])
        last_change[vm_id] = now

    def start_next(vm_id, now):
        if busy[vm_id] or not queues[vm_id]:
            return
        account_queue(vm_id, now)
        rec = records[queues[vm_id].popleft()]
        busy[vm_id] = True
        rec.start = now
        service = rec.demand / capacity[vm_id]
        stats[vm_id].busy_time += service
        log(now, START, rec)
        push(now + service, FINISH, rec.task_id)

    if horizon_order:
        if horizon_spacing is None:
            schedule_horizon(0, 0.0)
        else:
            for i in range(len(horizon_order)):
                schedule_horizon(i, i * float(horizon_spacing))

    now = 0.0
    next_horizon = 1
    while heap:
        now, _, _, kind, task_id = heapq.heappop(heap)
        rec = records[task_id]
        vm_id = rec.vm_id
        if kind == ARRIVAL:
            rec.arrival = now
            account_queue(vm_id, now)
            queues[vm_id].append(task_id)
            log(now, ARRIVAL, rec)
            start_next(vm_id, now)
            # Waiting tasks only: a task that starts at once never waits.
            stats[vm_id].max_queue_length = max(stats[vm_id].max_queue_length,
                                                len(queues[vm_id]))
        else:  # FINISH
            rec.finish = now
            busy[vm_id] = False
            stats[vm_id].tasks += 1
            log(now, FINISH, rec)
            remaining[rec.horizon_id] -= 1
            if remaining[rec.horizon_id] == 0:
                horizon_end[rec.horizon_id] = now
                if horizon_spacing is None and next_horizon < len(horizon_order):
                    schedule_horizon(next_horizon, now)
                    next_horizon += 1
            start_next(vm_id, now)

    for vm_id in capacity:
        account_queue(vm_id, now)
    ordered = [records[t.task_id] for h in horizon_order for t in by_horizon[h]]
    return SimulationResult(ordered, events, stats, now, horizon_end)


def vm_energy_proxy(vm, busy_time, end_time):
    """VM-level power proxy: idle_power for the whole run plus (max-idle) while busy."""
    return vm.idle_power * end_time + (vm.max_power - vm.idle_power) * busy_time
