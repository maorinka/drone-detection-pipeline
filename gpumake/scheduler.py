"""Run a job graph: independent jobs side by side on free GPUs, dependencies respected.

* At most ``max_jobs`` processes at once. A card running our job takes a second one only after
  ``settle_seconds`` (the first needs time to allocate memory before the card is measured again).
* After the first failure nothing new starts; running jobs finish, then the run stops.
* If the spare-disk rule is reached nothing new starts; running jobs finish. No data is deleted.
* If the scheduler itself is stopped (Ctrl-C, SIGTERM, an internal error), it stops the jobs it
  started and records them, so a rerun does not run duplicates beside orphans. It never touches
  any other process.
"""
from __future__ import annotations

import os
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Set

from . import gpu as gpu_module
from .spec import Job
from .util import log, read, save, space_ok


class JobFailed(RuntimeError):
    pass


def _update_state(run: Path, **changes) -> None:
    path = run / "state.json"
    state = read(path) if path.exists() else {}
    state.update(changes)
    save(path, state)


def run_jobs(run: Path, project: Path, jobs: List[Job], *, satisfied: Iterable[str] = (), gpu: str = "auto",
             max_jobs: int = 8, per_gpu: int = 1, min_free_mib: int = 8192, shared_free_mib: int = 3072,
             max_utilization: int = 20, min_free_disk_fraction: float = 0.30, poll: float = 5.0,
             settle_seconds: float = 30.0, devices: Callable[[], List[dict]] = gpu_module.list_devices,
             launch: Optional[Callable] = None) -> List[str]:
    """Run ``jobs`` (those not yet done); ``satisfied`` = labels already complete. Returns the labels run."""
    launch = launch or _launch
    gpu = str(gpu)
    if gpu.isdigit():
        gpu = str(int(gpu))
    logs = run / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    pending = list(jobs)
    running: dict = {}
    done: Set[str] = set(satisfied)
    finished: List[str] = []
    failed: List[str] = []
    low_disk = False
    waiting_logged = 0.0
    last_start: dict = {}
    cpu_only = gpu == "cpu"

    def record(label, item, code, note=None):
        save(logs / (label + ".exit.json"), {"argv": list(item["job"].argv), "pid": item["process"].pid, "gpu": item["gpu"],
                                             "started": item["started"], "finished": time.time(), "exit_code": code,
                                             **({"note": note} if note else {})})

    try:
        while pending or running:
            # ---- collect finished jobs
            for label, item in list(running.items()):
                code = item["process"].poll()
                if code is None:
                    continue
                item["log"].close()
                job: Job = item["job"]
                note = None
                if code == 0 and not (run / job.done).exists():
                    code, note = -1000, f"exited 0 but did not create {job.done}"
                    with (logs / (label + ".log")).open("a", encoding="utf-8") as f:
                        f.write(f"\n[gpumake] {note}\n")
                record(label, item, code, note)
                del running[label]
                minutes = (time.time() - item["started"]) / 60
                log(f"{'FAILED' if code else 'done'}: {label} ({_where(item['gpu'])}, {minutes:.1f} min)")
                if code:
                    failed.append(label)
                else:
                    done.add(label)
                    finished.append(label)
                _update_state(run, active_jobs=_active(running), completed=sorted(done),
                              **({"failure": {"job": label, "exit_code": code}} if code else {}))
            if failed and not running:
                raise JobFailed(f"{', '.join(failed)} failed; see {logs / (failed[0] + '.log')}")
            if low_disk and not running:
                raise JobFailed(f"spare-disk rule reached ({min_free_disk_fraction:.0%} must stay free); "
                                "nothing more was started and no data was deleted")
            started = False
            ready = [j for j in pending if set(j.deps) <= done]
            if not failed and not low_disk and ready and len(running) < max_jobs:
                if not space_ok(run, min_free_disk_fraction):
                    low_disk = True
                    log(f"spare-disk rule reached; starting nothing more, waiting for {len(running)} running job(s)")
                    continue
                launches = []
                ours = Counter(item["gpu"] for item in running.values() if item["gpu"] is not None)
                for item in running.values():
                    if item["job"].whole_gpu and item["gpu"] is not None:
                        ours[item["gpu"]] = max(ours[item["gpu"]], per_gpu)  # nothing joins a whole-GPU job
                for job in [j for j in ready if cpu_only or not j.gpu]:  # CPU jobs need no card
                    if len(running) + len(launches) >= max_jobs:
                        break
                    launches.append((job, None))
                gpu_ready = [j for j in ready if not (cpu_only or not j.gpu)]
                if gpu_ready and len(running) + len(launches) < max_jobs:
                    listed = devices()
                    if not listed:
                        raise JobFailed("no GPU found (nvidia-smi missing or failing); use --gpu cpu for CPU-only runs")
                    if gpu != "auto" and gpu not in {str(d["index"]) for d in listed}:
                        raise JobFailed(f"GPU {gpu} does not exist; nvidia-smi lists "
                                        + ", ".join(str(d["index"]) for d in listed))
                    free = gpu_module.slots(listed, gpu, dict(ours), min_free_mib=min_free_mib,
                                            max_utilization=max_utilization, per_gpu=per_gpu,
                                            shared_free_mib=shared_free_mib)
                    now = time.time()
                    # a card that already runs our job takes another only once that job has settled
                    free = [d for d in free if ours.get(str(d["index"]), 0) == 0
                            or now - last_start.get(str(d["index"]), 0) >= settle_seconds]
                    for device in free:
                        if len(running) + len(launches) >= max_jobs:
                            break
                        shared = ours.get(str(device["index"]), 0) > 0
                        job = next((j for j in gpu_ready if not (shared and j.whole_gpu)), None)
                        if job is not None:
                            gpu_ready.remove(job)
                            launches.append((job, str(device["index"])))
                    if not launches and not running and time.time() - waiting_logged > 600:
                        log(f"waiting for a GPU with {min_free_mib} MiB free" + ("" if gpu == "auto" else f" (GPU {gpu})"))
                        waiting_logged = time.time()
                for job, device in launches:
                    env = os.environ.copy()
                    env.update(job.env)
                    env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="" if device is None else device,
                               PYTHONUNBUFFERED="1", GPUMAKE_RUN=str(run.resolve()), GPUMAKE_PROJECT=str(project.resolve()),
                               GPUMAKE_JOB=job.label)
                    handle = (logs / (job.label + ".log")).open("a", encoding="utf-8")
                    pending.remove(job)
                    try:
                        process = launch(list(job.argv), env, handle, project)
                    except OSError as error:  # e.g. the command does not exist
                        handle.write(f"[gpumake] could not start {job.argv[0]!r}: {error}\n")
                        handle.close()
                        log(f"FAILED: {job.label} could not start ({error})")
                        failed.append(job.label)
                        _update_state(run, failure={"job": job.label, "error": str(error)})
                        continue
                    running[job.label] = {"process": process, "gpu": device, "log": handle, "started": time.time(), "job": job}
                    if device is not None:
                        last_start[device] = time.time()
                    started = True
                    log(f"start: {job.label} on {_where(device)} ({len(running)} running, {len(pending)} waiting)")
                if launches:
                    _update_state(run, status="running", active_jobs=_active(running))
            elif not ready and not running and pending:
                raise JobFailed("unsatisfiable dependencies: " + ", ".join(j.label for j in pending))
            save(run / "heartbeat.json", {"pid": os.getpid(), "time": time.time(), "running": _active(running),
                                          "waiting": [j.label for j in pending]})
            if not started and (pending or running):
                time.sleep(poll)
    finally:
        if running:  # leaving abnormally: stop our own jobs so a rerun does not duplicate them
            for label, item in running.items():
                _stop(item["process"])
                item["log"].close()
                record(label, item, None, "stopped because gpumake itself stopped")
                log(f"stopped: {label}")
            _update_state(run, active_jobs=[])
            save(run / "heartbeat.json", {"pid": os.getpid(), "time": time.time(), "running": [], "waiting": []})
    return finished


def _stop(process) -> None:
    try:
        process.terminate()
        process.wait(timeout=30)
    except Exception:  # noqa: BLE001 - best effort; the process may already be gone
        try:
            process.kill()
        except Exception:  # noqa: BLE001
            pass


def _launch(argv, env, handle, cwd):
    return subprocess.Popen(argv, cwd=str(cwd), env=env, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT)


def _active(running: dict) -> list:
    return [{"job": label, "pid": item["process"].pid, "gpu": item["gpu"]} for label, item in running.items()]


def _where(device) -> str:
    return "CPU" if device is None else f"GPU {device}"
