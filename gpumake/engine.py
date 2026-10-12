"""Plan and run: data revision, unit identities, superseding, then the job graph.

Layout inside the project (paths configurable in ``make_config.json``)::

    make/                     state.json (last run, input hashes, unit identities, history),
                              plan.json, cache/hashes.json, make.lock
    runs/<revision>/          inputs.json (frozen: input hashes + data settings),
                              make_units.json, state.json, heartbeat.json, logs/, complete.json,
                              superseded/<time>/ (outputs of stale units, moved, never deleted)
                              ... and whatever the jobs write
"""
from __future__ import annotations

import os
import shutil
import signal
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from . import scheduler
from .hashing import HashCache
from .spec import Job, Pipeline, Unit, check_jobs, check_units, load
from .util import identity, lock, log, read, save, sha

DEFAULTS = {
    "pipeline": "pipeline.py",
    "runs_dir": "runs",
    "state_dir": "make",
    "max_parallel_jobs": 8,
    "jobs_per_gpu": 1,
    "min_free_mib": 8192,
    "shared_free_mib": 3072,
    "max_utilization_percent": 20,
    "min_free_disk_fraction": 0.30,
    "settle_seconds": 30,
    "revision_length": 20,
}

# Top-level names inside a run folder that belong to gpumake; unit output patterns never match them.
RESERVED = frozenset({"inputs.json", "make_units.json", "state.json", "heartbeat.json", "complete.json", "logs",
                      "superseded"})


class MakeError(RuntimeError):
    pass


def load_config(project: Path, path: Optional[Path] = None) -> dict:
    path = Path(path) if path else project / "make_config.json"
    config = dict(DEFAULTS)
    if path.exists():
        user = read(path)
        if not isinstance(user, dict):
            raise MakeError(f"{path} must hold a JSON object")
        config.update(user)
    return config


class Project:
    def __init__(self, root, config: Optional[dict] = None, config_path=None):
        self.root = Path(root).resolve()
        self.config = config if config is not None else load_config(self.root, config_path)
        self.state_dir = self.root / self.config["state_dir"]
        self.runs_dir = self.root / self.config["runs_dir"]
        self._pipeline: Optional[Pipeline] = None

    @property
    def pipeline(self) -> Pipeline:
        if self._pipeline is None:
            self._pipeline = load(self.root / self.config["pipeline"])
        return self._pipeline

    def state(self) -> dict:
        path = self.state_dir / "state.json"
        return read(path) if path.exists() else {}

    # ------------------------------------------------------------------ identities
    def relative(self, path) -> str:
        path = Path(path)
        if not path.is_absolute():
            path = self.root / path
        try:
            return path.resolve().relative_to(self.root).as_posix()
        except ValueError:
            return str(path.resolve())

    def scan_inputs(self, hashes: HashCache) -> Dict[str, str]:
        files = {}
        for item in self.pipeline.inputs(self.root, self.config):
            path = Path(item) if Path(item).is_absolute() else self.root / item
            if not path.is_file():
                raise MakeError(f"input is not a file: {path}")
            files[self.relative(path)] = hashes(path)
        return dict(sorted(files.items()))

    def data_settings(self) -> dict:
        return {k: self.config.get(k) for k in self.pipeline.data_settings}

    def units(self) -> Dict[str, Unit]:
        return check_units(list(self.pipeline.units(self.root, self.config)))

    def unit_identities(self, units: Dict[str, Unit]) -> Dict[str, str]:
        result = {}
        for name, unit in units.items():
            files = {}
            for f in unit.files:
                path = self.root / f
                files[f] = sha(path) if path.is_file() else None
            result[name] = identity({"files": files, "extra": unit.extra})
        return result

    def jobs(self, run: Path) -> List[Job]:
        return check_jobs(list(self.pipeline.jobs(run, self.root, self.config)))


def stale_closure(changed: Set[str], units: Dict[str, Unit]) -> List[str]:
    """Changed units plus every unit that is (transitively) after one of them."""
    stale = set(changed)
    grew = True
    while grew:
        grew = False
        for unit in units.values():
            if unit.name not in stale and stale & set(unit.after):
                stale.add(unit.name)
                grew = True
    return sorted(stale)


def unit_outputs(run: Path, unit: Unit) -> List[Path]:
    """Existing paths a unit's output patterns match, never gpumake's own bookkeeping."""
    found = []
    for pattern in unit.outputs:
        for path in run.glob(pattern):
            if path.exists() and path.relative_to(run).parts[0] not in RESERVED:
                found.append(path)
    # drop paths nested inside another match (moving the parent moves them)
    found = sorted(set(found), key=lambda p: len(p.parts))
    kept: List[Path] = []
    for path in found:
        if not any(_inside(path, parent) for parent in kept):
            kept.append(path)
    return kept


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def plan(project: Project) -> Tuple[dict, dict]:
    """What a run would do. Returns (printable result, context for execute)."""
    hashes = HashCache(project.state_dir / "cache" / "hashes.json")
    files = project.scan_inputs(hashes)
    settings = project.data_settings()
    revision = identity({"inputs": files, "settings": settings})[: int(project.config["revision_length"])]
    units = project.units()
    unit_ids = project.unit_identities(units)
    previous = project.state()
    before = previous.get("files", {})
    changes = {"added": sorted(set(files) - set(before)),
               "changed": sorted(f for f in files if f in before and before[f] != files[f]),
               "deleted": sorted(set(before) - set(files))}
    run = project.runs_dir / revision
    exists = (run / "inputs.json").exists()
    recorded = read(run / "make_units.json") if (run / "make_units.json").exists() else {}
    stale = stale_closure({u for u in unit_ids if recorded.get(u) != unit_ids[u]}, units) if exists else []
    jobs = project.jobs(run)
    moved = [p for u in stale for p in unit_outputs(run, units[u])] if exists else []
    remaining = [j.label for j in jobs if not (run / j.done).exists() or any(_inside(run / j.done, m) for m in moved)]
    if not exists:
        mode = "new"
    elif stale:
        mode = "update"
    elif remaining:
        mode = "resume"
    else:
        mode = "nothing"
    result = {"revision": revision, "run": str(run), "mode": mode, "changes": changes, "settings": settings,
              "inputs": len(files), "stale_units": stale, "jobs_to_run": remaining, "jobs_total": len(jobs),
              "time": time.time()}
    context = {"files": files, "units": units, "unit_ids": unit_ids, "hashes": hashes, "run": run, "previous": previous}
    return result, context


def print_plan(result: dict) -> None:
    c = result["changes"]
    print("=== plan ===")
    print(f"inputs: {result['inputs']} files ({len(c['added'])} added, {len(c['changed'])} changed, "
          f"{len(c['deleted'])} deleted since the last run)")
    for kind in ("added", "changed", "deleted"):
        for f in c[kind][:20]:
            print(f"  {kind}: {f}")
        if len(c[kind]) > 20:
            print(f"  ... {len(c[kind]) - 20} more {kind}")
    print(f"data revision: {result['revision']}  ({result['mode']})")
    if result["stale_units"]:
        print("stale units (outputs move to superseded/, then rebuild): " + ", ".join(result["stale_units"]))
    if result["mode"] == "nothing":
        print("Nothing to do.")
    else:
        print(f"jobs to run: {len(result['jobs_to_run'])} of {result['jobs_total']}")
        for label in result["jobs_to_run"][:40]:
            print(f"  {label}")


def supersede(run: Path, unit: Unit, stamp: str) -> List[str]:
    moved = []
    for target in unit_outputs(run, unit):
        destination = run / "superseded" / stamp / target.relative_to(run)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(target), str(destination))
        moved.append(target.relative_to(run).as_posix())
    return moved


def execute(project: Project, gpu: str = "auto", poll: float = 5.0, devices=None, launch=None) -> dict:
    """Plan, then build: create or update ``runs/<revision>`` and run the jobs that are not done."""
    project.state_dir.mkdir(parents=True, exist_ok=True)
    with lock(project.state_dir / "make.lock"):
        result, context = plan(project)
        print_plan(result)
        save(project.state_dir / "plan.json", result)
        run: Path = context["run"]
        state = context["previous"]
        if result["mode"] == "nothing":
            context["hashes"].save()
            state.update(run=result["revision"], files=context["files"], units=context["unit_ids"], complete=True)
            save(project.state_dir / "state.json", state)
            return result
        _refuse_if_jobs_alive(run)
        frozen = {"inputs": context["files"], "settings": result["settings"]}
        run.mkdir(parents=True, exist_ok=True)
        if (run / "inputs.json").exists():
            if read(run / "inputs.json") != frozen:
                raise MakeError("revision collision: runs/%s holds different inputs" % result["revision"])
        else:
            save(run / "inputs.json", frozen)
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        superseded = []
        for name in result["stale_units"]:
            superseded += supersede(run, context["units"][name], stamp)
        if (run / "complete.json").exists():
            (run / "complete.json").unlink()
        # Recorded before any job starts: an interrupted run resumes instead of superseding again.
        save(run / "make_units.json", context["unit_ids"])
        state.update(run=result["revision"], files=context["files"], units=context["unit_ids"], complete=False)
        save(project.state_dir / "state.json", state)
        context["hashes"].save()
        run_state = read(run / "state.json") if (run / "state.json").exists() else {}
        for key in ("failure", "error", "finished", "completed"):
            run_state.pop(key, None)
        run_state.update(status="running", mode=result["mode"], started=time.time(), superseded=superseded,
                         active_jobs=[])
        save(run / "state.json", run_state)
        jobs = project.jobs(run)
        todo = [j for j in jobs if not (run / j.done).exists()]
        satisfied = [j.label for j in jobs if (run / j.done).exists()]
        c = project.config
        log(f"=== {len(todo)} job(s), up to {c['max_parallel_jobs']} at once, run {run} ===")
        kwargs = {}
        if devices is not None:
            kwargs["devices"] = devices
        if launch is not None:
            kwargs["launch"] = launch
        previous_handler = _raise_on_sigterm()
        try:
            scheduler.run_jobs(run, project.root, todo, satisfied=satisfied, gpu=gpu,
                               settle_seconds=float(c["settle_seconds"]),
                               max_jobs=int(c["max_parallel_jobs"]), per_gpu=int(c["jobs_per_gpu"]),
                               min_free_mib=int(c["min_free_mib"]), shared_free_mib=int(c["shared_free_mib"]),
                               max_utilization=int(c["max_utilization_percent"]),
                               min_free_disk_fraction=float(c["min_free_disk_fraction"]), poll=poll, **kwargs)
        except BaseException as error:
            failed = read(run / "state.json")
            failed.update(status="failed", error=str(error), finished=time.time())
            save(run / "state.json", failed)
            raise
        finally:
            if previous_handler is not None:
                signal.signal(signal.SIGTERM, previous_handler)
        finished = read(run / "state.json")
        finished.update(status="complete", active_jobs=[], finished=time.time())
        save(run / "state.json", finished)
        save(run / "complete.json", {"revision": result["revision"], "units": context["unit_ids"], "time": time.time()})
        history = (state.get("history") or [])[-49:] + [{"time": time.time(), "revision": result["revision"],
                                                          "mode": result["mode"], "stale_units": result["stale_units"]}]
        state.update(complete=True, history=history, finished=time.time())
        save(project.state_dir / "state.json", state)
        log(f"complete: {run}")
        return result


def _raise_on_sigterm():
    """SIGTERM becomes an exception, so the run is recorded as stopped and its jobs are stopped too."""
    if threading.current_thread() is not threading.main_thread():
        return None

    def handler(signum, frame):
        raise SystemExit(f"stopped by signal {signum}")
    return signal.signal(signal.SIGTERM, handler)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except (PermissionError, OSError):
        return True
    return True


def _refuse_if_jobs_alive(run: Path) -> None:
    """A run killed hard (SIGKILL, power loss) may have left its jobs running: never start duplicates."""
    path = run / "heartbeat.json"
    if not path.exists() or os.name == "nt":
        return
    beat = read(path)
    alive = [j for j in beat.get("running", []) if _alive(int(j["pid"]))]
    if alive:
        raise MakeError("jobs of an earlier run may still be running: "
                        + ", ".join(f"{j['job']} (pid {j['pid']})" for j in alive)
                        + "; wait for them or stop them, then run again")


def status(project: Project) -> dict:
    state = project.state()
    out = {"last_run": state.get("run"), "complete": state.get("complete")}
    if state.get("run"):
        run = project.runs_dir / state["run"]
        for name in ("state.json", "heartbeat.json"):
            if (run / name).exists():
                out[name[:-5]] = read(run / name)
    return out
