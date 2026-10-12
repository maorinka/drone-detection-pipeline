"""The pipeline contract: a project describes its data, units and jobs in a ``pipeline.py``.

A pipeline module defines three functions (and optionally ``DATA_SETTINGS``)::

    DATA_SETTINGS = ("seed", "folds")          # config keys that change the data revision

    def inputs(project, config):               # -> iterable of data file paths
        ...

    def units(project, config):                # -> iterable of Unit
        ...

    def jobs(run, project, config):            # -> iterable of Job (the whole graph)
        ...

gpumake hashes the inputs (plus the DATA_SETTINGS values) into a *data revision* and builds
it in ``runs/<revision>/``. Each Unit is a piece of work whose result depends on some code
files; when their contents change, the unit's outputs inside the current run are moved to
``superseded/`` and only its jobs run again. A job is skipped when its ``done`` marker exists.
"""
from __future__ import annotations

import importlib.util
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,119}$")


@dataclass(frozen=True)
class Unit:
    """Something whose outputs must be rebuilt when its code changes.

    files:   code/config files (relative to the project) whose contents define the unit.
    extra:   any JSON value that also defines it (a library version, a checkpoint hash).
    outputs: glob patterns, relative to the run folder, of everything the unit produces
             (including its jobs' ``done`` markers). They are moved aside when it is stale.
    after:   units whose staleness also makes this one stale (e.g. reports after training).
    """
    name: str
    files: Tuple[str, ...] = ()
    extra: object = None
    outputs: Tuple[str, ...] = ()
    after: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Job:
    """One process to run, typically on one GPU.

    argv:      the command; run with the project as working directory.
    deps:      labels of jobs that must finish first.
    done:      path relative to the run folder that the job creates when it succeeds;
               if it already exists the job is skipped (this is how runs resume).
    gpu:       False for CPU-only jobs (no GPU is reserved for them).
    whole_gpu: the job gets a GPU to itself (nothing else starts beside it).
    env:       extra environment variables.
    """
    label: str
    argv: Tuple[str, ...]
    done: str
    deps: Tuple[str, ...] = ()
    gpu: bool = True
    whole_gpu: bool = False
    env: Dict[str, str] = field(default_factory=dict)


@dataclass
class Pipeline:
    inputs: Callable
    units: Callable
    jobs: Callable
    data_settings: Sequence[str] = ()
    path: Optional[Path] = None


class SpecError(ValueError):
    pass


def load(path) -> Pipeline:
    """Import a pipeline module from a file path."""
    path = Path(path).resolve()
    if not path.is_file():
        raise SpecError(f"pipeline file not found: {path}")
    name = "gpumake_pipeline_" + re.sub(r"\W", "_", path.stem)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
    for required in ("inputs", "units", "jobs"):
        if not callable(getattr(module, required, None)):
            raise SpecError(f"{path.name} must define a function {required}()")
    return Pipeline(module.inputs, module.units, module.jobs, tuple(getattr(module, "DATA_SETTINGS", ())), path)


def check_units(units: List[Unit]) -> Dict[str, Unit]:
    by_name = {}
    for unit in units:
        if not isinstance(unit, Unit):
            raise SpecError(f"units() must return Unit objects, got {type(unit).__name__}")
        if not LABEL.match(unit.name) or unit.name in by_name:
            raise SpecError(f"invalid or duplicate unit name: {unit.name!r}")
        by_name[unit.name] = unit
    for unit in units:
        for other in unit.after:
            if other not in by_name:
                raise SpecError(f"unit {unit.name!r} is after unknown unit {other!r}")
    return by_name


def check_jobs(jobs: List[Job]) -> List[Job]:
    labels = set()
    for job in jobs:
        if not isinstance(job, Job):
            raise SpecError(f"jobs() must return Job objects, got {type(job).__name__}")
        if not LABEL.match(job.label) or job.label in labels:
            raise SpecError(f"invalid or duplicate job label: {job.label!r}")
        if not job.argv or not all(isinstance(a, str) for a in job.argv):
            raise SpecError(f"job {job.label!r} needs a non-empty argv of strings")
        if not job.done or Path(job.done).is_absolute() or ".." in Path(job.done).parts:
            raise SpecError(f"job {job.label!r}: done must be a path inside the run folder")
        labels.add(job.label)
    for job in jobs:
        for dep in job.deps:
            if dep not in labels:
                raise SpecError(f"job {job.label!r} depends on unknown job {dep!r}")
    _check_acyclic(jobs)
    return jobs


def _check_acyclic(jobs: List[Job]) -> None:
    deps = {j.label: set(j.deps) for j in jobs}
    done: set = set()
    while deps:
        ready = [label for label, d in deps.items() if d <= done]
        if not ready:
            raise SpecError("job dependencies form a cycle: " + ", ".join(sorted(deps)))
        for label in ready:
            done.add(label)
            del deps[label]
