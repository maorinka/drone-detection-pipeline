# gpumake

A make-style pipeline runner for **shared GPU servers**. You describe your data, your code units
and your jobs in one `pipeline.py`; gpumake works out what changed and rebuilds only that, running
independent jobs side by side on whichever GPUs are idle. A workstation can drive it over SSH; runs
are detached on the server and survive a closed laptop.

- **Data revisions.** The input files (by sha256) plus the settings you name form a *revision*.
  Each revision is built in its own folder, `runs/<revision>/`, and old revisions are kept.
- **Incremental rebuilds.** Code is grouped into *units*. When a unit's files (or an `extra` value
  such as a hyperparameter) change, its outputs in the current run move to `superseded/<time>/` and
  only its jobs, and the units declared `after` it, run again. Nothing is deleted.
- **Resume for free.** Every job declares a `done` marker. A job whose marker exists is skipped, so
  rerunning after a crash, a reboot or a failed job continues where it stopped.
- **Polite GPU scheduling.** With `--gpu auto` a job starts only on a card that is idle
  (utilisation ≤ 20 %) with enough free memory (8 GiB by default); `--gpu N` uses that card
  whatever its load, if it has the memory. Up to `jobs_per_gpu` jobs can share a card, and a card
  takes a second job only after the first has had `settle_seconds` to allocate its memory.
  `max_parallel_jobs` caps the total. `whole_gpu` jobs get a card to themselves, and CPU jobs need
  none. Other users' processes are never touched; a card `nvidia-smi` cannot fully report is treated
  as busy. If no GPU exists (or the requested index doesn't), the run stops with an error instead
  of waiting.
- **Spare-disk rule.** Before starting each job, gpumake checks the run's disk. If less than
  `min_free_disk_fraction` of it is free (30 % by default; on disks over 40 GiB at least 10 GiB),
  nothing new starts; running jobs finish.
- **Clean stops.** If gpumake itself is stopped (Ctrl-C, SIGTERM), it stops the jobs it started and
  records them, so a rerun resumes instead of running duplicates. If it was killed hard and its jobs
  may still be alive, the next run refuses to start until they are gone.
- **Remote mode.** Code is synced to the server by sha256, and the run starts as a detached
  supervisor with a heartbeat. You can follow its log or check its status at any time.

Python ≥ 3.8; no dependencies on the server. `paramiko` is needed on the workstation for remote mode.

## Quick start

```bash
pip install -e .[remote,test]
cd examples/toy
python -m gpumake plan            # what would be built (writes only make/plan.json and the hash cache)
python -m gpumake run --gpu cpu   # build (auto = idle GPUs, an index = that GPU, cpu = no GPU)
python -m gpumake run --gpu cpu   # "Nothing to do."
sed -i 's/"lr": 0.1/"lr": 0.2/' make_config.json
python -m gpumake plan            # stale units: model, report (features are kept)
```

## The pipeline contract

`pipeline.py` defines three functions and, optionally, `DATA_SETTINGS`:

```python
from gpumake import Job, Unit

DATA_SETTINGS = ("seeds",)              # config keys that change the data revision

def inputs(project, config):            # data files (hashed; a change = a new revision)
    return sorted((project / "data").glob("*.csv"))

def units(project, config):             # what to rebuild when code changes
    return [
        Unit("features", files=("scripts/features.py",), outputs=("features",)),
        Unit("model", files=("scripts/train.py",), extra={"lr": config["lr"]},
             outputs=("models",), after=("features",)),
        Unit("report", files=("scripts/report.py",), outputs=("report",), after=("model",)),
    ]

def jobs(run, project, config):         # the whole graph; done jobs are skipped
    return [
        Job("features-a", ("python", "scripts/features.py", "data/a.csv", str(run / "features/a.json")),
            done="features/a.json", gpu=False),
        Job("train", ("python", "scripts/train.py", str(run)), done="models/done.json",
            deps=("features-a",)),
        ...
    ]
```

| Field | Meaning |
| --- | --- |
| `Unit.files` | Files (relative to the project) whose contents define the unit |
| `Unit.extra` | Any JSON value that also defines it: a hyperparameter, a library version, a checkpoint hash |
| `Unit.outputs` | Glob patterns (relative to the run folder) of everything the unit writes, including its jobs' `done` markers. gpumake's own files (`inputs.json`, `state.json`, `logs/`, `superseded/`, ...) are never matched |
| `Unit.after` | Units whose staleness makes this one stale too |
| `Job.argv` | The command, run with the project folder as working directory |
| `Job.done` | Marker the job writes when it succeeds; if it exists, the job is skipped. Exit 0 without it counts as a failure |
| `Job.deps` | Labels of jobs that must finish first |
| `Job.gpu` / `Job.whole_gpu` | `False` = CPU job (no card reserved) / a card to itself |
| `Job.env` | Extra environment variables |

Every job gets `GPUMAKE_RUN` (the run folder), `GPUMAKE_PROJECT`, `GPUMAKE_JOB`, and
`CUDA_VISIBLE_DEVICES` set to its card (with `CUDA_DEVICE_ORDER=PCI_BUS_ID`, so indices match
`nvidia-smi`; empty for CPU jobs).

## What a change triggers

| Change | Result |
| --- | --- |
| An input file added, changed or removed; a `DATA_SETTINGS` value changed | A different revision. If `runs/<revision>/` is new, every job runs there; if you return to an older revision, it is reused (and updated or resumed as needed) |
| A unit's file or `extra` changed | Inside the current revision, that unit's outputs (and those of units `after` it) move to `superseded/<time>/`; their jobs rerun |
| A `done` marker missing (crash, failure, deleted) | Only those jobs run (`resume`) |
| Nothing | "Nothing to do." |

## Folder layout

```
make_config.json         settings (below)
pipeline.py              the contract above
make/                    state.json (last run, input hashes, unit identities, history),
                         plan.json, cache/hashes.json (hash cache by size + mtime), make.lock
runs/<revision>/         inputs.json (frozen), make_units.json, state.json, heartbeat.json,
                         logs/<job>.log + .exit.json, complete.json, superseded/<time>/, your outputs
```

`runs/` may be a symlink to a bigger disk.

## `make_config.json`

| Key | Default | Meaning |
| --- | --- | --- |
| `pipeline` | `pipeline.py` | The pipeline file |
| `runs_dir`, `state_dir` | `runs`, `make` | Where runs and state live |
| `max_parallel_jobs` | 8 | Jobs at once (all cards and CPU) |
| `jobs_per_gpu` | 1 | Jobs that may share one card |
| `min_free_mib` | 8192 | Free memory a card needs before its first job |
| `shared_free_mib` | 3072 | Free memory needed to add a job beside ours |
| `settle_seconds` | 30 | Wait after starting a job on a card before adding another beside it |
| `max_utilization_percent` | 20 | A card busier than this is someone else's; not used |
| `min_free_disk_fraction` | 0.30 | Spare-disk rule (0 disables) |
| `revision_length` | 20 | Characters of the revision hash |

Any other keys are yours: read them in `pipeline.py`, and name the ones that define the data in
`DATA_SETTINGS`.

## Remote mode (workstation → GPU server)

Copy `examples/toy/remote.example.json` to `remote.json` in your project and fill it in (keep it out
of git). Then:

```bash
python -m gpumake.remote enroll          # once: compare and type the host key fingerprint
python -m gpumake.remote plan            # sync code, show the plan from the server
python -m gpumake.remote run --follow    # sync, start detached, print the log
python -m gpumake.remote status          # state + log tail of the latest job
python -m gpumake.remote follow [JOB]
```

- **Sync** uploads the files matching `sync` patterns plus the gpumake package itself, only when
  their sha256 differs. Each file is written to a temporary name and renamed into place.
- **Jobs** live in `<remote_root>/.gpumake/jobs/<id>/` (`state.json`, `output.log`).
  `.gpumake/jobs/latest` points to the newest one.
- **Authentication** uses SSH keys or the agent, or `GPUMAKE_SSH_PASSWORD`, or `--ask-password`.
  Unknown host keys are rejected. `enroll` appends the server's key to `known_hosts` (a file in
  the project by default, kept out of git); `~/.ssh/known_hosts` is read but never written.
- **`follow`** stops when the job ends, or reports it if the job's supervisor has disappeared
  (server restart).

## Tests

```bash
python -m pytest -q
```
