"""Example pipeline: per-file statistics -> one small model per seed -> a report.

Everything is plain Python so it runs anywhere (use ``--gpu cpu`` on a machine without GPUs).
Try: change a number in data/, a line in scripts/train.py, or "lr" in make_config.json, then
``python -m gpumake plan`` and see what would be rebuilt.
"""
import sys

from gpumake import Job, Unit

# Config keys that define the data revision (a change makes a new runs/<revision>).
DATA_SETTINGS = ("seeds",)


def inputs(project, config):
    return sorted((project / "data").glob("*.csv"))


def units(project, config):
    return [
        Unit("features", files=("scripts/features.py",), outputs=("features",)),
        # A hyperparameter in `extra`: changing "lr" rebuilds the models (and the report), not the features.
        Unit("model", files=("scripts/train.py",), extra={"lr": config.get("lr", 0.1)}, outputs=("models",),
             after=("features",)),
        Unit("report", files=("scripts/report.py",), outputs=("report",), after=("model",)),
    ]


def jobs(run, project, config):
    py = sys.executable
    out = []
    for path in inputs(project, config):
        out.append(Job(f"features-{path.stem}", (py, "scripts/features.py", str(path), str(run / "features" / f"{path.stem}.json")),
                       done=f"features/{path.stem}.json", gpu=False))
    features = tuple(j.label for j in out)
    trains = []
    for seed in config.get("seeds", [0, 1, 2]):
        label = f"train-seed{seed}"
        out.append(Job(label, (py, "scripts/train.py", str(run), str(seed), str(config.get("lr", 0.1))),
                       done=f"models/seed{seed}/done.json", deps=features))
        trains.append(label)
    out.append(Job("report", (py, "scripts/report.py", str(run)), done="report/report.json", deps=tuple(trains), gpu=False))
    return out
