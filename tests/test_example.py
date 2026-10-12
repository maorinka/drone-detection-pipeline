import json
import shutil
from pathlib import Path

from gpumake.engine import Project, execute

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "toy"


def test_toy_example_end_to_end(tmp_path):
    project = tmp_path / "toy"
    shutil.copytree(EXAMPLE, project, ignore=shutil.ignore_patterns("runs", "make", "__pycache__"))
    config = json.loads((project / "make_config.json").read_text())
    config["min_free_disk_fraction"] = 0
    (project / "make_config.json").write_text(json.dumps(config))
    result = execute(Project(project), gpu="cpu", poll=0.05)
    report = json.loads((project / "runs" / result["revision"] / "report" / "report.json").read_text())
    assert len(report["models"]) == 3 and abs(report["mean_w"] - 51 / 8) < 1e-6
    config["lr"] = 0.2
    (project / "make_config.json").write_text(json.dumps(config))
    again = execute(Project(project), gpu="cpu", poll=0.05)
    assert again["mode"] == "update" and again["stale_units"] == ["model", "report"]
