import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PIPELINE = '''
import sys
from gpumake import Job, Unit

DATA_SETTINGS = ("n",)

def inputs(project, config):
    return sorted((project / "data").glob("*.txt"))

def units(project, config):
    return [Unit("a", files=("a.py",), outputs=("a",)),
            Unit("b", files=("b.py",), extra=config.get("hp"), outputs=("b",), after=("a",)),
            Unit("c", files=("c.py",), outputs=("c",))]

def jobs(run, project, config):
    py = sys.executable
    return [Job("a", (py, "a.py"), done="a/done"),
            Job("b", (py, "b.py"), done="b/done", deps=("a",)),
            Job("c", (py, "c.py"), done="c/done", gpu=False)]
'''

SCRIPT = '''
import os, pathlib
run = pathlib.Path(os.environ["GPUMAKE_RUN"])
name = pathlib.Path(__file__).stem
counter = pathlib.Path(os.environ["GPUMAKE_PROJECT"]) / ("count_" + name)
counter.write_text(str(int(counter.read_text()) + 1 if counter.exists() else 1))
if os.environ.get("FAIL_" + name.upper()):
    raise SystemExit(3)
if not os.environ.get("NOMARK_" + name.upper()):
    (run / name).mkdir(parents=True, exist_ok=True)
    (run / name / "done").write_text("ok")
'''


@pytest.fixture
def project(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "x.txt").write_text("1")
    (tmp_path / "pipeline.py").write_text(textwrap.dedent(PIPELINE))
    for name in "abc":
        (tmp_path / f"{name}.py").write_text(SCRIPT)
    (tmp_path / "make_config.json").write_text('{"min_free_disk_fraction": 0, "n": 1, "hp": 1}')
    return tmp_path


def count(project, name):
    path = project / f"count_{name}"
    return int(path.read_text()) if path.exists() else 0
