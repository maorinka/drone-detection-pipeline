import json
import os
import subprocess
import sys
import textwrap

import pytest

from conftest import count
from gpumake import gpu as gpu_module
from gpumake.engine import MakeError, Project, execute, plan
from gpumake.scheduler import JobFailed, run_jobs
from gpumake.spec import Job
from test_scheduler import FakeProcess, device


def test_nvidia_smi_values_it_cannot_report_make_a_card_unusable(monkeypatch):
    class Result:
        stdout = "0, 11000, 11264, [N/A], 00000000:01:00.0\n1, 9000, 11264, 3, 00000000:02:00.0\n"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: Result())
    devices = gpu_module.list_devices()
    assert devices[0]["utilization_percent"] == 100 and devices[1]["utilization_percent"] == 3
    assert [d["index"] for d in gpu_module.slots(devices, "auto", {})] == [1]


def test_launch_error_fails_the_job_cleanly(tmp_path):
    def launch(argv, env, handle, cwd):
        if env["GPUMAKE_JOB"] == "bad":
            raise FileNotFoundError("no such program")
        return FakeProcess(tmp_path / "run" / f"{env['GPUMAKE_JOB']}.done")
    jobs = [Job("good", ("x",), done="good.done", gpu=False), Job("bad", ("x",), done="bad.done", gpu=False)]
    with pytest.raises(JobFailed):
        run_jobs(tmp_path / "run", tmp_path, jobs, launch=launch, poll=0, min_free_disk_fraction=0)
    assert json.loads((tmp_path / "run" / "logs" / "good.exit.json").read_text())["exit_code"] == 0


def test_no_gpu_or_unknown_index_fails_instead_of_waiting(tmp_path):
    jobs = [Job("g", ("x",), done="g.done")]
    with pytest.raises(JobFailed, match="no GPU"):
        run_jobs(tmp_path / "run", tmp_path, jobs, devices=lambda: [], poll=0, min_free_disk_fraction=0)
    with pytest.raises(JobFailed, match="does not exist"):
        run_jobs(tmp_path / "run2", tmp_path, jobs, gpu="07", devices=lambda: [device(0)], poll=0, min_free_disk_fraction=0)


def test_a_card_settles_before_it_takes_a_second_job(tmp_path):
    starts = []

    class Slow(FakeProcess):
        def poll(self):
            self.polls += 1
            if self.polls < 4:
                return None
            self.marker.parent.mkdir(parents=True, exist_ok=True)
            self.marker.write_text("ok")
            return 0

    def launch(argv, env, handle, cwd):
        starts.append(env["GPUMAKE_JOB"])
        return Slow(tmp_path / "run" / f"{env['GPUMAKE_JOB']}.done")
    jobs = [Job("a", ("x",), done="a.done"), Job("b", ("x",), done="b.done")]
    run_jobs(tmp_path / "run", tmp_path, jobs, devices=lambda: [device(0)], launch=launch, poll=0.01, per_gpu=2,
             settle_seconds=60, min_free_disk_fraction=0)
    # b could share the card (per_gpu=2) but the card had not settled, so b waited for a to finish
    a_exit = json.loads((tmp_path / "run" / "logs" / "a.exit.json").read_text())["finished"]
    b_start = json.loads((tmp_path / "run" / "logs" / "b.exit.json").read_text())["started"]
    assert b_start >= a_exit


def test_output_patterns_never_touch_bookkeeping(project):
    pipeline = project / "pipeline.py"
    pipeline.write_text(pipeline.read_text().replace('Unit("c", files=("c.py",), outputs=("c",))',
                                                     'Unit("c", files=("c.py",), outputs=("*",))'))
    first = execute(Project(project), gpu="cpu", poll=0.05)
    (project / "c.py").write_text((project / "c.py").read_text() + "\n#x\n")
    second = execute(Project(project), gpu="cpu", poll=0.05)
    assert second["stale_units"] == ["c"]
    run = project / "runs" / first["revision"]
    assert (run / "inputs.json").exists() and (run / "logs").is_dir()


def test_resume_clears_old_completion(project, monkeypatch):
    first = execute(Project(project), gpu="cpu", poll=0.05)
    run = project / "runs" / first["revision"]
    (run / "c" / "done").unlink()
    monkeypatch.setenv("FAIL_C", "1")
    with pytest.raises(JobFailed):
        execute(Project(project), gpu="cpu", poll=0.05)
    assert not (run / "complete.json").exists()
    assert json.loads((run / "state.json").read_text())["status"] == "failed"


def test_refuses_to_start_beside_jobs_of_a_killed_run(project):
    first = execute(Project(project), gpu="cpu", poll=0.05)
    run = project / "runs" / first["revision"]
    (run / "c" / "done").unlink()
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        (run / "heartbeat.json").write_text(json.dumps({"running": [{"job": "c", "pid": sleeper.pid, "gpu": None}]}))
        with pytest.raises(MakeError, match="still be running"):
            execute(Project(project), gpu="cpu", poll=0.05)
    finally:
        sleeper.kill()
        sleeper.wait()
    assert execute(Project(project), gpu="cpu", poll=0.05)["mode"] == "resume"


def test_nothing_mode_records_the_current_revision(project):
    first = execute(Project(project), gpu="cpu", poll=0.05)
    (project / "data" / "y.txt").write_text("2")
    execute(Project(project), gpu="cpu", poll=0.05)
    (project / "data" / "y.txt").unlink()
    back = execute(Project(project), gpu="cpu", poll=0.05)
    assert back["mode"] == "nothing"
    assert json.loads((project / "make" / "state.json").read_text())["run"] == first["revision"]
