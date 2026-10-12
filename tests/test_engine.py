import json
import os

import pytest

from conftest import count
from gpumake.engine import Project, execute, plan
from gpumake.scheduler import JobFailed


def run(project, **kw):
    return execute(Project(project), gpu="cpu", poll=0.05, **kw)


def test_new_then_nothing(project):
    first = run(project)
    assert first["mode"] == "new"
    assert [count(project, n) for n in "abc"] == [1, 1, 1]
    second = run(project)
    assert second["mode"] == "nothing" and second["revision"] == first["revision"]
    assert [count(project, n) for n in "abc"] == [1, 1, 1]
    rundir = project / "runs" / first["revision"]
    assert (rundir / "complete.json").exists()
    assert json.loads((rundir / "state.json").read_text())["status"] == "complete"


def test_data_change_makes_new_revision(project):
    first = run(project)
    (project / "data" / "y.txt").write_text("2")
    second = run(project)
    assert second["mode"] == "new" and second["revision"] != first["revision"]
    assert second["changes"]["added"] == ["data/y.txt"]
    assert (project / "runs" / first["revision"] / "complete.json").exists()  # old run kept


def test_data_setting_changes_revision_but_other_settings_do_not(project):
    first = run(project)
    config = json.loads((project / "make_config.json").read_text())
    config["max_parallel_jobs"] = 2
    (project / "make_config.json").write_text(json.dumps(config))
    assert plan(Project(project))[0]["revision"] == first["revision"]
    config["n"] = 2
    (project / "make_config.json").write_text(json.dumps(config))
    assert plan(Project(project))[0]["revision"] != first["revision"]


def test_code_change_supersedes_unit_and_followers_only(project):
    first = run(project)
    (project / "a.py").write_text((project / "a.py").read_text() + "\n# changed\n")
    result = run(project)
    assert result["mode"] == "update" and result["stale_units"] == ["a", "b"]
    assert [count(project, n) for n in "abc"] == [2, 2, 1]
    superseded = list((project / "runs" / first["revision"] / "superseded").iterdir())
    assert len(superseded) == 1 and sorted(p.name for p in superseded[0].iterdir()) == ["a", "b"]


def test_extra_value_is_part_of_unit_identity(project):
    run(project)
    config = json.loads((project / "make_config.json").read_text())
    config["hp"] = 2
    (project / "make_config.json").write_text(json.dumps(config))
    result = run(project)
    assert result["stale_units"] == ["b"]
    assert [count(project, n) for n in "abc"] == [1, 2, 1]


def test_resume_runs_only_missing_jobs(project):
    first = run(project)
    (project / "runs" / first["revision"] / "c" / "done").unlink()
    result = run(project)
    assert result["mode"] == "resume" and result["jobs_to_run"] == ["c"]
    assert [count(project, n) for n in "abc"] == [1, 1, 2]


def test_failure_stops_dependents_and_marks_run(project, monkeypatch):
    monkeypatch.setenv("FAIL_A", "1")
    with pytest.raises(JobFailed):
        run(project)
    assert count(project, "b") == 0
    revision = plan(Project(project))[0]["revision"]
    state = json.loads((project / "runs" / revision / "state.json").read_text())
    assert state["status"] == "failed" and state["failure"]["job"] == "a"
    monkeypatch.delenv("FAIL_A")
    assert run(project)["mode"] == "resume"
    assert count(project, "b") == 1


def test_exit_zero_without_marker_is_a_failure(project, monkeypatch):
    monkeypatch.setenv("NOMARK_C", "1")
    with pytest.raises(JobFailed):
        run(project)
