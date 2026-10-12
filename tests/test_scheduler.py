from pathlib import Path

from gpumake.gpu import slots
from gpumake.scheduler import run_jobs
from gpumake.spec import Job


class FakeProcess:
    """Finishes on the second poll and writes its done marker."""
    count = 0

    def __init__(self, marker):
        FakeProcess.count += 1
        self.pid = 1000 + FakeProcess.count
        self.marker = marker
        self.polls = 0

    def poll(self):
        self.polls += 1
        if self.polls < 2:
            return None
        self.marker.parent.mkdir(parents=True, exist_ok=True)
        self.marker.write_text("ok")
        return 0


def device(index, free=11000, util=0):
    return {"index": index, "free_mib": free, "total_mib": 11264, "utilization_percent": util}


def test_slots_skip_busy_and_low_memory_cards():
    devices = [device(0, util=90), device(1, free=2000), device(2), device(3)]
    assert [d["index"] for d in slots(devices, "auto", {})] == [2, 3]


def test_slots_share_up_to_per_gpu():
    devices = [device(0, free=6000), device(1)]
    assert [d["index"] for d in slots(devices, "auto", {"0": 1}, per_gpu=2)] == [1, 0]
    assert [d["index"] for d in slots(devices, "auto", {"0": 2}, per_gpu=2)] == [1]


def test_requested_index_ignores_utilisation_but_not_memory():
    devices = [device(0, util=90), device(1)]
    assert [d["index"] for d in slots(devices, "0", {})] == [0]
    assert slots([device(0, free=100)], "0", {}) == []


def test_jobs_spread_over_gpus_and_respect_dependencies(tmp_path):
    placed = {}

    def launch(argv, env, handle, cwd):
        placed[env["GPUMAKE_JOB"]] = env["CUDA_VISIBLE_DEVICES"]
        return FakeProcess(tmp_path / "run" / f"{env['GPUMAKE_JOB']}.done")

    jobs = [Job(f"j{i}", ("x",), done=f"j{i}.done") for i in range(3)] + \
           [Job("after", ("x",), done="after.done", deps=("j0", "j1", "j2"), gpu=False)]
    finished = run_jobs(tmp_path / "run", tmp_path, jobs, devices=lambda: [device(0), device(1), device(2)],
                        launch=launch, poll=0, settle_seconds=0, min_free_disk_fraction=0)
    assert sorted(placed[f"j{i}"] for i in range(3)) == ["0", "1", "2"]
    assert placed["after"] == "" and finished[-1] == "after"


def test_whole_gpu_job_gets_a_card_to_itself(tmp_path):
    seen = []

    def launch(argv, env, handle, cwd):
        seen.append((env["GPUMAKE_JOB"], env["CUDA_VISIBLE_DEVICES"]))
        return FakeProcess(tmp_path / "run" / f"{env['GPUMAKE_JOB']}.done")

    jobs = [Job("big", ("x",), done="big.done", whole_gpu=True), Job("small", ("x",), done="small.done")]
    run_jobs(tmp_path / "run", tmp_path, jobs, devices=lambda: [device(0)], launch=launch, poll=0, settle_seconds=0,
             per_gpu=2, min_free_disk_fraction=0)
    assert seen[0] == ("big", "0") and seen[1] == ("small", "0")
    # small started only after big finished: two separate scheduling passes on the single card


def test_max_jobs_caps_concurrency(tmp_path):
    active, peak = [], [0]

    class Tracked(FakeProcess):
        def poll(self):
            code = super().poll()
            if code is not None and self in active:
                active.remove(self)
            return code

    def launch(argv, env, handle, cwd):
        p = Tracked(tmp_path / "run" / f"{env['GPUMAKE_JOB']}.done")
        active.append(p)
        peak[0] = max(peak[0], len(active))
        return p

    jobs = [Job(f"j{i}", ("x",), done=f"j{i}.done", gpu=False) for i in range(6)]
    run_jobs(tmp_path / "run", tmp_path, jobs, launch=launch, poll=0, settle_seconds=0, max_jobs=2, min_free_disk_fraction=0)
    assert peak[0] == 2
