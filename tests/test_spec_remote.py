import json
from pathlib import Path

import pytest

from gpumake.hashing import HashCache
from gpumake.remote import RemoteConfig, RemoteError, manifest, safe_relative
from gpumake.spec import Job, SpecError, Unit, check_jobs, check_units


def test_cycle_and_unknown_dependency_are_rejected():
    with pytest.raises(SpecError):
        check_jobs([Job("a", ("x",), done="a", deps=("b",)), Job("b", ("x",), done="b", deps=("a",))])
    with pytest.raises(SpecError):
        check_jobs([Job("a", ("x",), done="a", deps=("zzz",))])
    with pytest.raises(SpecError):
        check_jobs([Job("a", ("x",), done="../escape")])


def test_unit_after_must_exist():
    with pytest.raises(SpecError):
        check_units([Unit("a", after=("missing",))])


def test_hash_cache_reuses_unchanged_files(tmp_path):
    f = tmp_path / "f"
    f.write_text("hello")
    cache = HashCache(tmp_path / "h.json")
    first = cache(f)
    cache.save()
    again = HashCache(tmp_path / "h.json")
    assert again(f) == first and again.hashed_bytes == 0
    f.write_text("hello!")
    assert again(f) != first


def config(tmp_path, **kw):
    data = {"host": "gpu.example.org", "user": "me", "remote_root": "/srv/project", "sync": ["*.py", "scripts/**/*.py"]}
    data.update(kw)
    path = tmp_path / "remote.json"
    path.write_text(json.dumps(data))
    return RemoteConfig.load(path)


def test_remote_config_validation(tmp_path):
    assert config(tmp_path).port == 22
    for bad in ({"remote_root": "relative/path"}, {"remote_root": "/"}, {"remote_root": "/a/../b"},
                {"host": "two words"}, {"sync": ["../outside/*.py"]}, {"surprise": 1}):
        with pytest.raises(RemoteError):
            config(tmp_path, **bad)


def test_safe_relative():
    assert safe_relative("a/b.py") == "a/b.py"
    for bad in ("/abs", "../x", "a/../b", "a\\b", ""):
        with pytest.raises(RemoteError):
            safe_relative(bad)


def test_manifest_includes_patterns_and_the_package(tmp_path):
    (tmp_path / "pipeline.py").write_text("")
    (tmp_path / "scripts" / "deep").mkdir(parents=True)
    (tmp_path / "scripts" / "deep" / "x.py").write_text("")
    (tmp_path / "data.bin").write_text("")
    names = [rel for _, rel in manifest(tmp_path, config(tmp_path))]
    assert "pipeline.py" in names and "scripts/deep/x.py" in names and "data.bin" not in names
    assert "gpumake/engine.py" in names and "gpumake/remote.py" in names
