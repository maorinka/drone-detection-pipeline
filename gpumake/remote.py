"""Drive gpumake on a GPU server from a workstation over SSH.

    python -m gpumake.remote enroll            # trust the server's host key (type its fingerprint)
    python -m gpumake.remote sync              # upload changed code (by sha256)
    python -m gpumake.remote plan              # sync, then show the plan on the server
    python -m gpumake.remote run [--gpu auto] [--follow]
    python -m gpumake.remote status [JOB]      # state + log tail (default: the latest job)
    python -m gpumake.remote follow [JOB]      # print the log until the job ends

Settings live in ``remote.json`` next to the project (never commit it). A run is started as a
detached supervisor on the server (its own session), so closing this window, losing the VPN or
sleeping the workstation does not stop it. Authentication: SSH keys/agent, or a password from
the GPUMAKE_SSH_PASSWORD environment variable, or ``--ask-password``. Unknown host keys are
rejected; ``enroll`` once adds the server's key to ``known_hosts`` (a file in the project by
default; your ~/.ssh/known_hosts is also read but never written).
"""
from __future__ import annotations

import argparse
import base64
import codecs
import getpass
import glob
import hashlib
import json
import os
import posixpath
import re
import shlex
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Tuple

from .util import sha

JOB_ID = re.compile(r"^[0-9a-f]{32}$")
SMALL_SYNC_BYTES = 64 * 1024 ** 2
RESERVE_BYTES = 1 * 1024 ** 3
PACKAGE = Path(__file__).resolve().parent


class RemoteError(RuntimeError):
    pass


@dataclass(frozen=True)
class RemoteConfig:
    host: str
    user: str
    remote_root: str
    port: int = 22
    python: str = "python3"
    key: Optional[str] = None
    known_hosts: str = "known_hosts"
    sync: Tuple[str, ...] = ("pipeline.py", "make_config.json")
    min_free_disk_fraction: float = 0.30
    env: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, path) -> "RemoteConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict):
            raise RemoteError("remote config must be a JSON object")
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise RemoteError("unsupported remote config fields: " + ", ".join(sorted(unknown)))
        if "sync" in data:
            data["sync"] = tuple(data["sync"])
        try:
            config = cls(**data)
        except TypeError as exc:
            raise RemoteError(f"invalid remote config: {exc}") from None
        config.validate()
        return config

    def validate(self) -> None:
        for name in ("host", "user"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or any(c.isspace() for c in value):
                raise RemoteError(f"{name} must be a non-empty string without spaces")
        if not isinstance(self.port, int) or isinstance(self.port, bool) or not 1 <= self.port <= 65535:
            raise RemoteError("port must be between 1 and 65535")
        root = PurePosixPath(self.remote_root) if isinstance(self.remote_root, str) else None
        if root is None or not root.is_absolute() or ".." in root.parts or "\\" in self.remote_root or root == PurePosixPath("/"):
            raise RemoteError("remote_root must be an absolute POSIX path (not /) without '..'")
        if not self.python or "\x00" in self.python:
            raise RemoteError("python must name the server's interpreter")
        for pattern in self.sync:
            if not isinstance(pattern, str) or not pattern or Path(pattern).is_absolute() or ".." in Path(pattern).parts:
                raise RemoteError(f"sync patterns must be relative and stay inside the project: {pattern!r}")
        if not all(isinstance(k, str) and k.isidentifier() and isinstance(v, str) for k, v in self.env.items()):
            raise RemoteError("env must map variable names to strings")

    def path(self, relative: str) -> str:
        return posixpath.join(self.remote_root, safe_relative(relative))

    def known_hosts_path(self, project: Path) -> Path:
        path = Path(self.known_hosts).expanduser()
        return path if path.is_absolute() else Path(project) / path


def safe_relative(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise RemoteError("remote path must be a non-empty POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in (".", "..") for part in value.split("/")):
        raise RemoteError(f"unsafe remote path: {value!r}")
    return str(path)


# --------------------------------------------------------------------------- what to upload

def manifest(project: Path, config: RemoteConfig) -> List[Tuple[Path, str]]:
    """(local file, remote relative path): the sync patterns plus this gpumake package itself."""
    project = project.resolve()
    files: Dict[str, Path] = {}
    for pattern in config.sync:
        for name in sorted(glob.glob(str(project / pattern), recursive=True)):
            path = Path(name).resolve()
            if path.is_file() and "__pycache__" not in path.parts:
                try:
                    rel = path.relative_to(project).as_posix()
                except ValueError:
                    raise RemoteError(f"sync pattern {pattern!r} reaches outside the project: {path}") from None
                files[safe_relative(rel)] = path
    for path in sorted(PACKAGE.rglob("*.py")):
        if "__pycache__" not in path.parts:
            files.setdefault("gpumake/" + path.relative_to(PACKAGE).as_posix(), path)
    return [(files[k], k) for k in sorted(files)]


INVENTORY = r"""
import hashlib, json, os, sys
root = os.path.realpath(sys.argv[1]); out = {}
os.makedirs(root, exist_ok=True)
for rel in json.loads(sys.stdin.read()):
    path = os.path.realpath(os.path.join(root, rel))
    if os.path.commonpath([root, path]) != root:
        out[rel] = 'outside'; continue
    if not os.path.isfile(path):
        continue
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''): h.update(b)
    out[rel] = h.hexdigest()
st = os.statvfs(root)
print(json.dumps({'files': out, 'free': st.f_bavail * st.f_frsize, 'total': st.f_blocks * st.f_frsize}))
"""

SUPERVISOR = r"""
import datetime, json, os, pathlib, subprocess, sys, time, traceback
job, root, python = sys.argv[1:4]; args = sys.argv[4:]
path = pathlib.Path(job); state_file = path / 'state.json'
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(state):
    tmp = path / ('.state.%d.tmp' % os.getpid()); tmp.write_text(json.dumps(state, sort_keys=True)); os.replace(str(tmp), str(state_file))
state = {'status': 'starting', 'supervisor_pid': os.getpid(), 'started_at': now(), 'child_pid': None, 'exit_code': None, 'args': args}
save(state)
try:
    env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONPATH=root + os.pathsep + os.environ.get('PYTHONPATH', ''))
    env.update(json.loads((path / 'env.json').read_text()))
    with open(str(path / 'output.log'), 'ab') as out:
        proc = subprocess.Popen([python, '-m', 'gpumake', '--project', root] + args, cwd=root, env=env,
                                stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT)
        state.update(status='running', child_pid=proc.pid, heartbeat_at=now()); save(state)
        while True:
            try:
                code = proc.wait(timeout=5); break
            except subprocess.TimeoutExpired:
                state['heartbeat_at'] = now(); save(state)
    state.update(status='finished', exit_code=code, finished_at=now()); save(state)
except BaseException as exc:
    with open(str(path / 'output.log'), 'ab') as out:
        out.write(traceback.format_exc().encode('utf-8', 'replace'))
    state.update(status='failed', exit_code=1, error=str(exc), finished_at=now()); save(state)
"""

LAUNCH = r"""
import json, pathlib, subprocess, sys
source, job, root, python, env = sys.argv[1:6]
p = pathlib.Path(job); p.mkdir(parents=True, exist_ok=False); (p / 'env.json').write_text(env)
proc = subprocess.Popen([python, '-c', source, job, root, python] + sys.argv[6:], cwd=root, stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
print(json.dumps({'pid': proc.pid}))
"""


class Session:
    """An SSH connection plus the commands gpumake needs."""

    def __init__(self, config: RemoteConfig, project: Path, password: Optional[str] = None):
        self.config = config
        self.password = password
        self.client = connect(config, config.known_hosts_path(project), password)

    def close(self) -> None:
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def run_bytes(self, command: str, input_data: Optional[bytes] = None) -> bytes:
        stdin, stdout, stderr = self.client.exec_command(command)
        if input_data is not None:
            stdin.write(input_data)
            stdin.flush()
            stdin.channel.shutdown_write()
        out = stdout.read()
        err = stderr.read().decode("utf-8", "replace")
        code = stdout.channel.recv_exit_status()
        if code:
            message = err.strip() or out.decode("utf-8", "replace").strip()
            raise RemoteError(f"remote command failed (exit {code}): {self.redact(message)}")
        return out

    def redact(self, text: str) -> str:
        return text.replace(self.password, "[REDACTED]") if self.password else text

    def run(self, command: str, input_data: Optional[bytes] = None) -> str:
        return self.redact(self.run_bytes(command, input_data).decode("utf-8", "replace"))

    def python(self, code: str, *args: str, input_data: Optional[bytes] = None) -> str:
        command = " ".join(shlex.quote(x) for x in (self.config.python, "-c", code, *args))
        return self.run(command, input_data)

    def stream(self, command: str) -> int:
        channel = self.client.get_transport().open_session()
        channel.set_combine_stderr(True)
        channel.exec_command(command)
        while True:
            data = channel.recv(4096)
            if not data:
                break
            text = data.decode("utf-8", "replace")
            if self.password:
                text = text.replace(self.password, "[REDACTED]")
            sys.stdout.write(text)
            sys.stdout.flush()
        return channel.recv_exit_status()

    # ------------------------------------------------------------------ sync
    def sync(self, project: Path) -> Dict[str, int]:
        files = manifest(project, self.config)
        remote = json.loads(self.python(INVENTORY, self.config.remote_root,
                                        input_data=json.dumps([rel for _, rel in files]).encode()))
        outside = [rel for rel, value in remote["files"].items() if value == "outside"]
        if outside:
            raise RemoteError("remote paths resolve outside remote_root: " + ", ".join(outside[:5]))
        todo = [(local, rel) for local, rel in files if remote["files"].get(rel) != sha(local)]
        size = sum(local.stat().st_size for local, _ in todo)
        keep = max(RESERVE_BYTES, int(self.config.min_free_disk_fraction * remote["total"])) if size > SMALL_SYNC_BYTES \
            else RESERVE_BYTES
        if remote["free"] - size < keep:
            raise RemoteError(f"upload of {size / 1024 ** 2:.0f} MiB would break the spare-disk rule on the server")
        sftp = self.client.open_sftp()
        try:
            for local, rel in todo:
                destination = self.config.path(rel)
                self.run("mkdir -p " + shlex.quote(posixpath.dirname(destination)))
                temporary = destination + ".part-" + uuid.uuid4().hex[:8]
                sftp.put(str(local), temporary)
                sftp.posix_rename(temporary, destination)
                print(f"uploaded {rel}", flush=True)
        finally:
            sftp.close()
        return {"files": len(files), "uploaded": len(todo), "bytes": size}

    # ------------------------------------------------------------------ jobs
    def start(self, args: List[str]) -> str:
        job_id = uuid.uuid4().hex
        job_dir = self.config.path(".gpumake/jobs/" + job_id)
        result = json.loads(self.python(LAUNCH, SUPERVISOR, job_dir, self.config.remote_root, self.config.python,
                                        json.dumps(self.config.env), *args))
        self.run("ln -sfn " + shlex.quote(job_id) + " " + shlex.quote(self.config.path(".gpumake/jobs/latest")))
        print(json.dumps({"job": job_id, "supervisor_pid": result["pid"]}), flush=True)
        return job_id

    def job(self, job_id: Optional[str]) -> str:
        if job_id is None:
            job_id = self.run("readlink " + shlex.quote(self.config.path(".gpumake/jobs/latest")) + " || true").strip()
        if not JOB_ID.match(job_id or ""):
            raise RemoteError("no such job (give the 32-character job id)")
        return job_id

    def status(self, job_id: Optional[str] = None, lines: int = 30) -> dict:
        job_id = self.job(job_id)
        folder = self.config.path(".gpumake/jobs/" + job_id)
        state = json.loads(self.run("cat " + shlex.quote(folder + "/state.json")))
        tail = self.run("tail -n %d %s 2>/dev/null || true" % (int(lines), shlex.quote(folder + "/output.log")))
        return {"job": job_id, "state": state, "log_tail": tail}

    def supervisor_alive(self, state: dict) -> bool:
        pid = state.get("supervisor_pid")
        if not isinstance(pid, int):
            return False
        return self.run("kill -0 %d 2>/dev/null && echo yes || echo no" % pid).strip() == "yes"

    def follow(self, job_id: Optional[str] = None, interval: float = 10.0) -> int:
        """Print the job's log as it grows until the job ends (or its supervisor is gone)."""
        job_id = self.job(job_id)
        folder = self.config.path(".gpumake/jobs/" + job_id)
        shown = 0
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        while True:
            data = self.run_bytes("tail -c +%d %s 2>/dev/null || true" % (shown + 1, shlex.quote(folder + "/output.log")))
            if data:
                shown += len(data)  # counted in raw bytes, so nothing is skipped or repeated
                sys.stdout.write(self.redact(decoder.decode(data)))
                sys.stdout.flush()
            state = json.loads(self.run("cat " + shlex.quote(folder + "/state.json")))
            if state.get("status") in ("finished", "failed"):
                print(f"\n[job {job_id}: {state['status']}, exit {state.get('exit_code')}]")
                return int(state.get("exit_code") or 0)
            if not self.supervisor_alive(state):
                print(f"\n[job {job_id}: its supervisor is gone (server restart or killed); state was {state.get('status')!r}]")
                return 1
            time.sleep(interval)


def _paramiko():
    try:
        import paramiko
    except ImportError:
        raise RemoteError("remote commands need paramiko: pip install paramiko") from None
    return paramiko


def connect(config: RemoteConfig, known_hosts: Path, password: Optional[str] = None):
    paramiko = _paramiko()
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    if known_hosts.exists():
        client.get_host_keys().load(str(known_hosts))  # read only: paramiko never rewrites it
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect(hostname=config.host, port=config.port, username=config.user,
                   key_filename=str(Path(config.key).expanduser()) if config.key else None,
                   password=password, look_for_keys=True, allow_agent=True,
                   timeout=15, auth_timeout=15, banner_timeout=15)
    return client


def enroll(config: RemoteConfig, known_hosts: Path) -> str:
    """Show the server's host key fingerprint; append it to ``known_hosts`` only if typed back exactly."""
    import socket
    paramiko = _paramiko()
    sock = socket.create_connection((config.host, config.port), timeout=15)
    transport = paramiko.Transport(sock)
    try:
        transport.start_client(timeout=15)
        key = transport.get_remote_server_key()
        fingerprint = "SHA256:" + base64.b64encode(hashlib.sha256(key.asbytes()).digest()).decode().rstrip("=")
        print(f"{config.host}:{config.port} {key.get_name()} {fingerprint}")
        print("Compare it with the server admin's value, then type it exactly to trust this host:", flush=True)
        if input().strip() != fingerprint:
            raise RemoteError("fingerprint not accepted; nothing was saved")
        host = config.host if config.port == 22 else f"[{config.host}]:{config.port}"
        line = f"{host} {key.get_name()} {key.get_base64()}"
        existing = known_hosts.read_text(encoding="utf-8").splitlines() if known_hosts.exists() else []
        for entry in existing:
            fields = entry.split()
            if len(fields) >= 3 and host in fields[0].split(",") and fields[1] == key.get_name():
                if fields[2] == key.get_base64():
                    return fingerprint  # already trusted
                raise RemoteError(f"{known_hosts} holds a different {key.get_name()} key for {host}; "
                                  "refusing to overwrite it (the server key changed, or this is not the same server)")
        known_hosts.parent.mkdir(parents=True, exist_ok=True)
        with known_hosts.open("a", encoding="utf-8") as f:  # append one line; nothing else in the file changes
            if existing and not known_hosts.read_text(encoding="utf-8").endswith("\n"):
                f.write("\n")
            f.write(line + "\n")
        return fingerprint
    finally:
        transport.close()
        sock.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="gpumake.remote", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", default=".", help="local project folder (default: current folder)")
    parser.add_argument("--config", help="remote settings (default: <project>/remote.json)")
    parser.add_argument("--ask-password", action="store_true", help="prompt for the SSH password")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("enroll")
    sub.add_parser("sync")
    sub.add_parser("plan")
    runner = sub.add_parser("run")
    runner.add_argument("--gpu", default="auto")
    runner.add_argument("--follow", action="store_true")
    for name in ("status", "follow"):
        p = sub.add_parser(name)
        p.add_argument("job", nargs="?")
    args = parser.parse_args(argv)
    project = Path(args.project).resolve()
    try:
        config = RemoteConfig.load(Path(args.config) if args.config else project / "remote.json")
        if args.command == "enroll":
            print("trusted " + enroll(config, config.known_hosts_path(project)))
            return 0
        password = getpass.getpass("SSH password: ") if args.ask_password else os.environ.get("GPUMAKE_SSH_PASSWORD")
        with Session(config, project, password) as session:
            if args.command == "sync":
                print(json.dumps(session.sync(project)))
            elif args.command == "plan":
                session.sync(project)
                command = "cd %s && %s -m gpumake plan" % (shlex.quote(config.remote_root), shlex.quote(config.python))
                return session.stream(command)
            elif args.command == "run":
                if args.gpu not in ("auto", "cpu") and not args.gpu.isdigit():
                    raise RemoteError("--gpu must be auto, cpu or a GPU index")
                session.sync(project)
                job_id = session.start(["run", "--gpu", args.gpu])
                return session.follow(job_id) if args.follow else 0
            elif args.command == "status":
                print(json.dumps(session.status(args.job), indent=2))
            else:
                return session.follow(args.job)
    except RemoteError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
