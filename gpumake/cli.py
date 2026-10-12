"""Command line on the machine that has the GPUs (run from the project folder).

    python -m gpumake plan                 # what would be built; changes nothing
    python -m gpumake run [--gpu auto|N|cpu]
    python -m gpumake status
"""
from __future__ import annotations

import argparse
import json
import sys

from .engine import MakeError, Project, execute, plan, print_plan, status
from .scheduler import JobFailed
from .spec import SpecError
from .util import LockError, python_version_ok, save


def main(argv=None) -> int:
    python_version_ok()
    parser = argparse.ArgumentParser(prog="gpumake", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", default=".", help="project folder (default: current folder)")
    parser.add_argument("--config", help="make_config.json (default: <project>/make_config.json)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan", help="scan inputs and code; show what a run would do")
    runner = sub.add_parser("run", help="build what changed")
    runner.add_argument("--gpu", default="auto", help="auto (idle GPUs), an index, or cpu")
    runner.add_argument("--poll", type=float, default=5.0)
    sub.add_parser("status", help="state of the last run")
    args = parser.parse_args(argv)
    try:
        project = Project(args.project, config_path=args.config)
        if args.command == "plan":
            result, context = plan(project)
            print_plan(result)
            context["hashes"].save()
            save(project.state_dir / "plan.json", result)
        elif args.command == "run":
            if args.gpu not in ("auto", "cpu") and not args.gpu.isdigit():
                raise MakeError("--gpu must be auto, cpu or a GPU index")
            execute(project, gpu=args.gpu, poll=args.poll)
        else:
            print(json.dumps(status(project), indent=2))
    except (MakeError, SpecError, JobFailed, LockError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
