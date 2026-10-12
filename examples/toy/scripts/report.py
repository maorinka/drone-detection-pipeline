"""Collect every model into one report."""
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
models = [json.loads(p.read_text()) for p in sorted((run / "models").glob("seed*/model.json"))]
report = {"models": models, "mean_w": sum(m["w"] for m in models) / len(models)}
(run / "report").mkdir(parents=True, exist_ok=True)
(run / "report" / "report.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
