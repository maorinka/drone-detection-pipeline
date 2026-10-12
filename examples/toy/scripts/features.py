"""Statistics of one CSV file of numbers (one value per line)."""
import json
import statistics
import sys
from pathlib import Path

source, target = Path(sys.argv[1]), Path(sys.argv[2])
values = [float(line) for line in source.read_text().split() if line.strip()]
target.parent.mkdir(parents=True, exist_ok=True)
stats = {"file": source.name, "n": len(values), "mean": statistics.fmean(values), "stdev": statistics.pstdev(values)}
tmp = target.with_suffix(".tmp")
tmp.write_text(json.dumps(stats, indent=2))
tmp.replace(target)  # the done marker appears only when the file is complete
print(stats)
