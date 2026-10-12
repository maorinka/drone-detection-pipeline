"""'Train' a one-parameter model (the overall mean) by gradient descent from a seeded start."""
import json
import os
import random
import sys
from pathlib import Path

run, seed, lr = Path(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
features = [json.loads(p.read_text()) for p in sorted((run / "features").glob("*.json"))]
target = sum(f["mean"] * f["n"] for f in features) / sum(f["n"] for f in features)
rng = random.Random(seed)
w = rng.uniform(-10, 10)
for step in range(200):
    w -= lr * 2 * (w - target)
out = run / "models" / f"seed{seed}"
out.mkdir(parents=True, exist_ok=True)
(out / "model.json").write_text(json.dumps({"seed": seed, "lr": lr, "w": w, "target": target}, indent=2))
(out / "done.json").write_text(json.dumps({"ok": True}))
print(f"seed {seed}: w={w:.4f} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r})")
