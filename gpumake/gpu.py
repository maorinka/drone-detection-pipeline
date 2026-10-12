"""Read-only GPU availability: which cards may take another job right now.

Nothing here ever stops or signals another process; a busy card is simply not used.
"""
from __future__ import annotations

import csv
import subprocess
from typing import Dict, List


def _number(text: str, fallback: int) -> int:
    try:
        return int(float(text.strip()))
    except ValueError:  # "[N/A]", "[Not Supported]" (MIG and some other cards)
        return fallback


def list_devices() -> List[dict]:
    """All GPUs as ``nvidia-smi`` reports them, in PCI bus order (empty if there is no nvidia-smi).

    A value nvidia-smi cannot report makes the card look full and busy, so it is not used.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free,memory.total,utilization.gpu,pci.bus_id",
             "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=True, timeout=30)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return []
    devices = []
    for row in csv.reader(result.stdout.splitlines()):
        if len(row) < 4:
            continue
        index = _number(row[0], -1)
        if index < 0:
            continue
        device = {"index": index, "free_mib": _number(row[1], 0), "total_mib": _number(row[2], 0),
                  "utilization_percent": _number(row[3], 100)}
        if len(row) > 4:
            device["pci_bus_id"] = row[4].strip()
        devices.append(device)
    return devices


def slots(devices: List[dict], requested: str, ours: Dict[str, int], *, min_free_mib: int = 8192,
          max_utilization: int = 20, per_gpu: int = 1, shared_free_mib: int = 3072) -> List[dict]:
    """GPUs that may take one more job now, best first.

    With ``requested="auto"`` a card gpumake is not using yet must be idle (at most
    ``max_utilization`` percent busy) with ``min_free_mib`` free. An explicitly requested index
    needs only the free memory (you chose the card). A card already running some of our jobs
    takes another while it has fewer than ``per_gpu`` of them and ``shared_free_mib`` free.
    """
    chosen = []
    for d in devices:
        key = str(d["index"])
        if requested not in ("auto", None) and key != str(requested):
            continue
        mine = ours.get(key, 0)
        if mine == 0:
            idle = d["utilization_percent"] <= max_utilization or requested not in ("auto", None)
            if idle and d["free_mib"] >= min_free_mib:
                chosen.append(d)
        elif mine < per_gpu and d["free_mib"] >= shared_free_mib:
            chosen.append(d)
    return sorted(chosen, key=lambda d: (ours.get(str(d["index"]), 0), -d["free_mib"], d["utilization_percent"], d["index"]))
