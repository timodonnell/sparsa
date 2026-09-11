"""Stage final artifacts in a short-lived CPU task for workstation recovery.

After the READY line, use kubectl cp on this task's /app/outputs/recovered. The
1,200-second grace period consumes no GPUs and ends at the Iris task timeout.
"""

import argparse
import time
from pathlib import Path

from sparsa.train import storage

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    args = parser.parse_args()
    fs, root = storage(args.source)
    local = Path("/app/outputs/recovered")
    local.mkdir(parents=True, exist_ok=True)
    files = fs.find(root)
    total_bytes = 0
    for file in files:
        relative = file.removeprefix(root.rstrip("/") + "/")
        target = local / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        fs.get_file(file, str(target))
        total_bytes += target.stat().st_size
    print(f"READY {local} {total_bytes} bytes", flush=True)
    time.sleep(1200)
