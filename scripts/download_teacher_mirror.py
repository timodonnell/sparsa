"""Mirror an explicitly authorized teacher manifest using scoped signed URLs.

Keep the manifest private and outside the repository. Logs never include URLs.
Each completed file is size-checked, SHA256-hashed, and atomically published.
"""

import argparse
import concurrent.futures
import hashlib
import http.client
import json
import os
import time
import urllib.request
from pathlib import Path


def download_one(record, root):
    relative = Path(record["relative_path"])
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or relative.parts[0] not in {"afdb", "esm"}
    ):
        raise ValueError("Invalid relative teacher path")
    dest = root / relative
    dest.parent.mkdir(parents=True, exist_ok=True)
    receipt = dest.with_suffix(dest.suffix + ".sha256")
    if dest.exists() and dest.stat().st_size == record["size"] and receipt.exists():
        return record["size"]
    temp = dest.with_suffix(dest.suffix + ".partial")
    for attempt in range(6):
        try:
            digest = hashlib.sha256()
            size = 0
            with (
                urllib.request.urlopen(record["url"], timeout=60) as source,
                temp.open("wb") as target,
            ):
                while chunk := source.read(4 * 1024 * 1024):
                    digest.update(chunk)
                    target.write(chunk)
                    size += len(chunk)
            if size != record["size"]:
                raise ValueError("Unexpected object size")
            os.replace(temp, dest)
            receipt.write_text(digest.hexdigest() + "\n")
            return size
        except (OSError, ValueError, http.client.HTTPException) as exc:
            if attempt == 5:
                raise RuntimeError(
                    f"Teacher download failed: {relative}; {type(exc).__name__}"
                ) from None
            time.sleep(min(2**attempt, 20))
    raise AssertionError("Unreachable")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    records = json.loads(args.manifest.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    count = size = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download_one, r, args.out) for r in records]
        for future in concurrent.futures.as_completed(futures):
            size += future.result()
            count += 1
            if count % 100 == 0 or count == len(records):
                print(
                    json.dumps(
                        {
                            "files": count,
                            "total_files": len(records),
                            "bytes": size,
                            "elapsed_seconds": round(time.monotonic() - start, 1),
                        }
                    ),
                    flush=True,
                )
    # This public receipt deliberately omits the private URLs.
    manifest = [{k: v for k, v in r.items() if k != "url"} for r in records]
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("TEACHER_MIRROR_COMPLETE", count, size, flush=True)


if __name__ == "__main__":
    main()
