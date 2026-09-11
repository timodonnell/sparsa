"""Fetch a recovered result directory without requiring tar inside the pod."""

import argparse
import subprocess
import tarfile
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pod", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if not args.pod.startswith("iris-bizon-sparsa-"):
        raise ValueError("Expected our own Sparsa recovery task")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    archive = out / "results.tar.gz.partial"
    cmd = [
        "kubectl",
        "--kubeconfig",
        str(Path.home() / ".kube/coreweave-iris"),
        "--context",
        "marin-rn02a_RNO2A",
        "-n",
        "iris",
        "exec",
        args.pod,
        "-c",
        "task",
        "--",
        "/app/.venv/bin/python",
        "-c",
        'import tarfile,sys; t=tarfile.open(fileobj=sys.stdout.buffer,mode="w|gz"); t.add("/app/outputs/recovered",arcname="sparsa-results"); t.close()',
    ]
    with archive.open("wb") as f:
        subprocess.run(cmd, stdout=f, check=True)
    with tarfile.open(archive) as f:
        f.extractall(out, filter="data")
    archive.unlink()
    print("Recovered", out / "sparsa-results")
