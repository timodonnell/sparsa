"""Read a Sparsa artifact through our own live Iris task's colocated S3 access.

No credentials leave the pod. Use while training is running, or with an explicit
artifact-recovery CPU task after training. The destination is a workstation file.
"""

import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pod", required=True)
    parser.add_argument("--uri", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--kubeconfig", default=str(Path.home() / ".kube/coreweave-iris")
    )
    parser.add_argument("--context", default="marin-rn02a_RNO2A")
    parser.add_argument("--namespace", default="iris")
    args = parser.parse_args()
    if not args.uri.startswith(
        "s3://marin-us-east-02a/marin/protein-structure/sparsa/"
    ):
        raise ValueError("This helper only reads Sparsa artifacts")
    if not args.pod.startswith("iris-bizon-sparsa-"):
        raise ValueError("Use a task belonging to this project")
    command = [
        "kubectl",
        "--kubeconfig",
        args.kubeconfig,
        "--context",
        args.context,
        "-n",
        args.namespace,
        "exec",
        args.pod,
        "-c",
        "task",
        "--",
        "/app/.venv/bin/python",
        "-c",
        'import fsspec,sys,shutil; f=fsspec.open(sys.argv[1],"rb").open(); shutil.copyfileobj(f,sys.stdout.buffer)',
        args.uri,
    ]
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as f:
        subprocess.run(command, stdout=f, check=True)
    temporary.replace(path)
    print(f"Recovered {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
