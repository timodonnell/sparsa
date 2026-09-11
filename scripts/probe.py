"""Check GPU and colocated teacher storage using Iris-injected credentials."""

import json

import fsspec
import pyarrow.parquet as pq
import torch

print("torch", torch.__version__, "cuda", torch.cuda.is_available(), flush=True)
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(), flush=True)
fs = fsspec.filesystem("s3")
base = "marin-us-east-02a/MarinFold/exp232_sweep_cv1_decontam/data"
for source in ["afdb", "esm"]:
    files = fs.glob(f"{base}/{source}/**/*.parquet")
    print(source, len(files), files[:2], flush=True)
    with fs.open(files[0], "rb") as f:
        q = pq.ParquetFile(f)
        print(q.schema_arrow, flush=True)
        row = next(q.iter_batches(batch_size=1)).to_pylist()[0]
        print({k: str(v)[:1600] for k, v in row.items()}, flush=True)
    path = f"s3://marin-us-east-02a/marin/protein-structure/sparsa/data/{source}_shards.json"
    with fsspec.open(path, "wt") as f:
        json.dump(["s3://" + p for p in files], f)
    print("wrote", path, flush=True)
