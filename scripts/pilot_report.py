"""Collect pilot decisions from durable artifacts, including interrupted runs."""

import json

from sparsa.train import storage, write_json

ROOT = "s3://marin-us-east-02a/marin/protein-structure/sparsa"
RUNS = [
    "pilot-20260911",
    "pilot-no-triangle-20260911",
    "pilot-large-20260911",
    "pilot-weight16-20260911",
    "pilot-v2-weight4-20260911",
    "pilot-v2-weight16-20260911",
    "resume-smoke-20260911",
]

if __name__ == "__main__":
    result = {}
    for name in RUNS:
        fs, path = storage(f"{ROOT}/runs/{name}")
        record = {}
        for filename in [
            "best.json",
            "latest.json",
            "complete.json",
            "provenance.json",
        ]:
            if fs.exists(f"{path}/{filename}"):
                value = json.loads(fs.cat_file(f"{path}/{filename}"))
                if filename == "provenance.json":
                    value = {
                        k: value[k]
                        for k in [
                            "parameters",
                            "model_config",
                            "training_config",
                            "world_size",
                        ]
                    }
                record[filename.removesuffix(".json")] = value
        vals = sorted(fs.glob(f"{path}/validation/*.json"))
        record["validation"] = [json.loads(fs.cat_file(p)) for p in vals]
        result[name] = record
    write_json(result, ROOT + "/data/pilot_summary.json")
    print("PILOT_SUMMARY " + json.dumps(result), flush=True)
