"""CIFAR-10 data for E001 (pixel-space MAE pretraining).

Source: Hugging Face dataset `uoft-cs/cifar10`, pinned revision below (PNG-encoded,
lossless). The UToronto tarball was too slow from this cluster (~150 KB/s).

  download  : snapshot the pinned HF parquet into $CIFAR10_PATH/hf_<repo>_<rev7>/. Needs internet.
  convert   : decode parquet -> $CIFAR10_PATH/cifar10_{train,test}.npz (uint8 NHWC + labels),
              write SOURCE.json with sha256s. Needs pyarrow (wflow env lacks it; base Anaconda has it).

Run from the repo root after `source scripts/common.sh`:  python -m scripts.prepare_cifar10 <cmd>
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np

HF_REPO = "uoft-cs/cifar10"
HF_REVISION = "0b2714987fa478483af9968de7c934580d0bb9a2"


def _root() -> Path:
    from utils.env import CIFAR10_PATH, require

    return Path(require(CIFAR10_PATH, "CIFAR10_PATH"))


def _hf_dir() -> Path:
    return _root() / f"hf_{HF_REPO.replace('/', '_')}_{HF_REVISION[:7]}"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_download(_args):
    from huggingface_hub import snapshot_download

    snapshot_download(HF_REPO, repo_type="dataset", revision=HF_REVISION,
                      allow_patterns=["plain_text/*.parquet", "README.md"], local_dir=str(_hf_dir()))
    print(f"HF {HF_REPO}@{HF_REVISION} -> {_hf_dir()}")


def cmd_convert(_args):
    import pyarrow
    import pyarrow.parquet as pq
    from PIL import Image

    source = {"hf_repo": HF_REPO, "hf_revision": HF_REVISION, "pyarrow": pyarrow.__version__,
              "numpy": np.__version__, "converted_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "splits": {}}
    for split in ("train", "test"):
        pq_path = _hf_dir() / "plain_text" / f"{split}-00000-of-00001.parquet"
        table = pq.read_table(pq_path)
        imgs = table.column("img").to_pylist()
        labels = np.asarray(table.column("label").to_pylist(), dtype=np.int64)
        data = np.stack([np.asarray(Image.open(io.BytesIO(r["bytes"])).convert("RGB"), dtype=np.uint8) for r in imgs])
        assert data.shape == (len(labels), 32, 32, 3), data.shape
        assert set(np.unique(labels).tolist()) == set(range(10))
        out = _root() / f"cifar10_{split}.npz"
        np.savez(out, data=data, labels=labels)
        source["splits"][split] = {
            "parquet": str(pq_path), "parquet_sha256": _sha256(pq_path),
            "npz": str(out), "npz_sha256": _sha256(out), "n": int(len(labels)),
            "per_class": np.bincount(labels, minlength=10).tolist(),
            "data_sha256": hashlib.sha256(data.tobytes()).hexdigest(),
        }
        print(split, data.shape, np.bincount(labels, minlength=10).tolist())
    (_root() / "SOURCE.json").write_text(json.dumps(source, indent=2) + "\n")
    print(f"wrote {_root() / 'SOURCE.json'}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("download").set_defaults(fn=cmd_download)
    sub.add_parser("convert").set_defaults(fn=cmd_convert)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
