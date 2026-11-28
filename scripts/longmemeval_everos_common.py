"""Pinned inputs and atomic checkpoints for the MemoryH / EverOS comparison."""

import fcntl
import hashlib
import importlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "target/lme-s-everos-20261128"
UP = OUT / "input/everos"
DATA = OUT / "input/longmemeval_s.json"
DATA_REV = "2ec2a557f339b6c0369619b1ed5793734cc87533"
EVEROS_REV = "d2b74780a443c6d7304ca9aa1bfdf36fdc6d7c9b"
SHA = "08d8dad4be43ee2049a22ff5674eb86725d0ce5ff434cde2627e5e8e7e117894"
SOURCE_HASHES = {
    "adapters/base.py": "841aa73492b42462bd40b7232d5c2c6f01d9708d0ce8265e6f7dfd9ace8fa348",
    "adapters/longmemeval.py": "53d77dfd38688a9c9d6154fcf74cddd19c12972295784c959d24855b44febc77",
    "run.py": "1799ae749e9e072028d4d6ea373e17945537b31774c3f36b1ea118b2b3ad1230",
    "configs/longmemeval.toml": "fe5ec450c69e0b28ea2f4a46280b426ba1923fd05c41a81ee5ed39cd6849cef6",
}


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as tmp:
        json.dump(value, tmp, ensure_ascii=False, indent=2)
        tmp.flush()
        os.fsync(tmp.fileno())
    os.replace(tmp.name, path)


def read(path):
    return json.loads(path.read_text())


def fetch(path, url, expected):
    if path.exists():
        if digest(path) != expected:
            raise ValueError(f"unexpected content in {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        temp = Path(tmp.name)
        try:
            with urlopen(url, timeout=120) as src:
                shutil.copyfileobj(src, tmp)
            tmp.flush()
            if digest(temp) != expected:
                raise ValueError(f"unexpected release from {url}")
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)


def load_adapter():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "inputs.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        fetch(
            DATA,
            f"https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/{DATA_REV}/longmemeval_s",
            SHA,
        )
        for name, sha in SOURCE_HASHES.items():
            fetch(
                UP / name,
                f"https://raw.githubusercontent.com/EverMind-AI/EverOS/{EVEROS_REV}/benchmarks/{name}",
                sha,
            )
    sys.path.insert(0, str(UP))
    return importlib.import_module("adapters.longmemeval")
