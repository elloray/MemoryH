"""Replay one single-query LoCoMo cohort with the native Yuanbao MPS worker.

Run with the LoCoMo environment and scripts/prepare-yuanbao-mps.py outputs.
Compare against the normal width-32 CPU checkpoints, never grade these replays.
"""

import argparse
import asyncio
import fcntl
import importlib.util
import json
import shutil
import tempfile
from pathlib import Path

from locomo_mps_report import COUNT, FIRST, report

spec = importlib.util.spec_from_file_location(
    "locomo_compare", Path(__file__).with_name("locomo-compare.py")
)
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
ROOT = h.OUT / "yuanbao-gpu"


async def replay(mps_dir):
    h.locomo = h.setup()
    data, _ = h.load_data()
    conv = data[0]
    sessions = h.locomo.get_sorted_sessions(conv["conversation"])
    reference = sessions[-1][1]
    index = h.OUT / "stores/0"
    if any(not (index / (s + ".done")).is_file() for s, _, _ in sessions):
        raise RuntimeError("conversation indexing incomplete")
    with (
        h.runtime_context() as (runtime, address, library),
        tempfile.TemporaryDirectory(dir=ROOT) as tmp,
    ):
        snapshot = shutil.copytree(index, Path(tmp) / "store")
        with h.open_native(
            snapshot, h.iso(reference), runtime, address, library, 32, mps_dir=mps_dir
        ) as engine:
            h.status("retrieve", "GPU 单路实验 · 预热 Yuanbao")
            await h.search_one(engine, conv["qa"][5], 5, 0, reference, 32)
            for i in range(FIRST, FIRST + COUNT):
                qid = f"conv0_q{i}"
                path = ROOT / "rows" / (qid + ".json")
                if path.is_file():
                    continue
                h.status(
                    "retrieve",
                    f"GPU 单路实验 {i - FIRST + 1}/{COUNT} · {qid} · 每路 32 条",
                )
                row = await h.search_one(engine, conv["qa"][i], i, 0, reference, 32)
                h.put(path, row)
                h.put(ROOT / "report.json", report(h.OUT))
    h.status("retrieve", "20 道 GPU 单路检索已完成")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mps-dir", type=Path, default=h.ROOT / "target/yuanbao-mps")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    if args.report:
        result = report(h.OUT)
        h.put(ROOT / "report.json", result)
        print(json.dumps(result["summary"], indent=2))
    else:
        with (h.OUT / "retrieve.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            asyncio.run(replay(args.mps_dir))
