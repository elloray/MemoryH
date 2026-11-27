# /// script
# requires-python = ">=3.12"
# dependencies = ["aiohttp", "aiolimiter", "openai", "pydantic", "python-dotenv", "requests", "tqdm"]
# ///
"""Run with uv run scripts/locomo-compare.py retrieve and grade in separate processes."""

import argparse
import asyncio
import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from contextlib import ExitStack, contextmanager
from datetime import UTC
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "target/locomo-compare"
UP = OUT / "upstream"
REV = "4b61c5d31b9c668a12b4f5e78064248a02c82d2b"
DATA_SHA = "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"
DATA = OUT / "locomo10.json"
DATA_REV = "3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376"


def setup():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "setup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not UP.exists():
            with TemporaryDirectory(dir=OUT) as temp:
                repo = Path(temp) / "repo"
                subprocess.run(
                    [
                        "git",
                        "clone",
                        "-q",
                        "https://github.com/mem0ai/memory-benchmarks.git",
                        repo,
                    ],
                    check=True,
                )
                subprocess.run(
                    ["git", "-C", repo, "checkout", "-q", "--detach", REV], check=True
                )
                repo.rename(UP)
        if not DATA.exists():
            url = f"https://raw.githubusercontent.com/snap-research/locomo/{DATA_REV}/data/locomo10.json"
            with urlopen(url, timeout=60) as response:
                raw = response.read()
            if hashlib.sha256(raw).hexdigest() != DATA_SHA:
                raise RuntimeError("incorrect LoCoMo download")
            with NamedTemporaryFile("wb", dir=OUT, delete=False) as f:
                f.write(raw)
            Path(f.name).replace(DATA)
        rev = subprocess.check_output(
            ["git", "-C", UP, "rev-parse", "HEAD"], text=True
        ).strip()
        if (
            rev != REV
            or subprocess.run(
                [
                    "git",
                    "-C",
                    UP,
                    "diff",
                    "--quiet",
                    REV,
                    "--",
                    "benchmarks",
                    "results/platform/locomo_results.json",
                ],
                check=False,
            ).returncode
        ):
            raise RuntimeError("unexpected upstream revision or local scoring changes")
    sys.path.insert(0, str(UP))
    return importlib.import_module("benchmarks.locomo.run")


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", dir=path.parent, delete=False) as f:
        json.dump(value, f, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(f.name, path)


def read(path):
    return json.loads(path.read_bytes())


def status(stage, detail):
    put(OUT / f"{stage}-status.json", {"detail": detail, "updated": time.time()})
    print(detail, flush=True)


def file_at(group, qid):
    return OUT / group / (qid + ".json")


def load_data():
    if hashlib.sha256(DATA.read_bytes()).hexdigest() != DATA_SHA:
        raise RuntimeError("dataset or harness revision mismatch")
    data = locomo.load_dataset(DATA)
    items = locomo.expected_locomo_question_items(data, range(10), [1, 2, 3, 4], None)
    supplied = read(UP / "results/platform/locomo_results.json")["evaluations"]
    refs = {
        r["question_id"]: {
            "question": r["question"],
            "correct": r["cutoff_results"]["top_200"]["judgment"] == "CORRECT",
            "latency_ms": r["search_latency_ms"],
        }
        for r in supplied
    }
    if (
        len(items) != 1540
        or len(refs) != 1540
        or any(refs[q]["question"] != qa["question"] for q, _, _, qa in items)
    ):
        raise RuntimeError("Mem0 public rows and dataset do not match")
    put(OUT / "reference.json", refs)
    return data, items


class Native:
    def __init__(self, process, date, reranker):
        self.reranker = reranker
        self.process, self.date = process, date

    def request(self, cmd):
        p = self.process
        p.stdin.write(json.dumps(cmd, ensure_ascii=False) + "\n")
        p.stdin.flush()
        line = p.stdout.readline()
        if not line:
            raise RuntimeError(f"Qwen worker stopped ({p.poll()}); inspect native.log")
        r = json.loads(line)
        if "error" in r:
            raise RuntimeError(r["error"])
        return r

    async def search(self, question, user_id, top_k, score_debug):
        assert top_k == 200 and not score_debug
        reply = self.request({"op": "search", "question": question, "date": self.date})
        self.profile = reply["profile"]
        return reply["results"]


@contextmanager
def open_native(index, date, runtime, address, library, width, *, mps_dir=None):
    env = os.environ.copy()
    env.pop("MEMORYH_YUANBAO_MPS_DIR", None)
    if mps_dir is not None:
        env["MEMORYH_YUANBAO_MPS_DIR"] = str(mps_dir.resolve())
    with (OUT / "native.log").open("a") as log:
        p = subprocess.Popen(
            [
                ROOT / "target/release/examples/locomo_qwen",
                index,
                ROOT / "target/memoryh-models",
                runtime,
                str(width),
            ],
            env={
                **env,
                "MEMORYH_SPLITTER_ADDR": address,
                "MEMORYH_SCANN_LIB": str(library),
            },
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log,
            text=True,
        )
        try:
            engine = Native(p, date, "mps" if mps_dir is not None else "cpu")
            engine.request({"op": "insert", "turns": []})
            yield engine
        finally:
            p.stdin.close()
            p.wait()
            p.stdout.close()


def iso(date):
    return locomo.parse_locomo_date(date).replace(tzinfo=UTC).isoformat()


@contextmanager
def runtime_context():
    spec = importlib.util.spec_from_file_location(
        "scann_bench", ROOT / "scripts/benchmark-scann.py"
    )
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    status("retrieve", "正在加载 MemoryH 本机模型")
    llama, splitmodel, library, runtime = bench.prepare(
        ROOT / "target/memoryh-models", example="locomo_qwen"
    )
    with bench.local_splitter(llama, splitmodel, OUT) as address:
        yield runtime, address, library


async def search_one(engine, qa, qi, conversation, reference, width):
    row = await locomo.process_question(
        qa,
        qi,
        conversation,
        f"locomo_{conversation}",
        engine,
        None,
        None,
        [200],
        200,
        reference,
        None,
        None,
        True,
        None,
    )
    row["retrieval"].update(
        profile=engine.profile, recall_width=width, reranker=engine.reranker
    )
    return row


def publish_search(qid, row):
    put(file_at("searches", qid), row)
    save_search_metrics(qid, row)


def adopt_gpu_rows(items, width):
    """Publish identical completed pilot retrievals once; preserve earlier CPU answers."""
    for qid, _, _, qa in items:
        probe = file_at("yuanbao-gpu/rows", qid)
        if file_at("searches", qid).exists() or not probe.is_file():
            continue
        row = read(probe)
        r = row["retrieval"]
        if (
            r.get("recall_width") != width
            or row["question_id"] != qid
            or row["question"] != qa["question"]
            or r.get("reranker", "mps") != "mps"
        ):
            raise ValueError(f"GPU replay configuration mismatch for {qid}")
        r["reranker"] = "mps"
        publish_search(qid, row)


async def retrieve(data, items, width, compare_id, backend, pilot):
    mps_dir = ROOT / "target/yuanbao-mps" if backend == "mps" else None
    if backend not in {"cpu", "mps"}:
        raise ValueError(f"Unknown Yuanbao reranker backend {backend}")
    device = "M4 GPU" if mps_dir else "CPU"
    if mps_dir is not None and pilot:
        adopt_gpu_rows(items, width)
    with runtime_context() as (runtime, address, library):
        for c, entry in enumerate(data):
            jobs = [(qid, qi, qa) for qid, ci, qi, qa in items if ci == c]
            pending = []
            for qid, qi, qa in jobs:
                path = file_at("searches", qid)
                if path.exists():
                    if not file_at("search_metrics", qid).exists():
                        save_search_metrics(qid, read(path))
                else:
                    pending.append((qid, qi, qa))
            if not pending:
                continue
            conv = entry["conversation"]
            sessions = locomo.get_sorted_sessions(conv)
            reference = sessions[-1][1]
            index = OUT / "stores" / str(c)
            index.mkdir(parents=True, exist_ok=True)
            with ExitStack() as stack:
                engine = stack.enter_context(
                    open_native(
                        index,
                        iso(reference),
                        runtime,
                        address,
                        library,
                        width,
                        mps_dir=mps_dir,
                    )
                )
                controls = []
                control_engine = None
                compare_devices = (
                    c == 0 and mps_dir is not None and pilot and width == 32
                )
                if compare_devices:
                    from locomo_mps_report import COUNT, FIRST
                    from locomo_mps_report import report as gpu_report

                    for i in range(FIRST, FIRST + COUNT):
                        q = f"conv0_q{i}"
                        official = file_at("search_metrics", q)
                        has_cpu = (
                            official.is_file()
                            and read(official).get("reranker", "cpu") == "cpu"
                        )
                        if not has_cpu and not file_at("yuanbao-gpu/cpu", q).is_file():
                            controls.append((q, i))
                control_resources = stack.enter_context(ExitStack())
                for ix, (session, date, turns) in enumerate(sessions, 1):
                    marker = index / (session + ".done")
                    if marker.exists():
                        continue
                    chunks = locomo.session_to_chunks(
                        turns, conv["speaker_a"], conv["speaker_b"]
                    )
                    rows = [
                        {
                            "id": f"{session}_c{i}",
                            "text": chunk[0]["content"],
                            "date": iso(date),
                        }
                        for i, chunk in enumerate(chunks)
                    ]
                    status(
                        "retrieve", f"对话 {c + 1}/10 入库，会话 {ix}/{len(sessions)}"
                    )
                    if engine.request({"op": "insert", "turns": rows})[
                        "inserted"
                    ] != len(rows):
                        raise RuntimeError("incomplete insert")
                    put(marker, {"turns": len(rows)})
                if compare_id and any(qid == compare_id for qid, _, _ in jobs):
                    probe_path = file_at(f"width{width}", compare_id)
                    previous = read(file_at("search_metrics", compare_id))
                    if (
                        previous.get("recall_width", 200) != width
                        and not probe_path.exists()
                    ):
                        _, qi, qa = next(job for job in jobs if job[0] == compare_id)
                        status("retrieve", f"同题测速 {compare_id} · 每路 {width} 条")
                        row = await search_one(engine, qa, qi, c, reference, width)
                        put(
                            probe_path,
                            {
                                "question_id": compare_id,
                                "wide": previous,
                                "narrow": metrics(row),
                            },
                        )
                for seq, (qid, qi, qa) in enumerate(pending, 1):
                    status(
                        "retrieve",
                        f"对话 {c + 1}/10 正在检索 {qid} · {device} · 每路 {width} 条",
                    )
                    row = await search_one(engine, qa, qi, c, reference, width)
                    if compare_devices and FIRST <= qi < FIRST + COUNT:
                        put(file_at("yuanbao-gpu/rows", qid), row)
                    publish_search(qid, row)
                    if controls and seq % 2 == 0:
                        key, ix = controls.pop(0)
                        status("retrieve", f"独立 CPU 对照 {key} · 正式检索使用 M4 GPU")
                        if control_engine is None:
                            tmp = control_resources.enter_context(
                                TemporaryDirectory(dir=OUT)
                            )
                            clone = shutil.copytree(index, Path(tmp) / "store")
                            control_engine = control_resources.enter_context(
                                open_native(
                                    clone,
                                    iso(reference),
                                    runtime,
                                    address,
                                    library,
                                    width,
                                )
                            )
                            await search_one(
                                control_engine, entry["qa"][5], 5, c, reference, width
                            )
                        baseline = await search_one(
                            control_engine, entry["qa"][ix], ix, c, reference, width
                        )
                        put(file_at("yuanbao-gpu/cpu", key), baseline)
                        put(OUT / "yuanbao-gpu/report.json", gpu_report(OUT))
                        if not controls:
                            control_resources.close()
    status("retrieve", "1,540 题检索完成")


def save_search_metrics(qid, row):
    put(file_at("search_metrics", qid), metrics(row))


def metrics(row):
    s = row["retrieval"]
    return {
        "latency_ms": s["search_latency_ms"],
        "count": s["total_results"],
        "profile": s.get("profile"),
        "reranker": s.get("reranker", "cpu"),
        "recall_width": s.get(
            "recall_width", 200
        ),  # Older checkpoints used 200 per lane.
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["retrieve", "grade"])
    parser.add_argument("--recall-width", type=int, default=32)
    parser.add_argument(
        "--compare-id", help="Time a completed question at this width without regrading"
    )
    parser.add_argument(
        "--reranker",
        choices=["cpu", "mps"],
        help="Override the saved run configuration",
    )
    args = parser.parse_args()
    if args.recall_width < 1:
        parser.error("--recall-width must be positive")
    locomo = setup()
    with (OUT / (args.stage + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            data, items = load_data()
            if args.stage == "retrieve":
                protocol_path = OUT / "protocol.json"
                protocol = read(protocol_path) if protocol_path.is_file() else {}
                backend = args.reranker or protocol.get("reranker_default", "cpu")
                asyncio.run(
                    retrieve(
                        data,
                        items,
                        args.recall_width,
                        args.compare_id,
                        backend,
                        protocol.get("gpu_comparison", False),
                    )
                )
            else:
                from locomo_grade import grade

                asyncio.run(grade(OUT, locomo, items, read, put, status))
        except BaseException as exc:
            status(args.stage, f"已中断：{type(exc).__name__} {str(exc)[:200]}")
            raise
