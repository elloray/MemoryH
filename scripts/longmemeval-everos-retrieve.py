"""Run native MemoryH retrieval on the original 500 isolated LongMemEval-S histories."""

import fcntl
import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import UTC, datetime
from threading import Lock

from longmemeval_everos_common import (
    DATA,
    DATA_REV,
    EVEROS_REV,
    OUT,
    ROOT,
    SHA,
    digest,
    load_adapter,
    put,
)

adapter = load_adapter()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def request(proc, obj):
    proc.stdin.write(json.dumps(obj, ensure_ascii=False) + "\n")
    proc.stdin.flush()
    reply = proc.stdout.readline()
    if not reply:
        raise RuntimeError(f"Native worker stopped ({proc.poll()}); see native.log")
    reply = json.loads(reply)
    if "error" in reply:
        raise RuntimeError(reply["error"])
    return reply


class Progress:
    def __init__(self):
        self.guard = Lock()
        self.workers = {}

    def update(self, number, phase=None, **fields):
        with self.guard:
            if phase is None:
                self.workers.pop(number, None)
            else:
                self.workers[number] = {
                    "question": number,
                    "phase": phase,
                    "updated": time.time(),
                    **fields,
                }
            workers = [self.workers[n] for n in sorted(self.workers)]
            put(
                OUT / "retrieval-status.json",
                {**(workers[0] if workers else {}), "workers": workers},
            )


def build_index(unit, address, binary, resources, progress):
    number = unit["index"] + 1
    turns = []
    for session in adapter.sessions_of(unit):
        for msg in session["messages"]:
            date = datetime.fromtimestamp(msg["timestamp_ms"] / 1000, UTC).isoformat()
            turns.append(
                {
                    "id": msg["dia_id"],
                    "text": f"{msg['speaker']}: {msg['text']}",
                    "date": date,
                }
            )
    env = dict(
        os.environ,
        MEMORYH_YUANBAO_MPS_DIR=str(ROOT / "target/yuanbao-mps"),
        MEMORYH_SPLITTER_ADDR=address,
        MEMORYH_SCANN_EXACT_THRESHOLD="19999",
        MEMORYH_SCANN_LIB=str(
            ROOT / "target/scann-eval/deps/build/libscann_core.dylib"
        ),
    )
    directory = resources.enter_context(tempfile.TemporaryDirectory(dir=OUT))
    log = resources.enter_context((OUT / "native.log").open("a"))
    args = [
        binary,
        directory,
        ROOT / "target/memoryh-models",
        ROOT / "target/ort-cpu/build/Release/libonnxruntime.1.32.0.dylib",
        "128",
        "32",
    ]
    index_start = time.time()
    proc = resources.enter_context(
        subprocess.Popen(
            args,
            text=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log,
            env=env,
        )
    )
    resources.callback(proc.stdin.close)
    for first in range(0, len(turns), 16):
        chunk = turns[first : first + 16]
        progress.update(number, "index", written=first, turns=len(turns))
        if request(proc, {"op": "insert", "turns": chunk})["inserted"] != len(chunk):
            raise ValueError("incomplete native ingest")
    index_finish = time.time()
    progress.update(number, "ready", written=len(turns), turns=len(turns))
    return (
        proc,
        turns,
        {"index_started_at": index_start, "index_finished_at": index_finish},
    )


def search_and_checkpoint(unit, prepared, resources, progress):
    with resources:
        proc, turns, times = prepared
        index = unit["index"]
        number = index + 1
        qa = unit["qa"][0]
        progress.update(number, "search", turns=len(turns))
        search_start = time.time()
        reply = request(
            proc,
            {
                "op": "search",
                "question": qa["question"],
                "date": qa["question_date"],
                "top_k": 20,
            },
        )
        search_finish = time.time()
        hits = reply["results"]
        ids = {t["id"]: t for t in turns}
        if (
            len(hits) > 20
            or len({h["id"] for h in hits}) != len(hits)
            or any(
                h["memory"] != ids[h["id"]]["text"]
                or h["created_at"] != ids[h["id"]]["date"]
                for h in hits
            )
        ):
            raise ValueError("invalid native Top20 results")
        put(
            OUT / "retrieval" / f"{index:03}.json",
            {
                "index": index,
                "question_id": qa["question_id"],
                "turns": len(turns),
                "sessions": len(unit["haystack_sessions"]),
                "index_s": times["index_finished_at"] - times["index_started_at"],
                "search_s": search_finish - search_start,
                **times,
                "search_started_at": search_start,
                "search_finished_at": search_finish,
                **reply,
            },
        )
        progress.update(number)
        print(f"{number}/500 retrieved", flush=True)


def run_pipeline(units, addr, binary):
    # Each native store stays with its own history from ingestion through retrieval.
    # A single executor serializes GPU searches. The main thread ingests one history ahead.
    progress = Progress()
    previous = None
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="memoryh-search") as gpu:
        for unit in units:
            index = unit["index"]
            dst = OUT / "retrieval" / f"{index:03}.json"
            if dst.exists():
                row = json.loads(dst.read_text())
                if (
                    row["index"] != index
                    or row["question_id"] != unit["qa"][0]["question_id"]
                ):
                    raise ValueError("checkpoint mismatch")
                continue
            if previous is not None and previous.done():
                previous.result()
            print(f"{index + 1}/500 indexing", flush=True)
            with ExitStack() as resources:
                prepared = build_index(unit, addr, binary, resources, progress)
                if previous is not None:
                    previous.result()
                search_resources = resources.pop_all()
                try:
                    previous = gpu.submit(
                        search_and_checkpoint,
                        unit,
                        prepared,
                        search_resources,
                        progress,
                    )
                except BaseException:
                    search_resources.close()
                    raise
        if previous is not None:
            previous.result()


def main():
    units = adapter.load_units(str(DATA))
    if len(units) != 500 or len({u["qa"][0]["question_id"] for u in units}) != 500:
        raise ValueError("expected 500 unique questions")
    bench = module("scann_bench", ROOT / "scripts/benchmark-scann.py")
    binary = OUT / "memoryh-native"
    if not binary.exists():
        bench.prepare(ROOT / "target/memoryh-models", example="locomo_qwen")
        shutil.copy2(ROOT / "target/release/examples/locomo_qwen", binary)
    hashes = {
        str(p.relative_to(ROOT)): digest(p)
        for p in [
            *sorted((ROOT / "src").rglob("*.rs")),
            ROOT / "examples/locomo_qwen.rs",
            ROOT / "Cargo.toml",
            ROOT / "Cargo.lock",
        ]
    }
    manifest = {
        "data_sha256": SHA,
        "data_revision": DATA_REV,
        "everos_revision": EVEROS_REV,
        "native_sha256": digest(binary),
        "build_inputs_sha256": hashes,
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "top_k": 20,
        "vector_limit": 128,
        "keyword_limit": 32,
        "exact_threshold": 19_999,
        "unit": "one dated speaker turn",
        "reranker": "Yuanbao MPS",
    }
    mf = OUT / "retrieval-config.json"
    if mf.exists():
        prior = json.loads(mf.read_text())
        # The first run used this threshold in its frozen native worker's defaults.
        if prior.get("exact_threshold", 19_999) != manifest["exact_threshold"] or any(
            prior[k] != manifest[k]
            for k in (
                "native_sha256",
                "data_sha256",
                "top_k",
                "vector_limit",
                "keyword_limit",
                "unit",
                "reranker",
            )
        ):
            raise RuntimeError("different retrieval configuration")
    else:
        put(mf, manifest)
    execution = OUT / "parallel-start.json"
    if not execution.exists():
        put(
            execution,
            {
                "started_at": time.time(),
                "already_retrieved": len(
                    list((OUT / "retrieval").glob("[0-9][0-9][0-9].json"))
                ),
                "index_workers": 1,
                "search_workers": 1,
                "max_native_histories": 2,
                "native_sha256": manifest["native_sha256"],
            },
        )
    with bench.local_splitter(
        bench.DEPS / "llama/llama-server",
        ROOT / "target/memoryh-models/splitter" / bench.GGUF,
        OUT,
    ) as addr:
        run_pipeline(units, addr, binary)
    put(
        OUT / "retrieval-status.json",
        {"phase": "complete", "ranked": 500, "updated": time.time()},
    )


if __name__ == "__main__":
    with (OUT / "retrieval.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
