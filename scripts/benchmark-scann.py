#!/usr/bin/env python3
"""Prepare local models, run the default 21-question Qwen/ScaNN pipeline and report quality and latency."""

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "target/scann-eval"
DEPS = WORK / "deps"
GGUF = "Qwen3.5-2B-Q4_K_M.gguf"
MODEL_REPO = "unsloth/Qwen3.5-2B-GGUF"
MODEL_REV = "f6d5376be1edb4d416d56da11e5397a961aca8ae"
MODEL_SHA = "aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223"
LLAMA_URL = "https://github.com/ggml-org/llama.cpp/releases/download/b12779/llama-b12779-bin-macos-arm64.tar.gz"
LLAMA_SHA = "21a95d7e39071184ee27fa46ec36edc06e4214a2a5a991eb661b68f20bd084b1"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url, path, digest, cached_paths=()):
    if path.is_file() and sha(path) == digest:
        print("Reusing", path, flush=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
    try:
        cached = next(
            (item for item in cached_paths if item.is_file() and sha(item) == digest),
            None,
        )
        if cached is not None:
            shutil.copyfile(cached, temporary)
        else:
            print("Downloading", path.name, flush=True)
            subprocess.run(
                ["curl", "-fLsS", "--retry", "3", url, "-o", temporary], check=True
            )
        if sha(temporary) != digest:
            raise ValueError(f"SHA-256 mismatch for {path.name}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare(models, example="scann_benchmark"):
    subprocess.run(
        [ROOT / "scripts/fetch-qwen-models.sh", models, "yuanbao"], check=True
    )
    hub = Path(
        os.environ.get(
            "HF_HUB_CACHE",
            Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub",
        )
    )
    path = models / "splitter" / GGUF
    fetch(
        f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REV}/{GGUF}",
        path,
        MODEL_SHA,
        [
            ROOT / "target/memoryh-models/splitter" / GGUF,
            hub
            / ("models--" + MODEL_REPO.replace("/", "--"))
            / "snapshots"
            / MODEL_REV
            / GGUF,
        ],
    )
    llama = Path(os.environ.get("MEMORYH_LLAMA_SERVER", DEPS / "llama/llama-server"))
    if not llama.is_file():
        if "MEMORYH_LLAMA_SERVER" in os.environ or (
            platform.system(),
            platform.machine(),
        ) != ("Darwin", "arm64"):
            raise ValueError(
                "Set MEMORYH_LLAMA_SERVER to a compatible local llama-server (tested b12779)"
            )
        archive = DEPS / "llama.tar.gz"
        fetch(LLAMA_URL, archive, LLAMA_SHA)
        llama.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["tar", "-xzf", archive, "--strip-components=1", "-C", llama.parent],
            check=True,
        )
    runtime = Path(
        os.environ.get(
            "ONNX_RUNTIME_LIB",
            ROOT / "target/ort-cpu/build/Release/libonnxruntime.1.32.0.dylib",
        )
    )
    if not runtime.is_file():
        if "ONNX_RUNTIME_LIB" in os.environ or platform.system() != "Darwin":
            raise ValueError(
                "Set ONNX_RUNTIME_LIB to the local CPU ONNX Runtime library"
            )
        subprocess.run(["bash", ROOT / "scripts/build-onnxruntime-cpu.sh"], check=True)
    if "MEMORYH_SCANN_LIB" in os.environ:
        lib = Path(os.environ["MEMORYH_SCANN_LIB"]).resolve(strict=True)
    else:
        subprocess.run([ROOT / "scripts/build-scann.sh", DEPS], check=True)
        lib = (
            DEPS
            / "build"
            / (
                "libscann_core.dylib"
                if platform.system() == "Darwin"
                else "libscann_core.so"
            )
        )
    subprocess.run(
        [
            "cargo",
            "build",
            "--locked",
            "--release",
            "--features",
            "qwen",
            "--example",
            example,
        ],
        cwd=ROOT,
        check=True,
    )
    return llama, path, lib, runtime


@contextmanager
def local_splitter(binary, model, out):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    address = f"127.0.0.1:{port}"
    with (out / "splitter.log").open("w") as log:
        proc = subprocess.Popen(
            [
                binary,
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "-m",
                model,
                "-a",
                "memoryh-qwen3.5-2b",
                "-ngl",
                "0",
                "-t",
                "2",
                "-tb",
                "2",
                "-np",
                "1",
                "-c",
                "2048",
                "--reasoning",
                "off",
                "--no-cache-prompt",
                "--cache-ram",
                "0",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline and proc.poll() is None:
                try:
                    with urlopen(f"http://{address}/health", timeout=1) as response:
                        if json.load(response).get("status") == "ok":
                            break
                except (OSError, ValueError):
                    pass
                time.sleep(0.3)
            else:
                raise RuntimeError(f"Local splitter failed to start. See {log.name}")
            yield address
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()


def run_benchmark(models, lib, address, runtime, rounds, out, args):
    print(f"Running Qwen3.5-2B + ScaNN ({rounds} rounds)", flush=True)
    cmd = [
        ROOT / "target/release/examples/scann_benchmark",
        models / "qwen",
        models / "yuanbao",
        runtime,
        str(rounds),
    ]
    env = {
        **os.environ,
        "MEMORYH_SCANN_LIB": str(lib),
        "MEMORYH_SPLITTER_ADDR": address,
    }
    env["MEMORYH_EVAL_FIXTURE"] = str(args.fixture)
    for field, key in [
        ("vector_limit", "VECTOR_LIMIT"),
        ("keyword_limit", "KEYWORD_LIMIT"),
        ("exact_threshold", "EXACT_THRESHOLD"),
        ("probes", "PROBES"),
    ]:
        if (value := getattr(args, field)) is not None:
            env["MEMORYH_EVAL_" + key] = str(value)
    with tempfile.TemporaryFile() as data, (out / "queries.log").open("w") as log:
        proc = subprocess.Popen(
            cmd, stdout=data, stderr=subprocess.PIPE, text=True, env=env
        )
        try:
            for line in proc.stderr:
                log.write(line)
                log.flush()
                if line.startswith(("before round", "after round")):
                    print(line.rstrip(), flush=True)
            if proc.wait():
                raise RuntimeError(
                    f"Benchmark failed ({proc.returncode}). See {log.name}"
                )
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            proc.stderr.close()
        data.seek(0)
        report = json.load(data)
    (out / "queries.json").write_text(json.dumps(report, ensure_ascii=False) + "\n")
    return report


def percentile(values, p=0.95):
    return sorted(values)[math.ceil(len(values) * p) - 1]


def covers(groups, ids):
    return all(set(group).intersection(ids) for group in groups)


def distinct_facts(groups, extracted):
    # Every paraphrase of an instance must share an identity that no other instance has.
    facts = [[set(extracted.get(i, [])) for i in group] for group in groups]
    for n, group in enumerate(facts):
        others = set().union(
            *(s for i, other in enumerate(facts) if i != n for s in other)
        )
        if not (set.intersection(*group) - others):
            return False
    return True


def summary(run, out, fixture_path):
    fixture = json.loads(fixture_path.read_text())
    questions = fixture["questions"]
    rows = run["cases"]
    counts = {
        "candidate_all": 0,
        "top5": 0,
        "top1": 0,
        "top3": 0,
        "stale": 0,
        "cross": 0,
        "corrections": 0,
    }
    case_lines = []
    for question in questions:
        row = rows[question["id"]]
        groups = question.get("expected_groups", [[i] for i in question["expected"]])
        expected = {i for group in groups for i in group}

        forbidden = set(question.get("forbidden", []))
        top = row["top"]
        counts["candidate_all"] += covers(groups, row["pool"])
        counts["top5"] += covers(groups, row["top5"]) and not forbidden.intersection(
            row["top5"]
        )
        counts["top1"] += bool(top and top[0] in expected)
        complete = covers(groups, top) and not forbidden.intersection(top)
        counts["top3"] += complete
        counts["cross"] += complete and question["group"] == "跨分支"
        counts["corrections"] += complete and question["group"] == "记忆更正"
        counts["stale"] += bool(forbidden.intersection(row["pool"]))
        if question.get("check_fact_groups"):
            counts["inventory_questions"] = counts.get("inventory_questions", 0) + 1
            correct = distinct_facts(groups, row["facts"]) and all(
                not row["facts"].get(i) for i in question.get("excluded_facts", [])
            )
            counts["inventory_identities_correct"] = (
                counts.get("inventory_identities_correct", 0) + correct
            )
        clauses = (
            " / ".join(row["variants"][1:]).replace("|", "｜").replace("\n", " ")
            or "仅整题"
        )
        gold = ", ".join(" / ".join(group) for group in groups)
        case_lines.append(
            f"| {question['id']} | {gold} | {', '.join(top)} | {', '.join(row['top5'])} | {clauses} |"
        )
    samples = [s for c in rows.values() for s in c["samples"]]
    for field in samples[0] if samples else []:
        counts[field.replace("_ns", "_p95_ms")] = percentile(
            [s[field] / 1e6 for s in samples]
        )
    counts.update(
        pairs=sum(len(c["pool"]) for c in rows.values()),
        reopened=run["reopened"],
        samples=len(samples),
        questions=len(questions),
        fixture_sha256=sha(fixture_path),
    )
    if samples:
        counts["total_p50_ms"] = percentile([s["total_ns"] / 1e6 for s in samples], 0.5)
    if "recall" in run:
        counts["recall"] = run["recall"]
    result = json.dumps(counts, ensure_ascii=False, indent=2)
    (out / "summary.json").write_text(result + "\n")
    lines = [
        "# MemoryH 中文检索复测",
        "",
        "Qwen3.5-2B 拆题，Qwen3-Embedding-0.6B 编码，BM25 和 ScaNN 独立召回，Yuanbao-Reranker-0.6B 对去重候选重排。",
        "",
        f"候选池找齐 {counts['candidate_all']}/{len(questions)}，Top 1 命中 {counts['top1']}/{len(questions)}，Top 3 找齐 {counts['top3']}/{len(questions)}，Top 5 找齐 {counts['top5']}/{len(questions)}。重开后核对 {counts['reopened']} 题。",
        "",
        (
            f"预热后共 {len(samples)} 次查询，端到端 P95 为 {counts['total_p95_ms']:.2f} ms。"
            if samples
            else "本次只验证检索质量。"
        ),
    ]
    if counts.get("inventory_questions"):
        lines.extend(
            [
                "",
                f"逐条核对清点实例及排除项，{counts['inventory_identities_correct']}/{counts['inventory_questions']} 题通过。",
            ]
        )
    if samples:
        lines.extend(["", "| 阶段 | P95（ms） |", "|---|---:|"])
        for key, label in [
            ("split", "拆题"),
            ("encode", "编码"),
            ("recall", "双路召回及取原文"),
            ("vector", "其中向量检索"),
            ("rerank", "重排"),
        ]:
            lines.append(f"| {label} | {counts[key + '_p95_ms']:.3f} |")
    lines.extend(
        [
            "",
            "| 题号 | 标注原文 | Top 3 | Top 5 | 召回角度 |",
            "|---|---|---|---|---|",
            *case_lines,
        ]
    )
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(result, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models-dir", type=Path, default=ROOT / "target/memoryh-models")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=WORK
        / "runs"
        / (datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"),
    )
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--fixture", type=Path, default=ROOT / "cases.json")
    p.add_argument("--vector-limit", type=int)
    p.add_argument("--keyword-limit", type=int)
    p.add_argument("--exact-threshold", type=int)
    p.add_argument("--probes", type=int)
    args = p.parse_args()
    if args.rounds < 0:
        p.error("--rounds must be nonnegative (0 checks quality only)")
    models, out = args.models_dir.resolve(), args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    fixture_copy = out / "fixture.json"
    shutil.copyfile(args.fixture, fixture_copy)
    args.fixture = fixture_copy
    print("Run directory:", out, flush=True)
    binary, model, lib, runtime = prepare(models)
    with local_splitter(binary, model, out) as address:
        run = run_benchmark(models, lib, address, runtime, args.rounds, out, args)
    summary(run, out, args.fixture)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        sys.exit(str(exc))
