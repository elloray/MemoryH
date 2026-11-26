# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["fastembed==0.9.1", "onnxruntime==1.32.0"]
# ///

"""Compare a standalone Rust process with Python BGE and the same Rust store."""

import argparse
import hashlib
import json
from math import ceil
import os
from pathlib import Path
import platform
import subprocess
from time import perf_counter_ns

from benchmark import BenchmarkSearch, MODEL, QUERY_PREFIX, embed_texts, query_variants
from retrieval_storage import open_retrieval_store

TOP_K = 3
ROOT = Path(__file__).resolve().parent


def python_bge(fixture, model_dir, rounds):
    start = perf_counter_ns()
    from fastembed import TextEmbedding

    model = TextEmbedding(MODEL, threads=2, providers=["CPUExecutionProvider"],
                          specific_model_path=str(model_dir), local_files_only=True)
    load_ns = perf_counter_ns() - start
    cases = {}
    vectors = {}
    with open_retrieval_store("rust") as store:
        for stage, records in (("before", fixture["memories"]), ("after", fixture["corrections"])):
            by_text = embed_texts(model, (record["text"] for record in records))
            vectors.update((record["id"], by_text[record["text"]]) for record in records)
            store.apply(records, vectors)
            search = BenchmarkSearch(store, vectors, semantic=True)
            questions = [question for question in fixture["questions"] if question["stage"] == stage]

            def retrieve(question):
                variants = query_variants(question, storage="rust")
                texts = [QUERY_PREFIX + text for text in variants]
                query_vectors = embed_texts(model, texts)
                _, _, rankings = search.search(variants, [query_vectors[text] for text in texts], TOP_K)
                return [(record_id, *store.read(record_id)) for record_id in rankings["flat_vector_first"][:TOP_K]]

            for question in questions:
                hits = [{"id": record_id, "source": source, "text": text}
                        for record_id, source, text in retrieve(question["text"])]
                cases[question["id"]] = {"hits": hits, "latency_ns": []}
            for _ in range(rounds):
                for question in questions:
                    start = perf_counter_ns()
                    retrieve(question["text"])
                    elapsed = perf_counter_ns() - start
                    cases[question["id"]]["latency_ns"].append(elapsed)
    return {"load_ns": load_ns, "cases": cases}


def percentile(samples, fraction):
    ranked = sorted(samples)
    return ranked[ceil(len(ranked) * fraction) - 1] / 1_000_000


def print_report(fixture, baseline, native, rounds, model_path, *, reference_runtime=False):
    samples_total = len(fixture["questions"]) * rounds
    labels = ("对照 CPU 运行库", "替换 CPU 运行库") if reference_runtime else (
        "Python FastEmbed + Rust 检索", "原生 Rust",
    )
    print("# MemoryH 中文 BGE 原生推理对照\n")
    print(f"`{MODEL}`，{platform.machine()} CPU，各用 2 个 ONNX Runtime 推理线程，"
          "BM25 + 扁平向量，RRF 60，子问题向量首位保留，Top 3。"
          f"每道题预热一次后计时 {rounds} 轮，分别记录 {samples_total} 次查询。")
    with model_path.open("rb") as model:
        print(f"\n两边共用 ONNX 文件，SHA-256 为 `{hashlib.file_digest(model, 'sha256').hexdigest()}`。")
    print("\n查询计时包含拆题、分词和 BGE 前向、检索排序与原文读取。"
          "模型初始化包含导入或加载推理运行时与打开模型，进程启动和原文写入在计时外。\n")
    print("| 路线 | 模型初始化 ms | 查询 P50 ms | 查询 P95 ms |")
    print("|---|---:|---:|---:|")
    for label, report in zip(labels, (baseline, native), strict=True):
        samples = [sample for case in report["cases"].values() for sample in case["latency_ns"]]
        if len(samples) != samples_total:
            raise RuntimeError(f"{label} 的查询计时样本数错误")
        print(f"| {label} | {report['load_ns'] / 1_000_000:.2f} | "
              f"{percentile(samples, .5):.3f} | {percentile(samples, .95):.3f} |")

    changed = [question["id"] for question in fixture["questions"]
               if baseline["cases"][question["id"]]["hits"] != native["cases"][question["id"]]["hits"]]
    print(f"\nID、顺序、原文和来源均一致 {len(fixture['questions']) - len(changed)}/{len(fixture['questions'])} 题。")
    if changed:
        print(f"差异题号为 {', '.join(changed)}。")
    print(f"\n| 用例 | {labels[0]}找齐 | {labels[1]}找齐 |")
    print("|---|---:|---:|")
    groups = dict.fromkeys(question["group"] for question in fixture["questions"])
    for group in [*groups, "全部"]:
        questions = [question for question in fixture["questions"] if group in (question["group"], "全部")]
        row = []
        for report in (baseline, native):
            complete = 0
            for question in questions:
                ids = {hit["id"] for hit in report["cases"][question["id"]]["hits"]}
                complete += set(question["expected"]) <= ids and not set(question.get("forbidden", ())) & ids
            row.append(f"{complete}/{len(questions)}")
        print(f"| {group} | {' | '.join(row)} |")
    stale = []
    for report in (baseline, native):
        stale.append(sum(bool({hit["id"] for hit in report["cases"][question["id"]]["hits"]}
                              & set(question.get("forbidden", ()))) for question in fixture["questions"]))
    print(f"\n误召回禁止版本的题数分别为 {stale[0]} 和 {stale[1]}。")


def native_bge(binary, model_dir, runtime, rounds):
    return json.loads(subprocess.check_output([
        str(binary.resolve()), str(model_dir.resolve()), str(runtime.resolve()), str(rounds),
    ]))


def main():
    parser = argparse.ArgumentParser(description="比较 Python FastEmbed 与原生 Rust BGE 的在线查询")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--runtime-library", type=Path, required=True)
    parser.add_argument("--reference-runtime-library", type=Path, help="在同一 Rust 入口对照另一套 CPU 运行库")
    parser.add_argument("--native-bin", type=Path, default=ROOT / "target/release/examples/bge_benchmark")
    parser.add_argument("--rounds", type=int, default=30)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds 必须大于零")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    fixture = json.loads((ROOT / "cases.json").read_text())
    reference = args.reference_runtime_library
    if reference is None:
        native = native_bge(args.native_bin, args.model_dir, args.runtime_library, args.rounds)
        baseline = python_bge(fixture, args.model_dir.resolve(), args.rounds)
    else:
        baseline = native_bge(args.native_bin, args.model_dir, reference, args.rounds)
        native = native_bge(args.native_bin, args.model_dir, args.runtime_library, args.rounds)
    print_report(fixture, baseline, native, args.rounds, args.model_dir / "model_optimized.onnx",
                 reference_runtime=reference is not None)


if __name__ == "__main__":
    main()
