# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["fastembed==0.9.1"]
# ///

import argparse
from collections import Counter
from contextlib import closing
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from statistics import mean
import sys
from urllib.request import urlopen

from benchmark import (
    BEAM_WIDTH, MODEL, QUERY_PREFIX, RRF_CONSTANT, Node, combined_rankings, embed_texts,
    index_tree, initialize_database, query_variants, save_memories, search_query,
)


TOP_K = 3
BEAM_SWEEP = (2, 3, 4)
DATA_REVISION = "e209267490c686980b29c1bd9571a01edcde1877"
DATA_SHA256 = "3e3d9aa70c59c35b0254553314e15a0df4a340a2aea7d3781b95c7d1944c7f5d"
DATA_URL = f"https://huggingface.co/datasets/justis-xu/memory-eval-zh/resolve/{DATA_REVISION}/locomo/locomo10_zh.json"
SCORER_URL = "https://github.com/snap-research/locomo/blob/main/task_eval/evaluation.py#L228-L237"
LABELS = {2: "单跳", 1: "多跳", 3: "时间", 4: "开放域", 5: "对抗"}
METHODS = {
    "fts": "纯 FTS5",
    "flat_split": "扁平向量 + 子问题 RRF",
    "tree_split": "二叉树 + 子问题 RRF",
    "flat_vector_first": "扁平向量 + 向量首位保留",
    "tree_vector_first": "二叉树 + 向量首位保留",
}
ROOT = Node("根", (
    Node("0", (Node("00", key_prefix="00"), Node("01", key_prefix="01"))),
    Node("1", (Node("10", key_prefix="10"), Node("11", key_prefix="11"))),
))


def load_data(path):
    if path is None:
        with urlopen(DATA_URL, timeout=30) as response:
            raw = response.read()
    else:
        raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != DATA_SHA256:
        raise ValueError("LoCoMo 中文数据的 SHA-256 与固定版本不符")
    return json.loads(raw)


def dialogue_records(sample):
    conversation = sample["conversation"]
    session_keys = sorted(
        (key for key in conversation if re.fullmatch(r"session_\d+", key)),
        key=lambda key: int(key.removeprefix("session_")),
    )
    for session_key in session_keys:
        for turn in conversation[session_key]:
            text = f'{turn["speaker"]}说：「{turn["text"]}」'
            if turn.get("blip_caption"):
                text += f' 分享了图片：{turn["blip_caption"]}'
            yield {"id": turn["dia_id"], "text": text, "source": conversation[f"{session_key}_date_time"]}


def split_semantically(ids, vectors):
    center = sum(vectors[record_id] for record_id in ids) / len(ids)
    first = min(ids, key=lambda record_id: (float(vectors[record_id] @ center), record_id))
    second = min(ids, key=lambda record_id: (float(vectors[record_id] @ vectors[first]), record_id))
    axis = vectors[second] - vectors[first]
    ordered = sorted(ids, key=lambda record_id: (float(vectors[record_id] @ axis), record_id))
    midpoint = len(ordered) // 2
    return ordered[:midpoint], ordered[midpoint:]


def assign_leaf_keys(records, vectors):
    left, right = split_semantically([record["id"] for record in records], vectors)
    for branch, ids in enumerate((left, right)):
        for leaf, members in enumerate(split_semantically(ids, vectors)):
            for record_id in members:
                yield record_id, f"{branch}{leaf}.{record_id}"


def trace_evidence(gold, active_keys, searches, final_ids, has_subqueries):
    searched = {memory_id for result in searches for memory_id in result.tree_vector}
    flat_candidates = {memory_id for result in searches for memory_id in result.rankings["flat"]}
    tree_candidates = {memory_id for result in searches for memory_id in result.rankings["tree"]}
    misses = {}
    kept_only_by_fts = []
    for memory_id in dict.fromkeys(gold):
        if memory_id in final_ids:
            continue
        if memory_id not in active_keys:
            reason = "missing_id"
        elif memory_id in tree_candidates:
            reason = "subquery_merge" if has_subqueries else "rrf"
            if memory_id not in searched:
                kept_only_by_fts.append(memory_id)
        elif memory_id in searched:
            reason = "vector_cutoff"
        elif memory_id in flat_candidates:
            reason = "pruned"
        else:
            reason = "unvisited_not_flat_candidate"
        misses[memory_id] = reason
    return {
        "misses": misses,
        "candidate_via_fts_outside_tree": kept_only_by_fts,
        "scored_turns": sum(len(result.tree_vector) for result in searches),
        "flat_scored_turns": sum(len(result.flat_vector) for result in searches),
    }


def retrieve_sample(sample, model, beam_sweep=False):
    records = list(dialogue_records(sample))
    qa = sample["qa"]
    queries = [query_variants(item["question"]) for item in qa]
    by_text = embed_texts(model, (record["text"] for record in records))
    memory_vectors = {record["id"]: by_text[record["text"]] for record in records}
    if len(memory_vectors) != len(records):
        raise ValueError(f'重复的原文 ID：{sample["sample_id"]}')
    active_keys = dict(assign_leaf_keys(records, memory_vectors))
    leaves, _, node_vectors = index_tree(active_keys, memory_vectors, ROOT)
    query_vectors = embed_texts(model, (QUERY_PREFIX + text for variants in queries for text in variants))
    widths = BEAM_SWEEP if beam_sweep else (BEAM_WIDTH,)
    with closing(sqlite3.connect(":memory:")) as db:
        initialize_database(db)
        save_memories(db, (dict(record, key=active_keys[record["id"]]) for record in records))
        for index, (item, variants) in enumerate(zip(qa, queries, strict=True), start=1):
            vectors = [query_vectors[QUERY_PREFIX + text] for text in variants]
            result = {
                "sample_id": sample["sample_id"], "question_number": index,
                "question": item["question"], "category": item["category"],
                "evidence": item["evidence"], "subqueries": variants[1:],
                "rankings": {},
            }
            if beam_sweep:
                result["beam_diagnostics"] = {}
            for width in widths:
                whole = search_query(db, variants[0], vectors, active_keys, memory_vectors,
                                     leaves, node_vectors, TOP_K * 3, ROOT, beam_width=width)
                subqueries = {
                    text: search_query(db, text, [vector], active_keys, memory_vectors,
                                       leaves, node_vectors, TOP_K * 3, ROOT, beam_width=width)
                    for text, vector in zip(variants[1:], vectors[1:], strict=True)
                }
                rankings = combined_rankings(whole, subqueries, TOP_K)
                if width == BEAM_WIDTH:
                    result["rankings"].update({method: rankings[method][:TOP_K] for method in METHODS})
                if beam_sweep:
                    key = str(width)
                    final_ids = rankings["tree_vector_first"][:TOP_K]
                    result["rankings"][f"beam_{key}"] = final_ids
                    searches = list(subqueries.values()) or [whole]
                    result["beam_diagnostics"][key] = trace_evidence(
                        item["evidence"], active_keys, searches, final_ids, bool(subqueries),
                    )
            yield result


def recall(case, method):
    # Match the official dialogue scorer, including its empty-evidence convention.
    gold = case["evidence"]
    retrieved = set(case["rankings"][method])
    return round(sum(record_id in retrieved for record_id in gold) / len(gold), 3) if gold else 1.0


def found_all(case, method):
    return set(case["evidence"]) <= set(case["rankings"][method])


def print_report(results):
    annotated = [case for case in results if case["evidence"]]
    answerable = [case for case in annotated if case["category"] != 5]
    fits = [case for case in answerable if len(set(case["evidence"])) <= TOP_K]
    split = [case for case in answerable if case["subqueries"]]
    print("# MemoryH · LoCoMo 中文证据检索\n")
    print(f"{len(results):,} 题 · `{MODEL}` · RRF {RRF_CONSTANT} · Top {TOP_K} · 每通道最多 {TOP_K * 3} 条融合候选。")
    print(f"\n[中文版数据]({DATA_URL}) · [LoCoMo 官方证据评分代码]({SCORER_URL})\n")
    print("原文按发言轮次检索。每段对话单独建索引，包含发言者和已发布的图片文字描述。")
    print("二叉树用原文向量递归中位切分生成四个叶子，导航每层保留两个候选节点。")
    print("两条混合路线共用原型的子问题拆分、RRF、向量首位预留和去重代码。\n")
    print("| 方法 | 官方 Evidence Recall@3 | 有标注题 Evidence Hit@3 | 可答题 Evidence Recall@3 | 可答且证据 ≤3 条的全证据命中 | 拆分可答题 Recall@3 |")
    print("|---|---:|---:|---:|---:|---:|")
    for method, label in METHODS.items():
        official = mean(recall(case, method) for case in results)
        hit_rate = mean(bool(set(case["evidence"]).intersection(case["rankings"][method])) for case in annotated)
        raw = mean(recall(case, method) for case in answerable)
        complete = sum(found_all(case, method) for case in fits)
        split_value = f"{mean(recall(case, method) for case in split):.2%}" if split else "—"
        print(f"| {label} | {official:.2%} | {hit_rate:.2%} | {raw:.2%} | {complete}/{len(fits)} ({complete / len(fits):.2%}) | {split_value} |")
    print(f"\n官方口径纳入所有问题且对无证据题记 1。Hit@3 计算 {len(annotated):,} 道有标注题中至少命中一条证据的比例。"
          f"可答口径取前四类中带证据的 {len(answerable):,} 题。其中 {len(fits):,} 题的全部证据能放入 Top {TOP_K}，"
          f"{len(split):,} 题实际走了子问题检索。全量有 {sum(bool(case['subqueries']) for case in results):,} 题触发拆分。")
    print("\n## 官方口径分题型\n")
    print("| 题型 | 题数 | " + " | ".join(METHODS.values()) + " |")
    print("|---|---:|" + "---:|" * len(METHODS))
    counts = Counter(case["category"] for case in results)
    for category, label in LABELS.items():
        rows = [case for case in results if case["category"] == category]
        scores = [f"{mean(recall(case, method) for case in rows):.2%}" for method in METHODS]
        print(f"| {label} | {counts[category]} | " + " | ".join(scores) + " |")


def print_beam_report(results):
    flat = "flat_vector_first"
    multi_hop = [case for case in results if case["category"] == 1]
    total_work = sum(case["beam_diagnostics"][str(BEAM_SWEEP[0])]["flat_scored_turns"] for case in results)
    print("# MemoryH · LoCoMo 中文 Beam 宽度对照\n")
    print(f"{len(results):,} 道题，含 {len(multi_hop)} 道多跳题。MemoryH 使用 `{MODEL}`、RRF {RRF_CONSTANT}、"
          f"Top {TOP_K} 和子问题向量首位保留。词法和向量通道各取最多 {TOP_K * 3} 条进入 RRF。\n")
    print("## Beam 宽度与扁平基线\n")
    print("各行使用官方发言级 Evidence Recall@3，统一题集、发言 ID 和评分函数。"
          "打分量按最终使用的查询累计，子问题各计一次，树节点向量计算另计。\n")
    print("| 方法 | 全量 Recall@3 | 多跳 Recall@3 | 原文打分量 / 扁平路线 |")
    print("|---|---:|---:|---:|")
    methods = {flat: ("FTS5 + 扁平向量", "100.00%")}
    for width in BEAM_SWEEP:
        work = sum(case["beam_diagnostics"][str(width)]["scored_turns"] for case in results)
        methods[f"beam_{width}"] = f"FTS5 + 二叉树 · Beam {width}", f"{work / total_work:.2%}"
    for method, (label, coverage) in methods.items():
        overall = mean(recall(case, method) for case in results)
        multi = mean(recall(case, method) for case in multi_hop)
        print(f"| {label} | {overall:.2%} | {multi:.2%} | {coverage} |")

    print("\n## 扁平 Top 3 命中、树 Top 3 漏掉的标注发言\n")
    print("同一发言在不同题中分别计数，同一题内重复 ID 只计一次。仅 FTS5 入候选指发言的叶子未被访问，"
          "但词法通道将它送入了 RRF。末列是可证实失去扁平向量加分的下限，只计未拆分、FTS5 返回满三条且证据排在 FTS5 Top 3 之外的题。\n")
    print("| 范围 | Beam | 扁平独有命中 | 树独有命中 | 叶子被剪且未进候选 | 向量前 9 截断 | RRF 排出 Top 3 | 子问题合并漏掉 | 其中仅 FTS5 入候选 | 至少失去扁平向量加分 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    reasons = ("pruned", "vector_cutoff", "rrf", "subquery_merge")
    for label, cases in (("全量", results), ("多跳", multi_hop)):
        for width in BEAM_SWEEP:
            stages = Counter()
            flat_only = tree_only = fts_only = lost_vector_boost = 0
            for case in cases:
                gold = set(case["evidence"])
                flat_hits = gold.intersection(case["rankings"][flat])
                tree_hits = gold.intersection(case["rankings"][f"beam_{width}"])
                lost = flat_hits - tree_hits
                flat_only += len(lost)
                tree_only += len(tree_hits - flat_hits)
                trace = case["beam_diagnostics"][str(width)]
                stages.update(trace["misses"][memory_id] for memory_id in lost)
                lexical_rescues = lost.intersection(trace["candidate_via_fts_outside_tree"])
                fts_only += len(lexical_rescues)
                fts_top = case["rankings"]["fts"]
                if not case["subqueries"] and len(fts_top) == TOP_K:
                    lost_vector_boost += len(lexical_rescues.difference(fts_top))
            if sum(stages[reason] for reason in reasons) != flat_only:
                raise ValueError("扁平命中而树漏召的证据未能归因")
            print(f"| {label} | {width} | {flat_only} | {tree_only} | " +
                  " | ".join(str(stages[reason]) for reason in reasons) + f" | {fts_only} | {lost_vector_boost} |")

    print("\n## 多跳题的全部标注发言\n")
    print("| Beam | 命中 | 剪枝且扁平可进入候选 | 叶子未访问且扁平也未进候选 | 向量前 9 截断 | RRF 排出 Top 3 | 子问题合并漏掉 | ID 缺失 |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    all_reasons = ("pruned", "unvisited_not_flat_candidate", "vector_cutoff", "rrf", "subquery_merge", "missing_id")
    for width in BEAM_SWEEP:
        misses = Counter(reason for case in multi_hop
                         for reason in case["beam_diagnostics"][str(width)]["misses"].values())
        hit = sum(len(set(case["evidence"]).intersection(case["rankings"][f"beam_{width}"])) for case in multi_hop)
        print(f"| {width} | {hit} | " + " | ".join(str(misses[reason]) for reason in all_reasons) + " |")

    equal = sum(case["rankings"]["beam_4"] == case["rankings"][flat] for case in results)
    print(f"\nBeam 4 与扁平路线的 Top 3 发言 ID 及顺序一致 {equal:,}/{len(results):,} 题。")
    print(f"\n[中文版数据]({DATA_URL}) · [LoCoMo 官方证据评分代码]({SCORER_URL})")
    print("\nAML Cycle 2 文本榜使用多个题集、每题最多 100 条检索结果与端到端答案评分，"
          "Nano-Memory 的公开 Recall@3 使用会话作为检索单位。"
          " [AML 官方 FAQ](https://agentmemories.ai/blog?post=phase2-faq-20260920)"
          " · [Nano-Memory 原论文](https://arxiv.org/html/2604.11628v1)")


def main():
    parser = argparse.ArgumentParser(description="运行 LoCoMo 中文版的 MemoryH Top 3 原文召回对照")
    parser.add_argument("--data", type=Path, help="指定已下载的同版本中文 JSON")
    parser.add_argument("--predictions", type=Path, help="输出逐题证据 ID 和召回结果 JSONL")
    parser.add_argument("--beam-sweep", action="store_true", help="对照 Beam 2、3、4 并记录证据漏召阶段")
    args = parser.parse_args()

    from fastembed import TextEmbedding

    samples = load_data(args.data)
    model = TextEmbedding(MODEL, threads=2)
    results = []
    for sample in samples:
        results.extend(retrieve_sample(sample, model, args.beam_sweep))
        print(f'{sample["sample_id"]} · 累计 {len(results)} 题', file=sys.stderr, flush=True)
    if args.predictions:
        with args.predictions.open("w", encoding="utf-8") as output:
            for case in results:
                print(json.dumps(case, ensure_ascii=False), file=output)
    if args.beam_sweep:
        print_beam_report(results)
    else:
        print_report(results)


if __name__ == "__main__":
    main()
