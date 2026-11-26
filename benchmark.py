# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["fastembed==0.9.1"]
# ///

import argparse
from collections import Counter, defaultdict
from contextlib import closing
from dataclasses import dataclass
import json
from math import sqrt
from pathlib import Path
import re

from retrieval_storage import RustStorage, lexical_search, open_retrieval_store


MODEL = "BAAI/bge-small-zh-v1.5"
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
RRF_CONSTANT = 60
BEAM_WIDTH = 2
STAGES = {"before": "更正前", "after": "更正后"}
METHODS = {
    "fts": "FTS5",
    "flat": "FTS5 + 扁平向量",
    "flat_split": "FTS5 + 扁平向量（拆分）",
    "flat_vector_first": "FTS5 + 扁平向量（向量优先）",
    "tree": "FTS5 + 树",
    "tree_split": "FTS5 + 树（拆分）",
    "tree_vector_first": "FTS5 + 树（向量优先）",
}
HYBRID_METHODS = ("flat", "tree")


@dataclass(frozen=True)
class Node:
    label: str
    children: tuple["Node", ...] = ()
    key_prefix: str | None = None


@dataclass
class RetrievalResult:
    rankings: dict[str, list[str]]
    flat_vector: list[str]
    tree_vector: list[str]
    routes: list[str]


@dataclass
class QuestionResult:
    whole: RetrievalResult
    subqueries: dict[str, RetrievalResult]
    rankings: dict[str, list[str]]
    expected_leaves: list[str]


TREE = Node("记忆", (
    Node("工作", (
        Node("差旅财务", key_prefix="travel"),
        Node("项目工程", key_prefix="work"),
    )),
    Node("个人", (
        Node("饮食健康", key_prefix="health"),
        Node("居家日常", key_prefix="home"),
    )),
))


def descendants(node):
    for child in node.children:
        yield child
        yield from descendants(child)


def query_variants(question, *, storage="sqlite"):
    if storage == "rust":
        from memoryh_store import query_variants as native_variants

        return native_variants(question)
    clauses = re.split(r"[？?；;。！!]|另外|以及|并且|还有", question)
    clauses = [clause.strip(" ，,、") for clause in clauses]
    clauses = [clause for clause in clauses if len(clause) >= 4]
    return [question, *dict.fromkeys(clauses)] if len(clauses) > 1 else [question]


def normalized(vector):
    return vector / sqrt(float(vector @ vector))


def embed_texts(model, texts):
    texts = list(dict.fromkeys(texts))
    return {text: normalized(vector) for text, vector in zip(texts, model.embed(texts), strict=True)}


def route(node_vectors, vector, root=TREE, beam_width=BEAM_WIDTH):
    frontier = [(root, 0.0)]
    while any(node.children for node, _ in frontier):
        candidates = [
            (child, score + float(vector @ node_vectors[child]))
            for node, score in frontier
            for child in node.children
            if child in node_vectors
        ]
        frontier = sorted(candidates, key=lambda entry: -entry[1])[:beam_width]
    return [node.label for node, _ in frontier]


def assign_leaves(active_keys, root):
    labels = {node.key_prefix: node.label for node in descendants(root) if node.key_prefix}
    return {memory_id: labels[key.split(".", 1)[0]] for memory_id, key in active_keys.items()}


def index_tree(active_keys, memory_vectors, root=TREE):
    assignments = assign_leaves(active_keys, root)
    members = defaultdict(list)
    for memory_id, label in assignments.items():
        members[label].append(memory_id)

    node_vectors = {}
    for node in descendants(root):
        leaves = [child for child in descendants(node) if child.key_prefix] if node.children else [node]
        vectors = [memory_vectors[memory_id] for leaf in leaves for memory_id in members[leaf.label]]
        if vectors:
            node_vectors[node] = normalized(sum(vectors))
    return members, assignments, node_vectors


def rank_vectors(candidate_ids, memory_vectors, vectors):
    def rank(memory_id):
        score = max(float(vector @ memory_vectors[memory_id]) for vector in vectors)
        return -score, memory_id

    return sorted(candidate_ids, key=rank)


def fuse(rankings):
    scores = Counter()
    for ranking in rankings:
        for rank, memory_id in enumerate(ranking, start=1):
            scores[memory_id] += 1 / (RRF_CONSTANT + rank)
    return sorted(scores, key=lambda memory_id: (-scores[memory_id], memory_id))


def search_query(db, text, vectors, active_ids, memory_vectors, leaves, node_vectors, limit, root=TREE,
                 beam_width=BEAM_WIDTH):
    lexical = lexical_search(db, text, limit)
    routes = list(dict.fromkeys(
        label for vector in vectors for label in route(node_vectors, vector, root, beam_width)
    ))
    tree_ids = [memory_id for label in routes for memory_id in leaves.get(label, ())]
    flat_vector = rank_vectors(active_ids, memory_vectors, vectors)
    tree_vector = rank_vectors(tree_ids, memory_vectors, vectors)
    rankings = {
        "fts": lexical,
        "flat": fuse((lexical, flat_vector[:limit])),
        "tree": fuse((lexical, tree_vector[:limit])),
    }
    return RetrievalResult(rankings, flat_vector, tree_vector, routes)


def merge_subqueries(rankings, limit, reserved=()):
    # Each round gives every subquery one chance to contribute a new document.
    queues = [iter(ranking) for ranking in rankings]
    unique_reservations = list(dict.fromkeys(reserved))
    picked = dict.fromkeys(unique_reservations[:limit])
    while len(picked) < limit:
        count_before = len(picked)
        for queue in queues:
            next_id = next((memory_id for memory_id in queue if memory_id not in picked), None)
            if next_id is not None:
                picked[next_id] = None
            if len(picked) == limit:
                break
        if len(picked) == count_before:
            break
    return list(picked)


def combined_rankings(whole, subqueries, top_k):
    rankings = dict(whole.rankings)
    for method in HYBRID_METHODS:
        if not subqueries:
            rankings[f"{method}_split"] = whole.rankings[method]
            rankings[f"{method}_vector_first"] = whole.rankings[method]
            continue
        per_query_rrf = [result.rankings[method] for result in subqueries.values()]
        vector_rankings = (
            result.flat_vector if method == "flat" else result.tree_vector
            for result in subqueries.values()
        )
        reserved = [ranking[0] for ranking in vector_rankings if ranking]
        rankings[f"{method}_split"] = merge_subqueries(per_query_rrf, top_k)
        rankings[f"{method}_vector_first"] = merge_subqueries(per_query_rrf, top_k, reserved=reserved)
    return rankings


class BenchmarkSearch:
    """Adapt ranking output for reports while preserving the SQLite baseline."""

    def __init__(self, store, memory_vectors, root=TREE, *, semantic=False):
        self.store = store
        self.active_keys = store.active_keys()
        self.memory_vectors = memory_vectors
        self.root = root
        if isinstance(store, RustStorage):
            self.prefixes = [
                [(leaf.label, leaf.key_prefix) for leaf in branch.children]
                for branch in root.children
            ] if not semantic else None
        else:
            self.leaves, _, self.node_vectors = index_tree(self.active_keys, memory_vectors, root)

    def search(self, variants, vectors, top_k, beam_width=BEAM_WIDTH):
        if isinstance(self.store, RustStorage):
            whole, parts, flat_split, tree_split, flat_first, tree_first = self.store.search(
                variants, vectors, top_k, beam_width, self.prefixes,
            )

            def unpack(row):
                lexical, flat, tree, flat_vector, tree_vector, routes = row
                return RetrievalResult({"fts": lexical, "flat": flat, "tree": tree}, flat_vector, tree_vector, routes)

            whole = unpack(whole)
            subqueries = {text: unpack(row) for text, row in zip(variants[1:], parts, strict=True)}
            rankings = dict(whole.rankings, flat_split=flat_split, tree_split=tree_split,
                            flat_vector_first=flat_first, tree_vector_first=tree_first)
            return whole, subqueries, rankings

        def run(text, embeddings):
            return search_query(self.store.lexical_db, text, embeddings, self.active_keys, self.memory_vectors,
                                self.leaves, self.node_vectors, top_k * 3, self.root, beam_width=beam_width)

        whole = run(variants[0], vectors)
        subqueries = {text: run(text, [vector]) for text, vector in zip(variants[1:], vectors[1:], strict=True)}
        return whole, subqueries, combined_rankings(whole, subqueries, top_k)


def run_benchmark(fixture, model, top_k, *, storage="sqlite"):
    questions = fixture["questions"]
    records = fixture["memories"] + fixture["corrections"]
    variants = {question["id"]: query_variants(question["text"], storage=storage) for question in questions}
    by_text = embed_texts(model, (record["text"] for record in records))
    memory_vectors = {record["id"]: by_text[record["text"]] for record in records}
    query_vectors = embed_texts(model, (QUERY_PREFIX + text for group in variants.values() for text in group))
    results = {}
    evidence = {}
    leaf_counts = {}

    with open_retrieval_store(storage) as store:
        for stage, additions in (("before", fixture["memories"]), ("after", fixture["corrections"])):
            store.apply(additions, memory_vectors)
            search = BenchmarkSearch(store, memory_vectors)
            active_keys = search.active_keys
            assignments = assign_leaves(active_keys, TREE)
            leaf_counts[stage] = dict(Counter(assignments.values()))
            for question in questions:
                if question["stage"] != stage:
                    continue
                expected = set(question["expected"])
                forbidden = set(question.get("forbidden", []))
                if not expected or not expected <= active_keys.keys() or forbidden & active_keys.keys():
                    raise ValueError(f"证据标注与当前版本不符：{question['id']}")
                expected_leaves = [assignments[mid] for mid in question["expected"]]
                if question["group"] == "跨分支" and len(set(expected_leaves)) < 2:
                    raise ValueError(f"跨分支用例的证据落在同一叶子：{question['id']}")
                texts = variants[question["id"]]
                vectors = [query_vectors[QUERY_PREFIX + text] for text in texts]
                whole, subqueries, rankings = search.search(texts, vectors, top_k)
                selected = dict.fromkeys(memory_id for method in METHODS for memory_id in rankings[method][:top_k])
                evidence.update((memory_id, store.read(memory_id)) for memory_id in selected)
                results[question["id"]] = QuestionResult(whole, subqueries, rankings, expected_leaves)

        for record in records:
            memory_id = record["id"]
            if memory_id not in evidence:
                evidence[memory_id] = store.read(memory_id)
            if evidence[memory_id] is None:
                raise RuntimeError(f"存储中没有原文：{memory_id}")
        for record in fixture["corrections"]:
            old_id = record["supersedes"]
            old = store.read(old_id)
            if old_id in active_keys or not old or not old[1]:
                raise RuntimeError(f"更正后未保留旧版原文：{old_id}")
    return results, evidence, leaf_counts


def metrics(questions, results, method, top_k):
    complete, found, expected, stale = 0, 0, 0, 0
    for question in questions:
        retrieved = set(results[question["id"]].rankings[method][:top_k])
        required = set(question["expected"])
        blocked = set(question.get("forbidden", []))
        found += len(required & retrieved)
        expected += len(required)
        leaked = bool(blocked & retrieved)
        stale += leaked
        complete += required <= retrieved and not leaked
    return f"{complete}/{len(questions)}", f"{found}/{expected}", stale


def print_cross_branch_ranks(questions, results, top_k, lexical):
    print("\n## 整题检索的跨分支证据排名\n")
    print(f"数字越小排名越高。{lexical} 和向量每路最多前 {top_k * 3} 条进入 RRF，"
          f"最终只取前 {top_k} 条。`—` 表示这一步的列表中没有该证据。\n")
    print(f"| 问题 | 证据 | {lexical} 候选 | 扁平向量原始排名 | 树内向量原始排名 | 扁平 RRF | 树 RRF |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for question in questions:
        result = results[question["id"]].whole
        columns = (
            result.rankings["fts"], result.flat_vector, result.tree_vector,
            result.rankings["flat"], result.rankings["tree"],
        )
        positions = [{memory_id: rank for rank, memory_id in enumerate(ids, 1)} for ids in columns]
        for memory_id in question["expected"]:
            ranks = " | ".join(str(column.get(memory_id, "—")) for column in positions)
            print(f"| {question['id']} | `{memory_id}` | {ranks} |")


def print_subquery_ranks(results, top_k, lexical):
    print("\n## 子问题独立检索\n")
    print(f"各列显示该子问题的前 {top_k} 条原文。拆分方案按题面顺序从各子问题的 RRF 排名轮流取原文。"
          "向量优先方案先保留各子问题的向量第 1 名，再用同样的方式补齐。合并时按原文 ID 去重。\n")
    print(f"| 题号 | 子问题 | {lexical} | 扁平向量 | 扁平 RRF | 树向量 | 树 RRF |")
    print("|---|---|---|---|---|---|---|")
    for question_id, result in sorted(results.items()):
        for text, subquery in result.subqueries.items():
            rankings = (
                subquery.rankings["fts"], subquery.flat_vector, subquery.rankings["flat"],
                subquery.tree_vector, subquery.rankings["tree"],
            )
            cells = ["、".join(f"`{memory_id}`" for memory_id in ranking[:top_k]) or "—"
                     for ranking in rankings]
            print(f"| {question_id} | {text} | " + " | ".join(cells) + " |")


def print_report(fixture, results, evidence, leaf_counts, top_k, *, storage="sqlite"):
    questions = fixture["questions"]
    lexical = "Rust BM25" if storage == "rust" else "FTS5"
    methods = {method: label.replace("FTS5", lexical) for method, label in METHODS.items()}
    print("# MemoryH 中文记忆检索实验\n")
    print(f"{len(evidence)} 条原文版本，{len(questions)} 个问题或时点。最终结果预算均为 Top {top_k}。"
          f"混合路线共用 `{MODEL}` 和 RRF 常数 {RRF_CONSTANT}，每次查询每路至多 {top_k * 3} 条融合候选。\n")
    print("| 用例 | " + " | ".join(methods.values()) + " |")
    print("|---|" + "---:|" * len(METHODS))
    groups = {group: [q for q in questions if q["group"] == group]
              for group in dict.fromkeys(q["group"] for q in questions)}
    corrections = groups.pop("记忆更正", [])
    for stage, label in STAGES.items():
        if subset := [q for q in corrections if q["stage"] == stage]:
            groups[f"记忆更正（{label}）"] = subset
    totals = {method: metrics(questions, results, method, top_k) for method in METHODS}
    for group, subset in groups.items():
        row = [metrics(subset, results, method, top_k) for method in METHODS]
        print(f"| {group} | " + " | ".join(f"{complete} 题 · {found} 条" for complete, found, _ in row) + " |")
    print("| 全部 | " + " | ".join(f"{complete} 题 · {found} 条" for complete, found, _ in totals.values()) + " |")

    print("\n题数按找齐该题的全部原文且无禁止版本计算。条数按找回的证据原文计算。")
    stale = "、".join(f"{label} {totals[method][2]} 题" for method, label in methods.items())
    print(f"\n禁止版本误召回为 {stale}。\n")
    unsplit = [result for result in results.values() if not result.subqueries]
    print(f"有 {len(results) - len(unsplit)} 题识别出多个子问题。其余 {len(unsplit)} 题复用整题检索结果。\n")
    flat_examined = sum(len(result.whole.flat_vector) for result in results.values())
    tree_examined = sum(len(result.whole.tree_vector) for result in results.values())
    print(f"整题对照中，扁平路线逐条为 {flat_examined} 条原文打分，树路线逐条为 {tree_examined} 条原文打分。"
          "这个计数按题累加，分支节点打分另计。\n")
    for stage, counts in leaf_counts.items():
        print(f"- {STAGES[stage]}的叶子条数为 " + "、".join(f"{label} {count}" for label, count in counts.items()) + "。")

    if cross_branch := groups.get("跨分支"):
        print_cross_branch_ranks(cross_branch, results, top_k, lexical)
    if len(unsplit) < len(results):
        print_subquery_ranks(results, top_k, lexical)

    print("\n## 逐题召回原文")

    def format_evidence(memory_id):
        source, body = evidence[memory_id]
        return f"`{memory_id}` {source} · {body}"

    for question in questions:
        result = results[question["id"]]
        print(f"\n### {question['id']} · {question['text']}\n")
        expected_leaves = "、".join(dict.fromkeys(result.expected_leaves))
        print(f"{question['group']} · {STAGES[question['stage']]} · 证据分支 {expected_leaves}")
        print(f"\n整题对照的树搜索 {'、'.join(result.whole.routes)}（{len(result.whole.tree_vector)} 条原文），"
              f"扁平搜索全部 {len(result.whole.flat_vector)} 条有效原文。\n")
        evidence_lists = [("应找回", question["expected"])]
        if question.get("forbidden"):
            evidence_lists.append(("禁止版本", question["forbidden"]))
        evidence_lists.extend((label, result.rankings[method][:top_k]) for method, label in methods.items())
        for label, ids in evidence_lists:
            print(f"**{label}**\n")
            print("\n".join(f"- {format_evidence(memory_id)}" for memory_id in ids) or "无结果。")
            print()


def main():
    parser = argparse.ArgumentParser(description="对比中文记忆的关键词、扁平向量混合和二叉树混合召回")
    parser.add_argument("--top-k", type=int, default=3, help="每种方法返回的原文条数，默认 3")
    parser.add_argument("--storage", choices=("sqlite", "rust"), default="sqlite", help="存储与检索后端")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k 必须大于零")

    if args.storage == "sqlite":
        import sqlite3

        try:
            with closing(sqlite3.connect(":memory:")) as db:
                db.execute("CREATE VIRTUAL TABLE probe USING fts5(text)")
        except sqlite3.OperationalError as error:
            parser.exit(2, f"当前 Python 的 SQLite 无法启用 FTS5（{error}）。\n")

    from fastembed import TextEmbedding

    fixture = json.loads(Path(__file__).with_name("cases.json").read_text(encoding="utf-8"))
    model = TextEmbedding(MODEL, threads=2)
    report = run_benchmark(fixture, model, args.top_k, storage=args.storage)
    print_report(fixture, *report, args.top_k, storage=args.storage)


if __name__ == "__main__":
    main()
