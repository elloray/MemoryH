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
import sqlite3
import unicodedata


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


def lexical_terms(text):
    # Each overlapping Han bigram is one FTS token. Latin words stay whole.
    for part in re.findall(r"[\u3400-\u9fff]+|[a-zA-Z0-9]+", unicodedata.normalize("NFKC", text)):
        if len(part) > 1 and "\u3400" <= part[0] <= "\u9fff":
            yield from (part[i:i + 2] for i in range(len(part) - 1))
        else:
            yield part


def initialize_database(db):
    db.executescript("""
        CREATE TABLE memory (
            id TEXT PRIMARY KEY,
            memory_key TEXT NOT NULL,
            source TEXT NOT NULL,
            body TEXT NOT NULL,
            active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1))
        );
        CREATE UNIQUE INDEX active_memory_key ON memory(memory_key) WHERE active = 1;
        CREATE VIRTUAL TABLE memory_fts USING fts5(id UNINDEXED, terms, tokenize='unicode61');
    """)


def save_memories(db, records):
    with db:
        for record in records:
            old = record.get("supersedes")
            if old:
                updated = db.execute(
                    "UPDATE memory SET active = 0 WHERE id = ? AND memory_key = ? AND active = 1",
                    (old, record["key"]),
                )
                if updated.rowcount != 1:
                    raise ValueError(f"更正未找到有效的旧版本：{old}")
                db.execute("DELETE FROM memory_fts WHERE id = ?", (old,))
            db.execute(
                "INSERT INTO memory(id, memory_key, source, body) VALUES (?, ?, ?, ?)",
                (record["id"], record["key"], record["source"], record["text"]),
            )
            db.execute(
                "INSERT INTO memory_fts(id, terms) VALUES (?, ?)",
                (record["id"], " ".join(lexical_terms(record["text"]))),
            )


def lexical_search(db, question, limit):
    terms = dict.fromkeys(lexical_terms(question))
    if not terms:
        return []
    expression = " OR ".join(f'"{term}"' for term in terms)
    return [row[0] for row in db.execute(
        "SELECT f.id FROM memory_fts f JOIN memory m ON m.id = f.id "
        "WHERE memory_fts MATCH ? AND m.active = 1 ORDER BY f.rank, f.id LIMIT ?",
        (expression, limit),
    )]


def query_variants(question):
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


def index_tree(active_keys, memory_vectors, root=TREE):
    labels = {node.key_prefix: node.label for node in descendants(root) if node.key_prefix}
    assignments = {memory_id: labels[key.split(".", 1)[0]] for memory_id, key in active_keys.items()}
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


def run_benchmark(fixture, model, top_k):
    questions = fixture["questions"]
    records = fixture["memories"] + fixture["corrections"]
    variants = {question["id"]: query_variants(question["text"]) for question in questions}
    by_text = embed_texts(model, (record["text"] for record in records))
    memory_vectors = {record["id"]: by_text[record["text"]] for record in records}
    query_vectors = embed_texts(model, (QUERY_PREFIX + text for group in variants.values() for text in group))
    results = {}
    leaf_counts = {}
    candidate_limit = top_k * 3

    with closing(sqlite3.connect(":memory:")) as db:
        initialize_database(db)
        for stage, additions in (("before", fixture["memories"]), ("after", fixture["corrections"])):
            save_memories(db, additions)
            active_keys = dict(db.execute("SELECT id, memory_key FROM memory WHERE active = 1"))
            indexed_ids = [row[0] for row in db.execute("SELECT id FROM memory_fts")]
            if sorted(active_keys) != sorted(indexed_ids):
                raise RuntimeError("SQLite 原文与 FTS5 当前版本不一致")
            leaves, assignments, node_vectors = index_tree(active_keys, memory_vectors)
            leaf_counts[stage] = {label: len(ids) for label, ids in sorted(leaves.items())}
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
                whole_text, *clauses = variants[question["id"]]
                vectors = [query_vectors[QUERY_PREFIX + text] for text in (whole_text, *clauses)]
                whole = search_query(
                    db, whole_text, vectors, active_keys, memory_vectors, leaves, node_vectors, candidate_limit,
                )
                subqueries = {
                    clause: search_query(
                        db, clause, [vector], active_keys, memory_vectors, leaves, node_vectors, candidate_limit,
                    )
                    for clause, vector in zip(clauses, vectors[1:], strict=True)
                }
                rankings = combined_rankings(whole, subqueries, top_k)
                results[question["id"]] = QuestionResult(whole, subqueries, rankings, expected_leaves)

        for record in fixture["corrections"]:
            old = db.execute("SELECT active, body FROM memory WHERE id = ?", (record["supersedes"],)).fetchone()
            if not old or old[0] or not old[1]:
                raise RuntimeError(f"更正后未保留旧版原文：{record['supersedes']}")
        # Read both current and archived text back from the source of truth for the report.
        evidence = {row[0]: (row[1], row[2]) for row in db.execute("SELECT id, source, body FROM memory")}
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


def print_cross_branch_ranks(questions, results, top_k):
    print("\n## 整题检索的跨分支证据排名\n")
    print(f"数字越小排名越高。FTS5 和向量每路最多前 {top_k * 3} 条进入 RRF，"
          f"最终只取前 {top_k} 条。`—` 表示这一步的列表中没有该证据。\n")
    print("| 问题 | 证据 | FTS5 候选 | 扁平向量原始排名 | 树内向量原始排名 | 扁平 RRF | 树 RRF |")
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


def print_subquery_ranks(results, top_k):
    print("\n## 子问题独立检索\n")
    print(f"各列显示该子问题的前 {top_k} 条原文。拆分方案按题面顺序从各子问题的 RRF 排名轮流取原文。"
          "向量优先方案先保留各子问题的向量第 1 名，再用同样的方式补齐。合并时按原文 ID 去重。\n")
    print("| 题号 | 子问题 | FTS5 | 扁平向量 | 扁平 RRF | 树向量 | 树 RRF |")
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


def print_report(fixture, results, evidence, leaf_counts, top_k):
    questions = fixture["questions"]
    print("# MemoryH 中文记忆检索实验\n")
    print(f"{len(evidence)} 条原文版本，{len(questions)} 个问题或时点。最终结果预算均为 Top {top_k}。"
          f"混合路线共用 `{MODEL}` 和 RRF 常数 {RRF_CONSTANT}，每次查询每路至多 {top_k * 3} 条融合候选。\n")
    print("| 用例 | " + " | ".join(METHODS.values()) + " |")
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
    stale = "、".join(f"{label} {totals[method][2]} 题" for method, label in METHODS.items())
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
        print_cross_branch_ranks(cross_branch, results, top_k)
    if len(unsplit) < len(results):
        print_subquery_ranks(results, top_k)

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
        evidence_lists.extend((label, result.rankings[method][:top_k]) for method, label in METHODS.items())
        for label, ids in evidence_lists:
            print(f"**{label}**\n")
            print("\n".join(f"- {format_evidence(memory_id)}" for memory_id in ids) or "无结果。")
            print()


def main():
    parser = argparse.ArgumentParser(description="对比中文记忆的 FTS5、FTS5 + 扁平向量和 FTS5 + 语义二叉树召回")
    parser.add_argument("--top-k", type=int, default=3, help="每种方法返回的原文条数，默认 3")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k 必须大于零")

    try:
        with closing(sqlite3.connect(":memory:")) as db:
            db.execute("CREATE VIRTUAL TABLE probe USING fts5(text)")
    except sqlite3.OperationalError as error:
        parser.exit(2, f"当前 Python 的 SQLite 无法启用 FTS5（{error}）。\n")

    from fastembed import TextEmbedding

    fixture = json.loads(Path(__file__).with_name("cases.json").read_text(encoding="utf-8"))
    model = TextEmbedding(MODEL, threads=2)
    report = run_benchmark(fixture, model, args.top_k)
    print_report(fixture, *report, args.top_k)


if __name__ == "__main__":
    main()
