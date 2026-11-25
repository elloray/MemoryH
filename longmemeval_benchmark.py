# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["fastembed==0.9.1", "numpy==1.26.4", "rank-bm25==0.2.2"]
# ///

import argparse
from contextlib import closing
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
from statistics import mean
import sys
from tempfile import NamedTemporaryFile

import numpy as np
from rank_bm25 import BM25Okapi

from benchmark import (
    RRF_CONSTANT, combined_rankings, index_tree, initialize_database, normalized,
    query_variants, save_memories, search_query,
)
from locomo_benchmark import ROOT, assign_leaf_keys


DATA_REVISION = "98d7416c24c778c2fee6e6f3006e7a073259d48f"
DATA_SHA256 = "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
OFFICIAL_REVISION = "9e0b455f4ef0e2ab8f2e582289761153549043fc"
SCORER_SHA256 = "c98b8d1096877a15aa755c9de44fe33c195298466a2eb6f3c0f9f6bde8c72349"
MODEL = "BAAI/bge-small-en-v1.5"
MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
TOP_KS = (5, 10)
MAX_K = max(TOP_KS)
METHODS = {
    "official_bm25": "官方 BM25",
    "fts": "MemoryH FTS5",
    "flat": "MemoryH FTS5 + 扁平向量",
    "tree_2": "MemoryH FTS5 + 二叉树 · Beam 2",
    "tree_3": "MemoryH FTS5 + 二叉树 · Beam 3",
}


def read_data(path):
    with path.open("rb") as source:
        actual = hashlib.file_digest(source, "sha256").hexdigest()
    if actual != DATA_SHA256:
        raise ValueError("数据文件与官方 LongMemEval-S cleaned 英文版的固定版本不符")
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def load_official_scorer(path):
    if hashlib.sha256(path.read_bytes()).hexdigest() != SCORER_SHA256:
        raise ValueError("eval_utils.py 与固定版本的官方评分脚本不符")
    spec = importlib.util.spec_from_file_location("longmemeval_official_scorer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.evaluate_retrieval


def official_corpus(entry):
    texts, labels = [], []
    for session_id, session in zip(entry["haystack_session_ids"], entry["haystack_sessions"], strict=True):
        user_turns = [turn for turn in session if turn["role"] == "user"]
        texts.append(" ".join(turn["content"] for turn in user_turns))
        if "answer" in session_id and all(not turn["has_answer"] for turn in user_turns):
            session_id = session_id.replace("answer", "noans")
        labels.append(session_id)
    return texts, labels


def embed_corpus(model, texts, cache):
    texts = sorted(set(texts), key=lambda text: (len(text), text))
    embeddings = {}
    step = 1024
    for start in range(0, len(texts), step):
        batch = texts[start:start + step]
        path = cache / f"{start:05d}.npy" if cache else None
        if path and path.exists():
            vectors = np.load(path, allow_pickle=False)
        else:
            vectors = np.stack(list(model.embed(batch, batch_size=32)))
            if path:
                with NamedTemporaryFile(dir=cache, delete=False) as temporary:
                    np.save(temporary, vectors, allow_pickle=False)
                Path(temporary.name).replace(path)
        embeddings.update((text, normalized(vector)) for text, vector in zip(batch, vectors, strict=True))
        print(f"BGE 向量 {len(embeddings):,}/{len(texts):,}", file=sys.stderr, flush=True)
    return embeddings


class TorchBGE:
    def __init__(self, device):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.device = device
        self.torch = torch
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("当前环境无法使用 PyTorch MPS")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
        self.model = AutoModel.from_pretrained(MODEL, revision=MODEL_REVISION)
        self.model.eval().to(device, dtype=torch.float16 if device == "mps" else torch.float32)

    def embed(self, texts, batch_size):
        batch_size = min(batch_size, 16) if self.device == "mps" else batch_size
        for start in range(0, len(texts), batch_size):
            batch = self.tokenizer(texts[start:start + batch_size], padding=True, truncation=True,
                                   max_length=512, pad_to_multiple_of=32, return_tensors="pt")
            with self.torch.inference_mode():
                embeddings = self.model(**batch.to(self.device)).last_hidden_state[:, 0, :]
            yield from embeddings.detach().float().cpu().numpy()
        if self.device == "mps":
            del batch, embeddings
            self.torch.mps.empty_cache()


def memoryh_rankings(entry, texts, by_text):
    # Ranking uses neutral position IDs. IDs containing the official gold marker stay in scoring.
    records = [{"id": f"S{position:04d}", "text": text, "source": date}
               for position, (text, date) in enumerate(zip(texts, entry["haystack_dates"], strict=True))]
    memory_vectors = {record["id"]: by_text[record["text"]] for record in records}
    keys = dict(assign_leaf_keys(records, memory_vectors))
    leaves, _, node_vectors = index_tree(keys, memory_vectors, ROOT)
    variants = query_variants(entry["question"])
    vectors = [by_text[QUERY_PREFIX + variant] for variant in variants]
    rankings = {}
    with closing(sqlite3.connect(":memory:")) as db:
        initialize_database(db)
        save_memories(db, (dict(record, key=keys[record["id"]]) for record in records))
        for width in (2, 3):
            whole = search_query(db, variants[0], vectors, keys, memory_vectors, leaves, node_vectors,
                                 MAX_K * 3, ROOT, beam_width=width)
            parts = {
                text: search_query(db, text, [vector], keys, memory_vectors, leaves, node_vectors,
                                   MAX_K * 3, ROOT, beam_width=width)
                for text, vector in zip(variants[1:], vectors[1:], strict=True)
            }
            combined = combined_rankings(whole, parts, MAX_K)
            if width == 2:
                rankings["fts"] = combined["fts"]
                rankings["flat"] = combined["flat_vector_first"]
            rankings[f"tree_{width}"] = combined["tree_vector_first"]
    return {method: [int(item[1:]) for item in ranks[:MAX_K]] for method, ranks in rankings.items()}


def evaluate(entries, scorer, model, prediction_path, embedding_cache):
    corpora = [official_corpus(entry) for entry in entries]
    questions = [QUERY_PREFIX + text for entry in entries for text in query_variants(entry["question"])]
    texts = [text for corpus, _ in corpora for text in corpus] + questions
    by_text = embed_corpus(model, texts, embedding_cache)

    results = []
    with prediction_path.open("w", encoding="utf-8") as output:
        for index, (entry, (texts, labels)) in enumerate(zip(entries, corpora, strict=True), 1):
            official = BM25Okapi([text.split(" ") for text in texts])
            bm25 = official.get_scores(entry["question"].split(" "))
            rankings = {"official_bm25": np.argsort(bm25)[::-1][:MAX_K].tolist()}
            rankings.update(memoryh_rankings(entry, texts, by_text))
            correct_docs = list({corpus_id for corpus_id in labels if "answer" in corpus_id})
            methods = {}
            for method, positions in rankings.items():
                metrics = {}
                for k in TOP_KS:
                    values = scorer(positions, correct_docs, labels, k=k)
                    for key, score in zip(("recall_any", "recall_all", "ndcg_any"), values, strict=True):
                        metrics[f"{key}@{k}"] = float(score)
                methods[method] = {"positions": positions, "corpus_ids": [labels[i] for i in positions],
                                   "metrics": metrics}
            result = {
                "question_id": entry["question_id"], "question_type": entry["question_type"],
                "scored_evidence": sorted(correct_docs), "methods": methods,
            }
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), file=output)
            if index % 50 == 0:
                output.flush()
                print(f"检索评测 {index}/{len(entries)}", file=sys.stderr, flush=True)
    return results


def export_generation_input(entries, results, method, path):
    # The official generator resolves selected IDs against the source sessions itself.
    with path.open("w", encoding="utf-8") as output:
        for entry, result in zip(entries, results, strict=True):
            positions = result["methods"][method]["positions"]
            texts, labels = official_corpus(entry)
            source_ids = entry["haystack_session_ids"]
            selected_ids = {source_ids[position] for position in positions}
            selected = [position for position, session_id in enumerate(source_ids) if session_id in selected_ids]
            exported = {key: entry[key] for key in ("question_id", "question_type", "question", "answer", "question_date")}
            for key in ("haystack_dates", "haystack_session_ids", "haystack_sessions"):
                exported[key] = [entry[key][position] for position in selected]
            exported["retrieval_results"] = {
                "ranked_items": [{"corpus_id": labels[i], "text": texts[i], "timestamp": entry["haystack_dates"][i]}
                                 for i in positions],
            }
            print(json.dumps(exported, ensure_ascii=False), file=output)


def print_report(results):
    valid = [row for row in results if "_abs" not in row["question_id"]]
    has_gold = [row for row in valid if row["scored_evidence"]]
    print("# MemoryH · LongMemEval-S 官方检索评测\n")
    print(f"官方英文 cleaned 版，检索输出 {len(results)} 题，官方检索分数覆盖 {len(valid)} 题。"
          f"会话级检索，`{MODEL}`，RRF {RRF_CONSTANT}，每通道最多 {MAX_K * 3} 条候选。\n")
    print("每个问题独立建索引。按官方 session 流程，仅用用户发言生成检索键。"
          "扁平和树路线保留现有的子问题向量首位策略。Top 5 与 Top 10 取自同一排序结果。\n")
    print("| 方法 | Recall-All@5 | nDCG@5 | Recall-All@10 | nDCG@10 |")
    print("|---|---:|---:|---:|---:|")
    for method, label in METHODS.items():
        values = [mean(row["methods"][method]["metrics"][key] for row in valid)
                  for key in ("recall_all@5", "ndcg_any@5", "recall_all@10", "ndcg_any@10")]
        print(f"| {label} | " + " | ".join(f"{value:.2%}" for value in values) + " |")
    print(f"\n官方脚本跳过 {len(results) - len(valid)} 道 abstention 题。"
          f"会话重标后有 {len(valid) - len(has_gold)} 道可答题没有正例，官方评分器对此记 Recall-All 1、nDCG 0。\n")
    print("## 有会话正例的题\n")
    print(f"以下补充统计覆盖 {len(has_gold)} 题，展示 Recall-All@5 和 Recall-All@10。\n")
    print("| 方法 | Recall-All@5 | Recall-All@10 |")
    print("|---|---:|---:|")
    for method, label in METHODS.items():
        scores = [mean(row["methods"][method]["metrics"][f"recall_all@{k}"] for row in has_gold) for k in TOP_KS]
        print(f"| {label} | {scores[0]:.2%} | {scores[1]:.2%} |")


def main():
    parser = argparse.ArgumentParser(description="按官方 LongMemEval-S 会话级协议评估 MemoryH")
    parser.add_argument("--data", type=Path, required=True, help="官方英文 longmemeval_s_cleaned.json")
    parser.add_argument("--official-scorer", type=Path, help="固定版本的官方 src/retrieval/eval_utils.py")
    parser.add_argument("--predictions", type=Path, required=True, help="逐题排名与评分 JSONL")
    parser.add_argument("--reuse", action="store_true", help="读取已保存的排名，不再执行嵌入")
    parser.add_argument("--generation-input", type=Path, help="导出官方作答脚本使用的 Top 10 检索日志")
    parser.add_argument("--generation-method", choices=METHODS, default="flat")
    parser.add_argument("--embed-backend", choices=("fastembed", "mps", "torch-cpu"), default="fastembed")
    parser.add_argument("--embedding-cache", type=Path, help="分批保存 BGE 向量，便于中断后继续")
    args = parser.parse_args()
    if not args.reuse and not args.official_scorer:
        parser.error("新跑检索时需要 --official-scorer")

    entries = read_data(args.data)
    if args.reuse:
        with args.predictions.open(encoding="utf-8") as input_file:
            results = [json.loads(line) for line in input_file]
        if [row["question_id"] for row in results] != [entry["question_id"] for entry in entries]:
            raise ValueError("逐题结果与官方数据顺序不一致或尚未跑完")
    else:
        cache = None
        if args.embedding_cache:
            script_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
            cache = args.embedding_cache / f"{args.embed_backend}-{DATA_REVISION[:12]}-{script_digest}"
            cache.mkdir(parents=True, exist_ok=True)
        if args.embed_backend == "fastembed":
            from fastembed import TextEmbedding

            model = TextEmbedding(MODEL, threads=4)
        else:
            model = TorchBGE("mps" if args.embed_backend == "mps" else "cpu")
        scorer = load_official_scorer(args.official_scorer)
        results = evaluate(entries, scorer, model, args.predictions, cache)
    if args.generation_input:
        export_generation_input(entries, results, args.generation_method, args.generation_input)
    print_report(results)


if __name__ == "__main__":
    main()
