# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["numpy", "torch==2.9.1", "transformers==4.57.6"]
# ///

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
from statistics import mean
import sys
from urllib.request import urlopen

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


ORIGINAL_COMMIT = "3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376"
DATA_URL = f"https://raw.githubusercontent.com/snap-research/locomo/{ORIGINAL_COMMIT}/data/locomo10.json"
DATA_SHA256 = "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"
MODELS = {
    "context": "68074e7406bb0061b0d049b58592acafae00e9d4",
    "query": "2d3808c087119b953f8494b7638c216c71712cee",
}


def get_embeddings(groups, role, device):
    name = f"facebook/dragon-plus-{role}-encoder"
    tokenizer = AutoTokenizer.from_pretrained("facebook/dragon-plus-query-encoder", revision=MODELS["query"])
    encoder = AutoModel.from_pretrained(name, revision=MODELS[role]).eval().to(device)
    with torch.inference_mode():
        for group in groups:
            batches = []
            for start in range(0, len(group), 24):
                batch = tokenizer(group[start:start + 24], return_tensors="pt", padding=True, truncation=True)
                batches.append(encoder(**batch.to(device)).last_hidden_state[:, 0, :].cpu().numpy())
            yield np.concatenate(batches)
    del encoder
    if device == "mps":
        torch.mps.empty_cache()


def dialogue_documents(sample):
    conversation = sample["conversation"]
    sessions = sorted(
        (key for key in conversation if re.fullmatch(r"session_\d+", key)),
        key=lambda key: int(key.removeprefix("session_")),
    )
    docs = []
    ids = []
    for session in sessions:
        for turn in conversation[session]:
            document = f'{turn["speaker"]} said, "{turn["text"]}"'
            if "blip_caption" in turn:
                document += " and shared " + turn["blip_caption"]
            docs.append(document)
            ids.append(turn["dia_id"])
    return docs, ids


def run(samples, device):
    corpora, doc_ids = zip(*(dialogue_documents(sample) for sample in samples), strict=True)
    contexts = []
    for index, vectors in enumerate(get_embeddings(corpora, "context", device), 1):
        contexts.append(vectors)
        print(f"DRAGON contexts {index}/{len(samples)}", file=sys.stderr, flush=True)

    values = defaultdict(list)
    queries = [[item["question"] for item in sample["qa"]] for sample in samples]
    query_embeddings = get_embeddings(queries, "query", device)
    for sample, ids, context, query in zip(samples, doc_ids, contexts, query_embeddings, strict=True):
        order = np.argsort(query @ context.T, axis=1)[:, ::-1]
        for item, row in zip(sample["qa"], order, strict=True):
            gold = item["evidence"]
            for k in (3, 5):
                retrieved = {ids[index] for index in row[:k]}
                score = round(sum(dia_id in retrieved for dia_id in gold) / len(gold), 3) if gold else 1.0
                values[k, "all"].append(score)
                if item["category"] != 5 and gold:
                    values[k, "answerable"].append(score)
        print(sample["sample_id"], file=sys.stderr, flush=True)

    for (k, scope), recalls in sorted(values.items()):
        print(f"DRAGON English Evidence Recall@{k}, {scope}, n={len(recalls)}: {mean(recalls):.4%}")


def main():
    parser = argparse.ArgumentParser(description="复跑 LoCoMo 英文原版的 DRAGON 对话轮次检索")
    parser.add_argument("--data", type=Path, help="指定已下载的官方英文 JSON")
    args = parser.parse_args()
    if args.data is None:
        with urlopen(DATA_URL, timeout=30) as response:
            raw = response.read()
    else:
        raw = args.data.read_bytes()
    if hashlib.sha256(raw).hexdigest() != DATA_SHA256:
        raise ValueError("LoCoMo 英文数据的 SHA-256 与固定版本不符")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.set_num_threads(4)
    print(f"device={device}", file=sys.stderr, flush=True)
    run(json.loads(raw), device)


if __name__ == "__main__":
    main()
