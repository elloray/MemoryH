"""Paired live statistics for the Mem0 LoCoMo comparison."""

import csv
import io
import json
import math
from datetime import UTC, datetime
from functools import lru_cache
from statistics import mean, median


def read(path):
    return json.loads(path.read_text()) if path.is_file() else {}


@lru_cache(maxsize=8000)
def cached(path, mtime):
    return read(path)


def directory(path):
    return {p.stem: cached(p, p.stat().st_mtime_ns) for p in path.glob("*.json")}


def p95(values):
    return sorted(values)[math.ceil(0.95 * len(values)) - 1] if values else None


def latency_stats(search, refs, ids):
    ours = [search[q]["latency_ms"] for q in ids]
    theirs = [refs[q]["latency_ms"] for q in ids]
    return {
        "p50": median(ours) if ours else None,
        "p95": p95(ours),
        "ref_p50": median(theirs) if theirs else None,
        "ref_p95": p95(theirs),
    }


def snapshot(root):
    refs = read(root / "reference.json")
    # Collect later-stage files first. Each judgment has a durable answer and retrieval.
    judgments = directory(root / "judgments")
    answers = directory(root / "answers")
    search = directory(root / "search_metrics")
    correct = sum(v["correct"] for v in judgments.values())
    ref_correct = sum(refs[q]["correct"] for q in judgments)
    wins = sum(v["correct"] and not refs[q]["correct"] for q, v in judgments.items())
    losses = sum(not v["correct"] and refs[q]["correct"] for q, v in judgments.items())
    profiled = [q for q in refs if search.get(q, {}).get("profile")]
    latest = profiled[-1] if profiled else None
    n = len(judgments)
    tokens = [a["input_tokens"] for a in answers.values()]
    widths = {}
    for q, result in search.items():
        widths.setdefault(
            (result.get("recall_width", 200), result.get("reranker", "cpu")), []
        ).append(q)
    cohorts = []
    for (width, backend), ids in sorted(widths.items()):
        graded = [q for q in ids if q in judgments]
        usage = [answers[q]["input_tokens"] for q in ids if q in answers]
        cohorts.append(
            {
                "width": width,
                "reranker": backend,
                "searched": len(ids),
                "judged": len(graded),
                "correct": sum(judgments[q]["correct"] for q in graded),
                "ref_correct": sum(refs[q]["correct"] for q in graded),
                "input_tokens": mean(usage) if usage else None,
                **latency_stats(search, refs, ids),
            }
        )
    return {
        "time": datetime.now(UTC).isoformat(),
        "total": len(refs),
        "retrieval": read(root / "retrieve-status.json"),
        "grading": read(root / "grade-status.json"),
        "searched": len(search),
        "answered": len(answers),
        "judged": n,
        "correct": correct,
        "ref_correct": ref_correct,
        "wins": wins,
        "losses": losses,
        "accuracy": 100 * correct / n if n else None,
        "ref_accuracy": 100 * ref_correct / n if n else None,
        "input_tokens": mean(tokens) if tokens else None,
        **latency_stats(search, refs, search),
        "cohorts": cohorts,
        "width_comparisons": list(directory(root / "width32").values()),
        "profile_count": len(profiled),
        "yuanbao_p95_ms": p95([search[q]["profile"]["yuanbao_ms"] for q in profiled]),
        "last_profile": {"id": latest, **search[latest]} if latest else None,
        "recent": [
            {
                "id": q,
                "question": refs[q]["question"],
                "ours": judgments[q]["correct"],
                "ref": refs[q]["correct"],
                "tokens": answers[q]["input_tokens"],
            }
            for q in sorted(
                judgments, key=lambda q: judgments[q]["finished"], reverse=True
            )[:6]
        ],
    }


def per_question_csv(root):
    refs = read(root / "reference.json")
    search, answers, judged = [
        directory(root / d) for d in ("search_metrics", "answers", "judgments")
    ]
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(
        [
            "id",
            "question",
            "memoryh_correct",
            "mem0_correct",
            "answer_input_tokens",
            "memoryh_retrieval_ms",
            "mem0_retrieval_ms",
            "memoryh_returned",
            "memoryh_yuanbao_ms",
            "memoryh_scored_pairs",
            "memoryh_recall_width_per_lane",
            "memoryh_reranker",
        ]
    )
    for q, r in refs.items():
        s = search.get(q, {})
        writer.writerow(
            [
                q,
                r["question"],
                judged.get(q, {}).get("correct"),
                r["correct"],
                answers.get(q, {}).get("input_tokens"),
                s.get("latency_ms"),
                r["latency_ms"],
                s.get("count"),
                (s.get("profile") or {}).get("yuanbao_ms"),
                (s.get("profile") or {}).get("scored_pairs"),
                s.get("recall_width", 200) if s else None,
                s.get("reranker", "cpu") if s else None,
            ]
        )
    return out.getvalue().encode("utf-8-sig")
