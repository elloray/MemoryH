"""Matched single-stream LoCoMo CPU/GPU comparisons; replays are excluded from QA."""

from functools import lru_cache
from statistics import median

from locomo_progress import p95, read

FIRST, COUNT = 6, 20


def paired(cpu, gpu):
    a, b = cpu["retrieval"], gpu["retrieval"]
    if (
        a.get("reranker", "cpu") != "cpu"
        or b.get("reranker", "mps") != "mps"
        or a["recall_width"] != 32
        or b["recall_width"] != 32
        or cpu["question"] != gpu["question"]
    ):
        raise ValueError("CPU and GPU must use the same question and width 32")

    def docs(r):
        return [(s["id"], s["memory"], s["created_at"]) for s in r["search_results"]]

    left, right = docs(a), docs(b)
    complete = (
        len(left) == a["profile"]["candidates"]
        and len(right) == b["profile"]["candidates"]
    )
    cpu_ids, gpu_ids = [[r[0] for r in ranking] for ranking in (left, right)]
    return {
        "id": cpu["question_id"],
        "cpu_ms": a["search_latency_ms"],
        "gpu_ms": b["search_latency_ms"],
        "cpu_yuanbao_ms": a["profile"]["yuanbao_ms"],
        "gpu_yuanbao_ms": b["profile"]["yuanbao_ms"],
        "cpu_pairs": a["profile"]["scored_pairs"],
        "gpu_pairs": b["profile"]["scored_pairs"],
        "cpu_candidates": a["profile"]["candidates"],
        "gpu_candidates": b["profile"]["candidates"],
        "complete_pool": complete,
        "same_returned_documents": sorted(left) == sorted(right),
        "same_order": left == right,
        "first_difference": next(
            (k for k, (x, y) in enumerate(zip(cpu_ids, gpu_ids), 1) if x != y), None
        ),
        **{f"top{k}_order": cpu_ids[:k] == gpu_ids[:k] for k in (1, 3, 5)},
        **{f"top{k}_set": set(cpu_ids[:k]) == set(gpu_ids[:k]) for k in (3, 5)},
        "cpu_top5": cpu_ids[:5],
        "gpu_top5": gpu_ids[:5],
    }


@lru_cache(maxsize=64)
def read_pair(cpu, cpu_mtime, gpu, gpu_mtime):
    return paired(read(cpu), read(gpu))


def report(root):
    experiment = root / "yuanbao-gpu"
    rows = []
    for i in range(FIRST, FIRST + COUNT):
        qid = f"conv0_q{i}"
        cpu, gpu = (
            root / "searches" / (qid + ".json"),
            experiment / "rows" / (qid + ".json"),
        )
        control = experiment / "cpu" / (qid + ".json")
        if control.is_file():
            cpu = control
        else:
            metric = root / "search_metrics" / (qid + ".json")
            if not metric.is_file() or read(metric).get("reranker", "cpu") != "cpu":
                continue
        if cpu.exists() and gpu.exists():
            rows.append(
                read_pair(cpu, cpu.stat().st_mtime_ns, gpu, gpu.stat().st_mtime_ns)
            )

    def summary(rows):
        if not rows:
            return {}
        return {
            "n": len(rows),
            **{
                f"{c}_{key}": f([r[c + "_ms"] for r in rows])
                for c in ("cpu", "gpu", "cpu_yuanbao", "gpu_yuanbao")
                for key, f in (("p50_ms", median), ("p95_ms", p95))
            },
            **{
                k: sum(r[k] for r in rows)
                for k in (
                    "complete_pool",
                    "same_returned_documents",
                    "same_order",
                    "top1_order",
                    "top3_order",
                    "top3_set",
                    "top5_order",
                    "top5_set",
                )
            },
        }

    return {
        "planned": COUNT,
        "gpu_complete": len(list((experiment / "rows").glob("*.json"))),
        "summary": summary(rows),
        "original_pair": summary(
            [r for r in rows if r["id"] in ("conv0_q6", "conv0_q7")]
        ),
        "rows": rows,
    }
