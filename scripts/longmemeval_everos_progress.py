"""Read only the live checkpoints for the MemoryH/EverOS comparison page."""

from datetime import UTC, datetime
from functools import lru_cache
from statistics import median

from longmemeval_everos_common import read


@lru_cache(maxsize=2048)
def checkpoint(path, mtime):
    return read(path)


def snapshot(out, locked):
    def stage(name):
        paths = sorted((out / name).glob("[0-9][0-9][0-9].json"))
        return {int(p.stem): checkpoint(p, p.stat().st_mtime_ns) for p in paths}

    # A judgment has a saved answer, and an answer has saved retrieval results.
    # Read downstream first so a concurrent checkpoint cannot outrun its inputs.
    judged, answers, rank = (stage(s) for s in ("judgments", "answers", "retrieval"))
    total = 500
    status = read(out / "retrieval-status.json")
    retrieving = len(rank) < total and locked(out / "retrieval.lock")
    grading = len(judged) < total and locked(out / "qa.lock")
    active = status.get("question") if retrieving else None
    items = []
    for i in range(total):
        if i in judged:
            state = "yes" if judged[i]["label"] == "CORRECT" else "no"
        elif i in answers:
            state = "answer"
        elif i in rank:
            state = "rank"
        elif active == i + 1:
            state = "work"
        else:
            state = ""
        items.append(state)
    last = None
    if judged:
        i = max(judged)
        last = {
            "number": i + 1,
            "label": judged[i]["label"],
            "question_type": judged[i]["category"],
            "input_tokens": answers[i]["usage"].get("prompt_tokens", 0),
        }
    return {
        "time": datetime.now(UTC).isoformat(),
        "total": total,
        "retrieved": len(rank),
        "answered": len(answers),
        "judged": len(judged),
        "correct": items.count("yes"),
        "active": status if retrieving else None,
        "retrieving": retrieving,
        "grading": grading,
        "last": last,
        "items": items,
        "complete": len(judged) == total,
        "median_index_s": median(r["index_s"] for r in rank.values()) if rank else None,
        "median_search_s": median(r["search_s"] for r in rank.values())
        if rank
        else None,
    }
