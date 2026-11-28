"""Score new MemoryH Top20 checkpoints with EverOS's published LongMemEval prompts."""

import ast
import fcntl
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from longmemeval_everos_common import (
    DATA,
    OUT,
    SOURCE_HASHES,
    UP,
    load_adapter,
    put,
    read,
)

adapter = load_adapter()

ANSWER = "gpt-5.6-terra"
JUDGE = "gpt-4o-mini-2024-07-18"
KEY = Path.home() / ".config/memoryh/longmemeval-api-key"
ENDPOINT = "https://api.openai.com/v1/chat/completions"
CONFIG = {
    "answer": ANSWER,
    "reasoning_effort": "xhigh",
    "answer_temperature": "provider default 1; API rejects 0",
    "answer_max_completion_tokens": 32768,
    "judge": JUDGE,
    "judge_temperature": 0,
    "judge_runs": 1,
    "top_k": 20,
    "include_profile": False,
    "official_speaker_aliases": True,
    "official_source_sha256": SOURCE_HASHES,
}

# Reuse the exact final-answer extractor from the verified EverOS runner.
source = ast.parse((UP / "run.py").read_text())
extract_node = next(
    n
    for n in source.body
    if isinstance(n, ast.FunctionDef) and n.name == "_extract_final_answer"
)
ns = {}
exec(  # noqa: S102 -- only the checksum-pinned upstream final-answer function.
    compile(
        ast.Module(body=[extract_node], type_ignores=[]), str(UP / "run.py"), "exec"
    ),
    ns,
)
extract = ns["_extract_final_answer"]


def call(key, model, messages, **extra):
    request = Request(
        ENDPOINT,
        data=json.dumps(dict(model=model, messages=messages, **extra)).encode(),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
    )
    for attempt in range(5):
        try:
            with urlopen(request, timeout=300 if model == ANSWER else 120) as response:
                body = json.load(response)
            choice = body["choices"][0]
            if body["model"] != model or choice["finish_reason"] != "stop":
                raise RuntimeError(
                    f"Unexpected model or finish condition: {body['model']}, {choice['finish_reason']}"
                )
            return (
                choice["message"]["content"],
                body["usage"],
                body.get("system_fingerprint"),
            )
        except HTTPError as e:
            details = json.loads(e.read())["error"]
            if (
                e.code not in (408, 409, 429, 500, 502, 503, 504)
                or details.get("code") == "insufficient_quota"
                or attempt == 4
            ):
                raise RuntimeError(
                    f"OpenAI {model}: HTTP {e.code}, {details.get('code')}"
                ) from None
            retry = e.headers.get("Retry-After", "")
            time.sleep(
                max(
                    2**attempt,
                    float(retry) if retry.replace(".", "", 1).isdigit() else 0,
                )
            )
        except (TimeoutError, URLError):
            if attempt == 4:
                raise
            time.sleep(2**attempt)


def grade(key, unit, row):
    index = unit["index"]
    qa = unit["qa"][0]
    qid = qa["question_id"]
    if row["index"] != index or row["question_id"] != qid:
        raise ValueError("retrieval checkpoint does not match dataset")
    search_error = row.get("search_error")
    if search_error and row["results"]:
        raise ValueError("failed search contains evidence")
    answer_path = OUT / "answers" / f"{index:03}.json"
    source_hash = hashlib.sha256(
        json.dumps(row["results"], sort_keys=True).encode()
    ).hexdigest()
    if answer_path.exists():
        answer = read(answer_path)
        if (answer["model"], answer["evidence_sha256"], answer["question_id"]) != (
            ANSWER,
            source_hash,
            qid,
        ):
            raise ValueError("incompatible answer checkpoint")
        if answer.get("search_error") != search_error:
            raise ValueError("incompatible search error in answer checkpoint")
    else:
        hits = row["results"]
        usage, fingerprint, prompt_hash = {}, None, None
        if hits:
            a, b = adapter.speakers_of(unit)
            episodes = "\n\n".join(
                f"{h['created_at']}: {h['memory']}\n---" for h in hits
            )
            context = adapter.CONTEXT_TEMPLATE.format(
                speaker_a=a, speaker_b=b, episodes=episodes
            )
            date = qa["question_date"]
            prompt = adapter.ANSWER_PROMPT.format(
                context=context,
                current_date_line=f"Current Date: {date}\n" if date else "",
                question=qa["question"],
            )
            prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
            raw, usage, fingerprint = call(
                key,
                ANSWER,
                [{"role": "user", "content": prompt}],
                reasoning_effort="xhigh",
                max_completion_tokens=32768,
            )
            answer_text = extract(raw)
            if not answer_text or not answer_text.strip():
                raise ValueError("empty reader answer")
        else:
            answer_text = (
                "[SEARCH_FAILED]" if search_error else adapter.EMPTY_RETRIEVAL_MARKER
            )
        answer = {
            "index": index,
            "question_id": qid,
            "model": ANSWER,
            "final_answer": answer_text,
            "evidence_sha256": source_hash,
            "prompt_sha256": prompt_hash,
            "usage": usage,
            "fingerprint": fingerprint,
        }
        if search_error:
            answer["search_error"] = search_error
        put(answer_path, answer)
    if search_error or answer["final_answer"] == adapter.EMPTY_RETRIEVAL_MARKER:
        verdict, usage, fingerprint = "WRONG", {}, None
    else:
        prompt = adapter.JUDGE_USER_PROMPT.format(
            question=qa["question"],
            golden_answer=qa["answer"],
            generated_answer=answer["final_answer"],
        )
        clause = adapter.leniency_clause(qa)
        if clause:
            prompt = f"{clause}\n\n{prompt}"
        for _ in range(4):
            raw, usage, fingerprint = call(
                key,
                JUDGE,
                [
                    {"role": "system", "content": adapter.JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )
            verdict = adapter.parse_judge_label(raw)
            if verdict is not None:
                break
        else:
            raise ValueError("Four unreadable judgments")
    return {
        "index": index,
        "question_id": qid,
        "category": qa["category"],
        "label": verdict,
        "judge": JUDGE,
        "usage": usage,
        "fingerprint": fingerprint,
        "answer_prompt_sha256": answer["prompt_sha256"],
        **({"search_error": search_error} if search_error else {}),
    }


def main():
    cfg = OUT / "qa-config.json"
    if cfg.exists() and read(cfg) != CONFIG:
        raise ValueError("QA configuration changed")
    put(cfg, CONFIG)
    key = KEY.read_text().strip()
    units = adapter.load_units(str(DATA))
    if len(units) != 500:
        raise ValueError("expected 500 units")
    correct, counts, hits = 0, Counter(), Counter()
    for unit in units:
        index = unit["index"]
        rank = OUT / "retrieval" / f"{index:03}.json"
        dst = OUT / "judgments" / f"{index:03}.json"
        while not rank.exists():
            time.sleep(4)
        if not dst.exists():
            put(dst, grade(key, unit, read(rank)))
        judged = read(dst)
        if judged["question_id"] != unit["qa"][0]["question_id"] or judged[
            "label"
        ] not in ("CORRECT", "WRONG"):
            raise ValueError("invalid judgment checkpoint")
        cat = judged["category"]
        correct += judged["label"] == "CORRECT"
        hits[cat] += judged["label"] == "CORRECT"
        counts[cat] += 1
        summary = {
            "total": 500,
            "graded": index + 1,
            "correct": correct,
            "complete": index == 499,
            "by_category": {
                c: {"correct": hits[c], "graded": counts[c]} for c in counts
            },
        }
        if summary["complete"]:
            summary.update(
                accuracy_percent=correct / 5,
                everos_reference_percent=94.0,
                difference_pp=round(correct / 5 - 94, 1),
                strictly_exceeds_reference=correct > 470,
            )
        put(OUT / "score.json", summary)
        print(f"{index + 1}/500 judged, {correct} correct", flush=True)


if __name__ == "__main__":
    with (OUT / "qa.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
