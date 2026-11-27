"""Grade checkpointed retrievals with the upstream prompts, recording actual OpenAI usage."""

import asyncio
import hashlib
import json
import math
import os
import time
from contextvars import ContextVar
from pathlib import Path
from statistics import mean

from benchmarks.common.llm_client import LLMClient

current_question = ContextVar("current_question")
answer_usage = ContextVar("answer_usage", default=None)


class Answers:
    def __init__(self, out, key, read, put):
        self.directory = out / "answers"
        self.read = read
        self.put = put
        self.client = LLMClient(model="gpt-5", api_key=key, timeout=180)
        create = self.client._client.chat.completions.create

        async def capture(**kwargs):
            response = await create(**kwargs)
            answer_usage.get().update(
                model=response.model, tokens=response.usage.prompt_tokens
            )
            return response

        self.client._client.chat.completions.create = capture

    async def generate(self, system, user):
        path = self.directory / (current_question.get() + ".json")
        signature = hashlib.sha256(json.dumps([system, user]).encode()).hexdigest()
        if path.exists():
            saved = self.read(path)
            if saved["prompt_sha256"] != signature:
                raise ValueError("cached answer used a different prompt")
            return saved["answer"]
        answer_usage.set({})
        answer = await self.client.generate(system=system, user=user)
        usage = answer_usage.get()
        if not answer or not usage or not usage["tokens"]:
            raise RuntimeError("GPT-5 returned no usable answer or prompt-token count")
        self.put(
            path,
            {
                "answer": answer,
                "model": usage["model"],
                "input_tokens": usage["tokens"],
                "prompt_sha256": signature,
            },
        )
        return answer


class Judge:
    def __init__(self, key):
        self.client = LLMClient(model="gpt-5", api_key=key, timeout=180)

    async def generate_structured(self, **kwargs):
        result = await self.client.generate_structured(**kwargs)
        if not isinstance(result, dict) or str(result.get("label", "")).upper() not in {
            "CORRECT",
            "WRONG",
        }:
            raise RuntimeError(
                "GPT-5 produced no valid judge label; question remains pending"
            )
        return result


async def grade(out, locomo, items, read, put, status):
    key = (
        os.environ.get("OPENAI_API_KEY")
        or (Path.home() / ".config/memoryh/longmemeval-api-key").read_text().strip()
    )
    answers, judge = Answers(out, key, read, put), Judge(key)
    status("grade", "GPT-5 已就绪，等待首批检索结果")
    completed = {p.stem for p in (out / "judgments").glob("*.json")}

    async def one(qid, ci, qa):
        token = current_question.set(qid)
        try:
            row = read(out / "searches" / (qid + ".json"))
            await locomo.apply_locomo_judge_to_saved_result(
                row, qa, ci, answers, judge, [200], None
            )
            result = row["cutoff_results"]["top_200"]
            put(
                out / "judgments" / (qid + ".json"),
                {
                    "finished": time.time(),
                    "correct": result["judgment"] == "CORRECT",
                    "result": result,
                    "question_id": qid,
                    "category": qa["category"],
                },
            )
            completed.add(qid)
        finally:
            current_question.reset(token)

    try:
        while len(completed) < len(items):
            batch = [
                (qid, ci, qa)
                for qid, ci, _, qa in items
                if qid not in completed
                and (out / "search_metrics" / (qid + ".json")).is_file()
            ][:4]
            if not batch:
                await asyncio.sleep(2)
                continue
            status("grade", "GPT-5 作答与判分中：" + ", ".join(row[0] for row in batch))
            await asyncio.gather(*(one(*row) for row in batch))
            status("grade", f"GPT-5 已判 {len(completed)}/1540 题")
        evaluations = []
        tokens = []
        latencies = []
        for qid, _, _, qa in items:
            j = read(out / "judgments" / (qid + ".json"))
            evaluations.append(
                {
                    "question_id": qid,
                    "category": qa["category"],
                    "category_name": locomo.CATEGORY_NAMES[qa["category"]],
                    "cutoff_results": {"top_200": j["result"]},
                }
            )
            tokens.append(read(out / "answers" / (qid + ".json"))["input_tokens"])
            latencies.append(
                read(out / "search_metrics" / (qid + ".json"))["latency_ms"]
            )
        result = locomo.compute_locomo_metrics(evaluations, [200])
        put(
            out / "memoryh-result.json",
            {
                "model": "gpt-5",
                "top_k": 200,
                "metrics_by_cutoff": result,
                "answer_input_tokens_mean": mean(tokens),
                "retrieval_p95_ms": sorted(latencies)[
                    math.ceil(0.95 * len(latencies)) - 1
                ],
                "evaluations": evaluations,
            },
        )
        status("grade", "1,540 题作答与判分完成")
    finally:
        await answers.client._client.close()
        await judge.client._client.close()
