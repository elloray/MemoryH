# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["numpy==1.26.4", "tiktoken==0.12.0"]
# ///

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Lock
from time import monotonic, sleep
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import tiktoken


MODEL = "gpt-4o-2024-08-06"
ENDPOINT = "https://api.openai.com/v1/chat/completions"
ANSWER_TOKENS = 800
JUDGE_TOKENS = 10
MAX_HISTORY_TOKENS = 128_000 - ANSWER_TOKENS - 1_000
OFFICIAL_FILES = {
    "run_generation.py": "4f1eb3c69d7ad40f04065b9c0bc86f6582441018fc6ff751d162d66c95baf672",
    "evaluate_qa.py": "ecce9c4c79dc89d99534ac17b383a5cbb5b9f0c69ee98adaf0684742e3d95251",
    "print_qa_metrics.py": "e9283933a0cefb7a0ded7365e436ae3d1be5aac41853325e6155d83bf07607f0",
}


def official_source(path):
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != OFFICIAL_FILES[path.name]:
        raise ValueError(f"官方源码版本不符 {path.name}")
    return raw


def official_function(path, name):
    # Load the pinned prompt definition without importing unrelated inference dependencies.
    source = ast.parse(official_source(path))
    function = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {"json": json}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def request_completion(api_key, prompt, max_tokens):
    body = {
        "model": MODEL, "messages": [{"role": "user", "content": prompt}],
        "n": 1, "temperature": 0, "max_tokens": max_tokens,
    }
    request = Request(ENDPOINT, data=json.dumps(body).encode(), headers={
        "Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
    })
    for attempt in range(6):
        try:
            with urlopen(request, timeout=120) as response:
                result = json.load(response)
            if result["model"] != MODEL:
                raise RuntimeError(f"Unexpected response model {result['model']}")
            return result["choices"][0]["message"]["content"].strip(), result["usage"]
        except HTTPError as error:
            if error.status not in (408, 409, 429, 500, 502, 503, 504) or attempt == 5:
                raise RuntimeError(f"OpenAI API returned HTTP {error.status}") from None
            delay = error.headers.get("Retry-After", "")
            sleep(max(2 ** attempt, float(delay) if delay.replace(".", "", 1).isdigit() else 0))
        except (URLError, TimeoutError):
            if attempt == 5:
                raise RuntimeError("OpenAI API connection failed after retries") from None
            sleep(2 ** attempt)


def estimated_cost(prompt_tokens, max_completion_tokens):
    return (prompt_tokens * 2.50 + max_completion_tokens * 10.00) / 1_000_000


def billed_cost(usage):
    cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
    return estimated_cost(usage["prompt_tokens"] - cached, usage["completion_tokens"]) + cached * 1.25 / 1_000_000


def fingerprint(prompt):
    return sha256(prompt.encode()).hexdigest()


def load_rows(path):
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source]


class TokenPacer:
    def __init__(self, tokens_per_minute):
        self.tokens_per_second = tokens_per_minute / 60
        self.lock = Lock()
        self.next_slot = monotonic()

    def reserve(self, tokens):
        with self.lock:
            now = monotonic()
            slot = max(now, self.next_slot)
            self.next_slot = slot + tokens / self.tokens_per_second
        sleep(max(0, slot - monotonic()))


def prepare_jobs(entries, top_k, prepare, tokenizer, label):
    jobs = []
    judge_tokens = 0
    answer_tokens = 0
    for entry in entries:
        prompt = prepare(entry, "flat-session", top_k, False, "json", True,
                         tokenizer, "openai", MAX_HISTORY_TOKENS, "none")
        tokens = len(tokenizer.encode_ordinary(prompt)) + 16
        jobs.append((entry, prompt, fingerprint(prompt), tokens))
        answer_tokens += tokens
        judge_prompt = label(entry["question_type"], entry["question"], entry["answer"], "",
                             abstention="_abs" in entry["question_id"])
        judge_tokens += len(tokenizer.encode_ordinary(judge_prompt)) + ANSWER_TOKENS + 16
    ceiling = estimated_cost(answer_tokens + judge_tokens, len(entries) * (ANSWER_TOKENS + JUDGE_TOKENS))
    print(f"Top {top_k} · {len(entries)} 题 · 作答输入约 {answer_tokens:,} tokens · "
          f"按输出上限估算 US${ceiling:.2f}", flush=True)
    return jobs, ceiling


def evaluate(jobs, label, api_key, output_path, workers, pacer, tokenizer):
    saved = {row["question_id"]: row for row in load_rows(output_path)} if output_path.exists() else {}
    compatible = {entry["question_id"] for entry, _, digest, _ in jobs
                  if entry["question_id"] in saved and saved[entry["question_id"]]["prompt_sha256"] == digest}
    if compatible != set(saved):
        raise ValueError("已有结果对应不同提示词或检索输入")

    def run_one(entry, prompt, digest, tokens):
        question_id = entry["question_id"]
        pacer.reserve(tokens + ANSWER_TOKENS)
        hypothesis, answer_usage = request_completion(api_key, prompt, ANSWER_TOKENS)
        judge_prompt = label(entry["question_type"], entry["question"], entry["answer"], hypothesis,
                             abstention="_abs" in question_id)
        pacer.reserve(len(tokenizer.encode_ordinary(judge_prompt)) + 16 + JUDGE_TOKENS)
        decision, judge_usage = request_completion(api_key, judge_prompt, JUDGE_TOKENS)
        return {
            "question_id": question_id, "hypothesis": hypothesis, "prompt_sha256": digest,
            "autoeval_label": {"model": MODEL, "label": "yes" in decision.lower()},
            "usage": {"answer": answer_usage, "judge": judge_usage},
        }

    remaining = [job for job in jobs if job[0]["question_id"] not in saved]
    errors = []
    with output_path.open("a", encoding="utf-8") as output, ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_one, *job): job[0]["question_id"] for job in remaining}
        for future in as_completed(futures):
            try:
                row = future.result()
            except (RuntimeError, ValueError, KeyError, AttributeError) as error:
                errors.append(f"{futures[future]}: {error}")
                continue
            saved[row["question_id"]] = row
            print(json.dumps(row, ensure_ascii=False), file=output, flush=True)
            if len(saved) % 25 == 0 or len(saved) == len(jobs):
                print(f"作答和判分 {len(saved)}/{len(jobs)} · {output_path.name}", flush=True)
    cost = sum(billed_cost(usage) for row in saved.values() for usage in row["usage"].values())
    print(f"已完成 {len(saved)}/{len(jobs)} · 按实际 token 用量估算 US${cost:.2f}", flush=True)
    if errors:
        raise RuntimeError(f"{len(errors)} 题调用失败，保留已评分结果。{errors[:3]}")


def main():
    parser = argparse.ArgumentParser(description="用固定版本的 LongMemEval 官方作答和判分提示词跑 GPT-4o")
    parser.add_argument("--official-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="会话级检索日志")
    parser.add_argument("--output", type=Path, help="逐题答案和官方格式评分 JSONL")
    parser.add_argument("--top-k", type=int, choices=(5, 10), required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--tokens-per-minute", type=int, default=1_500_000)
    parser.add_argument("--estimate", action="store_true")
    parser.add_argument("--budget", type=float, help="本次评测允许的费用估算上限（美元）")
    parser.add_argument("--reference", type=Path, help="提供时用官方聚合脚本打印全 500 题准确率")
    parser.add_argument("--key-file", type=Path, help="API 密钥的本机路径")
    args = parser.parse_args()

    if args.workers < 1 or args.tokens_per_minute < 1:
        parser.error("--workers 和 --tokens-per-minute 必须为正数")
    if not args.estimate and (not args.output or args.budget is None):
        parser.error("执行 API 评测时需要 --output 和 --budget")
    if args.reference:
        official_source(args.official_dir / "print_qa_metrics.py")

    entries = load_rows(args.input)
    if len(entries) != 500 or len({entry["question_id"] for entry in entries}) != 500:
        raise ValueError("完整 LongMemEval-S 需要 500 个独立题目")
    if args.reference:
        with args.reference.open(encoding="utf-8") as reference:
            ids = {entry["question_id"] for entry in json.load(reference)}
        if {entry["question_id"] for entry in entries} != ids:
            raise ValueError("检索日志题目与参考文件不一致")

    prepare = official_function(args.official_dir / "run_generation.py", "prepare_prompt")
    label = official_function(args.official_dir / "evaluate_qa.py", "get_anscheck_prompt")
    tokenizer = tiktoken.get_encoding("o200k_base")
    jobs, ceiling = prepare_jobs(entries, args.top_k, prepare, tokenizer, label)
    if args.estimate:
        return
    if ceiling > args.budget:
        raise ValueError(f"费用估算超过本次预算 US${args.budget:.2f}")

    api_key = args.key_file.read_text().strip() if args.key_file else os.environ.get("OPENAI_API_KEY")
    if not api_key:
        parser.error("提供 OPENAI_API_KEY 或通过 --key-file 加载本机密钥")
    evaluate(jobs, label, api_key, args.output, args.workers, TokenPacer(args.tokens_per_minute), tokenizer)
    if args.reference:
        subprocess.run([sys.executable, str(args.official_dir / "print_qa_metrics.py"),
                        str(args.output), str(args.reference)], check=True)


if __name__ == "__main__":
    main()
