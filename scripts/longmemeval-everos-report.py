"""Write the verified comparison once both single-run workers have finished."""

import fcntl
from collections import Counter
from statistics import mean, median

from longmemeval_everos_common import OUT as base
from longmemeval_everos_common import load_adapter, read


def main():
    with (base / "qa.lock").open("r") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        run = read(base / "score.json")
        if run.get("graded") != 500 or run.get("complete") is not True:
            raise RuntimeError("500 verified judgments are required")
    adapter = load_adapter()
    units = adapter.load_units(base / "input/longmemeval_s.json")
    expected = {f"{j:03}.json" for j in range(500)}
    if any(
        {p.name for p in (base / d).glob("*.json")} != expected
        for d in ("retrieval", "answers", "judgments")
    ):
        raise ValueError("Missing or unexpected checkpoint files")
    counts, right = Counter(), Counter()
    recall_hits, recall_n, abs_hits, abs_n = 0, 0, 0, 0
    latencies, ingestion, tokens = [], [], []
    for unit in units:
        i = unit["index"]
        qa = unit["qa"][0]
        label = qa["question_id"]
        p = f"{i:03}.json"
        ret, ans, jud = (
            read(base / f / p) for f in ("retrieval", "answers", "judgments")
        )
        if (
            label != ret["question_id"]
            or label != ans["question_id"]
            or label != jud["question_id"]
        ):
            raise ValueError(f"Mismatched checkpoints {i}")
        if (ans["model"], jud["judge"]) != (
            "gpt-5.6-terra",
            "gpt-4o-mini-2024-07-18",
        ) or jud["label"] not in ("CORRECT", "WRONG"):
            raise ValueError(f"Invalid answer/judge {i}")
        good = jud["label"] == "CORRECT"
        counts[qa["category"]] += 1
        right[qa["category"]] += good
        if label.endswith("_abs"):
            abs_n += 1
            abs_hits += good
        else:
            gold = adapter.gold_of(unit, qa)
            if gold:
                seen = {
                    f"session_{int(h['id'][1:].split(':')[0])}" for h in ret["results"]
                }
                recall_hits += int(gold.issubset(seen))
                recall_n += 1
        tokens.append(ans["usage"].get("prompt_tokens", 0))
        latencies.append(ret["search_s"])
        ingestion.append(ret["index_s"])
    correct = sum(right.values())
    if sum(counts.values()) != 500 or correct != run["correct"]:
        raise ValueError("Final count mismatch")
    accuracy = correct / 5
    cfg = "https://github.com/EverMind-AI/EverOS/blob/d2b74780a443c6d7304ca9aa1bfdf36fdc6d7c9b/benchmarks/configs/longmemeval.toml"
    ref = "https://github.com/EverMind-AI/EverOS/blob/d2b74780a443c6d7304ca9aa1bfdf36fdc6d7c9b/benchmarks/README.md"
    report = [
        "# MemoryH · LongMemEval-S",
        "",
        f"MemoryH 答对 **{correct}/500（{accuracy:.1f}%）**。相对 EverOS 官宣 94.0%，差值为 **{accuracy - 94:+.1f} 个百分点**。高于官宣分数需至少 471 题正确。",
        "",
        "| 系统 | 作答模型 | 分数 |",
        "|---|---|---:|",
        f"| MemoryH 新版 | GPT-5.6 Terra（xhigh） | {accuracy:.1f}% |",
        f"| [EverOS 官宣]({ref}) | Gemini 3.6 Flash | 94.0% |",
        "| 阿里云 OpenSearch | 本轮未测 | — |",
        "",
        "## 分题型",
        "",
        "| 题型 | 正确 / 总题数 | 正确率 |",
        "|---|---:|---:|",
    ]
    report += [
        f"| {cat} | {right[cat]} / {counts[cat]} | {right[cat] / counts[cat]:.1%} |"
        for cat in sorted(counts)
    ]
    report += [
        "",
        f"其中拒答题 {abs_hits}/{abs_n}。可作答且标有证据的题目中，Top 20 原始发言覆盖全部标注会话的题数为 {recall_hits}/{recall_n}。此项按会话 ID 核对。",
        "",
        "## 测评配置",
        "",
        "使用原版 LongMemEval-S 的 500 份隔离历史，包含双方发言和原始会话日期。MemoryH 使用 Qwen3.5-2B 拆问、Qwen3-Embedding-0.6B、ScaNN 128 条向量候选、BM25 32 条关键词候选和 Yuanbao-0.6B MPS 重排，最多返回 20 条原始发言。EverOS 使用多轮检索，最多返回 20 个归纳后的 episode。",
        "",
        f"双方使用 EverOS 的 precise 作答提示、hybrid 判分规则、温度 0 的 GPT-4o-mini 单次判分。MemoryH 作答温度采用 Terra API 默认值 1，EverOS 的 Gemini 作答温度为 0。两侧作答上限均设置为 32,768 token，Terra 上限包含推理 token。MemoryH 的提取器只向裁判传递最终答案。 [EverOS 配置]({cfg})",
        "",
        f"MemoryH 每题作答输入平均 {mean(tokens):,.0f} token。检索耗时中位数 {median(latencies):.1f} 秒，本地写入中位数 {median(ingestion):.1f} 秒。",
        "",
        "本轮沿用了 EverOS 的说话人别名。官方模板在别名中保留题号，30 道不可作答题的题号带有 _abs 后缀。MemoryH 的写入、检索和重排只见历史内容、日期和查询，题型和标准答案只用于判分与统计。",
        "",
        "这是与公开端到端分数的跨作答模型比较。阿里云本轮按约定留空。",
        "",
    ]
    tmp = base / "report.md.tmp"
    tmp.write_text("\n".join(report))
    tmp.replace(base / "report.md")
    print(f"{correct}/500 ({accuracy:.1f}%), delta {accuracy - 94:+.1f} pp", flush=True)


if __name__ == "__main__":
    main()
