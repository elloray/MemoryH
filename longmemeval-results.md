# MemoryH · LongMemEval-S

2026 年 11 月 25 日。所有作答和判分使用官方 cleaned 英文数据的 500 题，含 30 道拒答题。每个问题独立建索引，完整 Top 5 会话按日期排序后提供给同一个回答模型。作答与逐题判分共用 `gpt-4o-2024-08-06`，使用固定版本的官方提示词、JSON Chain-of-Note 和准确率汇总脚本。 [数据](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/tree/98d7416c24c778c2fee6e6f3006e7a073259d48f) · [官方评测说明](https://github.com/xiaowu0162/LongMemEval#testing-your-system)

## 端到端问答

| 检索路线 | Overall Accuracy | Task-averaged Accuracy | 拒答准确率 | 平均作答输入 tokens |
|---|---:|---:|---:|---:|
| 官方 BM25 | 366/500（73.20%） | 75.06% | 26/30（86.67%） | 14,312 |
| MemoryH FTS5 + 扁平向量 | 421/500（84.20%） | 85.70% | 24/30（80.00%） | 14,745 |
| MemoryH FTS5 + 二叉树 · Beam 3 | 383/500（76.60%） | 77.98% | 25/30（83.33%） | 14,625 |

扁平路线比 Beam 3 多答对 38 题。扁平答对而树答错 49 题，树答对而扁平答错 11 题。相对官方 BM25，这两个方向分别有 75 题和 20 题。

| 任务 | 官方 BM25 | MemoryH 扁平 | MemoryH 二叉树 Beam 3 |
|---|---:|---:|---:|
| 单会话用户信息 | 65/70（92.86%） | 70/70（100.00%） | 63/70（90.00%） |
| 单会话助手信息 | 45/56（80.36%） | 54/56（96.43%） | 54/56（96.43%） |
| 用户偏好 | 18/30（60.00%） | 22/30（73.33%） | 18/30（60.00%） |
| 跨会话综合 | 71/133（53.38%） | 96/133（72.18%） | 84/133（63.16%） |
| 时间推理 | 95/133（71.43%） | 108/133（81.20%） | 98/133（73.68%） |
| 记忆更正 | 72/78（92.31%） | 71/78（91.03%） | 66/78（84.62%） |

## 会话级检索

官方英文 cleaned 版，检索输出 500 题，官方检索分数覆盖 470 题。会话级检索，`BAAI/bge-small-en-v1.5`，RRF 60，每通道最多 30 条候选。

每个问题独立建索引。按官方 session 流程，仅用用户发言生成检索键。扁平和树路线保留现有的子问题向量首位策略。Top 5 与 Top 10 取自同一排序结果。

| 方法 | Recall-All@5 | nDCG@5 | Recall-All@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| 官方 BM25 | 77.02% | 68.78% | 84.26% | 70.78% |
| MemoryH FTS5 | 85.53% | 77.15% | 90.64% | 78.31% |
| MemoryH FTS5 + 扁平向量 | 89.36% | 80.18% | 94.68% | 81.49% |
| MemoryH FTS5 + 二叉树 · Beam 2 | 63.83% | 62.31% | 67.66% | 63.30% |
| MemoryH FTS5 + 二叉树 · Beam 3 | 79.57% | 74.59% | 84.04% | 75.78% |

官方脚本跳过 30 道 abstention 题。会话重标后有 51 道可答题没有正例，官方评分器对此记 Recall-All 1、nDCG 0。

### 有会话正例的题

以下补充统计覆盖 419 题，展示 Recall-All@5 和 Recall-All@10。

| 方法 | Recall-All@5 | Recall-All@10 |
|---|---:|---:|
| 官方 BM25 | 74.22% | 82.34% |
| MemoryH FTS5 | 83.77% | 89.50% |
| MemoryH FTS5 + 扁平向量 | 88.07% | 94.03% |
| MemoryH FTS5 + 二叉树 · Beam 2 | 59.43% | 63.72% |
| MemoryH FTS5 + 二叉树 · Beam 3 | 77.09% | 82.10% |

这 51 题均为单会话助手信息题。作答分数仍包含这类问题，模型可以看到候选会话里的助手发言。

两条混合路线共用英文 BGE、RRF 和每通道最多 30 条候选。本轮使用 Apple MPS 以 FP16 推理 BGE 的固定原版权重，会话键最多输入 512 tokens。树为四叶语义中位切分，端到端采用预先选定的 Beam 3。五条路线的 20 个统计量均与官方汇总一致，500 题 BM25 排序与原版函数逐题一致。完整和精简检索日志构建出的所有作答提示词逐字一致。 [BGE 模型](https://huggingface.co/BAAI/bge-small-en-v1.5/tree/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a) · [官方会话检索器](https://github.com/xiaowu0162/LongMemEval/blob/9e0b455f4ef0e2ab8f2e582289761153549043fc/src/retrieval/run_retrieval.py)

### 二叉树漏召位置

扁平路线 Top 5 找齐证据、Beam 3 漏掉证据的有 48 题，其中跨会话综合 18 题。漏掉的 56 条逐题证据都在 FTS5 候选里。树导航剪掉了对应叶子，证据少了向量通道的 RRF 加分，因此跌出最终 Top 5。

在 121 道可答跨会话题中，Recall-All@5 从 Beam 2 的 41.32% 升到 Beam 3 的 65.29%，扁平为 79.34%。在 72 道可答记忆更正题中，Beam 3 为 86.11%，扁平为 97.22%。

## 公开成绩参照

LongMemEval-S 作者发布了数据和评分代码。公开项目各自报告结果，读取预算、数据修订版和作答模型会影响名次。 [官方仓库](https://github.com/xiaowu0162/LongMemEval)

| 配置 | Overall Accuracy | 阅读条件与来源 |
|---|---:|---|
| Recallium | 90.20% | cleaned 500 题，GPT-4o 作答和官方判分，平均 3.8 条。 [公开评测](https://recallium.ai/benchmarks) |
| MemoryH 扁平 | 84.20% | 本次 cleaned 500 题，Top 5 完整会话 |
| Engrava 0.6 | 81.60% | cleaned 500 题，同 GPT-4o 快照，Top 20 对话轮次。 [公开复现](https://dev.to/sovantica/a-longmemeval-s-number-you-can-reproduce-2l0n) |
| Zep 论文 | 71.20% | GPT-4o，平均约 1.6K 上下文 tokens，早期数据。 [论文表 2](https://arxiv.org/html/2501.13956v1) |
| 原论文 GPT-4o 全历史 + CoN | 64.00% | 早期 S 版，约 115K 上下文 tokens。 [论文图 3](https://proceedings.iclr.cc/paper_files/paper/2025/file/d813d324dbf0598bbdc9c8e79740ed01-Paper-Conference.pdf#page=6) |

## 运行记录

三条问答路线共 1,500 个独立作答与判分结果。按 API 返回的 token 用量和公开单价估算 US$54.76。准确率取自单次运行，模型响应可能有运行间波动。本轮范围是检索质量和问答准确率。 [GPT-4o 定价](https://developers.openai.com/api/docs/models/gpt-4o)

[检索逐题结果](longmemeval-predictions.jsonl) · [问答逐题结果](longmemeval-qa-top5.jsonl) · [复现命令](README.md#longmemeval-s)
