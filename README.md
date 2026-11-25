# MemoryH

MemoryH 是面向 LLM 的记忆检索原型和评测工具。它比较 **SQLite FTS5**、**FTS5 + 扁平向量**和 **FTS5 + 语义二叉树**，提供中文合成用例、LoCoMo 中文评测和官方 LongMemEval-S 英文评测的复现脚本与逐题结果。

在 LongMemEval-S 的 500 题上，FTS5 + 扁平向量的端到端准确率为 **84.2%**，同配置官方 BM25 为 **73.2%**，语义二叉树 Beam 3 为 **76.6%**。三条路线均将 Top 5 完整会话交给同一个 GPT-4o 快照作答并判分。[LongMemEval-S 评测报告](longmemeval-results.md)

中文合成用例对照整题检索与子问题独立检索，覆盖字面命中、同义改写、跨分支取证和记忆更正。[中文样本与证据标注](cases.json) · [逐题报告](results.md)

## 运行

安装 `uv`，在本目录运行。脚本需要带 FTS5 的 Python 3.12 或 3.13。首次运行会下载依赖和中文嵌入模型，模型推理在本机执行。

```sh
uv run --script benchmark.py
uv run --script benchmark.py --top-k 5
```

## LoCoMo 中文评测

```sh
uv run --script locomo_benchmark.py --predictions locomo-predictions.jsonl > locomo-results.md
```

脚本下载并校验固定版本的中文译本。每段对话独立建索引，按发言轮次召回。两条混合路线沿用 BGE、RRF 常数 60、Top 3 和子问题向量首位保留。中文译本没有原型的主题字段，树索引用发言向量递归中位切分成四个叶子。检索阶段只使用问题和对话原文，证据标注用于评分。[LoCoMo 原版](https://github.com/snap-research/locomo) · [中文译本](https://huggingface.co/datasets/justis-xu/memory-eval-zh)

报告按官方发言 ID 口径统计 Evidence Recall@3，并列出可答题和能在三条以内放下全部证据的子集。每题先按官方驱动取三位小数，再按题取平均。本次五路的 9,930 个逐题召回值都与原版评分函数一致。[中文报告](locomo-results.md) · [逐题结果](locomo-predictions.jsonl) · [LoCoMo 评分代码](https://github.com/snap-research/locomo/blob/main/task_eval/evaluation.py#L228-L237)

按原版检索代码在英文数据上本地复跑了 DRAGON。使用发言轮次、首 token 嵌入和点积检索，得到 Evidence Recall@3 **49.25%**、Recall@5 **58.32%**。论文 Table 3 报告的 Top 5 是 **56.7%**，因此这里的 Top 3 来自本地英文复跑。下面的命令会下载两个 DRAGON 编码器。[原版检索代码](https://github.com/snap-research/locomo/blob/main/task_eval/gpt_utils.py#L80-L169) · [原版论文 Table 3](https://aclanthology.org/2024.acl-long.747.pdf#page=7)

```sh
uv run --script locomo_dragon_reference.py
```

AML Cycle 2 文本榜汇总多个数据集的端到端 Answer/Eval 分数。LoCoMo 检索榜中的 Nano-Memory 以会话为 Top 3 单位，MemoryH 本轮以发言为单位。这些分数适合作为背景参照，正式名次需要同一数据、检索单位和评分流程。[AML Cycle 2 官方结果](https://agentmemories.ai/results/cycle-2/) · [AML 评测说明](https://agentmemories.ai/docs) · [Nano-Memory 原论文](https://arxiv.org/html/2604.11628v1)

### Beam 宽度对照

```sh
uv run --script locomo_benchmark.py --beam-sweep \
  --predictions locomo-beam-predictions.jsonl > locomo-beam-results.md
```

Beam 2、3、4 共用同一批中文发言、四叶树、BGE 向量、FTS5、RRF 和 Top 3 规则。报告使用原版的发言级 Evidence Recall@3，并逐条统计叶子剪枝、候选截断、RRF 和子问题合并。 [Beam 结果](locomo-beam-results.md) · [逐题漏召阶段](locomo-beam-predictions.jsonl)

## LongMemEval-S

使用官方英文 cleaned 版，各题独立建立会话索引。检索按官方聚合口径报告 Recall-All 和 nDCG 的 @5、@10。端到端评测覆盖 500 题，使用 Top 5 完整会话、官方 JSON Chain-of-Note 提示词，以及同一个 `gpt-4o-2024-08-06` 快照作答和判分。两条混合路线共用英文 BGE 嵌入和 RRF。 [官方协议](https://github.com/xiaowu0162/LongMemEval#testing-your-system) · [官方 cleaned 数据](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/tree/98d7416c24c778c2fee6e6f3006e7a073259d48f)

[评测报告](longmemeval-results.md) · [逐题检索结果](longmemeval-predictions.jsonl) · [逐题问答结果](longmemeval-qa-top5.jsonl)

下载 `longmemeval_s_cleaned.json`，并将固定版本的 `src/retrieval/eval_utils.py`、`src/generation/run_generation.py`、`src/evaluation/evaluate_qa.py` 和 `src/evaluation/print_qa_metrics.py` 放在同一评测目录。命令中的 `LME_DIR` 指向该目录。 [固定版本的官方源码](https://github.com/xiaowu0162/LongMemEval/tree/9e0b455f4ef0e2ab8f2e582289761153549043fc)

```sh
uv run --with torch==2.9.1 --with transformers==4.57.6 \
  --script longmemeval_benchmark.py --embed-backend mps \
  --data "$LME_DIR/longmemeval_s_cleaned.json" \
  --official-scorer "$LME_DIR/eval_utils.py" \
  --embedding-cache "$LME_DIR/embedding-cache" \
  --predictions longmemeval-predictions.jsonl \
  --generation-input "$LME_DIR/generation-flat.jsonl"

uv run --script longmemeval_qa.py --official-dir "$LME_DIR" \
  --input "$LME_DIR/generation-flat.jsonl" --top-k 5 --estimate
```

执行 API 问答时，移除 `--estimate`，传入 `--output` 和以美元计的 `--budget`，通过环境变量 `OPENAI_API_KEY` 或 `--key-file` 加载密钥。传入 `--reference "$LME_DIR/longmemeval_s_cleaned.json"` 可运行官方准确率汇总。复用同一个输出文件时，脚本核对提示词后跳过已完成的题。检索脚本的 `--reuse --generation-method` 可从逐题排名导出其他路线的作答输入。报告使用 Apple MPS，共用 BGE 的原版权重。

## 实验方法

- **FTS5** 将中文切成重叠双字，将字母数字串保留为整体。查询用同样的方式生成词项，由 SQLite BM25 排序。同一个查询文本的 FTS5 候选由扁平与树路线共用。[SQLite FTS5](https://sqlite.org/fts5.html)
- **扁平向量和语义二叉树** 共用 BGE 中文嵌入。扁平路线逐条比较所有有效原文。树路线先用节点向量导航，每层保留两个候选分支，再比较访问到的叶子中的原文。原文按 `memory_key` 的主题前缀归档，节点向量由当前有效原文计算。[BGE 中文模型](https://huggingface.co/BAAI/bge-small-zh-v1.5)
- **整题对照** 用完整问题查询 FTS5，对每条向量候选取整题和各分句余弦相似度的最大值，最后统一做一次 RRF。**子问题检索** 根据题面标点和「另外」「以及」等连接词拆分。各子问题分别跑 FTS5、向量召回和 RRF，再按题面顺序从各自的 RRF 排名轮流取一条尚未入选的原文。**向量优先** 在同一轮流补位规则之前，先按题面顺序为各子问题保留向量排名第 1 的原文，重复原文只占一个名额。每种方法都受 `K` 个名额限制，单一问题复用整题检索结果。[RRF 论文](https://doi.org/10.1145/1571941.1572114)
- **更正** 在一个 SQLite 事务中保留旧原文、停用旧版本、更新 FTS5，并写入带新出处的更正。语义树按同一个有效版本集合生成快照。报告检查新旧原文在各时点是否正确出现。旧文在 SQLite 中可查，当前召回候选使用有效版本。
- **指标** 每种方法最终各返回最多 `K` 条，默认 `K=3`。各次 RRF 的常数均为 60，每次融合的每个排序通道最多提供 `3K` 条候选。全证据命中要求找齐该题标注的全部原文，且结果里没有当时不可见的版本。报告列出整题和子问题的排名，以及最终召回原文和出处。

这版使用内存 SQLite 和固定的分叉，评估原文召回和需逐条打分的原文量。实际耗时还受节点导航、索引维护和语料规模影响。分叉自动生成、动态重平衡和真实对话上的表现有待验证。
