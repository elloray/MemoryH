# MemoryH

MemoryH 是面向 LLM 的嵌入式记忆存储与检索引擎。Rust 核心提供版本化持久存储、内存缓存、BM25 关键词召回、向量召回和混合排序。仓库另保留 SQLite FTS5 对照路线。

在 LongMemEval-S 的 500 题上，FTS5 + 扁平向量的端到端准确率为 **84.2%**，同配置官方 BM25 为 **73.2%**，语义二叉树 Beam 3 为 **76.6%**。三条路线均将 Top 5 完整会话交给同一个 GPT-4o 快照作答并判分。[LongMemEval-S 评测报告](longmemeval-results.md)

中文合成用例对照整题检索与子问题独立检索，覆盖字面命中、同义改写、跨分支取证和记忆更正。[中文样本与证据标注](cases.json) · [逐题报告](results.md)

## Rust 存储引擎

- **内存层**按记录 ID 缓存热原文，用字节预算约束 LRU。冷读按内存索引里的段号和偏移从磁盘取原文，再回填内存。可通过 `cache_stats()` 查看命中、磁盘读取和淘汰次数。
- **块缓存**可通过 `Store::open_with_block_cache` 从同一 RAM 预算划出一部分。原文 LRU 未命中时，小记录可复用同一段文件的 4 KiB 对齐块，解码后进入原文 LRU。超过 4 KiB 的编码记录走原有精确读取路径。
- **磁盘层**使用 MemoryH 自定义的二进制追加日志。每批操作共用一个修订号，通过长度与 CRC 校验，落盘后才对读操作可见。达到目标段大小后追加到新文件，旧段保留历史版本。
- **记忆版本**由逻辑键和不可复用的记录 ID 标识。更正和撤回要求传入当前记录 ID。用 `get_at` 读取某个修订号的可见版本，用 `get_by_id` 读取保留的历史原文。

多线程写入可用 `Store::open(path, options)?.into_shared()`，将返回的 `SharedStore` 克隆给各线程。`BgeStore` 已支持直接克隆，多个句柄共用编码器和提交队列。每次成功的非空 `apply` 拿到独立修订号，成功返回时该请求已完成磁盘同步，后续读取可见。引擎按入队顺序处理已到达的请求，每组最多 64 个请求、目标 1 MiB，单个大请求独立提交。段文件切换前同步旧段。单写者立即提交，多写者可共享一次同步。可用 `data_syncs()` 读取本次打开库后成功的数据同步次数。[组提交复测](group-commit-results.md)

安装 Rust 1.89 或更高版本，运行 `cargo test --locked`。以下示例展示基本嵌入方式，存储目录的父目录需要已存在。

```rust
use memoryh::{Memory, Mutation, Options, Query, SearchMethod, SearchOptions, Store};

fn main() -> memoryh::Result<()> {
    let mut store = Store::open("memoryh-data", Options::default())?;
    store.apply(&[Mutation::Put(Memory {
        id: "coffee-v1".into(),
        key: "preferences.coffee".into(),
        source: "2026-11-25 conversation".into(),
        body: "拿铁用无糖豆奶".into(),
    })])?;
    let query = Query { text: "无糖豆奶", embedding: None };
    let coffee = store.retrieve(&[query], &SearchOptions::default(), SearchMethod::Keywords)?;
    println!("{coffee:?}");
    Ok(())
}
```

同一存储目录使用一个独占句柄。本版启动时重放段文件重建内存索引，恢复会截去最后一段中未写完整的事务，并将完整事务校验失败报告为损坏。内存缓存默认 16 MiB，磁盘段目标大小默认 64 MiB，单批编码大小上限 64 MiB。当前测试运行在 macOS。

`Store::apply_with_embeddings` 将记录向量随原文写入同一事务。引擎从日志恢复当前版本的倒排索引和向量，语义树在查询时构建。更正或撤回会更新当前检索候选，旧原文仍可用 `get_by_id` 读取。旧格式段文件可直接打开并构建关键词索引。

`Store` 也接受调用方生成的向量，同一个库使用同一模型的记录与问题向量。`query_variants` 可拆分问题，`Store::search` 接收整题和子问题及其向量，返回关键词、扁平向量、树向量和融合排名。`Store::retrieve` 可直接返回指定路线的 Top K 原文。倒排索引和活跃向量常驻内存，单独计入引擎内存开销。

### Rust 原生中文 BGE

启用 Cargo 的 `bge` 功能后，`BgeStore` 在 Rust 进程中加载中文 BGE。它用 ONNX Runtime 的 CPU 后端推理，使用 512 token 截断、CLS 池化、L2 归一化和原有的中文查询前缀。`apply` 接受原文写入与更正，`retrieve` 接受问题文本并返回最多三条带来源的原文，默认走 BM25 + 扁平向量和子问题向量首位保留。 [评测用的模型快照](https://huggingface.co/Qdrant/bge-small-zh-v1.5/tree/46fbe35fd4374a00fee7de77dfddaeb6dd6a2c59)

应用启动时传入包含 `model_optimized.onnx` 和 `tokenizer.json` 的目录，以及支持 API 24 或更高版本的 ONNX Runtime 动态库路径。模型实例可复用到后续请求中。

在 Apple Silicon 上，`bash scripts/build-onnxruntime-cpu.sh` 会构建 ONNX Runtime 1.32.0 的 CPU 动态库，关闭 1DS 遥测、CoreML 和加速器探测，使用标准错误输出日志。构建需要 `uv` 和 Apple C++ 工具链。默认产物在 `target/ort-cpu/build/Release/libonnxruntime.1.32.0.dylib`。 [ONNX Runtime 1.32.0 源码存档](https://sourceforge.net/projects/onnx-runtime.mirror/files/v1.32.0/)

```rust
use memoryh::{Options, Result};
use memoryh::bge::{BgeSmallZh, BgeStore};

fn main() -> Result<()> {
    let encoder = BgeSmallZh::load(
        "bge-small-zh-v1.5",
        "target/ort-cpu/build/Release/libonnxruntime.1.32.0.dylib",
        2,
    )?;
    let store = BgeStore::open("memoryh-data", Options::default(), encoder)?;
    let hits = store.retrieve("我点拿铁时用什么奶？")?;
    println!("{hits:?}");
    Ok(())
}
```

同一批中文用例可用下列命令核对。Rust 示例直接生成结果。`uv` 命令额外运行 Python FastEmbed 基线，使用同一份模型目录并对比 Top 3、模型初始化和在线查询延迟。

```sh
cargo build --locked --release --features bge --example bge_benchmark
target/release/examples/bge_benchmark "$BGE_MODEL_DIR" "$ONNX_RUNTIME_LIB" 30
uv run --with . --script bge_benchmark.py \
  --model-dir "$BGE_MODEL_DIR" --runtime-library "$ONNX_RUNTIME_LIB" --rounds 30
```

用同一个 Rust 可执行文件比较两套运行库时，传入 `--reference-runtime-library`。这条命令只用 Python 汇总报告，两个检索进程都运行原生 Rust 示例。

```sh
python3 bge_benchmark.py --model-dir "$BGE_MODEL_DIR" \
  --reference-runtime-library "$REFERENCE_ONNX_RUNTIME_LIB" \
  --runtime-library "$ONNX_RUNTIME_LIB" --rounds 100
```

## 运行检索对照

三个检索脚本均支持 `--storage rust`。这条路线由 Python 运行本地 BGE 模型、调用 Rust 和评分，由 Rust 负责写入、分词、BM25、语义树、向量打分、RRF、子问题合并和原文读取。`--storage sqlite` 使用带 FTS5 的 Python，保留原先的对照流程。

```sh
uv run --with . --script benchmark.py --storage rust
uv run --script benchmark.py --storage sqlite
```

### 存储延迟和缓存

延迟测试回放合成用例中 FTS5 + 扁平向量路线返回的 Top 3 原文 ID。它保留更正前后的检索顺序并逐条核对原文，比较真实磁盘文件上的首次回放、引擎冷读和内存热读。[对照结果](storage-results.md)

同一命令也对照了 Rust 原文 LRU 加 4 KiB 块缓存，记录整次读取命中、块命中与读块次数。[块缓存复测](storage-block-results.md)

```sh
uv run --with . --script storage_benchmark.py --rounds 60
```

macOS 上可加 `--uncached-io`，在同一文件系统上测两边启用 `F_NOCACHE` 的冷读。用 `--work-dir` 指定两个临时库共同使用的父目录。[macOS 文件缓存说明](https://developer.apple.com/library/archive/documentation/Performance/Conceptual/FileSystem/Articles/FilePerformance.html)

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

上述 Python 检索基线使用内存 SQLite 和固定的分叉，评估原文召回和需逐条打分的原文量。实际耗时还受节点导航、索引维护和语料规模影响。分叉自动生成、动态重平衡和真实对话上的表现有待验证。
