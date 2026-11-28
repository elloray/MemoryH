# MemoryH

MemoryH 是面向 LLM 的嵌入式记忆存储与检索引擎。Rust 核心提供版本化持久存储、内存缓存、BM25 关键词召回、向量召回和混合排序。仓库另保留 SQLite FTS5 对照路线。

此前的 BGE 基线在 LongMemEval-S 的 500 题上，FTS5 + 扁平向量的端到端准确率为 **84.2%**，同配置官方 BM25 为 **73.2%**，语义二叉树 Beam 3 为 **76.6%**。三条路线均将 Top 5 完整会话交给同一个 GPT-4o 快照作答并判分。[LongMemEval-S 评测报告](longmemeval-results.md)

中文合成用例对照整题检索与子问题独立检索，覆盖字面命中、同义改写、跨分支取证和记忆更正。[中文样本与证据标注](cases.json) · [逐题报告](results.md)

## 一键复测 Qwen 拆题和 ScaNN

在 Apple Silicon 上从仓库根目录运行。脚本检查本地三套模型，已有且校验通过的文件直接复用，缺失时下载固定快照。拆题使用 Qwen3.5-2B 的 Q4_K_M GGUF，编码和重排分别沿用 Qwen3-Embedding-0.6B INT8 ONNX 和 Yuanbao-Reranker-0.6B FP16 ONNX。脚本优先检查指定模型目录、项目默认模型目录和 Hugging Face 对应快照缓存。Yuanbao 缺少可用的 ONNX 导出时会从固定版本的原始权重在本机转换并验证。首次准备还会按需下载 llama.cpp、构建 ScaNN C++ 库及 CPU ONNX Runtime。需要 Rust、Python 3、uv、git 和 C++ 工具链。

```sh
python3 scripts/benchmark-scann.py
```

脚本通过 `QwenModels::load` 和 `QwenStore` 运行全部 21 道题，每题预热后默认计时 10 次。`--rounds 1` 可进行短测。每次运行在 `target/scann-eval/runs/` 下新建结果目录，包含 `report.md`、`summary.json`、逐题候选和计时数据 `queries.json`。可用 `--models-dir` 指向已有模型目录，或用 `--output-dir` 指定新的结果目录。[历史四组对照](scann-results.md)

默认每个检索角度分别取最多 32 条向量候选和 32 条 BM25 候选，再按有效记录 ID 去重。向量索引由原生 ScaNN 构建，最多 64 条活跃向量时直接精确打分，超过阈值时用 k-means 分区，默认搜索半数分区（向上取整）。更新后按有效版本重新建索引。用 `QwenModels::with_recall_options(RecallOptions { vector_limit, keyword_limit, exact_threshold, partitions_to_search })` 配置。分区数超过实际叶子数时搜索全部分区。已有安装需要重新运行 `scripts/build-scann.sh` 构建新版桥接库，一键复测脚本会自动处理。[ScaNN 上游](https://github.com/google-research/google-research/tree/master/scann)

测试脚本自动启停本地 llama.cpp 服务。应用集成时，查询默认连接 `127.0.0.1:8080` 的拆题服务。打开 `QwenStore` 时从当前目录的 `target/scann-eval/deps/build/` 加载 ScaNN 动态库。可用 `MEMORYH_SPLITTER_ADDR` 指定其他本机地址，用 `MEMORYH_SCANN_LIB` 指定动态库文件。每个实例也可用 `with_splitter` 和 `with_scann` 修改地址和路径。库加载失败时打开存储报错，拆题请求失败时查询返回错误。准备好上述模型后，可用以下命令在独立终端启动服务。

```sh
target/scann-eval/deps/llama/llama-server \
  -m target/memoryh-models/splitter/Qwen3.5-2B-Q4_K_M.gguf \
  -a memoryh-qwen3.5-2b --host 127.0.0.1 --port 8080 \
  -ngl 0 -t 2 -tb 2 -np 1 -c 2048 --reasoning off --no-cache-prompt --cache-ram 0
```

## 本地 Qwen 检索

调用方把记忆原文传给 `QwenStore::apply`，把问题文本传给 `retrieve`，普通问题默认拿回三条带 ID、来源和版本的原文，清点问题默认拿回五条。`retrieve_top_k` 可显式指定条数。拆题由本机 llama.cpp 服务执行，编码和重排在 Rust 进程中通过 ONNX Runtime CPU 执行。

1. 写入时，Qwen3-Embedding-0.6B 用模型自带的 tokenizer 直接编码原文。引擎取最后一个有效 token 的隐藏状态，L2 归一化成 1,024 维向量，随原文一起提交。BM25 另外用 Rust 的 Jieba 词典分词，并保留连续的字母数字编号。
2. 查询时，Qwen3.5-2B 先判断是否需要拆题。编码器随后处理整题和实际生成的子问题。文档和问题都使用原始文本，无额外指令前缀。本地编码入口最多接收 2,048 个 token，较长输入会截断，存储的原文保持完整。
3. 在同一批当前有效版本上，整题及各子句分别做 BM25 和向量近邻搜索，每路默认各取最多 32 条。合并候选时按不可复用的记录 ID 去重。
4. Yuanbao-Reranker-0.6B 在本地用记忆检索提示词逐条判断候选，输入包含整题和候选原文。取最后一个有效 token 的 `yes` 与 `no` 的 logit 差作为分数。这个顺序与官方双标签 softmax 概率相同。含模板的输入上限为 512 token，截断时保留判断后缀。保留整题逐条打分，再用各个子问题分别重排候选，优先保留各角度的首条证据。个性化建议额外从已有设备、习惯和偏好补召回并独立重排。清点题逐段抽取物品或事件并校验原文引句。Qwen 先检查每个条目的类别和状态，再比较它与已有条目是否指向同一实例。对于单日活动，原文中可核实的日期或相对日用于区分不同场次。选取前五条时，每一步优先选能补充最多新事实的原文，新增数相同时沿用重排名次。[Qwen 模型卡](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) · [Yuanbao 重排模型卡](https://huggingface.co/Yuanbao-Search/Yuanbao-Reranker-0.6B)

```rust
use memoryh::{Memory, Mutation, Options};
use memoryh::qwen::{QwenModels, QwenStore};

fn search(runtime: &std::path::Path) -> memoryh::Result<()> {
    let models = QwenModels::load("target/memoryh-models/qwen",
        "target/memoryh-models/yuanbao", runtime, 2)?;
    let store = QwenStore::open("memoryh-data", Options::default(), models)?;
    store.apply(&[Mutation::Put(Memory {
        id: "coffee-v1".into(), key: "coffee".into(), source: "对话 12".into(),
        body: "午后喝低咖啡因拿铁。".into(),
    })])?;
    for hit in store.retrieve("下午怕睡不着，咖啡怎么点？")? {
        println!("{} {}", hit.memory.source, hit.memory.body);
    }
    Ok(())
}
```

时间题可以调用 `retrieve_with_context(question, &context)`。给 `QueryContext.question_date` 传提问日期。候选的会话日期优先从 `context.session_dates` 按不可变记录 ID 获取，其次读取 `Memory.source` 开头的 ISO 或 LongMemEval 日期。重排输入携带双方日期及相差天数，返回原文保持原样。未提供的日期不做推测。

```rust
use memoryh::qwen::{QueryContext, session_date};

let mut context = QueryContext { question_date: session_date("2025-06-20"), ..Default::default() };
context.session_dates.insert("session-1".into(), session_date("2025-06-10").unwrap());
let hits = store.retrieve_with_context("十天前我去了哪个公园？", &context)?;
```

复测入口支持 `--vector-limit`、`--keyword-limit`、`--exact-threshold`、`--probes`。`--rounds 0` 仅测质量，报告分别给出候选池、Top 3、Top 5 的完整证据命中。运行以下命令可单独验证日期、比较、建议和清点的中文开发用例。

```sh
python3 scripts/benchmark-scann.py --rounds 0 --fixture tests/fixtures/evidence-zh.json
python3 scripts/benchmark-scann.py --rounds 0 --fixture tests/fixtures/events-zh.json
```

在 Apple Silicon 上，准备脚本校验固定权重，首次通过 `uv` 将 Yuanbao 的 BF16 权重导出为 FP16 ONNX 并对照 Transformers 校验。查询可离线执行。已有 Qwen 向量库可直接更换 reranker。[Yuanbao 提示词与 21 题复测](yuanbao-results.md) · [先前 MiniLM 基线](qwen-results.md)

```sh
scripts/fetch-qwen-models.sh
scripts/build-scann.sh
bash scripts/build-onnxruntime-cpu.sh
export ONNX_RUNTIME_LIB="$PWD/target/ort-cpu/build/Release/libonnxruntime.1.32.0.dylib"
cargo run --locked --release --features qwen --example qwen_benchmark -- \
  target/memoryh-models/qwen target/memoryh-models/yuanbao "$ONNX_RUNTIME_LIB" 10
```

可用 `scripts/fetch-qwen-models.sh target/memoryh-models minilm` 准备旧 mMARCO 对照，加载目录为 `target/memoryh-models/reranker`。要替换 Yuanbao 的任务提示词，可调用 `QwenModels::load_with_instruction`。基准程序支持把提示词文件路径作为最后一个可选参数。

已有 BGE 或来源未知的向量库需要从原文重编码。关闭源库后运行下面的命令，`OUTPUT_PARENT` 指向已存在的输出父目录。工具按事务顺序导入原文、更正和撤回，成功后打印新库目录。应用切换到这个目录即可使用 Qwen。

```sh
cargo run --locked --release --features qwen --example qwen_reindex -- \
  "$SOURCE_STORE" "$OUTPUT_PARENT" target/memoryh-models/qwen \
  target/memoryh-models/yuanbao "$ONNX_RUNTIME_LIB"
```

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

### LoCoMo 英文评测与 M4 GPU 重排

评测选取 LoCoMo 的 1,540 道英文题（类别 1–4），使用固定版本的 Mem0 作答与判分脚本。MemoryH 默认每路召回 32 条，GPT-5 每题最多接收 200 条原文。逐题记录实际作答输入 token、检索耗时与重排设备。进度页按相同题号对比 Mem0 公开的逐题结果。

在 Apple Silicon 上，先准备 Yuanbao 的原生 Metal 打分进程与 FP16 权重。Rust 端沿用 CPU 路径的分词、补齐和排序逻辑，仅将 Yuanbao 打分交给 GPU。GPU 运行依赖准备时使用的 LibTorch 动态库。检索和判分命令可在两个终端独立运行，判分使用环境变量 `OPENAI_API_KEY`。

```sh
uv run scripts/prepare-yuanbao-mps.py target/yuanbao-mps
uv run scripts/locomo-compare.py retrieve --reranker mps --recall-width 32
uv run scripts/locomo-compare.py grade
python3 scripts/longmemeval-progress.py --out target/lme-qwen-main --port 8765
```

本机进度页位于 `http://127.0.0.1:8765/`。CPU 对照可显式指定 `--reranker cpu`。检查点位于 `target/locomo-compare/`，续跑会复用已完成的题目。

### LongMemEval-S 与 EverOS 对照

本轮使用 EverOS 固定版本指定的原版 500 题，每题独立建库，保留双方发言及会话日期。MemoryH 取 128 条向量候选和 32 条关键词候选，在不超过 19,999 条向量时精确检索，Yuanbao 重排后返回最多 20 条发言。Apple Silicon 上一题建库可与上一题重排并行，云端同时作答或判分。

作答模型为 GPT-5.6 Terra（`xhigh`，API 默认温度 1），裁判为 GPT-4o-mini（温度 0，单次判分）。EverOS 的 94.0% 使用 Gemini 3.6 Flash（温度 0）读取最多 20 个 episode。照用上游提示词时，说话人别名中包含题号，30 道不可作答题的题号含 `_abs`。检索阶段只接收日期、发言和问题。[EverOS 配置](https://github.com/EverMind-AI/EverOS/blob/d2b74780a443c6d7304ca9aa1bfdf36fdc6d7c9b/benchmarks/configs/longmemeval.toml)

通过本机文件 `~/.config/memoryh/longmemeval-api-key` 提供可调用两个模型的 OpenAI API 密钥。准备好 Metal 重排模型后，分别启动评测和进度页。

```sh
uv run scripts/prepare-yuanbao-mps.py target/yuanbao-mps
python3 scripts/longmemeval-everos-supervise.py
```

```sh
python3 scripts/longmemeval-progress.py --out target/lme-s-everos-20261128 --port 8766
```

[本机实时进度](http://127.0.0.1:8766/everos) 显示正在处理的题目、已判样本量和官宣参照。检查点保存在 `target/lme-s-everos-20261128/`，续跑会复用已完成的题目。收齐 500 道判分后生成 `report.md`，答对至少 471 题即超过 94.0%。

## 实验方法

- **FTS5** 将中文切成重叠双字，将字母数字串保留为整体。查询用同样的方式生成词项，由 SQLite BM25 排序。同一个查询文本的 FTS5 候选由扁平与树路线共用。[SQLite FTS5](https://sqlite.org/fts5.html)
- **扁平向量和语义二叉树** 共用 BGE 中文嵌入。扁平路线逐条比较所有有效原文。树路线先用节点向量导航，每层保留两个候选分支，再比较访问到的叶子中的原文。原文按 `memory_key` 的主题前缀归档，节点向量由当前有效原文计算。[BGE 中文模型](https://huggingface.co/BAAI/bge-small-zh-v1.5)
- **整题对照** 用完整问题查询 FTS5，对每条向量候选取整题和各分句余弦相似度的最大值，最后统一做一次 RRF。**子问题检索** 根据题面标点和「另外」「以及」等连接词拆分。各子问题分别跑 FTS5、向量召回和 RRF，再按题面顺序从各自的 RRF 排名轮流取一条尚未入选的原文。**向量优先** 在同一轮流补位规则之前，先按题面顺序为各子问题保留向量排名第 1 的原文，重复原文只占一个名额。每种方法都受 `K` 个名额限制，单一问题复用整题检索结果。[RRF 论文](https://doi.org/10.1145/1571941.1572114)
- **更正** 在一个 SQLite 事务中保留旧原文、停用旧版本、更新 FTS5，并写入带新出处的更正。语义树按同一个有效版本集合生成快照。报告检查新旧原文在各时点是否正确出现。旧文在 SQLite 中可查，当前召回候选使用有效版本。
- **指标** 每种方法最终各返回最多 `K` 条，默认 `K=3`。各次 RRF 的常数均为 60，每次融合的每个排序通道最多提供 `3K` 条候选。全证据命中要求找齐该题标注的全部原文，且结果里没有当时不可见的版本。报告列出整题和子问题的排名，以及最终召回原文和出处。

上述 Python 检索基线使用内存 SQLite 和固定的分叉，评估原文召回和需逐条打分的原文量。实际耗时还受节点导航、索引维护和语料规模影响。分叉自动生成、动态重平衡和真实对话上的表现有待验证。
