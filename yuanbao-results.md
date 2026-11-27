# Yuanbao-Reranker-0.6B 记忆检索复测

在 Apple M4 本地 CPU 上，用相同的 28 条初始记忆、4 条更正和 21 道中文问题比较三种重排设置。编码器为 Qwen3-Embedding-0.6B INT8，整题和子句分别走 BM25 和精确向量召回，每路最多 9 条。在同一有效版本集合中按记忆 ID 去重，统一用整题重排，取前三条原文。[用例与标注](cases.json)

逐题核对了三组的完整候选 ID 集合，21/21 相同。每题去重后有 9 至 22 条候选，中位数为 11 条，21 题合计 267 个问题与候选原文对。返回的每一条原文和来源都与样本一致。更正前后的禁用版本进入候选池的次数为 0。每组均关闭并重新打开数据库，验证了更正后阶段 17 道题的 Top 3、顺序、原文和版本一致。

## 提示词与打分

> 判断这条记忆原文能否为问题提供直接证据。若问题有多个独立子问题，能回答其中任意一个就算相关。核对所问对象、条件和具体事实；仅仅话题相近不足以判为相关。

这段话放在官方模板的 `<Instruct>` 字段，`<Query>` 放整道原始问题，`<Document>` 每次放一条候选原文。系统指令和 assistant 判断后缀沿用官方约定。取最后一个有效 token 的 yes、no 两个 logits，按两者之差降序排列，与按双标签 softmax 概率排序等价。模型逐条打分，各条证据共用三个名额。通用提示词对照使用官方模型卡的英文网页检索指令。三组的重排输入上限均为 512 token。对全部问题与 32 条原文的组合计数，Yuanbao 记忆提示词最长 158 token，通用提示词最长 125 token，两组均无截断。任务提示词根据此前观察到的跨分支漏证据设计，在本轮推理前固定。成绩属于同一批开发用例的回归结果。

Yuanbao 的官方 BF16 快照为 `0101299a7937a016848d8d55b092f6e31cfa312d`。本机使用 FP16 ONNX 导出，仅计算 yes/no 两个输出。单条和不同长度混合批次都与同权重 FP16 Transformers 前向对照过。另用中文问题和两条长短不同的原文测试完整 Rust 模板到分数的路径，两条 yes/no 对数几率均在 Transformers 参考值的 0.15 以内。MiniLM 使用原 mMARCO ARM64 INT8 模型。[Yuanbao 官方权重和评分协议](https://huggingface.co/Yuanbao-Search/Yuanbao-Reranker-0.6B/tree/0101299a7937a016848d8d55b092f6e31cfa312d) · [MiniLM 模型卡](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1)

## Top 3 完整证据命中

一题计为找齐，要求 Top 3 包含该题的所有标注证据，且没有禁用版本。

| 类别 | MiniLM | Yuanbao 官方通用提示词 | Yuanbao 记忆提示词 |
|---|---:|---:|---:|
| 字面命中 | 2/2 | 2/2 | 2/2 |
| 换个说法 | 6/6 | 6/6 | 6/6 |
| 跨分支 | 3/5 | 5/5 | 5/5 |
| 记忆更正 | 8/8 | 8/8 | 8/8 |
| 全部 | 19/21 | 21/21 | 21/21 |

第一条原文即为任一标注证据的题数分别为 MiniLM 15/21、Yuanbao 通用提示词 20/21、Yuanbao 记忆提示词 18/21。两种 Yuanbao 提示词均将 Q09 的低因咖啡原文 M18 从 MiniLM 的第 10 名提到第 1，将 Q13 的替代运动原文 M19 从第 4 名提到第 1。这两题的另一条证据均排第 2。自定义提示词在 Top 3 找齐率上与官方通用提示词持平，Q07 和 Q16 的正确证据从通用提示词下的第 1 名降到第 2 名。

## 延迟

使用 Rust release 可执行文件，同一 ONNX Runtime CPU 运行库，各模型会话 2 线程，各组顺序执行。每个阶段先逐题预热。MiniLM 和 Yuanbao 记忆提示词每题计时 10 轮，各 210 次。官方提示词组只用于区分提示词对质量的影响，每题额外计时 1 次。计时包含问题编码、召回、取原文和整题重排，不含模型启动、写入和重开验证。分位数按全部样本的 nearest-rank 计算。

| 整条查询链路 | P50 | P95 |
|---|---:|---:|
| MiniLM | 57.47 ms | 161.78 ms |
| Yuanbao 记忆提示词 | 2,794.98 ms | 5,641.34 ms |

## 逐题证据名次

需求列列出所有应返回的原文 ID。三个名次列均按需求列的顺序列出；超过 3 表示已召回、未进 Top 3。

| 题号 | 所需证据 | MiniLM | Yuanbao 通用 | Yuanbao 记忆 | 记忆提示词的 Top 3 |
|---|---|---|---|---|---|
| Q01 | M02 | 1 | 1 | 1 | M02、M08、M09 |
| Q02 | M13 | 1 | 1 | 1 | M13、M15、M14 |
| Q03 | M01 | 1 | 1 | 1 | M01、M02、M29 |
| Q04 | M09 | 2 | 1 | 1 | M09、M14、M13 |
| Q05 | M03 | 1 | 1 | 1 | M03、M01、M08 |
| Q06 | M18 | 1 | 1 | 1 | M18、M21、M31 |
| Q07 | M10 | 1 | 1 | 2 | M15、M10、M12 |
| Q08 | M28 | 1 | 1 | 1 | M28、M32、M12 |
| Q09 | M29、M18 | 2、10 | 2、1 | 2、1 | M18、M29、M07 |
| Q10 | M30、M02 | 1、3 | 3、1 | 3、1 | M02、M15、M30 |
| Q11 | M17、M24 | 1、2 | 1、2 | 1、2 | M17、M24、M32 |
| Q12 | M14、M32 | 3、2 | 1、2 | 1、2 | M14、M32、M13 |
| Q13 | M06、M19 | 1、4 | 2、1 | 2、1 | M19、M06、M01 |
| Q14 | M05 | 1 | 1 | 1 | M05、M07、M08 |
| Q15 | M29 | 1 | 1 | 1 | M29、M07、M08 |
| Q16 | M11 | 2 | 1 | 2 | M15、M11、M25 |
| Q17 | M30 | 2 | 2 | 2 | M15、M30、M25 |
| Q18 | M20 | 1 | 1 | 1 | M20、M21、M27 |
| Q19 | M31 | 2 | 1 | 1 | M31、M21、M27 |
| Q20 | M23 | 1 | 1 | 1 | M23、M26、M24 |
| Q21 | M32 | 1 | 1 | 1 | M32、M26、M24 |

## 复现

按 [本地运行说明](README.md#本地-qwen-检索) 准备模型与运行库。从仓库根目录运行。基准 JSON 的 `ranked_pool` 含按重排分数排序的所有候选 ID，`evidence_ranks` 为标注证据的名次。

```sh
scripts/fetch-qwen-models.sh target/memoryh-models minilm
cargo build --locked --release --features qwen --example qwen_benchmark
bench=target/release/examples/qwen_benchmark
models=target/memoryh-models
"$bench" "$models/qwen" "$models/yuanbao" "$ONNX_RUNTIME_LIB" 10 > memory.json
"$bench" "$models/qwen" "$models/reranker" "$ONNX_RUNTIME_LIB" 10 > minilm.json
printf '%s\n' 'Given a web search query, retrieve relevant passages that answer the query' > official.txt
"$bench" "$models/qwen" "$models/yuanbao" "$ONNX_RUNTIME_LIB" 1 official.txt > official.json
```
