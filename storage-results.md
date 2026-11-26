# MemoryH · Rust 与 SQLite 原文读取对照

同一批 21 道中文题和相同 BGE 向量。所有候选排名、返回原文、出处和更正可见性逐题一致。计时回放扁平向量 + FTS5 路线的 Top 3，63 次原文读取，覆盖 28 个记录 ID，每档 60 轮。

环境为 macOS-27.0.1-arm64-arm-64bit，Python 3.12.14，SQLite 3.54.1（APSW），测试目录位于内置 APFS SSD 的 `/private/tmp` 下。两边缓存预算均为 16 MiB。计时包含 Python 调用和返回原文，仓库打开、索引建立、预热、清缓存和 FTS5 / 向量排名均在计时外。

| 场景 | 后端 | 读取样本 | P50（µs） | P95（µs） | 读取命中率 | 后端命中 / 未命中计数 |
|---|---|---:|---:|---:|---:|---:|
| 首次回放 | Rust | 3,780 | 0.38 | 9.08 | 55.56% | 2,100 / 1,680 |
| 首次回放 | SQLite | 3,780 | 3.21 | 4.29 | 98.41% | 11,220 / 120 |
| 引擎冷读 | Rust | 3,780 | 8.00 | 9.92 | 0.00% | 0 / 3,780 |
| 引擎冷读 | SQLite | 3,780 | 6.92 | 9.46 | 0.00% | 0 / 11,340 |
| 内存热读 | Rust | 3,780 | 0.33 | 0.38 | 100.00% | 3,780 / 0 |
| 内存热读 | SQLite | 3,780 | 3.21 | 3.83 | 100.00% | 11,340 / 0 |
| F_NOCACHE 冷读 | Rust | 3,780 | 8.67 | 8.96 | 0.00% | 0 / 3,780 |
| F_NOCACHE 冷读 | SQLite | 3,780 | 9.08 | 19.79 | 0.00% | 0 / 11,340 |

读取命中率按整次读取计算。Rust 采用原文 LRU 是否命中，SQLite 采用本次读取是否发生页缓存未命中。末列分别是 Rust 的记录计数和 SQLite 的页面计数。首次回放每轮重开句柄。引擎冷读关闭 Rust LRU 并在每次 SQLite 读取前清除页缓存，系统文件缓存仍参与。内存热读预先读取所有返回记录。

F_NOCACHE 冷读同时要求两边绕过 macOS 文件数据缓存。SQLite 使用基准专用的 APSW VFS，经 Python `pread` 读取页面。该数字包含 VFS 回调，也受文件系统元数据及设备内部缓存影响。 [Apple 文件缓存说明](https://developer.apple.com/library/archive/documentation/Performance/Conceptual/FileSystem/Articles/FilePerformance.html)

## 检索结果核对

- 中文合成用例 21 题的两个命令行报告与原报告一致。[原报告](results.md)
- LoCoMo 中文版 1,986 题的 Rust 路线逐题排名和 Beam 诊断与已发布文件一致。[逐题结果](locomo-beam-predictions.jsonl)
- LongMemEval-S 500 题共用已保存的 BGE 向量，FTS5、扁平向量、Beam 2 和 Beam 3 的前十排序与 SQLite 路线及已发布结果逐题一致。[逐题结果](longmemeval-predictions.jsonl)
