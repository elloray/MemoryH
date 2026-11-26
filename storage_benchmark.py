# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["apsw==3.54.1.0", "fastembed==0.9.1"]
# ///

"""Compare record reads for the same MemoryH Top 3 retrieval trace."""

import argparse
from contextlib import closing
from dataclasses import dataclass, field
from functools import partial
import json
from math import ceil
import os
from pathlib import Path
import platform
from statistics import median
import sys
from tempfile import TemporaryDirectory
from time import perf_counter_ns

import apsw
from fastembed import TextEmbedding

from benchmark import MODEL, run_benchmark
from retrieval_storage import READ_RECORD, SQLiteStorage, apply_native, open_native


CACHE_BYTES = 16 * 1024 * 1024
BLOCK_CACHE_BYTES = 4 * 1024 * 1024
TOP_K = 3
ROUTE = "flat_vector_first"


class SharedEmbeddings:
    def __init__(self):
        self.model = TextEmbedding(MODEL, threads=2)
        self.vectors = {}

    def embed(self, texts):
        missing = [text for text in texts if text not in self.vectors]
        if missing:
            self.vectors.update(zip(missing, self.model.embed(missing), strict=True))
        return (self.vectors[text] for text in texts)


class UncachedFile(apsw.VFSFile):
    def __init__(self, name, flags):
        import fcntl

        self.fd = None
        super().__init__("", name, flags)
        try:
            filename = name.filename() if isinstance(name, apsw.URIFilename) else name
            self.fd = os.open(filename, os.O_RDONLY)
            # CPython does not expose this Darwin fcntl constant on every build.
            fcntl.fcntl(self.fd, getattr(fcntl, "F_NOCACHE", 48), 1)
        except BaseException:
            self.xClose()
            raise

    def xRead(self, amount, offset):
        read = os.pread(self.fd, amount, offset)
        while len(read) < amount:
            remainder = os.pread(self.fd, amount - len(read), offset + len(read))
            if not remainder:
                break
            read += remainder
        return read

    def xClose(self):
        try:
            if self.fd is not None:
                fd, self.fd = self.fd, None
                os.close(fd)
        finally:
            super().xClose()


class UncachedVFS(apsw.VFS):
    name = "memoryh-benchmark-uncached"

    def __init__(self):
        super().__init__(self.name, "")

    def xOpen(self, name, flags):
        if flags[0] & apsw.SQLITE_OPEN_MAIN_DB:
            return UncachedFile(name, flags)
        return apsw.VFSFile("", name, flags)


class SQLiteReader:
    def __init__(self, path, vfs=None):
        self.db = apsw.Connection(str(path), flags=apsw.SQLITE_OPEN_READONLY, vfs=vfs)
        self.db.execute(f"PRAGMA cache_size=-{CACHE_BYTES // 1024}")
        self.db.execute("PRAGMA mmap_size=0").fetchall()

    def close(self):
        self.db.close()

    def read(self, record_id):
        return self.db.execute(READ_RECORD, (record_id,)).fetchone()

    def clear_cache(self):
        self.db.execute("PRAGMA shrink_memory")

    def counters(self):
        return tuple(self.db.status(metric)[0] for metric in (
            apsw.SQLITE_DBSTATUS_CACHE_HIT, apsw.SQLITE_DBSTATUS_CACHE_MISS,
        )) + (0, 0)


class RustReader:
    def __init__(self, path, cold=False, uncached=False, with_blocks=False):
        self.db = open_native(
            path, 0 if cold else CACHE_BYTES, uncached_disk=uncached,
            block_cache_bytes=BLOCK_CACHE_BYTES if with_blocks else 0,
        )

    def close(self):
        self.db.close()

    def read(self, record_id):
        return self.db.read(record_id)

    def clear_cache(self):
        # The cold reader was opened with a zero-byte LRU.
        pass

    def counters(self):
        hits, _, reads = self.db.cache_stats()
        return (hits, reads, *self.db.block_cache_stats())


@dataclass
class Measurement:
    durations: list = field(default_factory=list)
    hits: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    block_hits: int = 0
    block_reads: int = 0

    def run(self, reader, trace, originals, repeats=1, *, cold=False, warm=False):
        if warm:
            for record_id in dict.fromkeys(trace):
                if reader.read(record_id) != originals[record_id]:
                    raise ValueError(f"预热读取原文不符：{record_id}")
        hit_start, miss_start, block_hit_start, block_read_start = reader.counters()
        previous_misses = miss_start
        for _ in range(repeats):
            for record_id in trace:
                if cold:
                    reader.clear_cache()
                start = perf_counter_ns()
                original = reader.read(record_id)
                self.durations.append(perf_counter_ns() - start)
                misses = reader.counters()[1]
                self.hits += misses == previous_misses
                previous_misses = misses
                if original != originals[record_id]:
                    raise ValueError(f"读取原文不符：{record_id}")
        cache_hits, cache_misses, block_hits, block_reads = reader.counters()
        self.cache_hits += cache_hits - hit_start
        self.cache_misses += cache_misses - miss_start
        self.block_hits += block_hits - block_hit_start
        self.block_reads += block_reads - block_read_start


def compare_retrieval(fixture):
    model = SharedEmbeddings()
    baseline = run_benchmark(fixture, model, TOP_K, storage="sqlite")
    native = run_benchmark(fixture, model, TOP_K, storage="rust")
    if baseline != native:
        raise RuntimeError("SQLite 与 Rust 后端的排名、原文或分支统计不一致")
    results, originals, _ = native
    trace = [record_id for result in results.values() for record_id in result.rankings[ROUTE][:TOP_K]]
    return trace, originals


def seed_stores(root, fixture):
    rust_path, sqlite_path = root / "native", root / "baseline.sqlite3"
    with closing(open_native(rust_path)) as native, closing(SQLiteStorage(sqlite_path)) as sqlite:
        for stage in ("memories", "corrections"):
            apply_native(native, fixture[stage])
            sqlite.apply(fixture[stage])
    return rust_path, sqlite_path


def measure(open_reader, trace, originals, rounds, *, fresh_handles=False, cold=False, warm=False):
    result = Measurement()
    if fresh_handles:
        for _ in range(rounds):
            with closing(open_reader()) as reader:
                result.run(reader, trace, originals)
    else:
        with closing(open_reader()) as reader:
            result.run(reader, trace, originals, rounds, cold=cold, warm=warm)
    return result


def print_report(measurements, fixture, trace, rounds, root):
    print("# MemoryH · Rust 与 SQLite 原文读取对照\n")
    print(f"同一批 {len(fixture['questions'])} 道中文题和相同 BGE 向量。"
          "所有候选排名、返回原文、出处和更正可见性逐题一致。"
          f"计时回放关键词 + 扁平向量路线的 Top 3，{len(trace)} 次原文读取，"
          f"覆盖 {len(set(trace))} 个记录 ID，每档 {rounds} 轮。")
    print()
    print(f"环境为 {platform.platform()}，Python {platform.python_version()}，"
          f"SQLite {apsw.sqlitelibversion()}（APSW），测试目录位于 `{root}` 下。"
          "缓存预算均为 16 MiB，其中 Rust 块缓存配置分配最多 4 MiB 给 4 KiB 块、12 MiB 给原文。"
          "计时包含 Python 调用和返回原文，"
          "仓库打开、索引建立、预热、清缓存和关键词／向量排名均在计时外。")
    print()
    print("| 场景 | 后端 | 读取样本 | P50（µs） | P95（µs） | 读取命中率 | 主缓存命中 / 磁盘计数 | 块命中 / 读块次数 |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for stage, backend, result in measurements:
        ordered = sorted(result.durations)
        count = len(ordered)
        print(f"| {stage} | {backend} | {count:,} | {median(ordered) / 1000:.2f} | "
              f"{ordered[ceil(0.95 * count) - 1] / 1000:.2f} | {result.hits / count:.2%} | "
              f"{result.cache_hits:,} / {result.cache_misses:,} | "
              f"{result.block_hits:,} / {result.block_reads:,} |")
    print()
    print("读取命中率按整次读取计算。Rust 命中包括原文 LRU 或所需块全部命中，"
          "SQLite 命中要求本次读取没有页缓存未命中。"
          "主缓存列中，Rust 按原文命中和访问磁盘的请求计数，SQLite 按页面命中和未命中计数。"
          "首次回放每轮重开句柄。引擎冷读关闭 Rust LRU 并在每次 SQLite 读取前清除页缓存，"
          "系统文件缓存仍参与。内存热读预先读取所有返回记录。")
    if any(stage == "F_NOCACHE 冷读" for stage, _, _ in measurements):
        print()
        print("F_NOCACHE 两档均要求后端绕过 macOS 文件数据缓存。"
              "SQLite 使用基准专用的 APSW VFS，经 Python `pread` 读取页面。"
              "该数字包含 VFS 回调，也受文件系统元数据及设备内部缓存影响。"
              " [Apple 文件缓存说明](https://developer.apple.com/library/archive/documentation/Performance/Conceptual/FileSystem/Articles/FilePerformance.html)")


def main():
    parser = argparse.ArgumentParser(description="回放同一批 Top 3，比较 Rust 和 SQLite 原文读取与缓存")
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--work-dir", type=Path, help="两种数据库共同使用的临时文件父目录")
    parser.add_argument("--uncached-io", action="store_true", help="额外测量 macOS F_NOCACHE 冷读")
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds 必须大于零")
    if args.uncached_io and sys.platform != "darwin":
        parser.error("--uncached-io 需要 macOS")

    fixture = json.loads(Path(__file__).with_name("cases.json").read_text(encoding="utf-8"))
    trace, originals = compare_retrieval(fixture)
    with TemporaryDirectory(prefix="memoryh-read-bench-", dir=args.work_dir) as directory:
        root = Path(directory).resolve()
        rust_path, sqlite_path = seed_stores(root, fixture)
        measurements = []
        stages = (
            ("首次回放", {"fresh_handles": True}),
            ("引擎冷读", {"cold": True}),
            ("内存热读", {"warm": True}),
        )
        for name, options in stages:
            readers = {
                "Rust 原文": partial(RustReader, rust_path, cold=options.get("cold", False)),
                "SQLite": partial(SQLiteReader, sqlite_path),
            }
            if not options.get("cold"):
                readers["Rust 原文 + 块"] = partial(RustReader, rust_path, with_blocks=True)
            for backend, reader in readers.items():
                test = measure(reader, trace, originals, args.rounds, **options)
                measurements.append((name, backend, test))
        if args.uncached_io:
            vfs = UncachedVFS()
            try:
                readers = {
                    "Rust 原文": partial(RustReader, rust_path, cold=True, uncached=True),
                    "SQLite": partial(SQLiteReader, sqlite_path, vfs=vfs.name),
                }
                for backend, reader in readers.items():
                    test = measure(reader, trace, originals, args.rounds, cold=True)
                    measurements.append(("F_NOCACHE 冷读", backend, test))
                readers = {
                    "Rust 原文": partial(RustReader, rust_path, uncached=True),
                    "SQLite": partial(SQLiteReader, sqlite_path, vfs=vfs.name),
                    "Rust 原文 + 块": partial(RustReader, rust_path, uncached=True, with_blocks=True),
                }
                for backend, reader in readers.items():
                    test = measure(reader, trace, originals, args.rounds, fresh_handles=True)
                    measurements.append(("F_NOCACHE 首次回放", backend, test))
            finally:
                vfs.unregister()
        print_report(measurements, fixture, trace, args.rounds, root.parent)


if __name__ == "__main__":
    main()
