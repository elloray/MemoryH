"""Rust bindings and the SQLite comparison backend."""

from contextlib import closing, contextmanager
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unicodedata


CREATE_FTS = "CREATE VIRTUAL TABLE memory_fts USING fts5(id UNINDEXED, terms, tokenize='unicode61')"
READ_RECORD = "SELECT source, body FROM memory WHERE id = ?"


def lexical_terms(text):
    for part in re.findall(r"[\u3400-\u9fff]+|[a-zA-Z0-9]+", unicodedata.normalize("NFKC", text)):
        if len(part) > 1 and "\u3400" <= part[0] <= "\u9fff":
            yield from (part[i:i + 2] for i in range(len(part) - 1))
        else:
            yield part


def initialize_database(db):
    db.executescript("""
        CREATE TABLE memory (
            id TEXT PRIMARY KEY,
            memory_key TEXT NOT NULL,
            source TEXT NOT NULL,
            body TEXT NOT NULL,
            active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1))
        );
        CREATE UNIQUE INDEX active_memory_key ON memory(memory_key) WHERE active = 1;
    """)
    db.execute(CREATE_FTS)


def index_record(db, record):
    if old := record.get("supersedes"):
        db.execute("DELETE FROM memory_fts WHERE id = ?", (old,))
    db.execute(
        "INSERT INTO memory_fts(id, terms) VALUES (?, ?)",
        (record["id"], " ".join(lexical_terms(record["text"]))),
    )


def save_memories(db, records):
    with db:
        for record in records:
            if old := record.get("supersedes"):
                updated = db.execute(
                    "UPDATE memory SET active = 0 WHERE id = ? AND memory_key = ? AND active = 1",
                    (old, record["key"]),
                )
                if updated.rowcount != 1:
                    raise ValueError(f"更正未找到有效的旧版本：{old}")
            db.execute(
                "INSERT INTO memory(id, memory_key, source, body) VALUES (?, ?, ?, ?)",
                (record["id"], record["key"], record["source"], record["text"]),
            )
            index_record(db, record)


def lexical_search(db, question, limit):
    terms = dict.fromkeys(lexical_terms(question))
    if not terms:
        return []
    expression = " OR ".join(f'"{term}"' for term in terms)
    return [row[0] for row in db.execute(
        "SELECT f.id FROM memory_fts f JOIN memory m ON m.id = f.id "
        "WHERE memory_fts MATCH ? AND m.active = 1 ORDER BY f.rank, f.id LIMIT ?",
        (expression, limit),
    )]


def open_native(path, cache_bytes=None, *, uncached_disk=False, block_cache_bytes=0):
    try:
        from memoryh_store import Store
    except ImportError as error:
        raise RuntimeError("Rust 后端需要使用 uv run --with . --script 安装绑定") from error
    return Store(path, cache_bytes, uncached_disk=uncached_disk, block_cache_bytes=block_cache_bytes)


def apply_native(store, records, vectors=None):
    records = list(records)
    embeddings = None if vectors is None else [vectors[record["id"]].tolist() for record in records]
    return store.apply([
        (record["id"], record["key"], record["source"], record["text"], record.get("supersedes") or None)
        for record in records
    ], embeddings)


class SQLiteStorage:
    def __init__(self, path):
        import sqlite3

        self.lexical_db = sqlite3.connect(path)
        try:
            initialize_database(self.lexical_db)
        except Exception:
            self.close()
            raise

    def close(self):
        self.lexical_db.close()

    def apply(self, records, vectors=None):
        save_memories(self.lexical_db, records)

    def active_keys(self):
        return dict(self.lexical_db.execute("SELECT id, memory_key FROM memory WHERE active = 1"))

    def read(self, record_id):
        return self.lexical_db.execute(READ_RECORD, (record_id,)).fetchone()


class RustStorage:
    def __init__(self, native):
        self.native = native

    def apply(self, records, vectors=None):
        return apply_native(self.native, records, vectors)

    def search(self, variants, vectors, top_k, beam_width, prefixes):
        return self.native.search(
            [(text, vector.tolist()) for text, vector in zip(variants, vectors, strict=True)],
            top_k, beam_width, prefixes,
        )

    def active_keys(self):
        return dict(self.native.visible_keys())

    def read(self, record_id):
        return self.native.read(record_id)

@contextmanager
def open_retrieval_store(backend):
    if backend == "rust":
        with TemporaryDirectory(prefix="memoryh-retrieval-") as path:
            with closing(open_native(Path(path))) as native:
                yield RustStorage(native)
    elif backend == "sqlite":
        with closing(SQLiteStorage(":memory:")) as store:
            yield store
    else:
        raise ValueError(f"未知存储后端：{backend}")
