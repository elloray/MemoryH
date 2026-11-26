//! Embedded storage for versioned memories with a bounded RAM cache and a
//! durable, append-only disk tier. All mutations in a batch share one revision.

#[cfg(feature = "bge")]
pub mod bge;
mod cache;
mod format;
mod group;
mod log;
pub use group::SharedStore;
#[cfg(feature = "python")]
mod python;
mod retrieval;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;

pub use cache::CacheStats;
use cache::{BLOCK_SIZE, BlockCache, Cache};
use format::{Entry, Operation, decode_batch, decode_operation, encode_batch};
use log::{Location, Log};
pub use retrieval::{
    PrefixLeaf, Query, Ranking, SearchMethod, SearchOptions, SearchResult, TreeLayout,
    query_variants,
};
use retrieval::{RetrievalIndex, normalize};

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("database is already open")]
    Locked,
    #[error("memory mutation conflict: {0}")]
    Conflict(String),
    #[error("invalid search query: {0}")]
    InvalidQuery(String),
    #[cfg(feature = "bge")]
    #[error("BGE inference error: {0}")]
    Embedding(String),
    #[error("corrupt memory store: {0}")]
    Corrupt(String),
    #[error("encoded batch exceeds 64 MiB")]
    BatchTooLarge,
    #[error("sequence exhausted")]
    SequenceExhausted,
    #[error("write outcome is uncertain; reopen the store to recover")]
    RecoveryRequired,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Memory {
    pub id: String,
    pub key: String,
    pub source: String,
    pub body: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Mutation {
    /// A new ID for a key with no current memory.
    Put(Memory),
    /// Atomically supersede the current version of the same key.
    Replace { expected_id: String, memory: Memory },
    /// Make a key invisible to current reads, keeping prior versions readable.
    Retract { key: String, expected_id: String },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Record {
    pub memory: Memory,
    pub revision: u64,
    pub supersedes: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Version {
    pub revision: u64,
    /// A missing ID is a retraction.
    pub id: Option<String>,
}

#[derive(Clone, Debug)]
pub struct Options {
    /// RAM budget for caching. Zero sends all record reads to disk.
    pub cache_bytes: usize,
    /// Target disk segment length. Oversized batches occupy one whole segment.
    pub segment_bytes: u64,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            cache_bytes: 16 * 1024 * 1024,
            segment_bytes: 64 * 1024 * 1024,
        }
    }
}

/// The exclusive handle to a MemoryH directory. The handle releases its file
/// lock when dropped. Calls on the same handle are serialized by the caller.
pub struct Store {
    log: Log,
    index: Index,
    cache: Cache,
    blocks: Option<BlockCache>,
    recovery_required: bool,
}

impl Store {
    pub fn open(path: impl AsRef<Path>, options: Options) -> Result<Self> {
        Self::open_with_block_cache(path, options, 0)
    }

    /// Reserve part of the RAM budget for aligned 4 KiB disk blocks. Records
    /// larger than a block use the exact-read path. A zero block budget
    /// selects record-only caching.
    pub fn open_with_block_cache(
        path: impl AsRef<Path>,
        options: Options,
        block_cache_bytes: usize,
    ) -> Result<Self> {
        let record_cache_bytes = options
            .cache_bytes
            .checked_sub(block_cache_bytes)
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "block budget exceeds total cache budget",
                )
            })?;
        let blocks = BlockCache::new(block_cache_bytes)?;
        let mut index = Index::default();
        let log = Log::open(path.as_ref(), options.segment_bytes, |location, payload| {
            let (revision, entries) = decode_batch(payload)?;
            if revision != index.next_revision()? {
                return Err(Error::Corrupt("nonconsecutive revision".into()));
            }
            index.validate(&entries)?;
            index.install(revision, location, payload, entries);
            Ok(())
        })?;
        Ok(Self {
            log,
            index,
            cache: Cache::new(record_cache_bytes),
            blocks,
            recovery_required: false,
        })
    }

    pub fn revision(&self) -> u64 {
        self.index.revision
    }

    /// Validate a batch, sync it to disk, then advance the visible revision.
    /// If a write or sync fails, drop this handle and reopen before retrying.
    pub fn apply(&mut self, mutations: &[Mutation]) -> Result<u64> {
        self.commit(mutations, None)
    }

    /// Persist supplied embeddings in the same transaction as their records.
    /// Each entry corresponds to one mutation. A store uses one vector dimension.
    pub fn apply_with_embeddings(
        &mut self,
        mutations: &[Mutation],
        embeddings: &[Option<&[f32]>],
    ) -> Result<u64> {
        self.commit(mutations, Some(embeddings))
    }

    fn commit(
        &mut self,
        mutations: &[Mutation],
        embeddings: Option<&[Option<&[f32]>]>,
    ) -> Result<u64> {
        self.ensure_available()?;
        let mut payload = encode_batch(mutations, embeddings)?;
        if mutations.is_empty() {
            return Ok(self.revision());
        }
        self.append_batch(&mut payload, true)
    }

    // A group holds the store lock until its final sync. Its private index changes
    // let later requests validate against the latest accepted head in queue order.
    fn append_batch(&mut self, payload: &mut [u8], sync_now: bool) -> Result<u64> {
        self.ensure_available()?;
        let revision = self.index.next_revision()?;
        payload[..8].copy_from_slice(&revision.to_le_bytes());
        let (_, entries) = decode_batch(payload)?;
        self.index.validate(&entries)?;

        let appended = self.log.append_frame(payload).and_then(|location| {
            if sync_now {
                self.log.sync_pending()?;
            }
            Ok(location)
        });
        let location = match appended {
            Ok(location) => location,
            Err(error) => {
                self.recovery_required = true;
                return Err(error);
            }
        };
        self.index.install(revision, location, payload, entries);
        Ok(revision)
    }

    /// Number of successful data-file syncs since open, excluding segment creation.
    pub fn data_syncs(&self) -> u64 {
        self.log.data_syncs
    }

    /// Share the open directory between callers and group concurrent durable writes.
    pub fn into_shared(self) -> SharedStore {
        SharedStore::new(self)
    }

    /// Search current records by keyword with native BM25.
    pub fn search_keywords(&self, text: &str, limit: usize) -> Result<Vec<String>> {
        self.ensure_available()?;
        Ok(self.index.retrieval.lexical(text, limit))
    }

    /// Rank the whole question followed by optional subquestions. Vectors come
    /// from the selected embedding model. Results contain stable record IDs.
    pub fn search(
        &mut self,
        queries: &[Query<'_>],
        options: &SearchOptions,
    ) -> Result<SearchResult> {
        self.ensure_available()?;
        self.index.retrieval.search(queries, options)
    }

    /// Return the original records for one search method, up to `top_k`.
    /// Hybrid methods reserve each subquestion's first vector hit before RRF fill.
    pub fn retrieve(
        &mut self,
        queries: &[Query<'_>],
        options: &SearchOptions,
        method: SearchMethod,
    ) -> Result<Vec<Record>> {
        let ids = match method {
            SearchMethod::Keywords => {
                let question = queries
                    .first()
                    .ok_or_else(|| Error::InvalidQuery("missing question".into()))?;
                self.search_keywords(question.text, options.top_k)?
            }
            SearchMethod::Flat => self.search(queries, options)?.flat_vector_first,
            SearchMethod::Tree => self.search(queries, options)?.tree_vector_first,
        };
        ids.iter()
            .take(options.top_k)
            .map(|id| {
                self.get_by_id(id)?
                    .ok_or_else(|| Error::Corrupt(format!("missing indexed record {id:?}")))
            })
            .collect()
    }

    pub fn get(&mut self, key: &str) -> Result<Option<Record>> {
        self.get_at(key, self.revision())
    }

    /// Read the version visible at a committed revision. Revision 0 is empty.
    pub fn get_at(&mut self, key: &str, revision: u64) -> Result<Option<Record>> {
        self.ensure_available()?;
        match self.index.head_at(key, revision) {
            Some(id) => {
                let location = self.index.by_id[id];
                load_record(&self.log, &mut self.cache, &mut self.blocks, id, location).map(Some)
            }
            None => Ok(None),
        }
    }

    /// Look up a stable record ID, including superseded or retracted versions.
    pub fn get_by_id(&mut self, id: &str) -> Result<Option<Record>> {
        self.ensure_available()?;
        self.index
            .by_id
            .get(id)
            .copied()
            .map(|location| load_record(&self.log, &mut self.cache, &mut self.blocks, id, location))
            .transpose()
    }

    pub fn history(&self, key: &str) -> Result<&[Version]> {
        self.ensure_available()?;
        Ok(self.index.versions.get(key).map_or(&[], Vec::as_slice))
    }

    /// Enumerate visible keys and their record IDs for a retrieval index.
    pub fn visible_keys(&self, revision: u64) -> Result<impl Iterator<Item = (&str, &str)>> {
        self.ensure_available()?;
        Ok(self
            .index
            .versions
            .iter()
            .filter_map(move |(key, versions)| {
                visible_version(versions, revision).map(|id| (key.as_str(), id))
            }))
    }

    pub fn cache_stats(&self) -> CacheStats {
        self.cache.stats()
    }

    /// On macOS, request uncached I/O for disk-tier reads. The RAM cache remains
    /// controlled by `Options::cache_bytes`.
    pub fn set_uncached_reads(&mut self, enabled: bool) -> Result<()> {
        self.ensure_available()?;
        self.log.set_uncached_reads(enabled)?;
        Ok(())
    }

    fn ensure_available(&self) -> Result<()> {
        if self.recovery_required {
            return Err(Error::RecoveryRequired);
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct RecordLocation {
    location: Location,
    len: usize,
    checksum: u32,
    revision: u64,
}

#[derive(Default)]
struct Index {
    revision: u64,
    versions: BTreeMap<String, Vec<Version>>,
    by_id: HashMap<String, RecordLocation>,
    retrieval: RetrievalIndex,
}

impl Index {
    fn next_revision(&self) -> Result<u64> {
        self.revision.checked_add(1).ok_or(Error::SequenceExhausted)
    }

    fn head_at(&self, key: &str, revision: u64) -> Option<&str> {
        visible_version(self.versions.get(key)?, revision)
    }

    fn validate(&self, entries: &[Entry<'_>]) -> Result<()> {
        let mut heads = HashMap::new();
        let mut new_ids = HashSet::new();
        let mut dimension = self.retrieval.dimension;
        for entry in entries {
            let key = entry.operation.key();
            if key.is_empty() {
                return Err(Error::Conflict("empty memory key".into()));
            }
            let head = *heads
                .entry(key)
                .or_insert_with(|| self.head_at(key, self.revision));
            let (expected, next) = match entry.operation {
                Operation::Write { id, supersedes, .. } => {
                    if id.is_empty() || self.by_id.contains_key(id) || !new_ids.insert(id) {
                        return Err(Error::Conflict(format!(
                            "empty or duplicate record ID {id:?}"
                        )));
                    }
                    (supersedes, Some(id))
                }
                Operation::Retract { expected_id, .. } => (Some(expected_id), None),
            };
            if head != expected {
                return Err(Error::Conflict(format!(
                    "key {key:?} expected {expected:?}, current version {head:?}"
                )));
            }
            if !entry.embedding.is_empty() {
                let size = entry.embedding.len() / 4;
                let mut nonzero = false;
                for value in embedding_values(entry.embedding) {
                    if !value.is_finite() {
                        return Err(Error::Conflict("embedding must be finite".into()));
                    }
                    nonzero |= value != 0.0;
                }
                if !nonzero || dimension.is_some_and(|previous| previous != size) {
                    return Err(Error::Conflict(
                        "embedding must be nonzero and match the store dimension".into(),
                    ));
                }
                dimension = Some(size);
            }
            heads.insert(key, next);
        }
        Ok(())
    }

    fn install(
        &mut self,
        revision: u64,
        location: Location,
        payload: &[u8],
        entries: Vec<Entry<'_>>,
    ) {
        for entry in entries {
            let key = entry.operation.key();
            let id = match entry.operation {
                Operation::Write {
                    id,
                    body,
                    supersedes,
                    ..
                } => {
                    if let Some(old) = supersedes {
                        self.retrieval.remove(old);
                    }
                    self.retrieval.insert(
                        id,
                        key,
                        body,
                        normalize(embedding_values(entry.embedding)),
                    );
                    let span = entry.span;
                    let record = RecordLocation {
                        location: Location {
                            segment: location.segment,
                            offset: location.offset + span.start as u64,
                        },
                        len: span.len(),
                        checksum: crc32fast::hash(&payload[span]),
                        revision,
                    };
                    self.by_id.insert(id.to_owned(), record);
                    Some(id.to_owned())
                }
                Operation::Retract { expected_id, .. } => {
                    self.retrieval.remove(expected_id);
                    None
                }
            };
            self.versions
                .entry(key.to_owned())
                .or_default()
                .push(Version { revision, id });
        }
        self.revision = revision;
    }
}

fn embedding_values(bytes: &[u8]) -> impl Iterator<Item = f32> + '_ {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four bytes")))
}

fn visible_version(versions: &[Version], revision: u64) -> Option<&str> {
    let end = versions.partition_point(|version| version.revision <= revision);
    versions.get(end.checked_sub(1)?)?.id.as_deref()
}

fn load_record(
    log: &Log,
    cache: &mut Cache,
    blocks: &mut Option<BlockCache>,
    id: &str,
    record: RecordLocation,
) -> Result<Record> {
    if let Some(record) = cache.get(id) {
        return Ok(record);
    }
    let (data, disk_read, block_reads) = match blocks {
        Some(blocks) if record.len <= BLOCK_SIZE => {
            let (data, loads) = blocks.read(log, record.location, record.len)?;
            (data, loads > 0, loads)
        }
        _ => (log.read(record.location, record.len)?, true, 0),
    };
    if crc32fast::hash(&data) != record.checksum {
        return Err(Error::Corrupt(format!(
            "record checksum mismatch for {id:?}"
        )));
    }
    let Operation::Write {
        id,
        key,
        source,
        body,
        supersedes,
    } = decode_operation(&data)?
    else {
        return Err(Error::Corrupt("record index points to a retraction".into()));
    };
    let record = Record {
        memory: Memory {
            id: id.into(),
            key: key.into(),
            source: source.into(),
            body: body.into(),
        },
        revision: record.revision,
        supersedes: supersedes.map(str::to_owned),
    };
    cache.loaded(&record, disk_read, block_reads);
    Ok(record)
}
