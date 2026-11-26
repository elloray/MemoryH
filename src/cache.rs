use std::io;
use std::mem::size_of;
use std::num::NonZeroUsize;

use lru::LruCache;

use crate::log::{Location, Log};
use crate::{Record, Result};

pub(crate) const BLOCK_SIZE: usize = 4096;

/// Counters cover this open handle. The byte charge includes record strings and
/// fixed entry fields, but excludes allocator overhead and the lookup index.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CacheStats {
    /// Reads served by the decoded-record cache.
    pub hits: u64,
    /// Misses in the decoded-record cache, including subsequent block hits.
    pub misses: u64,
    /// Successful record reads that needed at least one disk access.
    pub disk_reads: u64,
    /// Record reads served entirely by cached blocks after a decoded-record miss.
    pub block_hits: u64,
    /// Disk block loads. A straddling record can require two loads.
    pub block_reads: u64,
    pub evictions: u64,
    pub entries: usize,
    pub bytes: usize,
}

pub(crate) struct Cache {
    contents: LruCache<String, (Record, usize)>,
    budget: usize,
    stats: CacheStats,
}

impl Cache {
    pub(crate) fn new(budget: usize) -> Self {
        Self {
            contents: LruCache::unbounded(),
            budget,
            stats: CacheStats::default(),
        }
    }

    pub(crate) fn get(&mut self, id: &str) -> Option<Record> {
        match self.contents.get(id) {
            Some((record, _)) => {
                self.stats.hits += 1;
                Some(record.clone())
            }
            None => {
                self.stats.misses += 1;
                None
            }
        }
    }

    pub(crate) fn loaded(&mut self, record: &Record, disk_reads: bool, block_reads: u64) {
        self.stats.disk_reads += u64::from(disk_reads);
        self.stats.block_hits += u64::from(!disk_reads);
        self.stats.block_reads += block_reads;
        let memory = &record.memory;
        let charge = size_of::<(String, Record, usize)>()
            + 2 * memory.id.len()
            + memory.key.len()
            + memory.source.len()
            + memory.body.len()
            + record.supersedes.as_ref().map_or(0, String::len);
        if charge > self.budget {
            return;
        }
        while charge > self.budget - self.stats.bytes {
            let (_, (_, size)) = self.contents.pop_lru().expect("positive cache byte charge");
            self.stats.bytes -= size;
            self.stats.evictions += 1;
        }
        self.contents
            .put(memory.id.clone(), (record.clone(), charge));
        self.stats.bytes += charge;
    }

    pub(crate) fn stats(&self) -> CacheStats {
        CacheStats {
            entries: self.contents.len(),
            ..self.stats
        }
    }
}

pub(crate) struct BlockCache {
    blocks: LruCache<Location, Vec<u8>>,
}

impl BlockCache {
    pub(crate) fn new(budget: usize) -> io::Result<Option<Self>> {
        if budget == 0 {
            return Ok(None);
        }
        let charge = BLOCK_SIZE + size_of::<(Location, Vec<u8>)>();
        let capacity = NonZeroUsize::new(budget / charge).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "block budget cannot hold a 4 KiB block",
            )
        })?;
        Ok(Some(Self {
            blocks: LruCache::new(capacity),
        }))
    }

    pub(crate) fn read(
        &mut self,
        log: &Log,
        location: Location,
        len: usize,
    ) -> Result<(Vec<u8>, u64)> {
        let mut data = Vec::with_capacity(len);
        let mut disk_reads = 0;
        while data.len() < len {
            let offset = location.offset + data.len() as u64;
            let start = (offset % BLOCK_SIZE as u64) as usize;
            let count = (len - data.len()).min(BLOCK_SIZE - start);
            let end = start + count;
            let block_at = Location {
                segment: location.segment,
                offset: offset - start as u64,
            };
            let cache_has_range = self
                .blocks
                .peek(&block_at)
                .is_some_and(|block| block.len() >= end);
            if !cache_has_range {
                // Appending can extend a cached short tail. Reload it when a
                // new record needs bytes beyond the cached prefix.
                let block = log.read_block(block_at, BLOCK_SIZE)?;
                if block.len() < end {
                    return Err(
                        io::Error::new(io::ErrorKind::UnexpectedEof, "short record block").into(),
                    );
                }
                self.blocks.put(block_at, block);
                disk_reads += 1;
            }
            let block = self.blocks.get(&block_at).expect("block was loaded");
            data.extend_from_slice(&block[start..end]);
        }
        Ok((data, disk_reads))
    }
}
