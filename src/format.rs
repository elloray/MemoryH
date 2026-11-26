use std::ops::Range;

use crate::{Error, Mutation, Result};

pub(crate) const SEGMENT_HEADER: &[u8; 8] = b"MEMH\x01\0\0\0";
pub(crate) const FRAME_HEADER_SIZE: usize = 12;
pub(crate) const MAX_BATCH_BYTES: usize = 64 * 1024 * 1024;

pub(crate) fn frame_header(payload: &[u8]) -> [u8; FRAME_HEADER_SIZE] {
    let len = payload.len() as u32;
    let mut header = [0; FRAME_HEADER_SIZE];
    header[..4].copy_from_slice(&len.to_le_bytes());
    header[4..8].copy_from_slice(&(!len).to_le_bytes());
    header[8..].copy_from_slice(&crc32fast::hash(payload).to_le_bytes());
    header
}

pub(crate) fn frame_info(header: [u8; FRAME_HEADER_SIZE]) -> Result<(usize, u32)> {
    let len = u32::from_le_bytes(header[..4].try_into().unwrap());
    let guard = u32::from_le_bytes(header[4..8].try_into().unwrap());
    if guard != !len || !(12..=MAX_BATCH_BYTES).contains(&(len as usize)) {
        return Err(Error::Corrupt("invalid batch length".into()));
    }
    let checksum = u32::from_le_bytes(header[8..].try_into().unwrap());
    Ok((len as usize, checksum))
}

pub(crate) enum Operation<'a> {
    Write {
        id: &'a str,
        key: &'a str,
        source: &'a str,
        body: &'a str,
        supersedes: Option<&'a str>,
    },
    Retract {
        key: &'a str,
        expected_id: &'a str,
    },
}

impl<'a> Operation<'a> {
    pub(crate) fn key(&self) -> &'a str {
        match self {
            Self::Write { key, .. } | Self::Retract { key, .. } => key,
        }
    }
}

pub(crate) struct Entry<'a> {
    pub(crate) operation: Operation<'a>,
    pub(crate) span: Range<usize>,
    pub(crate) embedding: &'a [u8],
}

pub(crate) fn encode_batch(
    mutations: &[Mutation],
    embeddings: Option<&[Option<&[f32]>]>,
) -> Result<Vec<u8>> {
    if let Some(embeddings) = embeddings {
        if embeddings.len() != mutations.len() {
            return Err(Error::Conflict(
                "one embedding slot is required per mutation".into(),
            ));
        }
        if embeddings.iter().flatten().any(|vector| vector.is_empty()) {
            return Err(Error::Conflict("embedding must be nonempty".into()));
        }
    }
    let count = u32::try_from(mutations.len()).map_err(|_| Error::BatchTooLarge)?;
    // The writer assigns a revision after the request reaches the front of the queue.
    let mut payload = vec![0; 8];
    payload.extend_from_slice(&count.to_le_bytes());
    for (index, mutation) in mutations.iter().enumerate() {
        let embedding = embeddings.and_then(|values| values[index]);
        match mutation {
            Mutation::Put(memory) | Mutation::Replace { memory, .. } => {
                let supersedes = match mutation {
                    Mutation::Replace { expected_id, .. } => Some(expected_id.as_str()),
                    _ => None,
                };
                payload.push(match (supersedes.is_some(), embedding.is_some()) {
                    (false, false) => 1,
                    (true, false) => 2,
                    (false, true) => 4,
                    (true, true) => 5,
                });
                for value in [&memory.id, &memory.key, &memory.source, &memory.body] {
                    encode_str(&mut payload, value)?;
                }
                if let Some(id) = supersedes {
                    encode_str(&mut payload, id)?;
                }
                if let Some(vector) = embedding {
                    let remaining = MAX_BATCH_BYTES.saturating_sub(payload.len());
                    if remaining < 4 || vector.len() > (remaining - 4) / 4 {
                        return Err(Error::BatchTooLarge);
                    }
                    payload.extend_from_slice(&(vector.len() as u32).to_le_bytes());
                    for value in vector {
                        payload.extend_from_slice(&value.to_le_bytes());
                    }
                }
            }
            Mutation::Retract { key, expected_id } => {
                if embedding.is_some() {
                    return Err(Error::Conflict(
                        "a retraction cannot have an embedding".into(),
                    ));
                }
                payload.push(3);
                encode_str(&mut payload, key)?;
                encode_str(&mut payload, expected_id)?;
            }
        }
    }
    Ok(payload)
}

fn encode_str(payload: &mut Vec<u8>, text: &str) -> Result<()> {
    let remaining = MAX_BATCH_BYTES.saturating_sub(payload.len());
    if remaining < 4 || text.len() > remaining - 4 {
        return Err(Error::BatchTooLarge);
    }
    payload.extend_from_slice(&(text.len() as u32).to_le_bytes());
    payload.extend_from_slice(text.as_bytes());
    Ok(())
}

pub(crate) fn decode_batch(payload: &[u8]) -> Result<(u64, Vec<Entry<'_>>)> {
    let mut decoder = Decoder { payload, offset: 0 };
    let revision = u64::from_le_bytes(decoder.take(8)?.try_into().unwrap());
    let count = decoder.u32()? as usize;
    // A retraction is the smallest possible entry: one tag and two string lengths.
    if count == 0 || count > (payload.len() - decoder.offset) / 9 {
        return Err(Error::Corrupt("invalid mutation count".into()));
    }
    let mut entries = Vec::new();
    for _ in 0..count {
        let start = decoder.offset;
        let operation = decoder.operation()?;
        let span = start..decoder.offset;
        let embedding = if matches!(payload[start], 4 | 5) {
            let count = decoder.u32()? as usize;
            let len = count
                .checked_mul(4)
                .filter(|len| *len > 0)
                .ok_or_else(|| Error::Corrupt("invalid embedding length".into()))?;
            decoder.take(len)?
        } else {
            &[]
        };
        entries.push(Entry {
            operation,
            span,
            embedding,
        });
    }
    decoder.end()?;
    Ok((revision, entries))
}

pub(crate) fn decode_operation(payload: &[u8]) -> Result<Operation<'_>> {
    let mut decoder = Decoder { payload, offset: 0 };
    let operation = decoder.operation()?;
    decoder.end()?;
    Ok(operation)
}

struct Decoder<'a> {
    payload: &'a [u8],
    offset: usize,
}

impl<'a> Decoder<'a> {
    fn take(&mut self, len: usize) -> Result<&'a [u8]> {
        if len > self.payload.len() - self.offset {
            return Err(Error::Corrupt("truncated mutation".into()));
        }
        let start = self.offset;
        self.offset += len;
        Ok(&self.payload[start..self.offset])
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn string(&mut self) -> Result<&'a str> {
        let len = self.u32()? as usize;
        std::str::from_utf8(self.take(len)?)
            .map_err(|_| Error::Corrupt("invalid UTF-8 in mutation".into()))
    }

    fn operation(&mut self) -> Result<Operation<'a>> {
        let tag = self.take(1)?[0];
        match tag {
            1 | 2 | 4 | 5 => Ok(Operation::Write {
                id: self.string()?,
                key: self.string()?,
                source: self.string()?,
                body: self.string()?,
                supersedes: if matches!(tag, 2 | 5) {
                    Some(self.string()?)
                } else {
                    None
                },
            }),
            3 => Ok(Operation::Retract {
                key: self.string()?,
                expected_id: self.string()?,
            }),
            _ => Err(Error::Corrupt("unknown mutation type".into())),
        }
    }

    fn end(&self) -> Result<()> {
        if self.offset != self.payload.len() {
            return Err(Error::Corrupt("extra bytes after mutation data".into()));
        }
        Ok(())
    }
}
