//! Local Qwen semantic recall, independent keyword recall and cross-encoder reranking.
mod encoder;
mod rerank;
mod splitter;
use crate::retrieval::scann::ScannLibrary;
use std::time::{Duration, Instant};
mod yuanbao;
pub use yuanbao::MEMORY_INSTRUCTION;

use std::fs::{File, OpenOptions};
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::inference::{apply_encoded, inference_error};
use crate::{Mutation, Options, Query, Record, Result, SharedStore, Store};
pub use encoder::Qwen3Embedding;
use rerank::CrossEncoder;

enum Reranker {
    MiniLm(CrossEncoder),
    Yuanbao(yuanbao::Yuanbao),
}

impl Reranker {
    fn score(&mut self, question: &str, records: &[Record]) -> Result<Vec<f32>> {
        match self {
            Self::MiniLm(model) => model.score(question, records),
            Self::Yuanbao(model) => model.score(question, records),
        }
    }
}

const CHANNEL_LIMIT: usize = 9;
const TOP_K: usize = 3;
const MODEL_MARKER: &str = "QWEN3_EMBEDDING_06B_INT8_V1";

/// Resident CPU models can be shared by multiple store handles.
#[derive(Clone)]
pub struct QwenModels {
    encoder: Arc<Mutex<Qwen3Embedding>>,
    reranker: Arc<Mutex<Reranker>>,
    splitter: splitter::Splitter,
    scann_library: PathBuf,
}

impl QwenModels {
    /// Override the local Qwen3.5-2B endpoint for this set of models.
    pub fn with_splitter(mut self, address: std::net::SocketAddr) -> Result<Self> {
        self.splitter = splitter::Splitter::new(address)?;
        Ok(self)
    }

    /// Override the native ScaNN library location before opening a store.
    pub fn with_scann(mut self, library: impl AsRef<Path>) -> Result<Self> {
        self.scann_library = library.as_ref().canonicalize()?;
        Ok(self)
    }

    /// Configure Qwen3.5-2B and ScaNN alongside the resident ONNX models.
    pub fn load(
        encoder: impl AsRef<Path>,
        reranker: impl AsRef<Path>,
        runtime: impl AsRef<Path>,
        threads: usize,
    ) -> Result<Self> {
        Self::load_with_instruction(encoder, reranker, runtime, threads, None)
    }

    /// Use the memory prompt by default. A custom instruction applies only to Yuanbao.
    pub fn load_with_instruction(
        encoder: impl AsRef<Path>,
        reranker: impl AsRef<Path>,
        runtime: impl AsRef<Path>,
        threads: usize,
        instruction: Option<&str>,
    ) -> Result<Self> {
        let splitter = splitter::Splitter::from_env()?;
        let runtime = runtime.as_ref();
        let dir = reranker.as_ref();
        let ranker = if dir.join("yuanbao.onnx").try_exists()? {
            Reranker::Yuanbao(yuanbao::Yuanbao::load(
                dir,
                runtime,
                threads,
                instruction.unwrap_or(MEMORY_INSTRUCTION),
            )?)
        } else if instruction.is_some() {
            return Err(inference_error(
                "custom instruction requires a Yuanbao reranker",
            ));
        } else {
            Reranker::MiniLm(CrossEncoder::load(dir, runtime, threads)?)
        };
        Ok(Self {
            encoder: Arc::new(Mutex::new(Qwen3Embedding::load(
                encoder.as_ref(),
                runtime,
                threads,
            )?)),
            reranker: Arc::new(Mutex::new(ranker)),
            splitter,
            scann_library: ScannLibrary::default_path(),
        })
    }
}

#[derive(Default)]
pub struct SearchProfile {
    pub records: Vec<Record>,
    pub variants: Vec<String>,
    pub candidate_count: usize,
    pub split: Duration,
    pub encode: Duration,
    pub recall: Duration,
    /// Index acquisition plus the longest concurrent per-variant vector search, included in recall.
    pub vector: Duration,
    pub rerank: Duration,
}

#[derive(Clone)]
pub struct QwenStore {
    store: SharedStore,
    scann: Arc<ScannLibrary>,
    models: QwenModels,
}

impl QwenStore {
    pub fn open(path: impl AsRef<Path>, options: Options, models: QwenModels) -> Result<Self> {
        let path = path.as_ref();
        let scann = ScannLibrary::load(&models.scann_library)?;
        let store = Store::open(path, options)?;
        bind_model(path, &store)?;
        Ok(Self {
            store: store.into_shared(),
            scann,
            models,
        })
    }

    pub fn apply(&self, mutations: &[Mutation]) -> Result<u64> {
        apply_encoded(&self.store, mutations, |texts| {
            self.models.encoder.lock().unwrap().embed(texts)
        })
    }

    /// Rank every unique active candidate once against the original question.
    pub fn retrieve(&self, question: &str) -> Result<Vec<Record>> {
        self.retrieve_top_k(question, TOP_K)
    }

    /// Return up to k reranked originals from the same deduplicated candidate pool.
    pub fn retrieve_top_k(&self, question: &str, k: usize) -> Result<Vec<Record>> {
        Ok(self.retrieve_profiled(question, k)?.records)
    }

    pub fn retrieve_profiled(&self, question: &str, k: usize) -> Result<SearchProfile> {
        if k == 0 || question.trim().is_empty() {
            return Ok(SearchProfile::default());
        }
        let start = Instant::now();
        let variants = self.models.splitter.variants(question)?;
        let split = start.elapsed();
        let start = Instant::now();
        let texts: Vec<_> = variants.iter().map(String::as_str).collect();
        let vectors = self.models.encoder.lock().unwrap().embed(&texts)?;
        let encode = start.elapsed();
        let start = Instant::now();
        let queries: Vec<_> = texts
            .into_iter()
            .zip(&vectors)
            .map(|(text, vector)| Query {
                text,
                embedding: Some(vector),
            })
            .collect();
        let (records, vector) = self.store.recall(&queries, CHANNEL_LIMIT, &self.scann)?;
        let recall = start.elapsed();
        let candidate_count = records.len();
        let start = Instant::now();
        let scores = self
            .models
            .reranker
            .lock()
            .unwrap()
            .score(question, &records)?;
        let mut hits: Vec<_> = records.into_iter().zip(scores).collect();
        hits.sort_unstable_by(|(a, sa), (b, sb)| {
            sb.total_cmp(sa).then_with(|| a.memory.id.cmp(&b.memory.id))
        });
        let records = hits.into_iter().take(k).map(|(r, _)| r).collect();
        Ok(SearchProfile {
            records,
            variants,
            candidate_count,
            split,
            encode,
            recall,
            vector,
            rerank: start.elapsed(),
        })
    }
}

fn bind_model(path: &Path, store: &Store) -> Result<()> {
    let marker = path.join(MODEL_MARKER);
    if store
        .index
        .retrieval
        .dimension
        .is_some_and(|d| d != encoder::DIMENSION)
        || (store.revision() > 0 && !marker.is_file())
    {
        return Err(inference_error(
            "existing store has a different or unknown embedding model; re-encode its originals in a new Qwen store",
        ));
    }
    match OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&marker)
    {
        Ok(file) => {
            file.sync_all()?;
            File::open(path)?.sync_all()?;
        }
        Err(err) if err.kind() == ErrorKind::AlreadyExists && marker.is_file() => {}
        Err(err) => return Err(err.into()),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Memory;
    #[test]
    fn rejects_unmarked_vectors_even_with_a_matching_dimension() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = Store::open(dir.path(), Options::default()).unwrap();
        let memory = Memory {
            id: "1".into(),
            key: "k".into(),
            source: "s".into(),
            body: "b".into(),
        };
        store
            .apply_with_embeddings(
                &[Mutation::Put(memory)],
                &[Some(&[1.0; encoder::DIMENSION])],
            )
            .unwrap();
        assert!(bind_model(dir.path(), &store).is_err());
        assert!(!dir.path().join(MODEL_MARKER).exists());
    }
}
