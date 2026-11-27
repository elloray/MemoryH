//! Local Qwen semantic recall, independent keyword recall and cross-encoder reranking.
mod encoder;
mod evidence;
pub use evidence::{Date, QueryContext, session_date};
pub use splitter::QueryIntent;
use std::collections::{BTreeMap, BTreeSet};

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
    fn score(&mut self, question: &str, documents: &[String]) -> Result<Vec<f32>> {
        match self {
            Self::MiniLm(model) => model.score(question, documents),
            Self::Yuanbao(model) => model.score(question, documents),
        }
    }
}

pub use crate::retrieval::scann::RecallOptions;
const TOP_K: usize = 3;
const MODEL_MARKER: &str = "QWEN3_EMBEDDING_06B_INT8_V1";

/// Resident CPU models can be shared by multiple store handles.
#[derive(Clone)]
pub struct QwenModels {
    encoder: Arc<Mutex<Qwen3Embedding>>,
    reranker: Arc<Mutex<Reranker>>,
    splitter: splitter::Splitter,
    scann_library: PathBuf,
    recall: RecallOptions,
}

impl QwenModels {
    pub fn with_recall_options(mut self, options: RecallOptions) -> Self {
        self.recall = options;
        self
    }

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
            recall: RecallOptions::default(),
        })
    }
}

#[derive(Default)]
pub struct SearchProfile {
    pub records: Vec<Record>,
    pub variants: Vec<String>,
    pub candidate_count: usize,
    pub intent: QueryIntent,
    /// Evidence identities extracted for inventory selection, keyed by candidate ID.
    pub inventory_facts: BTreeMap<String, BTreeSet<String>>,
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
        let scann = ScannLibrary::load(&models.scann_library, models.recall)?;
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

    /// Inventory queries return up to five originals. Other intents default to three.
    pub fn retrieve(&self, question: &str) -> Result<Vec<Record>> {
        self.retrieve_with_context(question, &QueryContext::default())
    }

    /// Return up to k reranked originals from the same deduplicated candidate pool.
    pub fn retrieve_top_k(&self, question: &str, k: usize) -> Result<Vec<Record>> {
        Ok(self.retrieve_profiled(question, k)?.records)
    }

    pub fn retrieve_profiled(&self, question: &str, k: usize) -> Result<SearchProfile> {
        self.retrieve_profiled_with_context(question, Some(k), &QueryContext::default())
    }

    pub fn retrieve_with_context(
        &self,
        question: &str,
        context: &QueryContext,
    ) -> Result<Vec<Record>> {
        Ok(self
            .retrieve_profiled_with_context(question, None, context)?
            .records)
    }

    pub fn retrieve_profiled_with_context(
        &self,
        question: &str,
        k: Option<usize>,
        context: &QueryContext,
    ) -> Result<SearchProfile> {
        if k == Some(0) || question.trim().is_empty() {
            return Ok(SearchProfile::default());
        }
        let start = Instant::now();
        let plan = self.models.splitter.plan(question)?;
        let mut variants = plan.variants;
        if plan.intent == QueryIntent::Recommendation {
            let angle = if question
                .chars()
                .any(|c| ('\u{3400}'..='\u{9fff}').contains(&c))
            {
                "用户已经拥有的设备、习惯或偏好，哪些能让这个建议更有针对性？"
            } else {
                "Which devices, habits or preferences the user already has would make this advice more useful?"
            };
            variants.insert(1, format!("{question}\n{angle}"));
        }
        let k = k.unwrap_or(if plan.intent == QueryIntent::Inventory {
            5
        } else {
            TOP_K
        });
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
        let (records, vector) = self.store.recall(&queries, &self.scann)?;
        let recall = start.elapsed();
        let candidate_count = records.len();
        let start = Instant::now();
        let documents: Vec<_> = records
            .iter()
            .map(|record| context.document(record))
            .collect();
        let mut order = {
            let mut ranker = self.models.reranker.lock().unwrap();
            let direct = ranker.score(&context.query(question), &documents)?;
            let aspects = variants[1..]
                .iter()
                .map(|aspect| ranker.score(&context.query(aspect), &documents))
                .collect::<Result<Vec<_>>>()?;
            evidence::evidence_order(&direct, &aspects)
        };
        let mut inventory_facts = BTreeMap::new();
        if plan.intent == QueryIntent::Inventory {
            let asked = context.query(question);
            let mut known = Vec::new();
            let mut facts = vec![BTreeSet::new(); records.len()];
            for &i in &order {
                let record = &records[i].memory;
                let header = documents[i].strip_suffix(&record.body).unwrap_or_default();
                facts[i] = self.models.splitter.facts(
                    &asked,
                    &record.body,
                    header,
                    context.date_of(&records[i]),
                    &mut known,
                )?;
                inventory_facts.insert(record.id.clone(), facts[i].clone());
            }
            order = evidence::cover_facts(order, &facts, 5);
        }
        // Move the selected originals without cloning potentially long session bodies.
        let mut records: Vec<_> = records.into_iter().map(Some).collect();
        let records = order
            .into_iter()
            .take(k)
            .map(|i| records[i].take().unwrap())
            .collect();
        Ok(SearchProfile {
            records,
            variants,
            candidate_count,
            intent: plan.intent,
            inventory_facts,
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
