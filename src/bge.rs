//! Native CPU inference for the Chinese BGE Small v1.5 model.

use std::collections::HashMap;
use std::fmt::Display;
use std::path::Path;
use std::sync::{Arc, Mutex};

use ort::session::{
    Session,
    builder::{AutoDevicePolicy, GraphOptimizationLevel},
};
use ort::value::Tensor;
use tokenizers::{PaddingParams, Tokenizer, TruncationParams};

use crate::retrieval::{normalize, query_variants};
use crate::{
    Error, Mutation, Options, Query, Record, Result, SearchMethod, SearchOptions, SharedStore,
    Store,
};

pub const MODEL_NAME: &str = "BAAI/bge-small-zh-v1.5";
pub const MODEL_REVISION: &str = "46fbe35fd4374a00fee7de77dfddaeb6dd6a2c59";

const QUERY_PREFIX: &str = "为这个句子生成表示以用于检索相关文章：";
const DIMENSION: usize = 512;
const CONTEXT: usize = 512;
const BATCH_SIZE: usize = 32;

pub struct BgeSmallZh {
    tokenizer: Tokenizer,
    session: Session,
}

impl BgeSmallZh {
    /// Load the Qdrant ONNX export and tokenizer from a local model directory.
    /// libonnxruntime is a standalone dynamic library (API version 24 or later).
    pub fn load(
        model_dir: impl AsRef<Path>,
        runtime_library: impl AsRef<Path>,
        threads: usize,
    ) -> Result<Self> {
        if threads == 0 {
            return Err(Error::Embedding("thread count must be positive".into()));
        }
        let model_dir = model_dir.as_ref();
        let tokenizer_path = model_dir.join("tokenizer.json");
        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|error| inference_error(format!("{}: {error}", tokenizer_path.display())))?;
        let pad_id = tokenizer
            .token_to_id("[PAD]")
            .ok_or_else(|| Error::Embedding("BGE tokenizer is missing [PAD]".into()))?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: CONTEXT,
                ..Default::default()
            }))
            .map_err(inference_error)?;
        tokenizer.with_padding(Some(PaddingParams {
            pad_id,
            ..Default::default()
        }));

        ort::init_from(runtime_library.as_ref().canonicalize()?)
            .map_err(inference_error)?
            .commit();
        // Match FastEmbed's device selection and memory arena on CPU.
        let session = Session::builder()
            .map_err(inference_error)?
            .with_auto_device(AutoDevicePolicy::Default)
            .map_err(inference_error)?
            .with_no_environment_execution_providers()
            .map_err(inference_error)?
            .with_execution_providers([ort::ep::CPU::default().with_arena_allocator(true).build()])
            .map_err(inference_error)?
            .with_optimization_level(GraphOptimizationLevel::All)
            .map_err(inference_error)?
            .with_intra_threads(threads)
            .map_err(inference_error)?
            .with_inter_threads(threads)
            .map_err(inference_error)?
            .commit_from_file(model_dir.join("model_optimized.onnx"))
            .map_err(inference_error)?;
        Ok(Self { tokenizer, session })
    }

    /// Encode raw inputs. Documents use their original text; query callers supply the prefix.
    pub fn embed(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut embeddings = Vec::with_capacity(texts.len());
        for batch in texts.chunks(BATCH_SIZE) {
            let encoded = self
                .tokenizer
                .encode_batch(batch.to_vec(), true)
                .map_err(inference_error)?;
            let sequence = encoded[0].len();
            if sequence == 0 || encoded.iter().any(|item| item.len() != sequence) {
                return Err(Error::Embedding("invalid BGE token batch shape".into()));
            }
            let shape = [encoded.len(), sequence];
            let ids: Vec<_> = encoded
                .iter()
                .flat_map(|item| item.get_ids().iter().copied().map(i64::from))
                .collect();
            let masks: Vec<_> = encoded
                .iter()
                .flat_map(|item| item.get_attention_mask().iter().copied().map(i64::from))
                .collect();
            let segments = vec![0_i64; ids.len()];
            let output = self.session.run(ort::inputs! {
                "input_ids" => Tensor::from_array((shape, ids)).map_err(inference_error)?,
                "attention_mask" => Tensor::from_array((shape, masks)).map_err(inference_error)?,
                "token_type_ids" => Tensor::from_array((shape, segments)).map_err(inference_error)?,
            }).map_err(inference_error)?;
            let (shape, values) = output
                .get("last_hidden_state")
                .ok_or_else(|| Error::Embedding("missing BGE last_hidden_state output".into()))?
                .try_extract_tensor::<f32>()
                .map_err(inference_error)?;
            if **shape != [batch.len() as i64, sequence as i64, DIMENSION as i64] {
                return Err(inference_error(format!(
                    "unexpected BGE output shape {shape:?}"
                )));
            }

            // BGE uses the first token of each sequence, followed by L2 normalization.
            for row in values.chunks_exact(sequence * DIMENSION) {
                let vector = normalize(row[..DIMENSION].iter().copied())
                    .ok_or_else(|| Error::Embedding("BGE returned an invalid embedding".into()))?;
                embeddings.push(vector);
            }
        }
        Ok(embeddings)
    }
}

fn inference_error(error: impl Display) -> Error {
    Error::Embedding(error.to_string())
}

/// A MemoryH store with a resident BGE encoder for text input.
#[derive(Clone)]
pub struct BgeStore {
    store: SharedStore,
    encoder: Arc<Mutex<BgeSmallZh>>,
}

impl BgeStore {
    pub fn open(path: impl AsRef<Path>, options: Options, encoder: BgeSmallZh) -> Result<Self> {
        Self::with_store(Store::open(path, options)?, encoder)
    }

    /// Attach an encoder to an already opened store with custom cache options.
    pub fn with_store(store: Store, encoder: BgeSmallZh) -> Result<Self> {
        if store
            .index
            .retrieval
            .dimension
            .is_some_and(|dimension| dimension != DIMENSION)
        {
            return Err(Error::Embedding(
                "existing vectors do not match the BGE dimension".into(),
            ));
        }
        Ok(Self {
            store: store.into_shared(),
            encoder: Arc::new(Mutex::new(encoder)),
        })
    }

    /// Successful data-file synchronizations since opening this store.
    pub fn data_syncs(&self) -> Result<u64> {
        self.store.data_syncs()
    }

    /// Encode new and corrected text, then commit original records and vectors together.
    pub fn apply(&self, mutations: &[Mutation]) -> Result<u64> {
        let mut by_text = HashMap::new();
        let mut texts = Vec::new();
        let positions: Vec<_> = mutations
            .iter()
            .map(|mutation| {
                let memory = match mutation {
                    Mutation::Put(memory) | Mutation::Replace { memory, .. } => memory,
                    Mutation::Retract { .. } => return None,
                };
                Some(*by_text.entry(memory.body.as_str()).or_insert_with(|| {
                    texts.push(memory.body.as_str());
                    texts.len() - 1
                }))
            })
            .collect();
        let embeddings = self.encoder.lock().unwrap().embed(&texts)?;
        let vectors: Vec<_> = positions
            .into_iter()
            .map(|position| position.map(|index| embeddings[index].as_slice()))
            .collect();
        self.store.apply_with_embeddings(mutations, &vectors)
    }

    /// Search raw question text and return up to three originals via flat hybrid search.
    pub fn retrieve(&self, question: &str) -> Result<Vec<Record>> {
        self.retrieve_with_options(question, &SearchOptions::default(), SearchMethod::Flat)
    }

    pub fn retrieve_with_options(
        &self,
        question: &str,
        options: &SearchOptions,
        method: SearchMethod,
    ) -> Result<Vec<Record>> {
        if method == SearchMethod::Keywords {
            return self.store.retrieve(
                &[Query {
                    text: question,
                    embedding: None,
                }],
                options,
                method,
            );
        }

        let variants = query_variants(question);
        let prefixed: Vec<_> = variants
            .iter()
            .map(|text| format!("{QUERY_PREFIX}{text}"))
            .collect();
        let input: Vec<_> = prefixed.iter().map(String::as_str).collect();
        let vectors = self.encoder.lock().unwrap().embed(&input)?;
        let queries: Vec<_> = variants
            .into_iter()
            .zip(&vectors)
            .map(|(text, vector)| Query {
                text,
                embedding: Some(vector),
            })
            .collect();
        self.store.retrieve(&queries, options, method)
    }
}
