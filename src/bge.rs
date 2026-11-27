//! Native CPU inference for the Chinese BGE Small v1.5 model.

use std::path::Path;
use std::sync::{Arc, Mutex};

use ort::session::Session;
use ort::value::Tensor;
use tokenizers::{PaddingParams, Tokenizer, TruncationParams};

use crate::retrieval::{normalize, query_variants};
use crate::{
    Error, Mutation, Options, Query, Record, Result, SearchMethod, SearchOptions, SharedStore,
    Store,
};

use crate::inference::{apply_encoded, cpu_session, inference_error};

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

        let session = cpu_session(
            &model_dir.join("model_optimized.onnx"),
            runtime_library.as_ref(),
            threads,
        )?;
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
        apply_encoded(&self.store, mutations, |texts| {
            self.encoder.lock().unwrap().embed(texts)
        })
    }

    /// The historical BGE hybrid baseline.
    pub fn retrieve(&self, question: &str) -> Result<Vec<Record>> {
        self.retrieve_with_options(question, &SearchOptions::default(), SearchMethod::Flat)
    }

    fn embed_questions(&self, variants: &[&str]) -> Result<Vec<Vec<f32>>> {
        let prefixed: Vec<_> = variants
            .iter()
            .map(|s| format!("{QUERY_PREFIX}{s}"))
            .collect();
        self.encoder
            .lock()
            .unwrap()
            .embed(&prefixed.iter().map(String::as_str).collect::<Vec<_>>())
    }

    /// Explicit historical ranking modes for diagnostics and benchmark comparisons.
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
        let vectors = self.embed_questions(&variants)?;
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
