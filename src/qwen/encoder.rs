use std::path::Path;

use ort::{
    memory::Allocator,
    session::{OutputSelector, RunOptions, Session},
    value::Tensor,
};
use tokenizers::{PaddingDirection, PaddingParams, Tokenizer, TruncationParams};

use crate::Result;
use crate::inference::{cpu_session, inference_error};
use crate::retrieval::normalize;

pub(super) const DIMENSION: usize = 1024;
const CONTEXT: usize = 2048;
const CACHE_HEADS: usize = 8;
const HEAD_DIM: usize = 128;

/// Qwen3-Embedding-0.6B with the pinned CPU INT8 ONNX feature-extraction graph.
pub struct Qwen3Embedding {
    tokenizer: Tokenizer,
    session: Session,
    cache_inputs: Vec<String>,
}

impl Qwen3Embedding {
    pub fn load(dir: &Path, runtime: &Path, threads: usize) -> Result<Self> {
        let mut tokenizer =
            Tokenizer::from_file(dir.join("tokenizer.json")).map_err(inference_error)?;
        let pad = "<|endoftext|>";
        let pad_id = tokenizer
            .token_to_id(pad)
            .ok_or_else(|| inference_error("missing Qwen pad token"))?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: CONTEXT,
                ..Default::default()
            }))
            .map_err(inference_error)?;
        tokenizer.with_padding(Some(PaddingParams {
            direction: PaddingDirection::Left,
            pad_id,
            pad_token: pad.into(),
            ..Default::default()
        }));
        let session = cpu_session(&dir.join("model_int8.onnx"), runtime, threads)?;
        let cache_inputs = session
            .inputs()
            .iter()
            .map(|input| input.name())
            .filter(|name| name.starts_with("past_key_values."))
            .map(str::to_owned)
            .collect();
        Ok(Self {
            tokenizer,
            session,
            cache_inputs,
        })
    }

    /// Encode original text with the model tokenizer, last-token pooling and L2 normalization.
    pub fn embed(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut embeddings = Vec::with_capacity(texts.len());
        for batch in texts.chunks(4) {
            let enc = self
                .tokenizer
                .encode_batch(batch.to_vec(), true)
                .map_err(inference_error)?;
            let seq = enc[0].len();
            if seq == 0
                || enc
                    .iter()
                    .any(|e| e.len() != seq || e.get_attention_mask().last() != Some(&1))
            {
                return Err(inference_error(
                    "expected left-padded Qwen input with a final valid token",
                ));
            }
            let shape = [batch.len(), seq];
            let ids: Vec<i64> = enc
                .iter()
                .flat_map(|e| e.get_ids().iter().copied().map(i64::from))
                .collect();
            let masks: Vec<i64> = enc
                .iter()
                .flat_map(|e| e.get_attention_mask().iter().copied().map(i64::from))
                .collect();
            let mut positions = Vec::with_capacity(ids.len());
            for row in masks.chunks_exact(seq) {
                let mut index = 0;
                for &mask in row {
                    positions.push(if mask == 0 { 0 } else { index });
                    index += mask;
                }
            }
            let tensor = |v: Vec<i64>| Tensor::from_array((shape, v)).map_err(inference_error);
            let mut inputs = ort::inputs!["input_ids" => tensor(ids)?, "attention_mask" => tensor(masks)?, "position_ids" => tensor(positions)?];
            // Every text is an independent sequence. The decoder export requires empty KV inputs.
            let empty = Tensor::<f32>::new(
                &Allocator::default(),
                [batch.len(), CACHE_HEADS, 0, HEAD_DIM],
            )
            .map_err(inference_error)?;
            for name in &self.cache_inputs {
                inputs.push((name.as_str().into(), (&empty).into()));
            }
            let options = RunOptions::new()
                .map_err(inference_error)?
                .with_outputs(OutputSelector::no_default().with("last_hidden_state"));
            let output = self
                .session
                .run_with_options(inputs, &options)
                .map_err(inference_error)?;
            let (dims, values) = output["last_hidden_state"]
                .try_extract_tensor::<f32>()
                .map_err(inference_error)?;
            if **dims != [batch.len() as i64, seq as i64, DIMENSION as i64] {
                return Err(inference_error(format!(
                    "unexpected Qwen output shape {dims:?}"
                )));
            }
            for row in values.chunks_exact(seq * DIMENSION) {
                let last = &row[(seq - 1) * DIMENSION..];
                embeddings.push(
                    normalize(last.iter().copied())
                        .ok_or_else(|| inference_error("invalid Qwen embedding"))?,
                );
            }
        }
        Ok(embeddings)
    }
}
