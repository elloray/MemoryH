//! CPU cross-encoder inference for a multilingual sentence-pair classifier.
use std::path::Path;

use ort::{session::Session, value::Tensor};
use tokenizers::{PaddingParams, Tokenizer, TruncationParams};

use crate::Result;
use crate::inference::{cpu_session, inference_error};

pub struct CrossEncoder {
    tokenizer: Tokenizer,
    session: Session,
    type_ids: bool,
}

impl CrossEncoder {
    /// The model directory holds model.onnx and its matching tokenizer.json.
    pub fn load(dir: impl AsRef<Path>, runtime: impl AsRef<Path>, threads: usize) -> Result<Self> {
        let dir = dir.as_ref();
        let mut tokenizer =
            Tokenizer::from_file(dir.join("tokenizer.json")).map_err(inference_error)?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: 512,
                ..Default::default()
            }))
            .map_err(inference_error)?;
        let pad = ["<pad>", "[PAD]"]
            .into_iter()
            .find_map(|token| tokenizer.token_to_id(token).map(|id| (token, id)))
            .ok_or_else(|| inference_error("cross-encoder tokenizer has no pad token"))?;
        tokenizer.with_padding(Some(PaddingParams {
            pad_id: pad.1,
            pad_token: pad.0.into(),
            ..Default::default()
        }));
        let session = cpu_session(&dir.join("model.onnx"), runtime.as_ref(), threads)?;
        let type_ids = session
            .inputs()
            .iter()
            .any(|i| i.name() == "token_type_ids");
        Ok(Self {
            session,
            tokenizer,
            type_ids,
        })
    }

    pub(super) fn score(&mut self, query: &str, documents: &[String]) -> Result<Vec<f32>> {
        let mut scores = Vec::with_capacity(documents.len());
        for batch in documents.chunks(16) {
            let pairs: Vec<_> = batch.iter().map(|r| (query, r.as_str())).collect();
            let enc = self
                .tokenizer
                .encode_batch(pairs, true)
                .map_err(inference_error)?;
            let shape = [enc.len(), enc[0].len()];
            let tensor =
                |values: Vec<i64>| Tensor::from_array((shape, values)).map_err(inference_error);
            let ids = tensor(
                enc.iter()
                    .flat_map(|v| v.get_ids().iter().copied().map(i64::from))
                    .collect(),
            )?;
            let mask = tensor(
                enc.iter()
                    .flat_map(|v| v.get_attention_mask().iter().copied().map(i64::from))
                    .collect(),
            )?;
            let mut inputs = ort::inputs!["input_ids" => ids, "attention_mask" => mask];
            if self.type_ids {
                let types = tensor(
                    enc.iter()
                        .flat_map(|v| v.get_type_ids().iter().copied().map(i64::from))
                        .collect(),
                )?;
                inputs.push(("token_type_ids".into(), types.into()));
            }
            let output = self.session.run(inputs).map_err(inference_error)?;
            let (dims, values) = output
                .get("logits")
                .ok_or_else(|| inference_error("cross-encoder has no logits output"))?
                .try_extract_tensor::<f32>()
                .map_err(inference_error)?;
            if **dims != [batch.len() as i64, 1] || values.iter().any(|v| !v.is_finite()) {
                return Err(inference_error(
                    "expected one finite relevance logit per query/document pair",
                ));
            }
            scores.extend_from_slice(values);
        }
        Ok(scores)
    }
}
