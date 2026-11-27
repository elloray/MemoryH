//! Pointwise Yuanbao decision scoring with its trained chat template.
mod mps;

use std::path::Path;

use ort::{session::Session, value::Tensor};
use tokenizers::{Tokenizer, TruncationParams};

use crate::Result;
use crate::inference::{cpu_session, inference_error};

pub const MEMORY_INSTRUCTION: &str = "判断这条记忆原文能否为问题提供直接证据。若问题有多个独立子问题，能回答其中任意一个就算相关。核对所问对象、条件和具体事实；仅仅话题相近不足以判为相关。";
const PREFIX: &str = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n";
const SUFFIX: &str = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
const CONTEXT: usize = 512;

enum Backend {
    Cpu(Session),
    Mps(mps::Mps),
}

pub(super) struct Yuanbao {
    tokenizer: Tokenizer,
    backend: Backend,
    suffix: Vec<u32>,
    pad: i64,
    instruction: String,
}

impl Yuanbao {
    pub(super) fn load(
        dir: &Path,
        runtime: &Path,
        threads: usize,
        instruction: &str,
    ) -> Result<Self> {
        if instruction.trim().is_empty() {
            return Err(inference_error("empty Yuanbao task instruction"));
        }
        let mut tokenizer =
            Tokenizer::from_file(dir.join("tokenizer.json")).map_err(inference_error)?;
        tokenizer.with_padding(None);
        tokenizer.with_truncation(None).map_err(inference_error)?;
        let suffix = tokenizer
            .encode(SUFFIX, false)
            .map_err(inference_error)?
            .get_ids()
            .to_vec();
        let pad = tokenizer
            .token_to_id("<|endoftext|>")
            .ok_or_else(|| inference_error("missing Yuanbao pad token"))?;
        // Keep the decision suffix even when the document is truncated.
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: CONTEXT - suffix.len(),
                ..Default::default()
            }))
            .map_err(inference_error)?;
        let backend = match std::env::var_os("MEMORYH_YUANBAO_MPS_DIR") {
            Some(path) => Backend::Mps(mps::Mps::open(Path::new(&path))?),
            None => Backend::Cpu(cpu_session(&dir.join("yuanbao.onnx"), runtime, threads)?),
        };
        Ok(Self {
            tokenizer,
            backend,
            suffix,
            pad: i64::from(pad),
            instruction: instruction.to_owned(),
        })
    }

    pub(super) fn score(&mut self, query: &str, documents: &[String]) -> Result<Vec<f32>> {
        let mut scores = Vec::with_capacity(documents.len());
        let prefix = format!(
            "{PREFIX}<Instruct>: {}\n\n<Query>: {query}\n\n<Document>: ",
            self.instruction
        );
        for batch in documents.chunks(8) {
            let bodies: Vec<_> = batch.iter().map(|r| format!("{prefix}{r}")).collect();
            let enc = self
                .tokenizer
                .encode_batch(bodies, false)
                .map_err(inference_error)?;
            let width = enc.iter().map(|e| e.len()).max().unwrap() + self.suffix.len();
            let shape = [batch.len(), width];
            let mut ids = vec![self.pad; batch.len() * width];
            let mut mask = vec![0_i64; ids.len()];
            for (i, body) in enc.iter().enumerate() {
                let start = i * width;
                for (offset, &id) in body.get_ids().iter().chain(&self.suffix).enumerate() {
                    ids[start + offset] = i64::from(id);
                    mask[start + offset] = 1;
                }
            }
            let values = match &mut self.backend {
                Backend::Cpu(session) => {
                    let tensor =
                        |v: Vec<i64>| Tensor::from_array((shape, v)).map_err(inference_error);
                    let out = session
                        .run(ort::inputs!["input_ids" => tensor(ids)?, "attention_mask" => tensor(mask)?])
                        .map_err(inference_error)?;
                    let (dims, logits) = out
                        .get("decision_logits")
                        .ok_or_else(|| inference_error("missing Yuanbao decision logits"))?
                        .try_extract_tensor::<f32>()
                        .map_err(inference_error)?;
                    if **dims != [batch.len() as i64, 2] {
                        return Err(inference_error("invalid Yuanbao output shape"));
                    }
                    logits.to_vec()
                }
                Backend::Mps(worker) => worker.logits(shape, &ids, &mask)?,
            };
            if values.iter().any(|v| !v.is_finite()) {
                return Err(inference_error(
                    "expected finite [no, yes] logits per candidate",
                ));
            }
            // log odds orders identically to the official two-token softmax and avoids probability ties.
            scores.extend(values.chunks_exact(2).map(|pair| pair[1] - pair[0]));
        }
        Ok(scores)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{env, path::PathBuf};

    #[test]
    #[ignore = "requires Yuanbao FP16 export and CPU ONNX Runtime"]
    fn mixed_length_scores_match_transformers_reference() {
        let dir = PathBuf::from(env::var_os("YUANBAO_DIR").expect("YUANBAO_DIR"));
        let runtime = PathBuf::from(env::var_os("ONNX_RUNTIME_LIB").expect("ONNX_RUNTIME_LIB"));
        let mut model =
            Yuanbao::load(&dir, &runtime, 2, "为问题寻找能直接回答的记忆原文。").unwrap();
        let records =
            ["海边讲座在二层。", "观众可以从南门入场，现场供应饮用水。"].map(String::from);
        // Independent official Transformers forward with right padding and the same task instruction.
        let reference = [11.546875_f32, -4.390625];
        let scores = model.score("海边讲座在哪一层？", &records).unwrap();
        assert_eq!(scores.len(), reference.len());
        for (score, expected) in scores.iter().zip(reference) {
            assert!((score - expected).abs() < 0.15, "{score} vs {expected}");
        }
    }
}
