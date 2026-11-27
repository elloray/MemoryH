use crate::{Error, Mutation, Result, SharedStore};
use ort::session::{
    Session,
    builder::{AutoDevicePolicy, GraphOptimizationLevel},
};
use std::{collections::HashMap, fmt::Display, path::Path};

pub(crate) fn cpu_session(
    model_path: &Path,
    runtime_library: &Path,
    threads: usize,
) -> Result<Session> {
    if threads == 0 {
        return Err(Error::Embedding("thread count must be positive".into()));
    }
    ort::init_from(runtime_library.canonicalize()?)
        .map_err(inference_error)?
        .commit();
    Session::builder()
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
        .commit_from_file(model_path)
        .map_err(inference_error)
}

pub(crate) fn inference_error(error: impl Display) -> Error {
    Error::Embedding(error.to_string())
}

pub(crate) fn apply_encoded(
    store: &SharedStore,
    mutations: &[Mutation],
    embed: impl FnOnce(&[&str]) -> Result<Vec<Vec<f32>>>,
) -> Result<u64> {
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
    let embeddings = embed(&texts)?;
    let vectors: Vec<_> = positions
        .into_iter()
        .map(|position| position.map(|index| embeddings[index].as_slice()))
        .collect();
    store.apply_with_embeddings(mutations, &vectors)
}
