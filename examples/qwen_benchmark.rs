mod common;
#[path = "common/text_benchmark.rs"]
mod text_benchmark;
use memoryh::Options;
use memoryh::qwen::{QwenModels, QwenStore};
use std::{env, path::PathBuf, time::Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args_os().skip(1);
    let usage = "usage: qwen_benchmark MODEL_DIR RERANKER_DIR ONNXRUNTIME_LIBRARY ROUNDS [INSTRUCTION_FILE]";
    let model_dir = PathBuf::from(args.next().ok_or(usage)?);
    let reranker_dir = PathBuf::from(args.next().ok_or(usage)?);
    let runtime = PathBuf::from(args.next().ok_or(usage)?);
    let rounds: usize = args.next().ok_or(usage)?.to_string_lossy().parse()?;
    let instruction = args.next().map(std::fs::read_to_string).transpose()?;
    if rounds == 0 || args.next().is_some() {
        return Err(usage.into());
    }

    let directory = tempfile::tempdir()?;
    let start = Instant::now();
    let models = QwenModels::load_with_instruction(
        model_dir,
        reranker_dir,
        runtime,
        2,
        instruction.as_deref().map(str::trim),
    )?;
    let load_ns = start.elapsed().as_nanos() as u64;
    let store = QwenStore::open(directory.path(), Options::default(), models.clone())?;
    text_benchmark::run(
        rounds,
        load_ns,
        |m| store.apply(m),
        |q| store.retrieve(q),
        |q| store.retrieve_top_k(q, usize::MAX),
    )?;
    let fixture: common::Fixture<text_benchmark::Question> =
        serde_json::from_str(include_str!("../cases.json"))?;
    let mut checks = Vec::new();
    for question in fixture.questions.iter().filter(|q| q.stage == "after") {
        let hits = store.retrieve(&question.text)?;
        checks.push((&question.text, hits));
    }
    drop(store);
    let reopened = QwenStore::open(directory.path(), Options::default(), models)?;
    for (query, hits) in checks {
        assert_eq!(
            reopened.retrieve(query)?,
            hits,
            "reopened search changed for {query}"
        );
    }
    Ok(())
}
