mod common;
#[path = "common/text_benchmark.rs"]
mod text_benchmark;
use memoryh::Options;
use memoryh::bge::{BgeSmallZh, BgeStore};
use std::{env, path::PathBuf, time::Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args_os().skip(1);
    let usage = "usage: bge_benchmark MODEL_DIR ONNXRUNTIME_LIBRARY ROUNDS";
    let model_dir = PathBuf::from(args.next().ok_or(usage)?);
    let runtime = PathBuf::from(args.next().ok_or(usage)?);
    let rounds: usize = args.next().ok_or(usage)?.to_string_lossy().parse()?;
    if rounds == 0 || args.next().is_some() {
        return Err(usage.into());
    }

    let directory = tempfile::tempdir()?;
    let start = Instant::now();
    let encoder = BgeSmallZh::load(model_dir, runtime, 2)?;
    let load_ns = start.elapsed().as_nanos() as u64;
    let store = BgeStore::open(directory.path(), Options::default(), encoder)?;
    text_benchmark::run(
        rounds,
        load_ns,
        |m| store.apply(m),
        |q| store.retrieve(q),
        |q| store.retrieve(q),
    )
}
