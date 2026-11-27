use memoryh::qwen::{QwenModels, QwenStore};
use memoryh::{Error, Options, Store};
use std::{env, fs, path::PathBuf};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<PathBuf> = env::args_os().skip(1).map(PathBuf::from).collect();
    let [source, parent, model, reranker, runtime] = args.as_slice() else {
        return Err("usage: qwen_reindex SOURCE_STORE OUTPUT_PARENT MODEL_DIR RERANKER_DIR ONNXRUNTIME_LIBRARY".into());
    };
    if !source.join("0000000000000001.mhl").is_file() {
        return Err("source must be an existing MemoryH store".into());
    }
    let source = Store::open(source, Options::default())?;
    let directory = tempfile::Builder::new()
        .prefix("memoryh-qwen-")
        .tempdir_in(parent)?;
    let models = QwenModels::load(model, reranker, runtime, 2)?;
    let destination = QwenStore::open(directory.path(), Options::default(), models)?;
    source.visit_mutations(|revision, mutations| {
        if destination.apply(mutations)? != revision {
            return Err(Error::Corrupt("imported revision diverged".into()));
        }
        Ok(())
    })?;
    drop(destination);
    fs::File::open(parent)?.sync_all()?;
    println!("{}", directory.keep().display());
    Ok(())
}
