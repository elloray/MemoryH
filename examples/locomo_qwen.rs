//! Persistent QwenStore for one LoCoMo conversation. The worker sees no QA labels.
use memoryh::qwen::{QueryContext, QwenModels, QwenStore, RecallOptions, session_date};
use memoryh::{Memory, Mutation, Options};
use serde::Deserialize;
use serde_json::{Value, json};
use std::error::Error;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;

#[derive(Deserialize)]
struct Turn {
    id: String,
    text: String,
    date: String,
}

#[derive(Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
enum Request {
    Insert {
        turns: Vec<Turn>,
    },
    Search {
        question: String,
        date: String,
        top_k: Option<usize>,
    },
}

fn handle(store: &QwenStore, request: Request) -> Result<Value, Box<dyn Error>> {
    match request {
        Request::Insert { turns } => {
            let count = turns.len();
            let mutations: Vec<_> = turns
                .into_iter()
                .map(|turn| {
                    Mutation::Put(Memory {
                        key: turn.id.clone(),
                        id: turn.id,
                        source: turn.date,
                        body: turn.text,
                    })
                })
                .collect();
            for batch in mutations.chunks(4) {
                store.apply(batch)?;
            }
            Ok(json!({"inserted": count}))
        }
        Request::Search {
            question,
            date,
            top_k,
        } => {
            let context = QueryContext {
                question_date: Some(session_date(&date).ok_or("invalid question date")?),
                ..Default::default()
            };
            let profile = store.retrieve_profiled_with_context(
                &question,
                Some(top_k.unwrap_or(200)),
                &context,
            )?;
            let timings = json!({
                "candidates":profile.candidate_count,
                "variants":profile.variants.len(),
                "scored_pairs":profile.candidate_count * profile.variants.len(),
                "split_ms":profile.split.as_secs_f64()*1000.0,
                "encode_ms":profile.encode.as_secs_f64()*1000.0,
                "recall_ms":profile.recall.as_secs_f64()*1000.0,
                "rerank_ms":profile.rerank.as_secs_f64()*1000.0,
                "yuanbao_ms":profile.model_scoring.as_secs_f64()*1000.0
            });
            let hits = profile.records;
            // The upstream formatter sorts by score. Preserve our final reranker order.
            let records: Vec<_> = hits.into_iter().enumerate().map(|(rank, hit)| json!({
                "id": hit.memory.id, "memory": hit.memory.body, "created_at": hit.memory.source,
                "score": 1.0 / (1 + rank) as f64
            })).collect();
            Ok(json!({"results": records,"profile":timings}))
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    let [dir, model_dir, runtime, limits @ ..] = args.as_slice() else {
        return Err(
            "expected STORE MODEL_DIR ORT [RECALL_WIDTH | VECTOR_LIMIT KEYWORD_LIMIT]".into(),
        );
    };
    let mut recall = RecallOptions::default();
    let limit = |arg: &std::ffi::OsString| -> Result<usize, Box<dyn Error>> {
        let value = arg.to_str().ok_or("invalid recall limit")?.parse()?;
        if value == 0 {
            return Err("recall limit must be positive".into());
        }
        Ok(value)
    };
    match limits {
        [] => {}
        [both] => {
            recall.vector_limit = limit(both)?;
            recall.keyword_limit = recall.vector_limit;
        }
        [vector, keyword] => {
            recall.vector_limit = limit(vector)?;
            recall.keyword_limit = limit(keyword)?;
        }
        _ => return Err("expected at most two recall limits".into()),
    }
    if let Some(threshold) = std::env::var_os("MEMORYH_SCANN_EXACT_THRESHOLD") {
        recall.exact_threshold = threshold
            .to_str()
            .ok_or("invalid ScaNN threshold")?
            .parse()?;
    }
    let models = PathBuf::from(model_dir);
    let models = QwenModels::load(models.join("qwen"), models.join("yuanbao"), runtime, 4)?
        .with_recall_options(recall);
    let store = QwenStore::open(dir, Options::default(), models)?;
    let mut stdout = io::stdout().lock();
    for line in io::stdin().lock().lines() {
        let request = serde_json::from_str(&line?)?;
        let response = handle(&store, request).unwrap_or_else(|e| json!({"error":e.to_string()}));
        writeln!(stdout, "{response}")?;
        stdout.flush()?;
    }
    Ok(())
}
