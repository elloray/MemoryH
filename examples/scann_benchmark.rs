mod common;
use common::{Fixture, InputRecord};
use memoryh::qwen::{
    QueryContext, QwenModels, QwenStore, RecallOptions, SearchProfile, session_date,
};
use memoryh::{Mutation, Options};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, HashMap},
    env,
    time::Instant,
};

#[derive(Deserialize)]
struct Question {
    id: String,
    text: String,
    stage: String,
    question_date: Option<String>,
}

fn sample(start: Instant, p: &SearchProfile) -> Value {
    let total = start.elapsed();
    json!({
        "total_ns": total.as_nanos() as u64, "split_ns": p.split.as_nanos() as u64,
        "encode_ns": p.encode.as_nanos() as u64, "recall_ns": p.recall.as_nanos() as u64,
        "vector_ns": p.vector.as_nanos() as u64, "rerank_ns": p.rerank.as_nanos() as u64
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    let [encoder, reranker, runtime, rounds] = args.as_slice() else {
        return Err("usage: scann_benchmark ENCODER YUANBAO ORT ROUNDS".into());
    };
    let rounds: usize = rounds.parse()?;
    let mut options = RecallOptions::default();
    if let Some(v) = env::var_os("MEMORYH_EVAL_VECTOR_LIMIT") {
        options.vector_limit = v.to_string_lossy().parse()?;
    }
    if let Some(v) = env::var_os("MEMORYH_EVAL_KEYWORD_LIMIT") {
        options.keyword_limit = v.to_string_lossy().parse()?;
    }
    if let Some(v) = env::var_os("MEMORYH_EVAL_EXACT_THRESHOLD") {
        options.exact_threshold = v.to_string_lossy().parse()?;
    }
    if let Some(v) = env::var_os("MEMORYH_EVAL_PROBES") {
        options.partitions_to_search = Some(v.to_string_lossy().parse()?);
    }
    let models = QwenModels::load(encoder, reranker, runtime, 2)?.with_recall_options(options);
    let dir = tempfile::tempdir()?;
    let store = QwenStore::open(dir.path(), Options::default(), models.clone())?;
    let data = match env::var_os("MEMORYH_EVAL_FIXTURE") {
        Some(path) => std::fs::read_to_string(path)?,
        None => include_str!("../cases.json").to_owned(),
    };
    let fixture: Fixture<Question> = serde_json::from_str(&data)?;
    let mut context = QueryContext::default();
    let set_date =
        |context: &mut QueryContext, q: &Question| -> Result<(), Box<dyn std::error::Error>> {
            context.question_date = q
                .question_date
                .as_deref()
                .map(|d| session_date(d).ok_or("invalid question date"))
                .transpose()?;
            Ok(())
        };
    let mut live = HashMap::<String, InputRecord>::new();
    let mut cases = BTreeMap::new();
    let mut reopen_checks = Vec::new();
    for (stage, batch) in [("before", fixture.memories), ("after", fixture.corrections)] {
        let ops = batch
            .iter()
            .cloned()
            .map(Mutation::from)
            .collect::<Vec<_>>();
        store.apply(&ops)?;
        for record in batch {
            if let Some(old) = &record.supersedes {
                live.remove(old);
            }
            live.insert(record.id.clone(), record);
        }
        let questions: Vec<_> = fixture
            .questions
            .iter()
            .filter(|q| q.stage == stage)
            .collect();
        for q in &questions {
            set_date(&mut context, q)?;
            let p = store.retrieve_profiled_with_context(&q.text, Some(usize::MAX), &context)?;
            let pool: Vec<_> = p.records.iter().map(|r| &r.memory.id).collect();
            let unique: std::collections::HashSet<_> = pool.iter().collect();
            assert_eq!(pool.len(), unique.len(), "{} repeated ID", q.id);
            for r in &p.records {
                let expected = live.get(&r.memory.id).expect("active record");
                assert_eq!(r.memory.body, expected.text);
                assert_eq!(r.memory.source, expected.source);
            }
            let top = pool.iter().take(3).copied().collect::<Vec<_>>();
            cases.insert(
                q.id.clone(),
                json!({"intent":format!("{:?}",p.intent), "facts":p.inventory_facts, "variants":p.variants, "pool":pool, "top5":pool.iter().take(5).collect::<Vec<_>>(),"top":top,"samples":[]}),
            );
            if stage == "after" {
                reopen_checks.push((*q, p.records.into_iter().take(5).collect::<Vec<_>>()));
            }
        }
        for round in 0..rounds {
            for q in &questions {
                set_date(&mut context, q)?;
                let start = Instant::now();
                let p = store.retrieve_profiled_with_context(&q.text, Some(5), &context)?;
                let duration = sample(start, &p);
                let case = cases.get_mut(&q.id).unwrap();
                assert_eq!(
                    json!(p.variants),
                    case["variants"],
                    "{} decomposition changed",
                    q.id
                );
                assert_eq!(
                    json!(p.records.iter().map(|r| &r.memory.id).collect::<Vec<_>>()),
                    case["top5"],
                    "{} ranking changed",
                    q.id
                );
                case["samples"].as_array_mut().unwrap().push(duration);
            }
            eprintln!("{stage} round {}/{}", round + 1, rounds);
        }
    }
    drop(store);
    let store = QwenStore::open(dir.path(), Options::default(), models)?;
    for (q, top) in &reopen_checks {
        set_date(&mut context, q)?;
        assert_eq!(
            store
                .retrieve_profiled_with_context(&q.text, Some(5), &context)?
                .records,
            *top,
            "reopen changed {}",
            q.id
        );
    }
    println!(
        "{}",
        json!({"cases":cases,"reopened":reopen_checks.len(),"rounds":rounds,
            "recall":{"vector_limit":options.vector_limit,"keyword_limit":options.keyword_limit,
            "exact_threshold":options.exact_threshold,"partitions_to_search":options.partitions_to_search}})
    );
    Ok(())
}
