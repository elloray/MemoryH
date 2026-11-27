mod common;
use common::{Fixture, InputRecord};
use memoryh::qwen::{QwenModels, QwenStore, SearchProfile};
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
    let models = QwenModels::load(encoder, reranker, runtime, 2)?;
    let dir = tempfile::tempdir()?;
    let store = QwenStore::open(dir.path(), Options::default(), models.clone())?;
    let fixture: Fixture<Question> = serde_json::from_str(include_str!("../cases.json"))?;
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
            let p = store.retrieve_profiled(&q.text, usize::MAX)?;
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
                json!({"variants":p.variants, "pool":pool,"top":top,"samples":[]}),
            );
            if stage == "after" {
                reopen_checks.push((&q.text, p.records.into_iter().take(3).collect::<Vec<_>>()));
            }
        }
        for round in 0..rounds {
            for q in &questions {
                let start = Instant::now();
                let p = store.retrieve_profiled(&q.text, 3)?;
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
                    case["top"],
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
        assert_eq!(store.retrieve(q)?, *top, "reopen changed {q}");
    }
    println!(
        "{}",
        json!({"cases":cases,"reopened":reopen_checks.len(),"rounds":rounds})
    );
    Ok(())
}
