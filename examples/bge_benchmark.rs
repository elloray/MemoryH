mod common;
use common::Fixture;
use std::collections::BTreeMap;
use std::env;
use std::path::PathBuf;
use std::time::Instant;

use memoryh::bge::{BgeSmallZh, BgeStore};
use memoryh::{Memory, Mutation, Options, Record};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Question {
    id: String,
    stage: String,
    text: String,
}

#[derive(Serialize)]
struct Hit {
    id: String,
    source: String,
    text: String,
}

impl From<Record> for Hit {
    fn from(record: Record) -> Self {
        let Memory {
            id, source, body, ..
        } = record.memory;
        Self {
            id,
            source,
            text: body,
        }
    }
}

#[derive(Serialize)]
struct CaseResult {
    hits: Vec<Hit>,
    latency_ns: Vec<u64>,
}

#[derive(Serialize)]
struct Report {
    load_ns: u64,
    cases: BTreeMap<String, CaseResult>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args_os().skip(1);
    let usage = "usage: bge_benchmark MODEL_DIR ONNXRUNTIME_LIBRARY ROUNDS";
    let model_dir = PathBuf::from(args.next().ok_or(usage)?);
    let runtime = PathBuf::from(args.next().ok_or(usage)?);
    let rounds: usize = args.next().ok_or(usage)?.to_string_lossy().parse()?;
    if rounds == 0 || args.next().is_some() {
        return Err(usage.into());
    }

    let fixture: Fixture<Question> = serde_json::from_str(include_str!("../cases.json"))?;
    let directory = tempfile::tempdir()?;
    let start = Instant::now();
    let encoder = BgeSmallZh::load(model_dir, runtime, 2)?;
    let mut report = Report {
        load_ns: start.elapsed().as_nanos() as u64,
        cases: BTreeMap::new(),
    };
    let store = BgeStore::open(directory.path(), Options::default(), encoder)?;

    for (stage, records) in [("before", fixture.memories), ("after", fixture.corrections)] {
        let records: Vec<_> = records.into_iter().map(Mutation::from).collect();
        store.apply(&records)?;
        let questions: Vec<_> = fixture
            .questions
            .iter()
            .filter(|question| question.stage == stage)
            .collect();

        for question in &questions {
            let hits = store
                .retrieve(&question.text)?
                .into_iter()
                .map(Hit::from)
                .collect();
            report.cases.insert(
                question.id.clone(),
                CaseResult {
                    hits,
                    latency_ns: Vec::with_capacity(rounds),
                },
            );
        }
        for _ in 0..rounds {
            for question in &questions {
                let start = Instant::now();
                let hits = store.retrieve(&question.text)?;
                let elapsed = start.elapsed().as_nanos() as u64;
                report
                    .cases
                    .get_mut(&question.id)
                    .expect("warmed question")
                    .latency_ns
                    .push(elapsed);
                std::hint::black_box(hits);
            }
        }
    }
    println!("{}", serde_json::to_string(&report)?);
    Ok(())
}
