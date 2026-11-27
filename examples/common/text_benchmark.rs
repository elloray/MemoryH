use crate::common::Fixture;
use memoryh::{Memory, Mutation, Record};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::time::Instant;

#[derive(Deserialize)]
pub struct Question {
    pub id: String,
    pub stage: String,
    pub text: String,
    pub expected: Vec<String>,
    #[serde(default)]
    pub forbidden: Vec<String>,
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
    ranked_pool: Vec<String>,
    evidence_ranks: BTreeMap<String, Option<usize>>,
    hits: Vec<Hit>,
    latency_ns: Vec<u64>,
}

#[derive(Serialize)]
struct Report {
    load_ns: u64,
    cases: BTreeMap<String, CaseResult>,
}

pub fn run(
    rounds: usize,
    load_ns: u64,
    apply: impl Fn(&[Mutation]) -> memoryh::Result<u64>,
    retrieve: impl Fn(&str) -> memoryh::Result<Vec<Record>>,
    rank_pool: impl Fn(&str) -> memoryh::Result<Vec<Record>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let fixture: Fixture<Question> = serde_json::from_str(include_str!("../../cases.json"))?;
    let mut report = Report {
        load_ns,
        cases: BTreeMap::new(),
    };
    for (stage, records) in [("before", fixture.memories), ("after", fixture.corrections)] {
        let records: Vec<_> = records.into_iter().map(Mutation::from).collect();
        apply(&records)?;
        let questions: Vec<_> = fixture
            .questions
            .iter()
            .filter(|question| question.stage == stage)
            .collect();

        for question in &questions {
            let pool = rank_pool(&question.text)?;
            let evidence_ranks = question
                .expected
                .iter()
                .map(|id| {
                    (
                        id.clone(),
                        pool.iter().position(|r| &r.memory.id == id).map(|p| p + 1),
                    )
                })
                .collect();
            let ranked_pool = pool.iter().map(|record| record.memory.id.clone()).collect();
            let hits: Vec<_> = pool.into_iter().take(3).map(Hit::from).collect();
            assert!(
                hits.iter().all(|hit| !question.forbidden.contains(&hit.id)),
                "returned a forbidden version for {}",
                question.id
            );
            report.cases.insert(
                question.id.clone(),
                CaseResult {
                    ranked_pool,
                    evidence_ranks,
                    hits,
                    latency_ns: Vec::with_capacity(rounds),
                },
            );
        }
        eprintln!("{stage}: warmed {} questions", questions.len());
        for round in 0..rounds {
            for question in &questions {
                let start = Instant::now();
                let hits = retrieve(&question.text)?;
                let elapsed = start.elapsed().as_nanos() as u64;
                report
                    .cases
                    .get_mut(&question.id)
                    .expect("warmed question")
                    .latency_ns
                    .push(elapsed);
                let warm = &report.cases[&question.id].hits;
                assert!(
                    hits.len() == warm.len()
                        && hits.iter().zip(warm).all(|(h, w)| h.memory.id == w.id
                            && h.memory.body == w.text
                            && h.memory.source == w.source)
                );
            }
            eprintln!("{stage}: completed timing round {}/{}", round + 1, rounds);
        }
    }
    println!("{}", serde_json::to_string(&report)?);
    Ok(())
}
