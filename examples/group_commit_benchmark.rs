//! Compare independently acknowledged writes; also exercise real crash recovery.
mod common;
use common::InputRecord;
use memoryh::bge::{BgeSmallZh, BgeStore};
use memoryh::{Memory, Mutation, Options, SharedStore, Store};
use serde::Deserialize;
use serde_json::json;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::{Barrier, Mutex};
use std::thread;
use std::time::{Duration, Instant};

#[derive(Deserialize)]
struct Question {
    id: String,
    group: String,
    stage: String,
    text: String,
    expected: Vec<String>,
    #[serde(default)]
    forbidden: Vec<String>,
}

type Fixture = common::Fixture<Question>;
type BenchResult<T> = Result<T, Box<dyn std::error::Error>>;

enum Writer {
    Single(Box<Mutex<Store>>),
    Group(SharedStore),
    SingleText(Box<Mutex<(Store, BgeSmallZh)>>),
    GroupText(BgeStore),
}

impl Writer {
    fn apply(&self, mutation: &Mutation, vector: &[f32]) -> memoryh::Result<u64> {
        let one = std::slice::from_ref(mutation);
        match self {
            Self::Single(store) => store
                .lock()
                .unwrap()
                .apply_with_embeddings(one, &[Some(vector)]),
            Self::Group(store) => store.apply_with_embeddings(one, &[Some(vector)]),
            Self::SingleText(pair) => {
                let (store, encoder) = &mut *pair.lock().unwrap();
                let vectors = encoder.embed(&[&record(mutation).body])?;
                store.apply_with_embeddings(one, &[Some(&vectors[0])])
            }
            Self::GroupText(store) => store.apply(one),
        }
    }

    fn syncs(&self) -> memoryh::Result<u64> {
        Ok(match self {
            Self::Single(store) => store.lock().unwrap().data_syncs(),
            Self::Group(store) => store.data_syncs()?,
            Self::SingleText(pair) => pair.lock().unwrap().0.data_syncs(),
            Self::GroupText(store) => store.data_syncs()?,
        })
    }
}

fn record(mutation: &Mutation) -> &Memory {
    match mutation {
        Mutation::Put(memory) | Mutation::Replace { memory, .. } => memory,
        Mutation::Retract { .. } => unreachable!("fixture has no retractions"),
    }
}

fn copies(rows: &[InputRecord], count: usize) -> Vec<Vec<Mutation>> {
    (0..count)
        .map(|copy| {
            rows.iter()
                .cloned()
                .map(|mut row| {
                    row.id = format!("{copy}/{}", row.id);
                    row.key = format!("{copy}/{}", row.key);
                    row.supersedes = row.supersedes.map(|id| format!("{copy}/{id}"));
                    row.into()
                })
                .collect()
        })
        .collect()
}

fn measure(
    writer: &Writer,
    sets: &[Vec<Mutation>],
    vectors: &[Vec<f32>],
    workers: usize,
    burst: bool,
) -> (Duration, Vec<u64>) {
    let wave = Barrier::new(workers);
    let ready = Barrier::new(workers + 1);
    let start = Barrier::new(workers + 1);
    thread::scope(|scope| {
        let mut threads = Vec::new();
        for worker in 0..workers {
            let wave = &wave;
            let ready = &ready;
            let start = &start;
            threads.push(scope.spawn(move || {
                let mut samples = Vec::with_capacity(sets.len().div_ceil(workers) * vectors.len());
                ready.wait();
                start.wait();
                for set in sets.iter().skip(worker).step_by(workers) {
                    for (mutation, vector) in set.iter().zip(vectors) {
                        if burst {
                            wave.wait();
                        }
                        let before = Instant::now();
                        writer.apply(mutation, vector).unwrap();
                        samples.push(before.elapsed().as_nanos() as u64);
                    }
                }
                (Instant::now(), samples)
            }));
        }
        ready.wait();
        let before = Instant::now();
        start.wait();
        let mut end = before;
        let mut samples = Vec::new();
        for t in threads {
            let (finished, times) = t.join().unwrap();
            end = end.max(finished);
            samples.extend(times);
        }
        (end.duration_since(before), samples)
    })
}

fn bench(
    model: &Path,
    runtime: &Path,
    fixture: &Fixture,
    mode: &str,
    workers: usize,
    count: usize,
    root: &Path,
) -> BenchResult<()> {
    if workers == 0 || count < workers || !count.is_multiple_of(workers) {
        return Err("copies must be a positive multiple of workers".into());
    }
    let rows: Vec<_> = fixture
        .memories
        .iter()
        .chain(&fixture.corrections)
        .cloned()
        .collect();
    let sets = copies(&rows, count);
    let mut encoder = BgeSmallZh::load(model, runtime, 2)?;
    let vectors = encoder.embed(&rows.iter().map(|r| r.text.as_str()).collect::<Vec<_>>())?;
    let dir = tempfile::tempdir_in(root)?;
    let store = Store::open(dir.path(), Options::default())?;
    let workload = mode.strip_suffix("-burst").unwrap_or(mode);
    let writer = match workload {
        "single" => Writer::Single(Box::new(Mutex::new(store))),
        "group" => Writer::Group(store.into_shared()),
        "single-text" => Writer::SingleText(Box::new(Mutex::new((store, encoder)))),
        "group-text" => Writer::GroupText(BgeStore::with_store(store, encoder)?),
        _ => return Err("unknown mode".into()),
    };
    let (duration, times) = measure(&writer, &sets, &vectors, workers, workload != mode);
    let syncs = writer.syncs()?;
    drop(writer);
    // Every independently acknowledged revision and original must survive reopening.
    let mut store = Store::open(dir.path(), Options::default())?;
    assert_eq!(store.revision() as usize, times.len());
    for mutation in sets.iter().flatten() {
        let mem = record(mutation);
        assert_eq!(store.get_by_id(&mem.id)?.unwrap().memory, *mem);
        if matches!(mutation, Mutation::Replace { .. }) {
            assert_eq!(store.get(&mem.key)?.unwrap().memory, *mem);
        }
    }
    println!(
        "{}",
        json!({"mode":mode,"workers":workers,"writes":times.len(),
        "seconds":duration.as_secs_f64(),"data_syncs":syncs,"latency_ns":times})
    );
    Ok(())
}

// Query using the corrected original text as a probe, and verify exact source
// and body. This isolates storage visibility from paraphrase-recall quality.
fn check_hit(store: &BgeStore, row: &InputRecord) -> memoryh::Result<()> {
    let hits = store.retrieve(&row.text)?;
    assert!(
        hits.iter().any(|r| r.memory.id == row.id
            && r.memory.body == row.text
            && r.memory.source == row.source),
        "missing new original {}",
        row.id
    );
    assert!(
        !hits
            .iter()
            .any(|r| Some(&r.memory.id) == row.supersedes.as_ref()),
        "stale original for {}",
        row.id
    );
    Ok(())
}

fn correction_probes(store: &BgeStore, fixture: &Fixture) -> BenchResult<serde_json::Value> {
    let mut cases = serde_json::Map::new();
    for question in fixture
        .questions
        .iter()
        .filter(|q| q.group == "记忆更正" && q.stage == "after")
    {
        let ids: Vec<_> = store
            .retrieve(&question.text)?
            .into_iter()
            .map(|r| r.memory.id)
            .collect();
        assert!(question.forbidden.iter().all(|id| !ids.contains(id)));
        cases.insert(
            question.id.clone(),
            json!({
                "hits":ids, "complete":question.expected.iter().all(|id| ids.contains(id))
            }),
        );
    }
    Ok(cases.into())
}

fn verify_child(model: &Path, runtime: &Path, fixture: &Fixture, path: &Path) -> BenchResult<()> {
    let store = BgeStore::open(
        path,
        Options::default(),
        BgeSmallZh::load(model, runtime, 2)?,
    )?;
    for row in &fixture.memories {
        store.apply(&[row.clone().into()])?;
    }
    let start = Barrier::new(fixture.corrections.len());
    thread::scope(|scope| {
        for row in &fixture.corrections {
            let start = &start;
            let store = store.clone();
            scope.spawn(move || {
                start.wait();
                store.apply(&[row.clone().into()]).unwrap();
                check_hit(&store, row).unwrap();
            });
        }
    });
    let probes = correction_probes(&store, fixture)?;
    println!("ack {} {}", store.data_syncs()?, probes);
    std::io::stdout().flush()?;
    // Parent keeps stdin open and kills us after observing the acknowledgements.
    std::io::stdin().read_line(&mut String::new())?;
    Ok(())
}

fn verify(model: &Path, runtime: &Path, fixture: &Fixture, root: &Path) -> BenchResult<()> {
    let dir = tempfile::tempdir_in(root)?;
    let mut child = Command::new(std::env::current_exe()?)
        .arg(model)
        .arg(runtime)
        .arg("verify-child")
        .arg(dir.path())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    let mut ack = String::new();
    BufReader::new(child.stdout.take().unwrap()).read_line(&mut ack)?;
    if !ack.starts_with("ack ") {
        drop(child.stdin.take());
        return Err(format!("child failed: {}", child.wait()?).into());
    }
    child.kill()?;
    let status = child.wait()?;
    assert!(!status.success());
    let mut raw = Store::open(dir.path(), Options::default())?;
    assert_eq!(
        raw.revision() as usize,
        fixture.memories.len() + fixture.corrections.len()
    );
    for row in fixture.memories.iter().chain(&fixture.corrections) {
        let stored = raw.get_by_id(&row.id)?.unwrap();
        assert_eq!(stored.memory, *record(&row.clone().into()));
        assert_eq!(stored.supersedes, row.supersedes);
    }
    for row in &fixture.corrections {
        assert_eq!(raw.get(&row.key)?.unwrap().memory.id, row.id);
    }
    let store = BgeStore::with_store(raw, BgeSmallZh::load(model, runtime, 2)?)?;
    for row in &fixture.corrections {
        check_hit(&store, row)?;
    }
    let after = correction_probes(&store, fixture)?;
    let before: serde_json::Value =
        serde_json::from_str(ack.trim().splitn(3, ' ').nth(2).ok_or("missing probes")?)?;
    assert_eq!(after, before);
    println!(
        "{} originals recovered after {status}; {} corrections returned exact source/text before and after restart ({})",
        fixture.memories.len() + fixture.corrections.len(),
        fixture.corrections.len(),
        ack.trim()
    );
    Ok(())
}

fn main() -> BenchResult<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let fixture: Fixture = serde_json::from_str(include_str!("../cases.json"))?;
    match args.as_slice() {
        [model, rt, mode, dir] if mode == "verify-child" => verify_child(model.as_ref(), rt.as_ref(), &fixture, dir.as_ref()),
        [model, rt, mode, dir] if mode == "verify" => verify(model.as_ref(),rt.as_ref(),&fixture,dir.as_ref()),
        [model, rt, mode, workers, count, dir] => bench(model.as_ref(),rt.as_ref(),&fixture,mode,workers.parse()?,count.parse()?,dir.as_ref()),
        _ => Err("usage: MODEL RUNTIME MODE WORKERS COPIES DIR (MODE: single, group, single-text, group-text; optional -burst suffix), or MODEL RUNTIME verify DIR".into()),
    }
}
