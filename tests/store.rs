use std::fs::{self, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use memoryh::{Error, Memory, Mutation, Options, Query, SearchMethod, SearchOptions, Store};
use serde_json::Value;
use tempfile::tempdir;

#[test]
fn retrieval_tracks_committed_heads_and_rebuilds_vectors_on_reopen() {
    let dir = tempdir().unwrap();
    let whole = Query {
        text: "饮品和差旅",
        embedding: Some(&[1.0, 0.0]),
    };
    let coffee = Query {
        text: "无糖豆奶",
        embedding: Some(&[1.0, 0.0]),
    };
    let travel = Query {
        text: "报销规则",
        embedding: Some(&[0.0, 1.0]),
    };
    let options = SearchOptions {
        top_k: 2,
        ..SearchOptions::default()
    };
    let mut store = Store::open(dir.path(), Options::default()).unwrap();
    store
        .apply(&[Mutation::Put(memory("legacy", "note", "ＡＢＣ 字"))])
        .unwrap();
    let revision = store
        .apply_with_embeddings(
            &[
                Mutation::Put(memory("old", "coffee", "燕麦奶")),
                Mutation::Put(memory("travel", "travel", "差旅报销规则")),
            ],
            &[Some(&[1.0, 0.0]), Some(&[0.0, 1.0])],
        )
        .unwrap();
    assert_eq!(store.search_keywords("abc", 3).unwrap(), ["legacy"]);
    assert_eq!(store.search_keywords("燕麦奶", 3).unwrap(), ["old"]);

    let replacement = Mutation::Replace {
        expected_id: "old".into(),
        memory: memory("new", "coffee", "无糖豆奶"),
    };
    assert!(
        store
            .apply_with_embeddings(std::slice::from_ref(&replacement), &[Some(&[1.0])])
            .is_err()
    );
    assert_eq!(store.revision(), revision);
    assert_eq!(store.search_keywords("燕麦奶", 3).unwrap(), ["old"]);
    store
        .apply_with_embeddings(&[replacement], &[Some(&[1.0, 0.0])])
        .unwrap();

    let queries = [whole, coffee, travel];
    let ranked = store.search(&queries, &options).unwrap();
    assert_eq!(ranked.flat_vector_first, ["new", "travel"]);
    assert_eq!(ranked.tree_vector_first, ["new", "travel"]);
    let originals = store
        .retrieve(&queries, &options, SearchMethod::Tree)
        .unwrap();
    assert_eq!(
        originals
            .iter()
            .map(|record| record.memory.id.as_str())
            .collect::<Vec<_>>(),
        ["new", "travel"]
    );
    assert!(store.search_keywords("燕麦奶", 3).unwrap().is_empty());
    assert_eq!(
        store.get_by_id("new").unwrap().unwrap().memory.body,
        "无糖豆奶"
    );
    assert_eq!(
        store
            .get_at("coffee", revision)
            .unwrap()
            .unwrap()
            .memory
            .body,
        "燕麦奶"
    );

    drop(store);
    let mut store = Store::open(dir.path(), Options::default()).unwrap();
    assert_eq!(store.search(&queries, &options).unwrap(), ranked);
    store
        .apply(&[Mutation::Retract {
            key: "coffee".into(),
            expected_id: "new".into(),
        }])
        .unwrap();
    assert!(store.search_keywords("无糖豆奶", 3).unwrap().is_empty());
    assert_eq!(
        store.search(&queries, &options).unwrap().whole.flat_vector,
        ["travel"]
    );
    assert!(store.get_by_id("new").unwrap().is_some());
}

fn memory(id: &str, key: &str, body: &str) -> Memory {
    Memory {
        id: id.into(),
        key: key.into(),
        source: "test transcript".into(),
        body: body.into(),
    }
}

#[test]
fn historical_reads_survive_replacement_retraction_and_reopen() {
    let tmp = tempdir().unwrap();
    let path = tmp.path().join("memories");
    let mut store = Store::open(&path, Options::default()).unwrap();
    let old = memory("m1", "饮食.咖啡", "全脂牛奶");
    let new = memory("m2", &old.key, "无糖豆奶");
    let other = memory("m3", "travel.hotel", "南京");

    let before = store
        .apply(&[Mutation::Put(old.clone()), Mutation::Put(other.clone())])
        .unwrap();
    assert_eq!(store.get(&old.key).unwrap().unwrap().memory, old);
    let after = store
        .apply(&[Mutation::Replace {
            expected_id: old.id.clone(),
            memory: new.clone(),
        }])
        .unwrap();
    assert_eq!(store.get(&old.key).unwrap().unwrap().memory, new);
    let removed = store
        .apply(&[Mutation::Retract {
            key: new.key.clone(),
            expected_id: new.id.clone(),
        }])
        .unwrap();
    assert_eq!((before, after, removed), (1, 2, 3));
    drop(store);

    let mut store = Store::open(&path, Options::default()).unwrap();
    assert_eq!(store.revision(), removed);
    assert_eq!(store.get(&old.key).unwrap(), None);
    assert_eq!(store.get_at(&old.key, 0).unwrap(), None);
    let prior = store.get_at(&old.key, before).unwrap().unwrap();
    let updated = store.get_at(&old.key, after).unwrap().unwrap();
    assert_eq!(prior.memory, old);
    assert_eq!(updated.memory, new);
    assert_eq!(updated.supersedes.as_deref(), Some(old.id.as_str()));
    assert_eq!(store.get_by_id(&old.id).unwrap(), Some(prior));
    assert_eq!(store.get_by_id(&new.id).unwrap(), Some(updated));
    assert_eq!(
        store.visible_keys(removed).unwrap().collect::<Vec<_>>(),
        vec![(other.key.as_str(), other.id.as_str())]
    );
    assert_eq!(
        store
            .history(&old.key)
            .unwrap()
            .iter()
            .map(|version| (version.revision, version.id.as_deref()))
            .collect::<Vec<_>>(),
        vec![(before, Some("m1")), (after, Some("m2")), (removed, None)]
    );
    let restored = memory("m4", &old.key, "黑咖啡");
    store.apply(&[Mutation::Put(restored.clone())]).unwrap();
    assert_eq!(store.get(&old.key).unwrap().unwrap().memory, restored);
}

#[test]
fn stale_batch_is_atomic_and_ids_are_never_reused() {
    let tmp = tempdir().unwrap();
    let mut store = Store::open(tmp.path(), Options::default()).unwrap();
    let old = memory("m1", "coffee", "latte");
    let new = memory("m2", "coffee", "espresso");
    let other = memory("m3", "travel", "train");
    store.apply(&[Mutation::Put(old.clone())]).unwrap();
    let path = segment(tmp.path(), 1);
    let bytes = fs::read(&path).unwrap();

    let invalid = [
        Mutation::Put(other.clone()),
        Mutation::Replace {
            expected_id: "stale".into(),
            memory: new.clone(),
        },
    ];
    assert!(matches!(store.apply(&invalid), Err(Error::Conflict(_))));
    assert_eq!(store.revision(), 1);
    assert_eq!(store.get("travel").unwrap(), None);
    assert_eq!(fs::read(&path).unwrap(), bytes);

    let newer = memory("m4", "coffee", "filter");
    assert_eq!(
        store
            .apply(&[
                Mutation::Replace {
                    expected_id: old.id.clone(),
                    memory: new.clone()
                },
                Mutation::Replace {
                    expected_id: new.id.clone(),
                    memory: newer.clone()
                },
                Mutation::Put(other.clone()),
            ])
            .unwrap(),
        2
    );
    assert_eq!(store.get_at("coffee", 1).unwrap().unwrap().memory, old);
    assert_eq!(store.get_at("coffee", 2).unwrap().unwrap().memory, newer);
    assert_eq!(store.get_by_id("m2").unwrap().unwrap().memory, new);
    assert!(matches!(
        store.apply(&[Mutation::Put(memory("m1", "unused", "duplicate"))]),
        Err(Error::Conflict(_))
    ));
    drop(store);
    let mut store = Store::open(tmp.path(), Options::default()).unwrap();
    assert_eq!(store.get("coffee").unwrap().unwrap().memory, newer);
    assert_eq!(store.get("travel").unwrap().unwrap().memory, other);
}

#[test]
fn cold_reads_fill_ram_and_lru_evicts_by_byte_budget() {
    let tmp = tempdir().unwrap();
    let mut store = Store::open(
        tmp.path(),
        Options {
            cache_bytes: 5000,
            ..Options::default()
        },
    )
    .unwrap();
    let body = "x".repeat(2048);
    let records: Vec<_> = ["a", "b", "c"]
        .into_iter()
        .map(|id| Mutation::Put(memory(id, id, &body)))
        .collect();
    store.apply(&records).unwrap();

    for key in ["a", "b", "a", "c", "a", "b"] {
        assert_eq!(store.get(key).unwrap().unwrap().memory.body, body);
    }
    let stats = store.cache_stats();
    assert_eq!(
        (stats.hits, stats.misses, stats.disk_reads, stats.evictions),
        (2, 4, 4, 2)
    );
    assert_eq!(stats.entries, 2);
    assert!(stats.bytes <= 5000);
    drop(store);

    let mut store = Store::open(
        tmp.path(),
        Options {
            cache_bytes: 0,
            ..Options::default()
        },
    )
    .unwrap();
    assert_eq!(store.cache_stats().entries, 0);
    for _ in 0..2 {
        assert_eq!(store.get("a").unwrap().unwrap().memory.body, body);
    }
    let stats = store.cache_stats();
    assert_eq!((stats.hits, stats.disk_reads, stats.entries), (0, 2, 0));
}

#[test]
fn neighboring_records_share_one_disk_block_and_promote_to_record_lru() {
    let tmp = tempdir().unwrap();
    let mut store =
        Store::open_with_block_cache(tmp.path(), Options::default(), 16 * 1024).unwrap();
    let first = memory("first", "one", "咖啡");
    let second = memory("second", "two", "茶");
    store
        .apply(&[Mutation::Put(first.clone()), Mutation::Put(second.clone())])
        .unwrap();

    assert_eq!(store.get_by_id("first").unwrap().unwrap().memory, first);
    assert_eq!(store.get("two").unwrap().unwrap().memory, second);
    assert_eq!(store.get("two").unwrap().unwrap().memory, second);

    let stats = store.cache_stats();
    assert_eq!((stats.hits, stats.misses, stats.disk_reads), (1, 2, 1));
    assert_eq!((stats.block_hits, stats.block_reads), (1, 1));
}

#[test]
fn block_reads_span_boundaries_and_refresh_a_growing_tail() {
    let tmp = tempdir().unwrap();
    let mut store =
        Store::open_with_block_cache(tmp.path(), Options::default(), 16 * 1024).unwrap();
    let first = memory("pad", "pad", &"x".repeat(3970));
    store.apply(&[Mutation::Put(first.clone())]).unwrap();
    assert_eq!(store.get("pad").unwrap().unwrap().memory, first);
    assert_eq!(store.cache_stats().block_reads, 1);

    // The next batch starts inside the cached short tail and crosses 4 KiB.
    let crossing = memory("new", "cross", &"汉".repeat(100));
    store.apply(&[Mutation::Put(crossing.clone())]).unwrap();
    assert_eq!(store.get("cross").unwrap().unwrap().memory, crossing);
    assert_eq!(store.cache_stats().block_reads, 3);

    let tail = memory("tail", "tail", "追加");
    store.apply(&[Mutation::Put(tail.clone())]).unwrap();
    assert_eq!(store.get("tail").unwrap().unwrap().memory, tail);
    assert_eq!(store.cache_stats().block_reads, 4);
    assert_eq!(store.get_at("pad", 1).unwrap().unwrap().memory, first);
}

#[test]
fn block_cache_separates_segment_offsets_and_keeps_large_record_reads_exact() {
    let tmp = tempdir().unwrap();
    let options = Options {
        segment_bytes: 50,
        ..Options::default()
    };
    let mut store = Store::open_with_block_cache(tmp.path(), options, 16 * 1024).unwrap();
    let first = memory("m1", "one", "one segment");
    let second = memory("m2", "two", "two segment");
    let large = memory("m3", "large", &"x".repeat(8192));
    for record in [&first, &second, &large] {
        store.apply(&[Mutation::Put(record.clone())]).unwrap();
    }
    for record in [&first, &second, &large] {
        assert_eq!(&store.get(&record.key).unwrap().unwrap().memory, record);
    }
    let stats = store.cache_stats();
    assert_eq!(
        (stats.disk_reads, stats.block_reads, stats.block_hits),
        (3, 2, 0)
    );
}

#[test]
fn reads_across_sealed_segments_and_rejects_a_second_handle() {
    let tmp = tempdir().unwrap();
    let options = Options {
        segment_bytes: 64,
        ..Options::default()
    };
    let mut store = Store::open(tmp.path(), options.clone()).unwrap();
    assert!(matches!(
        Store::open(tmp.path(), options.clone()),
        Err(Error::Locked)
    ));
    for id in ["m1", "m2", "m3"] {
        store
            .apply(&[Mutation::Put(memory(id, id, &"x".repeat(80)))])
            .unwrap();
    }
    assert!(segment(tmp.path(), 3).is_file());
    drop(store);
    let mut store = Store::open(tmp.path(), options).unwrap();
    assert_eq!(store.revision(), 3);
    for id in ["m1", "m2", "m3"] {
        assert_eq!(store.get_by_id(id).unwrap().unwrap().memory.id, id);
    }
    assert_eq!(store.cache_stats().disk_reads, 3);
}

#[test]
fn recovers_every_truncated_suffix_of_an_uncommitted_batch() {
    let fixture = tempdir().unwrap();
    let mut store = Store::open(fixture.path(), Options::default()).unwrap();
    let first = memory("m1", "coffee", "latte");
    store.apply(&[Mutation::Put(first.clone())]).unwrap();
    let path = segment(fixture.path(), 1);
    let prefix = fs::metadata(&path).unwrap().len() as usize;
    store
        .apply(&[Mutation::Put(memory("m2", "travel", "train"))])
        .unwrap();
    drop(store);
    let journal = fs::read(path).unwrap();

    let tmp = tempdir().unwrap();
    let path = segment(tmp.path(), 1);
    for cutoff in prefix..journal.len() {
        fs::write(&path, &journal[..cutoff]).unwrap();
        let mut store = Store::open(tmp.path(), Options::default()).unwrap();
        assert_eq!(store.revision(), 1, "cutoff: {cutoff}");
        assert_eq!(store.get("coffee").unwrap().unwrap().memory, first);
        assert_eq!(store.get("travel").unwrap(), None);
        assert_eq!(fs::metadata(&path).unwrap().len(), prefix as u64);
        assert_eq!(
            store
                .apply(&[Mutation::Put(memory("m3", "travel", "flight"))])
                .unwrap(),
            2
        );
    }
    let mut store = Store::open(tmp.path(), Options::default()).unwrap();
    assert_eq!(store.get("travel").unwrap().unwrap().memory.id, "m3");
}

#[test]
fn rejects_corruption_in_a_complete_frame_and_an_incomplete_sealed_segment() {
    let tmp = tempdir().unwrap();
    let options = Options {
        segment_bytes: 64,
        ..Options::default()
    };
    let mut store = Store::open(tmp.path(), options.clone()).unwrap();
    for id in ["m1", "m2"] {
        store
            .apply(&[Mutation::Put(memory(id, id, &"x".repeat(80)))])
            .unwrap();
    }
    drop(store);
    let last = segment(tmp.path(), 2);
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&last)
        .unwrap();
    let length = file.metadata().unwrap().len();
    file.seek(SeekFrom::End(-1)).unwrap();
    let mut byte = [0];
    file.read_exact(&mut byte).unwrap();
    file.seek(SeekFrom::End(-1)).unwrap();
    file.write_all(&[byte[0] ^ 1]).unwrap();
    drop(file);
    assert!(matches!(
        Store::open(tmp.path(), options.clone()),
        Err(Error::Corrupt(_))
    ));
    assert_eq!(fs::metadata(last).unwrap().len(), length);

    let sealed = segment(tmp.path(), 1);
    let length = fs::metadata(&sealed).unwrap().len();
    OpenOptions::new()
        .write(true)
        .open(&sealed)
        .unwrap()
        .set_len(length - 1)
        .unwrap();
    assert!(
        matches!(Store::open(tmp.path(), options), Err(Error::Corrupt(message)) if message.contains("sealed"))
    );
    assert_eq!(fs::metadata(sealed).unwrap().len(), length - 1);
}

fn segment(directory: &Path, number: u64) -> std::path::PathBuf {
    directory.join(format!("{number:016x}.mhl"))
}

#[test]
fn imports_the_chinese_fixture_and_reopens_both_read_views() {
    let fixture: Value = serde_json::from_str(include_str!("../cases.json")).unwrap();
    let mutations = |name: &str| {
        fixture[name]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| {
                let record = Memory {
                    id: row["id"].as_str().unwrap().into(),
                    key: row["key"].as_str().unwrap().into(),
                    source: row["source"].as_str().unwrap().into(),
                    body: row["text"].as_str().unwrap().into(),
                };
                match row["supersedes"].as_str() {
                    Some(id) => Mutation::Replace {
                        expected_id: id.into(),
                        memory: record,
                    },
                    None => Mutation::Put(record),
                }
            })
            .collect::<Vec<_>>()
    };
    let root = tempdir().unwrap();
    let mut store = Store::open(root.path(), Options::default()).unwrap();
    let before = store.apply(&mutations("memories")).unwrap();
    let after = store.apply(&mutations("corrections")).unwrap();
    drop(store);

    let mut store = Store::open(root.path(), Options::default()).unwrap();
    let mut checked = 0;
    for (stage, revision) in [("before", before), ("after", after)] {
        let ids = store
            .visible_keys(revision)
            .unwrap()
            .map(|(_, id)| id.to_owned())
            .collect::<std::collections::HashSet<_>>();
        for question in fixture["questions"].as_array().unwrap() {
            if question["stage"] != stage {
                continue;
            }
            for evidence in question["expected"].as_array().unwrap() {
                let id = evidence.as_str().unwrap();
                assert!(ids.contains(id), "{stage} missing {id}");
                let row = store.get_by_id(id).unwrap().unwrap();
                assert_eq!(store.get_at(&row.memory.key, revision).unwrap(), Some(row));
            }
            if let Some(forbidden) = question["forbidden"].as_array() {
                for evidence in forbidden {
                    let id = evidence.as_str().unwrap();
                    assert!(!ids.contains(id), "{stage} exposed {id}");
                }
            }
            checked += 1;
        }
    }
    assert_eq!(checked, 21);
}

#[test]
fn competing_shared_corrections_publish_only_the_acknowledged_winner() {
    use std::sync::Barrier;
    use std::thread;

    let dir = tempdir().unwrap();
    let shared = Store::open(dir.path(), Options::default())
        .unwrap()
        .into_shared();
    shared
        .apply(&[Mutation::Put(memory("v1", "milk", "燕麦奶"))])
        .unwrap();
    let gate = Barrier::new(8);
    let winner = thread::scope(|scope| {
        let threads: Vec<_> = (0..8)
            .map(|i| {
                let gate = &gate;
                let store = shared.clone();
                scope.spawn(move || {
                    let new = memory(&format!("v2-{i}"), "milk", &format!("豆奶{i}"));
                    gate.wait();
                    match store.apply(&[Mutation::Replace {
                        expected_id: "v1".into(),
                        memory: new.clone(),
                    }]) {
                        Ok(revision) => {
                            assert_eq!(revision, 2);
                            assert_eq!(store.get("milk").unwrap().unwrap().memory, new);
                            Some(new)
                        }
                        Err(Error::Conflict(_)) => None,
                        other => panic!("unexpected result {other:?}"),
                    }
                })
            })
            .collect();
        let mut winners = threads
            .into_iter()
            .filter_map(|t| t.join().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(winners.len(), 1);
        winners.pop().unwrap()
    });
    assert_eq!(shared.apply(&[]).unwrap(), 2);
    assert_eq!(shared.data_syncs().unwrap(), 2);
    drop(shared);
    let mut reopened = Store::open(dir.path(), Options::default()).unwrap();
    assert_eq!(reopened.get("milk").unwrap().unwrap().memory, winner);
    assert_eq!(reopened.get_at("milk", 1).unwrap().unwrap().memory.id, "v1");
}
