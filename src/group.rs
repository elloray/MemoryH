//! Opportunistic group commit. The first waiting caller writes on behalf of
//! its peers. A lone caller never waits for a timer or a background thread.
use std::collections::VecDeque;
use std::sync::{Arc, Mutex, mpsc};

use crate::format::encode_batch;
use crate::{Error, Mutation, Query, Record, Result, SearchMethod, SearchOptions, Store};

const MAX_REQUESTS: usize = 64;
const TARGET_GROUP_BYTES: usize = 1024 * 1024;

/// Cloneable handle to a single open directory. Concurrent requests retain independent
/// revisions and acknowledge only after synchronization and publication.
#[derive(Clone)]
pub struct SharedStore {
    inner: Arc<Shared>,
}

struct Shared {
    store: Mutex<Store>,
    queue: Mutex<Queue>,
}

#[derive(Default)]
struct Queue {
    writing: bool,
    pending: VecDeque<Request>,
}

struct Request {
    payload: Vec<u8>,
    reply: mpsc::Sender<Reply>,
}

enum Reply {
    Lead,
    Done(Result<u64>),
}

impl SharedStore {
    pub(crate) fn new(store: Store) -> Self {
        Self {
            inner: Arc::new(Shared {
                store: Mutex::new(store),
                queue: Mutex::default(),
            }),
        }
    }

    pub fn apply(&self, mutations: &[Mutation]) -> Result<u64> {
        self.submit(mutations, None)
    }

    pub fn apply_with_embeddings(
        &self,
        mutations: &[Mutation],
        embeddings: &[Option<&[f32]>],
    ) -> Result<u64> {
        self.submit(mutations, Some(embeddings))
    }

    fn submit(&self, mutations: &[Mutation], embeddings: Option<&[Option<&[f32]>]>) -> Result<u64> {
        // The commit leader assigns the revision in FIFO order.
        let payload = encode_batch(mutations, embeddings)?;
        if mutations.is_empty() {
            return self.read(|store| Ok(store.revision()));
        }
        let (reply, receiver) = mpsc::channel();
        let lead = {
            let mut queue = self.inner.queue.lock().unwrap();
            queue.pending.push_back(Request { payload, reply });
            let lead = !queue.writing;
            queue.writing = true;
            lead
        };
        if lead {
            self.write_group();
        }
        loop {
            match receiver.recv().map_err(|_| Error::RecoveryRequired)? {
                Reply::Lead => self.write_group(),
                Reply::Done(result) => return result,
            }
        }
    }

    fn write_group(&self) {
        let group = {
            let mut queue = self.inner.queue.lock().unwrap();
            let mut group = Vec::new();
            let mut bytes = 0;
            while let Some(request) = queue.pending.front() {
                if !group.is_empty()
                    && (group.len() == MAX_REQUESTS
                        || bytes + request.payload.len() > TARGET_GROUP_BYTES)
                {
                    break;
                }
                bytes += request.payload.len();
                group.push(queue.pending.pop_front().unwrap());
            }
            group
        };
        let mut store = self.inner.store.lock().unwrap();
        commit_group(&mut store, group);
        drop(store);

        let mut queue = self.inner.queue.lock().unwrap();
        if let Some(next) = queue.pending.front() {
            next.reply.send(Reply::Lead).expect("waiting commit leader");
        } else {
            queue.writing = false;
        }
    }

    fn read<T>(&self, read: impl FnOnce(&mut Store) -> Result<T>) -> Result<T> {
        let mut store = self.inner.store.lock().unwrap();
        store.ensure_available()?;
        read(&mut store)
    }

    pub fn get(&self, key: &str) -> Result<Option<Record>> {
        self.read(|store| store.get(key))
    }

    pub fn get_by_id(&self, id: &str) -> Result<Option<Record>> {
        self.read(|store| store.get_by_id(id))
    }

    pub fn retrieve(
        &self,
        queries: &[Query<'_>],
        options: &SearchOptions,
        method: SearchMethod,
    ) -> Result<Vec<Record>> {
        self.read(|store| store.retrieve(queries, options, method))
    }

    #[cfg(feature = "qwen")]
    pub(crate) fn recall(
        &self,
        queries: &[Query<'_>],
        limit: usize,
        scann: &Arc<crate::retrieval::scann::ScannLibrary>,
    ) -> Result<(Vec<Record>, std::time::Duration)> {
        let snapshot = self.read(|store| Ok(Arc::clone(&store.index.retrieval)))?;
        let (ids, vector_time) = snapshot.candidates(queries, limit, scann)?;
        drop(snapshot);
        // Record IDs are immutable and remain readable even across a correction.
        self.read(|store| {
            ids.iter()
                .map(|id| {
                    store
                        .get_by_id(id)?
                        .ok_or_else(|| Error::Corrupt(format!("missing recalled record {id}")))
                })
                .collect::<Result<Vec<_>>>()
                .map(|records| (records, vector_time))
        })
    }

    pub fn data_syncs(&self) -> Result<u64> {
        self.read(|store| Ok(store.data_syncs()))
    }
}

fn commit_group(store: &mut Store, group: Vec<Request>) {
    let mut replies = Vec::with_capacity(group.len());
    for mut request in group {
        let result = store.append_batch(&mut request.payload, false);
        replies.push((request.reply, result));
    }
    // An append or rotation error makes the entire unpublished prefix uncertain.
    // Latch failures before releasing the mutex: no reader may observe that index.
    let mut sync_error = None;
    if !store.recovery_required
        && let Err(error) = store.log.sync_pending()
    {
        store.recovery_required = true;
        sync_error = Some(error);
    }
    for (reply, result) in replies {
        let result = if store.recovery_required && result.is_ok() {
            Err(sync_error.take().unwrap_or(Error::RecoveryRequired))
        } else {
            result
        };
        let _ = reply.send(Reply::Done(result));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Memory, Options};
    use tempfile::tempdir;

    fn put(id: &str, key: &str) -> Mutation {
        Mutation::Put(Memory {
            id: id.into(),
            key: key.into(),
            source: "test".into(),
            body: id.into(),
        })
    }

    fn run_group(store: &mut Store, batches: &[&[Mutation]]) -> Vec<Result<u64>> {
        let mut group = Vec::new();
        let mut receivers = Vec::new();
        for batch in batches {
            let (reply, receiver) = mpsc::channel();
            group.push(Request {
                payload: encode_batch(batch, None).unwrap(),
                reply,
            });
            receivers.push(receiver);
        }
        commit_group(store, group);
        receivers
            .into_iter()
            .map(|receiver| match receiver.recv().unwrap() {
                Reply::Done(result) => result,
                Reply::Lead => unreachable!(),
            })
            .collect()
    }

    #[test]
    fn one_sync_preserves_revisions_and_rejects_only_the_conflicting_request() {
        let dir = tempdir().unwrap();
        let mut store = Store::open(dir.path(), Options::default()).unwrap();
        let replace = Mutation::Replace {
            expected_id: "v1".into(),
            memory: Memory {
                id: "v2".into(),
                key: "k".into(),
                source: "corrected".into(),
                body: "新原文".into(),
            },
        };
        let results = run_group(
            &mut store,
            &[
                &[put("v1", "k")],
                &[put("other", "other"), put("duplicate", "k")],
                &[replace],
                &[put("other", "other")],
            ],
        );
        assert!(matches!(
            results.as_slice(),
            [Ok(1), Err(Error::Conflict(_)), Ok(2), Ok(3)]
        ));
        assert_eq!(store.data_syncs(), 1);
        assert_eq!(store.get("k").unwrap().unwrap().memory.body, "新原文");
        assert_eq!(store.get_at("k", 1).unwrap().unwrap().memory.id, "v1");
        drop(store);
        let mut reopened = Store::open(dir.path(), Options::default()).unwrap();
        assert_eq!(reopened.revision(), 3);
        assert_eq!(reopened.get("k").unwrap().unwrap().memory.id, "v2");
    }

    #[test]
    fn a_group_syncs_each_segment_before_rotation() {
        let dir = tempdir().unwrap();
        let options = Options {
            segment_bytes: 1,
            ..Options::default()
        };
        let mut store = Store::open(dir.path(), options.clone()).unwrap();
        let results = run_group(
            &mut store,
            &[&[put("a", "a")], &[put("b", "b")], &[put("c", "c")]],
        );
        assert!(matches!(results.as_slice(), [Ok(1), Ok(2), Ok(3)]));
        assert_eq!(store.data_syncs(), 3);
        drop(store);
        let mut reopened = Store::open(dir.path(), options).unwrap();
        for id in ["a", "b", "c"] {
            assert_eq!(reopened.get(id).unwrap().unwrap().memory.id, id);
        }
    }

    #[test]
    fn sync_error_withholds_entire_group_and_requires_recovery() {
        for segment_bytes in [1, 65536] {
            let dir = tempdir().unwrap();
            let mut store = Store::open(
                dir.path(),
                Options {
                    segment_bytes,
                    ..Options::default()
                },
            )
            .unwrap();
            store.log.fail_next_sync = true;
            let results = run_group(&mut store, &[&[put("a", "a")], &[put("b", "b")]]);
            assert!(results.iter().all(Result::is_err));
            let shared = store.into_shared();
            assert!(matches!(shared.get("a"), Err(Error::RecoveryRequired)));
            assert!(matches!(
                shared.apply(&[put("c", "c")]),
                Err(Error::RecoveryRequired)
            ));
        }
    }
}
