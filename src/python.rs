use std::path::PathBuf;

use pyo3::exceptions::{PyOSError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::{
    Error, Memory, Mutation, Options, PrefixLeaf, Query, Ranking, SearchOptions, Store, TreeLayout,
};

type Write = (String, String, String, String, Option<String>);
type PrefixLayout = Vec<Vec<(String, String)>>;
type PyRanking = (
    Vec<String>,
    Vec<String>,
    Vec<String>,
    Vec<String>,
    Vec<String>,
    Vec<String>,
);
type PySearch = (
    PyRanking,
    Vec<PyRanking>,
    Vec<String>,
    Vec<String>,
    Vec<String>,
    Vec<String>,
);

#[pyclass(name = "Store")]
struct PyStore(Option<Store>);

#[pymethods]
impl PyStore {
    #[new]
    #[pyo3(signature = (path, cache_bytes=None, *, uncached_disk=false, block_cache_bytes=0))]
    fn new(
        path: PathBuf,
        cache_bytes: Option<usize>,
        uncached_disk: bool,
        block_cache_bytes: usize,
    ) -> PyResult<Self> {
        let mut options = Options::default();
        if let Some(bytes) = cache_bytes {
            options.cache_bytes = bytes;
        }
        let mut store =
            Store::open_with_block_cache(path, options, block_cache_bytes).map_err(python_error)?;
        store
            .set_uncached_reads(uncached_disk)
            .map_err(python_error)?;
        Ok(Self(Some(store)))
    }

    fn close(&mut self) {
        self.0 = None;
    }

    #[pyo3(signature = (records, embeddings=None))]
    fn apply(
        &mut self,
        records: Vec<Write>,
        embeddings: Option<Vec<Option<Vec<f32>>>>,
    ) -> PyResult<u64> {
        let mutations = records
            .into_iter()
            .map(|(id, key, source, body, supersedes)| {
                let memory = Memory {
                    id,
                    key,
                    source,
                    body,
                };
                match supersedes {
                    Some(expected_id) => Mutation::Replace {
                        expected_id,
                        memory,
                    },
                    None => Mutation::Put(memory),
                }
            })
            .collect::<Vec<_>>();
        let store = self.store()?;
        match embeddings {
            Some(vectors) => {
                let vectors: Vec<_> = vectors.iter().map(Option::as_deref).collect();
                store.apply_with_embeddings(&mutations, &vectors)
            }
            None => store.apply(&mutations),
        }
        .map_err(python_error)
    }

    #[pyo3(signature = (queries, top_k, beam_width, prefixes=None))]
    fn search(
        &mut self,
        queries: Vec<(String, Option<Vec<f32>>)>,
        top_k: usize,
        beam_width: usize,
        prefixes: Option<PrefixLayout>,
    ) -> PyResult<PySearch> {
        let queries: Vec<_> = queries
            .iter()
            .map(|(text, vector)| Query {
                text,
                embedding: vector.as_deref(),
            })
            .collect();
        let tree = match prefixes {
            Some(branches) => TreeLayout::Prefixes(
                branches
                    .into_iter()
                    .map(|leaves| {
                        leaves
                            .into_iter()
                            .map(|(label, key_prefix)| PrefixLeaf { label, key_prefix })
                            .collect()
                    })
                    .collect(),
            ),
            None => TreeLayout::Semantic,
        };
        let result = self
            .store()?
            .search(
                &queries,
                &SearchOptions {
                    top_k,
                    beam_width,
                    tree,
                },
            )
            .map_err(python_error)?;
        Ok((
            ranking(result.whole),
            result.parts.into_iter().map(ranking).collect(),
            result.flat_split,
            result.tree_split,
            result.flat_vector_first,
            result.tree_vector_first,
        ))
    }

    fn visible_keys(&mut self) -> PyResult<Vec<(String, String)>> {
        let store = self.store()?;
        Ok(store
            .visible_keys(store.revision())
            .map_err(python_error)?
            .map(|(key, id)| (id.to_owned(), key.to_owned()))
            .collect())
    }

    fn read(&mut self, id: &str) -> PyResult<Option<(String, String)>> {
        Ok(self
            .store()?
            .get_by_id(id)
            .map_err(python_error)?
            .map(|record| (record.memory.source, record.memory.body)))
    }

    fn cache_stats(&mut self) -> PyResult<(u64, u64, u64)> {
        let stats = self.store()?.cache_stats();
        Ok((stats.hits, stats.misses, stats.disk_reads))
    }

    fn block_cache_stats(&mut self) -> PyResult<(u64, u64)> {
        let stats = self.store()?.cache_stats();
        Ok((stats.block_hits, stats.block_reads))
    }
}

impl PyStore {
    fn store(&mut self) -> PyResult<&mut Store> {
        self.0
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("MemoryH store is closed"))
    }
}

fn python_error(error: Error) -> PyErr {
    match error {
        Error::Conflict(_) | Error::InvalidQuery(_) | Error::BatchTooLarge => {
            PyValueError::new_err(error.to_string())
        }
        Error::Io(_) => PyOSError::new_err(error.to_string()),
        _ => PyRuntimeError::new_err(error.to_string()),
    }
}

fn ranking(result: Ranking) -> PyRanking {
    (
        result.lexical,
        result.flat,
        result.tree,
        result.flat_vector,
        result.tree_vector,
        result.routes,
    )
}

#[pyfunction]
fn query_variants(question: &str) -> Vec<String> {
    crate::query_variants(question)
        .into_iter()
        .map(str::to_owned)
        .collect()
}

#[pymodule]
fn memoryh_store(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyStore>()?;
    module.add_function(wrap_pyfunction!(query_variants, module)?)?;
    Ok(())
}
