//! Immutable ScaNN searcher owned by one generation of the Rust index.
use std::collections::BTreeMap;
use std::ffi::{CStr, c_char, c_void};
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::Arc;

use super::Document;
use crate::{Error, Result};
use libloading::Library;

type Build =
    unsafe extern "C" fn(*const f32, u32, u32, u32, u32, u32, *mut c_char, usize) -> *mut c_void;
type Search =
    unsafe extern "C" fn(*const c_void, *const f32, u32, i32, *mut u32, *mut c_char, usize) -> i32;
type Destroy = unsafe extern "C" fn(*mut c_void);

/// Each query variant independently retrieves up to these limits before ID deduplication.
#[derive(Clone, Copy, Debug)]
pub struct RecallOptions {
    pub vector_limit: usize,
    pub keyword_limit: usize,
    /// Use exact dot products through this many active vectors, inclusive.
    pub exact_threshold: u32,
    /// None probes half the partitions (rounded up). Values above the leaf count probe all.
    pub partitions_to_search: Option<u32>,
}

impl Default for RecallOptions {
    fn default() -> Self {
        Self {
            vector_limit: 32,
            keyword_limit: 32,
            exact_threshold: 64,
            partitions_to_search: None,
        }
    }
}

pub(crate) struct ScannLibrary {
    pub(super) options: RecallOptions,
    build: Build,
    search: Search,
    destroy: Destroy,
    _library: Library,
}

impl ScannLibrary {
    pub(crate) fn default_path() -> PathBuf {
        std::env::var_os("MEMORYH_SCANN_LIB")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                Path::new("target/scann-eval/deps/build")
                    .join(libloading::library_filename("scann_core"))
            })
    }

    pub(crate) fn load(path: &Path, options: RecallOptions) -> Result<Arc<Self>> {
        if !(1..=i32::MAX as usize).contains(&options.vector_limit)
            || options.keyword_limit == 0
            || options.partitions_to_search == Some(0)
        {
            return Err(scann_error("invalid recall limits or partition count"));
        }
        // The pinned bridge exports these three versioned C signatures.
        unsafe {
            let lib = Library::new(
                path.canonicalize()
                    .map_err(|err| scann_error(format!("{}: {err}", path.display())))?,
            )
            .map_err(scann_error)?;
            Ok(Arc::new(Self {
                options,
                build: *lib.get(b"memoryh_scann_build_v2").map_err(scann_error)?,
                search: *lib.get(b"memoryh_scann_search_v1").map_err(scann_error)?,
                destroy: *lib.get(b"memoryh_scann_drop_v1").map_err(scann_error)?,
                _library: lib,
            }))
        }
    }
}

fn scann_error(error: impl std::fmt::Display) -> Error {
    Error::InvalidQuery(format!("ScaNN: {error}"))
}

pub(super) struct ScannVectors {
    lib: Arc<ScannLibrary>,
    index: NonNull<c_void>,
    ids: Vec<String>,
    dim: u32,
}

// ScannInterface supports concurrent const Search. Rows and the handle are never mutated
// after publication, and the snapshot outlives all searches.
unsafe impl Send for ScannVectors {}
unsafe impl Sync for ScannVectors {}

impl ScannVectors {
    pub(super) fn build(
        library: &Arc<ScannLibrary>,
        docs: &BTreeMap<String, Arc<Document>>,
        dim: usize,
    ) -> Result<Option<Self>> {
        let mut ids = Vec::new();
        let mut rows = Vec::new();
        for (id, doc) in docs {
            if let Some(v) = &doc.embedding {
                if v.len() != dim || dim == 0 {
                    return Err(scann_error("inconsistent document vector dimensions"));
                }
                ids.push(id.clone());
                rows.extend_from_slice(v);
            }
        }
        if ids.is_empty() {
            return Ok(None);
        }
        let count = u32::try_from(ids.len()).map_err(scann_error)?;
        let dim = u32::try_from(dim).map_err(scann_error)?;
        let mut error = [0; 512];
        // Initialize copies the dense rows into the owned native index.
        let raw = unsafe {
            (library.build)(
                rows.as_ptr(),
                count,
                dim,
                library.options.vector_limit as u32,
                library.options.exact_threshold,
                library.options.partitions_to_search.unwrap_or(0),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        let index = NonNull::new(raw).ok_or_else(|| {
            scann_error(unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy())
        })?;
        Ok(Some(Self {
            lib: library.clone(),
            index,
            ids,
            dim,
        }))
    }

    pub(super) fn search(&self, vector: &[f32], limit: usize) -> Result<Vec<String>> {
        if vector.len() != self.dim as usize {
            return Err(scann_error("query dimension mismatch"));
        }
        let k = i32::try_from(limit.min(self.ids.len())).map_err(scann_error)?;
        if k == 0 {
            return Ok(Vec::new());
        }
        let mut rows = vec![0; k as usize];
        let mut error = [0; 512];
        let n = unsafe {
            (self.lib.search)(
                self.index.as_ptr(),
                vector.as_ptr(),
                self.dim,
                k,
                rows.as_mut_ptr(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if n < 0 || n > k {
            return Err(scann_error(
                unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy(),
            ));
        }
        rows[..n as usize]
            .iter()
            .map(|&i| {
                self.ids
                    .get(i as usize)
                    .cloned()
                    .ok_or_else(|| scann_error("invalid result row"))
            })
            .collect()
    }
}

impl Drop for ScannVectors {
    fn drop(&mut self) {
        unsafe { (self.lib.destroy)(self.index.as_ptr()) };
    }
}
