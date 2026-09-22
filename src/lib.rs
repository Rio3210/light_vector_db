//! A local-first vector database library.
//!
//! `light_vector_db` intentionally does not generate embeddings. Your
//! application supplies a `Vec<f32>` from any embedding model; this crate
//! validates, stores, indexes, persists, and searches those vectors.
//!
//! The code is organised in layers (see `docs/ARCHITECTURE.md`):
//!
//! - `record` / `error` — the data types and error type.
//! - `distance` — similarity metrics (cosine today).
//! - `index` — the `VectorIndex` trait with `BruteForce` and HNSW impls.
//! - `storage` — the `.lvdb` binary file format.
//! - `db` — [`VectorDb`], which ties the layers together.
//! - `mmap` — [`MmapDb`], read-only memory-mapped search over a `.lvdb` file.

mod db;
mod distance;
mod error;
mod index;
mod mmap;
mod record;
mod storage;

pub use db::{IndexKind, VectorDb};
pub use distance::cosine_similarity;
pub use error::VectorDbError;
pub use index::hnsw::{AnnIndex, AnnParams};
pub use mmap::MmapDb;
pub use record::{Metadata, Record, SearchResult};

// Crate-internal helper, reachable as `crate::validate_numbers` from submodules.
pub(crate) use record::validate_numbers;
