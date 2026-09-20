//! The index layer: one trait, interchangeable implementations.
//!
//! An index answers "which stored vectors are nearest this query?" — nothing
//! more. It owns only **ids and geometry**, never the payloads (text/metadata);
//! [`VectorDb`](crate::VectorDb) keeps those and resolves the ids an index
//! returns back into full [`Record`](crate::Record)s. That separation is what
//! lets the storage engine persist and memory-map vectors independently of the
//! variable-length payloads.
//!
//! Two implementations live behind [`VectorIndex`]:
//!
//! - [`BruteForce`] — exact linear scan. Simple, always correct, and the
//!   fastest choice below a few thousand vectors.
//! - [`AnnIndex`](hnsw::AnnIndex) — an approximate NSW graph for large
//!   collections (see [`hnsw`]).

mod brute_force;
pub mod hnsw;

pub use brute_force::BruteForce;

use crate::VectorDbError;

/// A nearest-neighbour index over `(id, vector)` pairs.
///
/// Implementations must agree on scoring (higher score = more similar) so the
/// query layer can treat them interchangeably. `Debug` is required so a
/// [`VectorDb`](crate::VectorDb) holding a `Box<dyn VectorIndex>` stays
/// printable.
pub trait VectorIndex: std::fmt::Debug {
    /// Add a vector under `id`. Returns an error for empty, non-finite, or
    /// wrong-dimension vectors.
    fn insert(&mut self, id: u64, vector: Vec<f32>) -> Result<(), VectorDbError>;

    /// Return up to `k` ids most similar to `query`, each with its score,
    /// sorted by descending score.
    ///
    /// `ef` is the search-effort hint used by graph indexes (larger = higher
    /// recall, slower); exact indexes ignore it.
    fn search(&self, query: &[f32], k: usize, ef: usize) -> Result<Vec<(u64, f32)>, VectorDbError>;

    /// Number of indexed vectors.
    fn len(&self) -> usize;

    /// Whether the index holds no vectors.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
