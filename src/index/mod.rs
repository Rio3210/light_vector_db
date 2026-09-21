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

pub(crate) use brute_force::BruteForce;

use crate::VectorDbError;

/// A persistable snapshot of a graph index's *structure* — not its vectors.
///
/// Vectors already live in the file's records, so the storage layer only needs
/// the graph: which id each node holds, each node's neighbour list (as node
/// indices), and the entry node. This lets a graph index be saved and reloaded
/// without rebuilding it from scratch.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct IndexData {
    /// Node index → record id (in the index's own node order).
    pub node_ids: Vec<u64>,
    /// Node index → its neighbours' node indices.
    pub neighbors: Vec<Vec<u32>>,
    /// The entry node index, if the graph is non-empty.
    pub entry: Option<u32>,
}

/// A nearest-neighbour index over `(id, vector)` pairs.
///
/// Implementations must agree on scoring (higher score = more similar) so the
/// query layer can treat them interchangeably. `Debug` is required so a
/// [`VectorDb`](crate::VectorDb) holding a `Box<dyn VectorIndex>` stays
/// printable.
pub(crate) trait VectorIndex: std::fmt::Debug {
    /// Add a vector under `id`. Returns an error for empty, non-finite, or
    /// wrong-dimension vectors.
    fn insert(&mut self, id: u64, vector: Vec<f32>) -> Result<(), VectorDbError>;

    /// Return up to `k` ids most similar to `query`, each with its score,
    /// sorted by descending score.
    ///
    /// `ef` is the search-effort hint used by graph indexes (larger = higher
    /// recall, slower); exact indexes ignore it.
    fn search(&self, query: &[f32], k: usize, ef: usize) -> Result<Vec<(u64, f32)>, VectorDbError>;

    /// A snapshot of the index's graph for persistence, if it has one worth
    /// storing. Exact indexes have nothing to persist and return `None` (the
    /// default), so they are simply rebuilt on load.
    fn persist(&self) -> Option<IndexData> {
        None
    }
}
