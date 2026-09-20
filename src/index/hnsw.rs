//! Experimental approximate nearest-neighbour (ANN) index.
//!
//! This is the first slice of the roadmap's HNSW work. It implements a single
//! layer navigable small-world (NSW) graph: nodes are connected to their
//! approximate nearest neighbours at insert time, and queries greedily walk the
//! graph toward the most similar vectors. HNSW adds a probabilistic layer
//! hierarchy on top of exactly this structure, so the search and neighbour
//! selection routines here are the foundation the hierarchy will reuse.
//!
//! Following the index-layer contract (see [`crate::index`]), the index stores
//! only **ids and vectors** — never payloads. It implements
//! [`VectorIndex`](crate::VectorIndex), so it is interchangeable with
//! [`BruteForce`](crate::BruteForce), and [`VectorDb`](crate::VectorDb)
//! resolves the ids it returns back into full records.
//!
//! Results are approximate: with the default parameters recall is high but not
//! guaranteed to match brute force. Raise [`AnnParams::ef_search`] to trade
//! speed for accuracy.
//!
//! ```
//! use light_vector_db::{AnnIndex, AnnParams};
//!
//! let mut index = AnnIndex::with_dimension(3, AnnParams::default()).unwrap();
//! index.insert(1, vec![1.0, 0.0, 0.0]).unwrap();
//! index.insert(2, vec![0.0, 1.0, 0.0]).unwrap();
//! let hits = index.search(&[0.9, 0.1, 0.0], 1).unwrap();
//! assert_eq!(hits[0].0, 1); // (id, score)
//! ```

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use serde::{Deserialize, Serialize};

use crate::index::VectorIndex;
use crate::{VectorDbError, cosine_similarity, validate_numbers};

/// Tuning parameters for an [`AnnIndex`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AnnParams {
    /// Maximum neighbours retained per node (`M` in HNSW terms).
    pub max_neighbors: usize,
    /// Size of the dynamic candidate list kept while inserting.
    pub ef_construction: usize,
    /// Default size of the dynamic candidate list kept while searching.
    pub ef_search: usize,
}

impl Default for AnnParams {
    fn default() -> Self {
        Self {
            max_neighbors: 16,
            ef_construction: 64,
            ef_search: 64,
        }
    }
}

/// One indexed vector and its id.
#[derive(Debug, Clone)]
struct Node {
    id: u64,
    vector: Vec<f32>,
}

/// A scored node used inside the graph's priority queues.
///
/// Ordered by cosine similarity (higher is closer), with the node index as a
/// deterministic tie-breaker so results are stable across runs.
#[derive(Debug, Clone, Copy)]
struct Candidate {
    similarity: f32,
    node: usize,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.node == other.node && self.similarity.total_cmp(&other.similarity).is_eq()
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.similarity
            .total_cmp(&other.similarity)
            .then(self.node.cmp(&other.node))
    }
}

/// An in-memory approximate nearest-neighbour index over `(id, vector)` pairs.
#[derive(Debug, Clone)]
pub struct AnnIndex {
    dimension: Option<usize>,
    params: AnnParams,
    nodes: Vec<Node>,
    neighbors: Vec<Vec<usize>>,
    entry: Option<usize>,
}

impl Default for AnnIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl AnnIndex {
    /// Create an empty index with default parameters. The dimension is fixed by
    /// the first inserted vector.
    pub fn new() -> Self {
        Self::with_params(AnnParams::default())
    }

    /// Create an empty index with the given parameters.
    pub fn with_params(params: AnnParams) -> Self {
        Self {
            dimension: None,
            params,
            nodes: Vec::new(),
            neighbors: Vec::new(),
            entry: None,
        }
    }

    /// Create an empty index that only accepts vectors of `dimension`.
    pub fn with_dimension(dimension: usize, params: AnnParams) -> Result<Self, VectorDbError> {
        if dimension == 0 {
            return Err(VectorDbError::InvalidVector(
                "dimension must be greater than zero",
            ));
        }
        Ok(Self {
            dimension: Some(dimension),
            ..Self::with_params(params)
        })
    }

    /// The fixed vector dimension, once known.
    pub fn dimension(&self) -> Option<usize> {
        self.dimension
    }

    /// Number of indexed vectors.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the index holds no vectors.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// The tuning parameters this index was built with.
    pub fn params(&self) -> AnnParams {
        self.params
    }

    /// Insert a vector under `id`, linking it into the graph near its
    /// approximate nearest neighbours.
    pub fn insert(&mut self, id: u64, vector: Vec<f32>) -> Result<(), VectorDbError> {
        self.validate_dimension(&vector)?;
        let node = self.nodes.len();
        self.nodes.push(Node { id, vector });
        self.neighbors.push(Vec::new());

        if self.entry.is_none() {
            self.entry = Some(node);
            return Ok(());
        }

        let vector = self.nodes[node].vector.clone();
        let m = self.params.max_neighbors;
        let candidates = self.search_layer(&vector, self.params.ef_construction.max(m));
        let selected: Vec<usize> = candidates
            .into_iter()
            .map(|candidate| candidate.node)
            .filter(|&other| other != node)
            .take(m)
            .collect();

        self.neighbors[node] = selected.clone();
        for neighbor in selected {
            self.neighbors[neighbor].push(node);
            if self.neighbors[neighbor].len() > m {
                self.prune(neighbor);
            }
        }
        Ok(())
    }

    /// Search for the `k` most similar vectors using the default `ef_search`,
    /// returning `(id, score)` pairs sorted by descending score.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u64, f32)>, VectorDbError> {
        self.search_with_ef(query, k, self.params.ef_search)
    }

    /// Search with an explicit candidate-list size. Larger `ef` improves recall
    /// at the cost of speed; it is clamped up to at least `k`.
    pub fn search_with_ef(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
    ) -> Result<Vec<(u64, f32)>, VectorDbError> {
        self.validate_query(query)?;
        if k == 0 || self.entry.is_none() {
            return Ok(Vec::new());
        }
        let mut found = self.search_layer(query, ef.max(k));
        found.truncate(k);
        Ok(found
            .into_iter()
            .map(|candidate| (self.nodes[candidate.node].id, candidate.similarity))
            .collect())
    }

    /// Greedy best-first walk of the graph, returning up to `ef` closest nodes
    /// sorted by descending similarity.
    fn search_layer(&self, query: &[f32], ef: usize) -> Vec<Candidate> {
        let Some(entry) = self.entry else {
            return Vec::new();
        };
        let ef = ef.max(1);
        let mut visited = vec![false; self.nodes.len()];
        // `to_explore` is a max-heap: always expand the most similar candidate.
        let mut to_explore: BinaryHeap<Candidate> = BinaryHeap::new();
        // `results` is a min-heap (via `Reverse`): the least similar kept node
        // sits on top so it is cheap to evict once we exceed `ef`.
        let mut results: BinaryHeap<Reverse<Candidate>> = BinaryHeap::new();

        let start = Candidate {
            similarity: cosine_similarity(query, &self.nodes[entry].vector),
            node: entry,
        };
        visited[entry] = true;
        to_explore.push(start);
        results.push(Reverse(start));

        while let Some(current) = to_explore.pop() {
            let worst = results.peek().map_or(f32::NEG_INFINITY, |r| r.0.similarity);
            if results.len() >= ef && current.similarity < worst {
                break;
            }
            for index in 0..self.neighbors[current.node].len() {
                let neighbor = self.neighbors[current.node][index];
                if visited[neighbor] {
                    continue;
                }
                visited[neighbor] = true;
                let candidate = Candidate {
                    similarity: cosine_similarity(query, &self.nodes[neighbor].vector),
                    node: neighbor,
                };
                let worst = results.peek().map_or(f32::NEG_INFINITY, |r| r.0.similarity);
                if results.len() < ef || candidate.similarity > worst {
                    to_explore.push(candidate);
                    results.push(Reverse(candidate));
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        let mut out: Vec<Candidate> = results.into_iter().map(|entry| entry.0).collect();
        out.sort_by(|a, b| b.cmp(a));
        out
    }

    /// Keep only the `max_neighbors` closest neighbours of `node`.
    fn prune(&mut self, node: usize) {
        let base = self.nodes[node].vector.clone();
        let mut neighbors = std::mem::take(&mut self.neighbors[node]);
        neighbors.sort_by(|&a, &b| {
            cosine_similarity(&base, &self.nodes[b].vector)
                .total_cmp(&cosine_similarity(&base, &self.nodes[a].vector))
                .then(a.cmp(&b))
        });
        neighbors.dedup();
        neighbors.truncate(self.params.max_neighbors);
        self.neighbors[node] = neighbors;
    }

    fn validate_dimension(&mut self, vector: &[f32]) -> Result<(), VectorDbError> {
        validate_numbers(vector)?;
        match self.dimension {
            Some(expected) if expected != vector.len() => Err(VectorDbError::DimensionMismatch {
                expected,
                actual: vector.len(),
            }),
            Some(_) => Ok(()),
            None => {
                self.dimension = Some(vector.len());
                Ok(())
            }
        }
    }

    fn validate_query(&self, vector: &[f32]) -> Result<(), VectorDbError> {
        validate_numbers(vector)?;
        if let Some(expected) = self.dimension
            && vector.len() != expected
        {
            return Err(VectorDbError::DimensionMismatch {
                expected,
                actual: vector.len(),
            });
        }
        Ok(())
    }
}

impl VectorIndex for AnnIndex {
    fn insert(&mut self, id: u64, vector: Vec<f32>) -> Result<(), VectorDbError> {
        AnnIndex::insert(self, id, vector)
    }

    fn search(&self, query: &[f32], k: usize, ef: usize) -> Result<Vec<(u64, f32)>, VectorDbError> {
        self.search_with_ef(query, k, ef)
    }

    fn len(&self) -> usize {
        AnnIndex::len(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random vector generator (splitmix64), no rng dep.
    fn sample_vector(seed: u64, dimension: usize) -> Vec<f32> {
        let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        (0..dimension)
            .map(|_| {
                state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                (z >> 40) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
            })
            .collect()
    }

    fn brute_force_top1(entries: &[(u64, Vec<f32>)], query: &[f32]) -> u64 {
        entries
            .iter()
            .map(|(id, vector)| (cosine_similarity(query, vector), *id))
            .max_by(|a, b| a.0.total_cmp(&b.0))
            .map(|(_, id)| id)
            .unwrap()
    }

    #[test]
    fn empty_index_returns_no_hits() {
        let index = AnnIndex::new();
        assert!(index.is_empty());
        assert!(index.search(&[1.0, 0.0], 5).unwrap().is_empty());
    }

    #[test]
    fn zero_k_returns_no_hits() {
        let mut index = AnnIndex::new();
        index.insert(1, vec![1.0, 0.0]).unwrap();
        assert!(index.search(&[1.0, 0.0], 0).unwrap().is_empty());
    }

    #[test]
    fn dimension_is_inferred_then_enforced() {
        let mut index = AnnIndex::new();
        index.insert(1, vec![1.0, 0.0]).unwrap();
        assert_eq!(index.dimension(), Some(2));
        assert!(matches!(
            index.insert(2, vec![1.0, 0.0, 0.0]),
            Err(VectorDbError::DimensionMismatch {
                expected: 2,
                actual: 3
            })
        ));
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn rejects_empty_and_non_finite_vectors() {
        let mut index = AnnIndex::with_dimension(2, AnnParams::default()).unwrap();
        assert!(matches!(
            index.insert(1, vec![]),
            Err(VectorDbError::InvalidVector(_))
        ));
        assert!(matches!(
            index.insert(1, vec![f32::NAN, 1.0]),
            Err(VectorDbError::InvalidVector(_))
        ));
        assert!(matches!(
            index.search(&[f32::INFINITY, 0.0], 1),
            Err(VectorDbError::InvalidVector(_))
        ));
    }

    #[test]
    fn search_rejects_dimension_mismatch() {
        let mut index = AnnIndex::with_dimension(3, AnnParams::default()).unwrap();
        index.insert(1, vec![1.0, 0.0, 0.0]).unwrap();
        assert!(matches!(
            index.search(&[1.0, 0.0], 1),
            Err(VectorDbError::DimensionMismatch {
                expected: 3,
                actual: 2
            })
        ));
    }

    #[test]
    fn finds_true_nearest_on_separated_points() {
        // Well-separated one-hot vectors: the nearest neighbour is unambiguous.
        let mut index = AnnIndex::with_dimension(4, AnnParams::default()).unwrap();
        index.insert(1, vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        index.insert(2, vec![0.0, 1.0, 0.0, 0.0]).unwrap();
        index.insert(3, vec![0.0, 0.0, 1.0, 0.0]).unwrap();
        index.insert(4, vec![0.0, 0.0, 0.0, 1.0]).unwrap();

        let hits = index.search(&[0.1, 0.0, 0.9, 0.05], 2).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].0, 3);
        assert!(hits[0].1 >= hits[1].1);
    }

    #[test]
    fn matches_brute_force_top1_with_high_ef() {
        // With ef >= number of nodes the greedy walk explores the whole
        // connected graph, so recall should match exact search.
        let dimension = 16;
        let count = 60;
        let entries: Vec<(u64, Vec<f32>)> = (0..count)
            .map(|i| (i as u64, sample_vector(i as u64, dimension)))
            .collect();

        let params = AnnParams {
            max_neighbors: 8,
            ef_construction: 64,
            ef_search: count,
        };
        let mut index = AnnIndex::with_dimension(dimension, params).unwrap();
        for (id, vector) in &entries {
            index.insert(*id, vector.clone()).unwrap();
        }
        assert_eq!(index.len(), count);

        for q in 0..10 {
            let query = sample_vector(1_000 + q as u64, dimension);
            let expected = brute_force_top1(&entries, &query);
            let hits = index.search(&query, 1).unwrap();
            assert_eq!(hits[0].0, expected, "query {q} mismatch");
        }
    }
}
