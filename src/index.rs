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
//! - [`AnnIndex`](crate::AnnIndex) — an approximate NSW graph for large
//!   collections (see [`crate::ann`]).

use crate::{VectorDbError, cosine_similarity, validate_numbers};

/// A nearest-neighbour index over `(id, vector)` pairs.
///
/// Implementations must agree on scoring (higher score = more similar) so the
/// query layer can treat them interchangeably.
pub trait VectorIndex {
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

/// An exact index: every query is scored against every stored vector.
///
/// O(n · d) per query, but with no graph to traverse it beats approximate
/// indexes on small collections and is always exactly correct.
#[derive(Debug, Clone, Default)]
pub struct BruteForce {
    dimension: Option<usize>,
    entries: Vec<(u64, Vec<f32>)>,
}

impl BruteForce {
    /// Create an empty index. The dimension is fixed by the first insert.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an empty index that only accepts vectors of `dimension`.
    pub fn with_dimension(dimension: usize) -> Result<Self, VectorDbError> {
        if dimension == 0 {
            return Err(VectorDbError::InvalidVector(
                "dimension must be greater than zero",
            ));
        }
        Ok(Self {
            dimension: Some(dimension),
            entries: Vec::new(),
        })
    }

    /// The fixed vector dimension, once known.
    pub fn dimension(&self) -> Option<usize> {
        self.dimension
    }

    fn check_dimension(&mut self, vector: &[f32]) -> Result<(), VectorDbError> {
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

    fn check_query(&self, query: &[f32]) -> Result<(), VectorDbError> {
        validate_numbers(query)?;
        if let Some(expected) = self.dimension
            && query.len() != expected
        {
            return Err(VectorDbError::DimensionMismatch {
                expected,
                actual: query.len(),
            });
        }
        Ok(())
    }
}

impl VectorIndex for BruteForce {
    fn insert(&mut self, id: u64, vector: Vec<f32>) -> Result<(), VectorDbError> {
        self.check_dimension(&vector)?;
        self.entries.push((id, vector));
        Ok(())
    }

    fn search(
        &self,
        query: &[f32],
        k: usize,
        _ef: usize,
    ) -> Result<Vec<(u64, f32)>, VectorDbError> {
        self.check_query(query)?;
        if k == 0 {
            return Ok(Vec::new());
        }
        let mut scored: Vec<(u64, f32)> = self
            .entries
            .iter()
            .map(|(id, vector)| (*id, cosine_similarity(query, vector)))
            .collect();
        // Sort by descending score, breaking ties by id for determinism.
        scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        scored.truncate(k);
        Ok(scored)
    }

    fn len(&self) -> usize {
        self.entries.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AnnIndex;

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

    #[test]
    fn brute_force_is_exact_and_sorted() {
        let mut index = BruteForce::with_dimension(4).unwrap();
        index.insert(1, vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        index.insert(2, vec![0.0, 1.0, 0.0, 0.0]).unwrap();
        index.insert(3, vec![0.0, 0.0, 1.0, 0.0]).unwrap();

        let hits = index.search(&[0.1, 0.0, 0.9, 0.0], 2, 0).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].0, 3);
        assert!(hits[0].1 >= hits[1].1);
    }

    #[test]
    fn brute_force_rejects_bad_vectors() {
        let mut index = BruteForce::with_dimension(2).unwrap();
        assert!(matches!(
            index.insert(1, vec![]),
            Err(VectorDbError::InvalidVector(_))
        ));
        assert!(matches!(
            index.insert(1, vec![f32::NAN, 0.0]),
            Err(VectorDbError::InvalidVector(_))
        ));
        assert!(matches!(
            index.insert(1, vec![1.0, 2.0, 3.0]),
            Err(VectorDbError::DimensionMismatch { .. })
        ));
    }

    /// The two indexes must agree on the top result: with a high `ef` the graph
    /// walk is effectively exhaustive, so HNSW should match brute force.
    #[test]
    fn brute_force_and_hnsw_agree_on_top1() {
        let dimension = 16;
        let count = 60usize;

        let mut exact = BruteForce::with_dimension(dimension).unwrap();
        let mut approx = AnnIndex::with_dimension(dimension, Default::default()).unwrap();
        for i in 0..count {
            let v = sample_vector(i as u64, dimension);
            exact.insert(i as u64, v.clone()).unwrap();
            approx.insert(i as u64, v).unwrap();
        }

        for q in 0..10 {
            let query = sample_vector(1_000 + q as u64, dimension);
            let truth = exact.search(&query, 1, 0).unwrap();
            let got = approx.search_with_ef(&query, 1, count).unwrap();
            assert_eq!(got[0].0, truth[0].0, "query {q} disagreed");
        }
    }
}
