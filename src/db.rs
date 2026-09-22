//! The `VectorDb` collection: records, a pluggable index, and persistence.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::index::hnsw::AnnIndex;
use crate::index::{BruteForce, IndexData, VectorIndex};
use crate::storage;
use crate::{
    AnnParams, Metadata, Record, SearchResult, VectorDbError, cosine_similarity, validate_numbers,
};

/// How a [`VectorDb`] indexes vectors for search.
///
/// The kind is chosen at construction and persisted with the database.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum IndexKind {
    /// Exact brute-force scan. Always correct; the best choice for small
    /// collections and the default.
    #[default]
    Exact,
    /// Approximate NSW-graph search. Faster on large collections, but results
    /// are approximate — raise [`AnnParams::ef_search`] for higher recall.
    Hnsw(AnnParams),
}

/// How vectors are encoded in the `.lvdb` file.
///
/// This is a storage concern only: vectors are always `f32` in memory. A
/// quantized file is smaller on disk and dequantized back to `f32` on load
/// (lossily). JSON export is always `Float32`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum Encoding {
    /// Full-precision little-endian `f32` (4 bytes per component).
    #[default]
    Float32,
    /// Scalar quantization: each component stored as one byte against a global
    /// min/max range. 4× smaller, with a small, bounded precision loss.
    ScalarU8,
}

/// A local-first collection of vectors with metadata and similarity search.
///
/// Records (the payloads) are the source of truth; the [`VectorIndex`] holds
/// only ids and vectors and is rebuilt from the records when needed, so it is
/// never serialized.
#[derive(Debug)]
pub struct VectorDb {
    dimension: Option<usize>,
    records: BTreeMap<u64, Record>,
    index_kind: IndexKind,
    index: Box<dyn VectorIndex>,
    /// Ids deleted since the last rebuild. Their nodes still sit in `index`
    /// (search skips them); `compact` reclaims them. Empty after any rebuild.
    tombstones: BTreeSet<u64>,
    /// How vectors are written to the `.lvdb` file (in-memory is always `f32`).
    encoding: Encoding,
}

/// Borrowed view of a [`VectorDb`] for serialization (no clone of records).
#[derive(Serialize)]
struct DbSnapshot<'a> {
    dimension: Option<usize>,
    records: &'a BTreeMap<u64, Record>,
    index_kind: IndexKind,
}

/// Owned form read back during deserialization. Missing `index_kind` in older
/// files defaults to [`IndexKind::Exact`].
#[derive(Deserialize)]
struct OwnedDbSnapshot {
    dimension: Option<usize>,
    #[serde(default)]
    records: BTreeMap<u64, Record>,
    #[serde(default)]
    index_kind: IndexKind,
}

impl Default for VectorDb {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for VectorDb {
    fn clone(&self) -> Self {
        let mut cloned = Self::empty(None, self.index_kind).expect("no dimension cannot fail");
        cloned.dimension = self.dimension;
        cloned.records = self.records.clone();
        cloned.encoding = self.encoding;
        cloned
            .rebuild_index()
            .expect("existing records are already valid");
        cloned
    }
}

/// Build an empty index of the requested kind for a (possibly unknown) dimension.
fn new_index(
    kind: IndexKind,
    dimension: Option<usize>,
) -> Result<Box<dyn VectorIndex>, VectorDbError> {
    Ok(match (kind, dimension) {
        (IndexKind::Exact, Some(d)) => Box::new(BruteForce::with_dimension(d)?),
        (IndexKind::Exact, None) => Box::new(BruteForce::new()),
        (IndexKind::Hnsw(params), Some(d)) => Box::new(AnnIndex::with_dimension(d, params)?),
        (IndexKind::Hnsw(params), None) => Box::new(AnnIndex::with_params(params)),
    })
}

impl VectorDb {
    /// Create an empty, exact-search collection. The dimension is fixed by the
    /// first inserted vector.
    pub fn new() -> Self {
        Self::empty(None, IndexKind::Exact).expect("no dimension cannot fail")
    }

    /// Create an empty, exact-search collection fixed to `dimension`.
    pub fn with_dimension(dimension: usize) -> Result<Self, VectorDbError> {
        Self::with_index(dimension, IndexKind::Exact)
    }

    /// Create an empty collection fixed to `dimension` with a chosen index kind.
    pub fn with_index(dimension: usize, index_kind: IndexKind) -> Result<Self, VectorDbError> {
        if dimension == 0 {
            return Err(VectorDbError::InvalidVector(
                "dimension must be greater than zero",
            ));
        }
        Self::empty(Some(dimension), index_kind)
    }

    /// Create an empty collection with a chosen index kind and an unfixed
    /// dimension (set by the first inserted vector).
    pub fn with_index_kind(index_kind: IndexKind) -> Self {
        Self::empty(None, index_kind).expect("no dimension cannot fail")
    }

    /// The index kind backing this collection's search.
    pub fn index_kind(&self) -> IndexKind {
        self.index_kind
    }

    fn empty(dimension: Option<usize>, index_kind: IndexKind) -> Result<Self, VectorDbError> {
        Ok(Self {
            dimension,
            records: BTreeMap::new(),
            index_kind,
            index: new_index(index_kind, dimension)?,
            tombstones: BTreeSet::new(),
            encoding: Encoding::default(),
        })
    }

    /// The on-disk vector encoding for this database.
    pub fn encoding(&self) -> Encoding {
        self.encoding
    }

    /// Choose how vectors are written to the `.lvdb` file. In-memory vectors
    /// stay `f32`; this only affects [`save_to_path`](Self::save_to_path).
    pub fn set_encoding(&mut self, encoding: Encoding) {
        self.encoding = encoding;
    }

    pub fn dimension(&self) -> Option<usize> {
        self.dimension
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub fn insert(&mut self, record: Record) -> Result<(), VectorDbError> {
        if self.records.contains_key(&record.id) {
            return Err(VectorDbError::DuplicateId(record.id));
        }
        self.validate_vector(&record.vector)?;
        let id = record.id;
        if self.tombstones.contains(&id) {
            // Reusing a just-deleted id: the index still holds a stale node for
            // it, so rebuild to drop it rather than duplicate the id.
            self.records.insert(id, record);
            self.rebuild_index()?;
        } else {
            self.index.insert(id, record.vector.clone())?;
            self.records.insert(id, record);
        }
        Ok(())
    }

    pub fn upsert(&mut self, record: Record) -> Result<(), VectorDbError> {
        self.validate_vector(&record.vector)?;
        let id = record.id;
        let was_tombstoned = self.tombstones.contains(&id);
        let replaces_existing = self.records.insert(id, record).is_some();
        if replaces_existing || was_tombstoned {
            // Replacing a live record, or reusing a deleted id, leaves a stale
            // node in the index; rebuild to drop it.
            self.rebuild_index()?;
        } else {
            let vector = self.records[&id].vector.clone();
            self.index.insert(id, vector)?;
        }
        Ok(())
    }

    pub fn get(&self, id: u64) -> Option<&Record> {
        self.records.get(&id)
    }

    pub fn delete(&mut self, id: u64) -> Result<Record, VectorDbError> {
        let removed = self
            .records
            .remove(&id)
            .ok_or(VectorDbError::NotFound(id))?;
        // Tombstone instead of rebuilding: the node stays in the index but is
        // skipped in results (its id is no longer in `records`), and `compact`
        // reclaims it later. Deletes stay O(log n).
        self.tombstones.insert(id);
        Ok(removed)
    }

    /// Reclaim deleted entries by rebuilding the index without the tombstoned
    /// nodes. Returns how many tombstones were reclaimed.
    ///
    /// Deletes are cheap (they only tombstone); `compact` pays the one-time
    /// rebuild. It also lets [`save_to_path`](Self::save_to_path) persist the
    /// graph again (a database with pending tombstones saves without one).
    pub fn compact(&mut self) -> Result<usize, VectorDbError> {
        let reclaimed = self.tombstones.len();
        if reclaimed > 0 {
            self.rebuild_index()?;
        }
        Ok(reclaimed)
    }

    /// Rebuild the search index from the records (the source of truth). Any
    /// tombstones are dropped, since the fresh index holds only live records.
    fn rebuild_index(&mut self) -> Result<(), VectorDbError> {
        let mut index = new_index(self.index_kind, self.dimension)?;
        for record in self.records.values() {
            index.insert(record.id, record.vector.clone())?;
        }
        self.index = index;
        self.tombstones.clear();
        Ok(())
    }

    pub fn search(&self, query: &[f32], limit: usize) -> Result<Vec<SearchResult>, VectorDbError> {
        self.search_filtered(query, limit, &Metadata::new())
    }

    pub fn search_filtered(
        &self,
        query: &[f32],
        limit: usize,
        filter: &Metadata,
    ) -> Result<Vec<SearchResult>, VectorDbError> {
        self.validate_query(query)?;
        if limit == 0 {
            return Ok(Vec::new());
        }

        // Unfiltered search routes through the configured index (exact or
        // approximate); the ids it returns are resolved back to full records.
        if filter.is_empty() {
            let ef = match self.index_kind {
                IndexKind::Hnsw(params) => params.ef_search,
                IndexKind::Exact => 0,
            };
            // Over-fetch by the tombstone count so deleted-but-not-compacted
            // nodes can't shrink the result below `limit`.
            let fetch = limit.saturating_add(self.tombstones.len());
            let hits = self.index.search(query, fetch, ef)?;
            return Ok(hits
                .into_iter()
                .filter_map(|(id, score)| {
                    self.records.get(&id).map(|record| SearchResult {
                        score,
                        record: record.clone(),
                    })
                })
                .take(limit)
                .collect());
        }

        // Filtered search stays exact: score the matching records directly. For
        // a selective filter this beats over-fetching from an approximate index
        // and sidesteps the filter-recall cliff.
        let mut scored: Vec<(f32, &Record)> = self
            .records
            .values()
            .filter(|record| {
                filter
                    .iter()
                    .all(|(key, value)| record.metadata.get(key) == Some(value))
            })
            .map(|record| (cosine_similarity(query, &record.vector), record))
            .collect();
        scored.sort_by(|left, right| right.0.total_cmp(&left.0).then(left.1.id.cmp(&right.1.id)));
        Ok(scored
            .into_iter()
            .take(limit)
            .map(|(score, record)| SearchResult {
                score,
                record: record.clone(),
            })
            .collect())
    }

    /// Build an approximate nearest-neighbour index from this collection.
    ///
    /// The collection keeps exact brute-force [`search`](Self::search) as its
    /// default; this returns a separate, experimental [`AnnIndex`] for faster
    /// approximate search over large collections.
    pub fn build_ann_index(&self, params: AnnParams) -> Result<AnnIndex, VectorDbError> {
        let mut index = match self.dimension {
            Some(dimension) => AnnIndex::with_dimension(dimension, params)?,
            None => AnnIndex::with_params(params),
        };
        for record in self.records.values() {
            index.insert(record.id, record.vector.clone())?;
        }
        Ok(index)
    }

    /// Save the database to its primary format: a compact, portable `.lvdb`
    /// binary file (conventionally a `.lvdb` extension).
    ///
    /// Written to a temporary file and atomically renamed, so an interrupted
    /// save never corrupts an existing file. The search index is derived state
    /// rebuilt on load, so only the records and configuration are stored.
    ///
    /// For a human-readable file, use [`export_json`](Self::export_json).
    pub fn save_to_path(&self, path: impl AsRef<Path>) -> Result<(), VectorDbError> {
        // Persist the graph only when it's clean. With pending tombstones the
        // index still references deleted ids, so we skip it and let load rebuild
        // from the (live) records; `compact` first for an instant-load file.
        // An exact index has nothing to store (`persist` returns None).
        let index = if self.tombstones.is_empty() {
            self.index.persist()
        } else {
            None
        };
        let bytes = storage::encode(
            self.dimension,
            self.index_kind,
            self.encoding,
            &self.records,
            index.as_ref(),
        );
        Self::atomic_write(path.as_ref(), &bytes)
    }

    /// Load a database from a `.lvdb` binary file, rebuilding the search index.
    ///
    /// Returns [`VectorDbError::Corrupt`] for a malformed or truncated file and
    /// [`VectorDbError::UnsupportedVersion`] for a newer major format version.
    pub fn load_from_path(path: impl AsRef<Path>) -> Result<Self, VectorDbError> {
        let (dimension, index_kind, encoding, records, index_data) =
            storage::decode(&fs::read(path)?)?;
        Self::from_parts(dimension, index_kind, encoding, records, index_data)
    }

    /// Export the database to human-readable JSON.
    ///
    /// JSON is portable, diffable, and easy to inspect or hand-edit — ideal for
    /// test fixtures and version control. Use [`save_to_path`](Self::save_to_path)
    /// for the compact primary format.
    pub fn export_json(&self, path: impl AsRef<Path>) -> Result<(), VectorDbError> {
        let snapshot = DbSnapshot {
            dimension: self.dimension,
            records: &self.records,
            index_kind: self.index_kind,
        };
        Self::atomic_write(path.as_ref(), &serde_json::to_vec_pretty(&snapshot)?)
    }

    /// Import a database from a JSON file written by
    /// [`export_json`](Self::export_json), rebuilding the search index.
    pub fn import_json(path: impl AsRef<Path>) -> Result<Self, VectorDbError> {
        let snapshot: OwnedDbSnapshot = serde_json::from_slice(&fs::read(path)?)?;
        // JSON carries no graph and is always full precision, so the index is
        // rebuilt from the records and the encoding is Float32.
        Self::from_parts(
            snapshot.dimension,
            snapshot.index_kind,
            Encoding::Float32,
            snapshot.records,
            None,
        )
    }

    /// Reconstruct a database from loaded parts: validate, then restore the
    /// index — reusing a persisted graph when present, otherwise rebuilding.
    fn from_parts(
        dimension: Option<usize>,
        index_kind: IndexKind,
        encoding: Encoding,
        records: BTreeMap<u64, Record>,
        index_data: Option<IndexData>,
    ) -> Result<Self, VectorDbError> {
        let mut db = Self::empty(None, index_kind).expect("no dimension cannot fail");
        db.dimension = dimension;
        db.records = records;
        db.encoding = encoding;
        db.validate_database()?;
        match (index_kind, index_data) {
            // A persisted graph is reconstructed directly — no rebuild.
            (IndexKind::Hnsw(params), Some(data)) => {
                db.index = Box::new(AnnIndex::from_persisted(
                    db.dimension,
                    params,
                    data,
                    &db.records,
                )?);
            }
            // Exact index, JSON, or an older file without a graph: rebuild.
            _ => db.rebuild_index()?,
        }
        Ok(db)
    }

    /// Write `bytes` to `path` via a temporary file and an atomic rename.
    fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), VectorDbError> {
        let parent = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        fs::create_dir_all(parent)?;
        let temporary = path.with_extension("tmp");
        fs::write(&temporary, bytes)?;
        // Rename is atomic, so an interrupted save never corrupts the existing file.
        fs::rename(temporary, path)?;
        Ok(())
    }

    fn validate_vector(&mut self, vector: &[f32]) -> Result<(), VectorDbError> {
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

    fn validate_database(&self) -> Result<(), VectorDbError> {
        for record in self.records.values() {
            validate_numbers(&record.vector)?;
            if let Some(expected) = self.dimension
                && record.vector.len() != expected
            {
                return Err(VectorDbError::DimensionMismatch {
                    expected,
                    actual: record.vector.len(),
                });
            }
        }
        if self.records.is_empty() && self.dimension == Some(0) {
            return Err(VectorDbError::InvalidVector(
                "dimension must be greater than zero",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn record(id: u64, vector: Vec<f32>) -> Record {
        Record::new(id, vector, format!("record {id}"))
    }

    fn temp_path(tag: &str, ext: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "light-vector-db-{tag}-{}.{ext}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    #[test]
    fn insert_search_and_filter_work() {
        let mut db = VectorDb::with_dimension(2).unwrap();
        db.insert(record(1, vec![1.0, 0.0]).with_metadata("topic", "rust"))
            .unwrap();
        db.insert(record(2, vec![0.0, 1.0]).with_metadata("topic", "web"))
            .unwrap();
        let mut filter = Metadata::new();
        filter.insert("topic".into(), "rust".into());
        let hits = db.search_filtered(&[1.0, 0.0], 5, &filter).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].record.id, 1);
    }

    #[test]
    fn dimensions_and_non_finite_numbers_are_rejected() {
        let mut db = VectorDb::with_dimension(2).unwrap();
        assert!(matches!(
            db.insert(record(1, vec![1.0])),
            Err(VectorDbError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            db.insert(record(1, vec![f32::NAN, 1.0])),
            Err(VectorDbError::InvalidVector(_))
        ));
    }

    #[test]
    fn duplicate_upsert_and_delete_behave_as_expected() {
        let mut db = VectorDb::new();
        db.insert(record(1, vec![1.0, 0.0])).unwrap();
        assert!(matches!(
            db.insert(record(1, vec![1.0, 0.0])),
            Err(VectorDbError::DuplicateId(1))
        ));
        db.upsert(record(1, vec![0.0, 1.0])).unwrap();
        assert_eq!(db.get(1).unwrap().vector, vec![0.0, 1.0]);
        assert_eq!(db.delete(1).unwrap().id, 1);
        assert!(matches!(db.delete(1), Err(VectorDbError::NotFound(1))));
    }

    #[test]
    fn hnsw_backed_db_returns_nearest() {
        let mut db = VectorDb::with_index(4, IndexKind::Hnsw(AnnParams::default())).unwrap();
        db.insert(record(1, vec![1.0, 0.0, 0.0, 0.0])).unwrap();
        db.insert(record(2, vec![0.0, 1.0, 0.0, 0.0])).unwrap();
        db.insert(record(3, vec![0.0, 0.0, 1.0, 0.0])).unwrap();
        assert_eq!(db.index_kind(), IndexKind::Hnsw(AnnParams::default()));

        let hits = db.search(&[0.05, 0.0, 0.95, 0.0], 1).unwrap();
        assert_eq!(hits[0].record.id, 3);
    }

    #[test]
    fn upsert_updates_the_index() {
        let mut db = VectorDb::with_dimension(2).unwrap();
        db.insert(record(1, vec![1.0, 0.0])).unwrap();
        db.insert(record(2, vec![-1.0, 0.0])).unwrap();
        assert_eq!(db.search(&[1.0, 0.0], 1).unwrap()[0].record.id, 1);

        // Re-point record 1; searching its new direction must return it with a
        // near-perfect score, proving the stale vector left the index.
        db.upsert(record(1, vec![0.0, 1.0])).unwrap();
        let top = db.search(&[0.0, 1.0], 1).unwrap();
        assert_eq!(top[0].record.id, 1);
        assert!((top[0].score - 1.0).abs() < 1e-6, "score {}", top[0].score);
    }

    #[test]
    fn delete_removes_from_search() {
        let mut db = VectorDb::with_dimension(2).unwrap();
        db.insert(record(1, vec![1.0, 0.0])).unwrap();
        db.insert(record(2, vec![0.0, 1.0])).unwrap();
        db.delete(1).unwrap();

        let hits = db.search(&[1.0, 0.0], 5).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].record.id, 2);
    }

    #[test]
    fn bulk_delete_still_returns_k_live() {
        // Delete the top-ranked records; over-fetch must still return k live hits.
        let mut db = VectorDb::with_index(2, IndexKind::Hnsw(AnnParams::default())).unwrap();
        for i in 0..10u64 {
            let angle = i as f32 * 0.05;
            db.insert(record(i, vec![1.0 - angle, angle])).unwrap();
        }
        for id in [0u64, 1, 2] {
            db.delete(id).unwrap();
        }
        let hits = db.search(&[1.0, 0.0], 3).unwrap();
        assert_eq!(
            hits.len(),
            3,
            "over-fetch should still return k live results"
        );
        for hit in &hits {
            assert!(![0, 1, 2].contains(&hit.record.id));
        }
    }

    #[test]
    fn delete_then_reinsert_has_no_duplicate() {
        let mut db = VectorDb::with_dimension(2).unwrap();
        db.insert(record(1, vec![1.0, 0.0])).unwrap();
        db.insert(record(2, vec![0.0, 1.0])).unwrap();
        db.delete(1).unwrap();
        db.insert(record(1, vec![0.5, 0.5])).unwrap(); // reuse the deleted id

        assert_eq!(db.len(), 2);
        let hits = db.search(&[0.5, 0.5], 10).unwrap();
        let ones = hits.iter().filter(|h| h.record.id == 1).count();
        assert_eq!(ones, 1, "id 1 must appear once, not duplicated");
        assert_eq!(db.get(1).unwrap().vector, vec![0.5, 0.5]);
    }

    #[test]
    fn compact_reclaims_tombstones() {
        let mut db = VectorDb::with_index(2, IndexKind::Hnsw(AnnParams::default())).unwrap();
        for i in 0..5u64 {
            db.insert(record(i, vec![i as f32, 1.0])).unwrap();
        }
        db.delete(0).unwrap();
        db.delete(3).unwrap();
        assert_eq!(db.compact().unwrap(), 2);
        assert_eq!(db.compact().unwrap(), 0); // already clean
        assert_eq!(db.len(), 3);
        for hit in &db.search(&[0.0, 1.0], 5).unwrap() {
            assert!(hit.record.id != 0 && hit.record.id != 3);
        }
    }

    #[test]
    fn save_with_tombstones_loads_clean() {
        let path = temp_path("tomb", "lvdb");
        let mut db = VectorDb::with_index(2, IndexKind::Hnsw(AnnParams::default())).unwrap();
        db.insert(record(1, vec![1.0, 0.0])).unwrap();
        db.insert(record(2, vec![0.0, 1.0])).unwrap();
        db.delete(2).unwrap(); // pending tombstone → saved without a persisted index

        db.save_to_path(&path).unwrap();
        let loaded = VectorDb::load_from_path(&path).unwrap();
        assert_eq!(loaded.len(), 1);
        assert!(loaded.get(2).is_none());
        assert_eq!(loaded.search(&[1.0, 0.0], 5).unwrap()[0].record.id, 1);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn binary_round_trip() {
        let path = temp_path("bin", "lvdb");
        let mut db = VectorDb::with_index(3, IndexKind::Hnsw(AnnParams::default())).unwrap();
        db.insert(record(1, vec![1.0, 0.0, 0.0]).with_metadata("topic", "rust"))
            .unwrap();
        db.insert(record(2, vec![0.0, 0.0, 1.0]).with_metadata("topic", "db"))
            .unwrap();
        db.save_to_path(&path).unwrap();

        let loaded = VectorDb::load_from_path(&path).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded.dimension(), Some(3));
        assert_eq!(loaded.index_kind(), IndexKind::Hnsw(AnnParams::default()));
        assert_eq!(loaded.get(1).unwrap().metadata["topic"], "rust");
        assert_eq!(loaded.get(1).unwrap().vector, vec![1.0, 0.0, 0.0]);
        // The index was rebuilt on load, so search works.
        assert_eq!(loaded.search(&[0.9, 0.0, 0.1], 1).unwrap()[0].record.id, 1);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn binary_rejects_corrupt_file() {
        let path = temp_path("corrupt", "lvdb");
        let mut db = VectorDb::with_dimension(2).unwrap();
        db.insert(record(1, vec![1.0, 0.0])).unwrap();
        db.save_to_path(&path).unwrap();

        let mut bytes = fs::read(&path).unwrap();
        bytes[0] = b'X'; // clobber the magic
        fs::write(&path, bytes).unwrap();
        assert!(matches!(
            VectorDb::load_from_path(&path),
            Err(VectorDbError::Corrupt(_))
        ));
        fs::remove_file(path).unwrap();
    }

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
    fn persisted_hnsw_index_matches_live_search() {
        let dimension = 8;
        let mut db =
            VectorDb::with_index(dimension, IndexKind::Hnsw(AnnParams::default())).unwrap();
        // Insert in a shuffled id order so node order differs from id order,
        // making a faithful graph round-trip meaningful.
        for id in [5u64, 1, 9, 3, 7, 2, 8, 0, 6, 4] {
            db.insert(record(id, sample_vector(id, dimension))).unwrap();
        }
        let query = sample_vector(100, dimension);
        let before = db.search(&query, 5).unwrap();

        let path = temp_path("persist", "lvdb");
        db.save_to_path(&path).unwrap();
        let loaded = VectorDb::load_from_path(&path).unwrap();
        let after = loaded.search(&query, 5).unwrap();

        // The reloaded graph must produce identical results (it was restored,
        // not rebuilt from scratch).
        assert_eq!(before.len(), after.len());
        for (b, a) in before.iter().zip(&after) {
            assert_eq!(b.record.id, a.record.id);
            assert!((b.score - a.score).abs() < 1e-6);
        }
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn scalar_quantized_file_is_smaller_and_searchable() {
        let dimension = 16;
        let mut db = VectorDb::with_dimension(dimension).unwrap();
        db.set_encoding(Encoding::ScalarU8);
        for id in 0..20u64 {
            db.insert(record(id, sample_vector(id, dimension))).unwrap();
        }
        let query = sample_vector(3, dimension); // matches record 3 exactly
        let before = db.search(&query, 1).unwrap()[0].record.id;

        let q_path = temp_path("q", "lvdb");
        let f_path = temp_path("f", "lvdb");
        db.save_to_path(&q_path).unwrap();
        let mut f32_db = db.clone();
        f32_db.set_encoding(Encoding::Float32);
        f32_db.save_to_path(&f_path).unwrap();

        // The quantized file is meaningfully smaller.
        let q_size = fs::metadata(&q_path).unwrap().len();
        let f_size = fs::metadata(&f_path).unwrap().len();
        assert!(
            q_size < f_size,
            "quantized {q_size} should be < f32 {f_size}"
        );

        // Round-trips: encoding preserved, and search still finds the same record.
        let loaded = VectorDb::load_from_path(&q_path).unwrap();
        assert_eq!(loaded.encoding(), Encoding::ScalarU8);
        assert_eq!(loaded.len(), 20);
        assert_eq!(loaded.search(&query, 1).unwrap()[0].record.id, before);

        fs::remove_file(q_path).unwrap();
        fs::remove_file(f_path).unwrap();
    }

    #[test]
    fn json_export_import_round_trip() {
        let path = temp_path("json", "json");
        let mut db = VectorDb::with_index(2, IndexKind::Hnsw(AnnParams::default())).unwrap();
        db.insert(record(1, vec![1.0, 0.0]).with_metadata("topic", "rust"))
            .unwrap();
        db.insert(record(2, vec![0.0, 1.0])).unwrap();
        db.export_json(&path).unwrap();

        let loaded = VectorDb::import_json(&path).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded.index_kind(), IndexKind::Hnsw(AnnParams::default()));
        assert_eq!(loaded.get(1).unwrap().metadata["topic"], "rust");
        assert_eq!(loaded.search(&[1.0, 0.0], 1).unwrap()[0].record.id, 1);
        fs::remove_file(path).unwrap();
    }
}
