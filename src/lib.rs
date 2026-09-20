//! A local-first vector database library.
//!
//! `light_vector_db` intentionally does not generate embeddings. Your
//! application supplies a `Vec<f32>` from any embedding model; this crate
//! validates, stores, persists, and searches those vectors.

use std::{collections::BTreeMap, error::Error, fmt, fs, path::Path};

use serde::{Deserialize, Serialize};

mod ann;
mod index;
mod storage;

pub use ann::{AnnIndex, AnnParams};
pub use index::{BruteForce, VectorIndex};

pub type Metadata = BTreeMap<String, String>;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Record {
    pub id: u64,
    pub vector: Vec<f32>,
    pub text: String,
    #[serde(default)]
    pub metadata: Metadata,
}

impl Record {
    pub fn new(id: u64, vector: Vec<f32>, text: impl Into<String>) -> Self {
        Self {
            id,
            vector,
            text: text.into(),
            metadata: Metadata::new(),
        }
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    pub score: f32,
    pub record: Record,
}

#[derive(Debug)]
#[non_exhaustive]
pub enum VectorDbError {
    DimensionMismatch {
        expected: usize,
        actual: usize,
    },
    DuplicateId(u64),
    InvalidVector(&'static str),
    NotFound(u64),
    Io(std::io::Error),
    Serialization(serde_json::Error),
    /// A `.lvdb` file is malformed or truncated.
    Corrupt(&'static str),
    /// A `.lvdb` file uses a newer major format version than this build supports.
    UnsupportedVersion {
        major: u16,
        minor: u16,
    },
}

impl fmt::Display for VectorDbError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch { expected, actual } => write!(
                f,
                "vector dimension mismatch: expected {expected}, got {actual}"
            ),
            Self::DuplicateId(id) => write!(f, "a record with id {id} already exists"),
            Self::InvalidVector(message) => write!(f, "invalid vector: {message}"),
            Self::NotFound(id) => write!(f, "no record with id {id}"),
            Self::Io(error) => write!(f, "I/O error: {error}"),
            Self::Serialization(error) => write!(f, "database serialization error: {error}"),
            Self::Corrupt(message) => write!(f, "corrupt .lvdb file: {message}"),
            Self::UnsupportedVersion { major, minor } => {
                write!(f, "unsupported .lvdb format version {major}.{minor}")
            }
        }
    }
}

impl Error for VectorDbError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Serialization(error) => Some(error),
            _ => None,
        }
    }
}

impl From<std::io::Error> for VectorDbError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<serde_json::Error> for VectorDbError {
    fn from(error: serde_json::Error) -> Self {
        Self::Serialization(error)
    }
}

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
        })
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
        self.index.insert(record.id, record.vector.clone())?;
        self.records.insert(record.id, record);
        Ok(())
    }

    pub fn upsert(&mut self, record: Record) -> Result<(), VectorDbError> {
        self.validate_vector(&record.vector)?;
        let id = record.id;
        let replaces_existing = self.records.insert(id, record).is_some();
        if replaces_existing {
            // The old vector is still in the index; rebuild to drop it.
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
        // Point removal from the index arrives in roadmap M5; rebuild for now.
        self.rebuild_index()?;
        Ok(removed)
    }

    /// Rebuild the search index from the records (the source of truth).
    fn rebuild_index(&mut self) -> Result<(), VectorDbError> {
        let mut index = new_index(self.index_kind, self.dimension)?;
        for record in self.records.values() {
            index.insert(record.id, record.vector.clone())?;
        }
        self.index = index;
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
            let hits = self.index.search(query, limit, ef)?;
            return Ok(hits
                .into_iter()
                .filter_map(|(id, score)| {
                    self.records.get(&id).map(|record| SearchResult {
                        score,
                        record: record.clone(),
                    })
                })
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

    pub fn save_to_path(&self, path: impl AsRef<Path>) -> Result<(), VectorDbError> {
        let path = path.as_ref();
        let parent = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        fs::create_dir_all(parent)?;
        let snapshot = DbSnapshot {
            dimension: self.dimension,
            records: &self.records,
            index_kind: self.index_kind,
        };
        let temporary = path.with_extension("tmp");
        fs::write(&temporary, serde_json::to_vec_pretty(&snapshot)?)?;
        // Rename is atomic, so an interrupted save never corrupts the existing file.
        fs::rename(temporary, path)?;
        Ok(())
    }

    pub fn load_from_path(path: impl AsRef<Path>) -> Result<Self, VectorDbError> {
        let snapshot: OwnedDbSnapshot = serde_json::from_slice(&fs::read(path)?)?;
        let mut db = Self::empty(None, snapshot.index_kind).expect("no dimension cannot fail");
        db.dimension = snapshot.dimension;
        db.records = snapshot.records;
        db.validate_database()?;
        db.rebuild_index()?;
        Ok(db)
    }

    /// Save the database to a compact, portable `.lvdb` binary file.
    ///
    /// Written to a temporary file and atomically renamed, so an interrupted
    /// save never corrupts an existing file. The search index is not stored
    /// (it is rebuilt on load); only the records and configuration are.
    pub fn save_lvdb(&self, path: impl AsRef<Path>) -> Result<(), VectorDbError> {
        let path = path.as_ref();
        let parent = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        fs::create_dir_all(parent)?;
        let bytes = storage::encode(self.dimension, self.index_kind, &self.records);
        let temporary = path.with_extension("lvdb.tmp");
        fs::write(&temporary, bytes)?;
        fs::rename(temporary, path)?;
        Ok(())
    }

    /// Load a database from a `.lvdb` binary file, rebuilding the search index.
    ///
    /// Returns [`VectorDbError::Corrupt`] for a malformed or truncated file and
    /// [`VectorDbError::UnsupportedVersion`] for a newer major format version.
    pub fn load_lvdb(path: impl AsRef<Path>) -> Result<Self, VectorDbError> {
        let (dimension, index_kind, records) = storage::decode(&fs::read(path)?)?;
        let mut db = Self::empty(None, index_kind).expect("no dimension cannot fail");
        db.dimension = dimension;
        db.records = records;
        db.validate_database()?;
        db.rebuild_index()?;
        Ok(db)
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

fn validate_numbers(vector: &[f32]) -> Result<(), VectorDbError> {
    if vector.is_empty() {
        return Err(VectorDbError::InvalidVector("vector cannot be empty"));
    }
    if vector.iter().any(|value| !value.is_finite()) {
        return Err(VectorDbError::InvalidVector("all values must be finite"));
    }
    Ok(())
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn record(id: u64, vector: Vec<f32>) -> Record {
        Record::new(id, vector, format!("record {id}"))
    }

    #[test]
    fn identical_vectors_score_one() {
        assert!((cosine_similarity(&[1.0, 2.0], &[1.0, 2.0]) - 1.0).abs() < 1e-6);
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
    fn persistence_round_trip() {
        let path = std::env::temp_dir().join(format!(
            "light-vector-db-{}.json",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut db = VectorDb::with_dimension(2).unwrap();
        db.insert(record(7, vec![0.4, 0.8]).with_metadata("source", "test"))
            .unwrap();
        db.save_to_path(&path).unwrap();
        let loaded = VectorDb::load_from_path(&path).unwrap();
        assert_eq!(loaded.get(7).unwrap().metadata["source"], "test");
        fs::remove_file(path).unwrap();
    }

    fn temp_path(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "light-vector-db-{tag}-{}.json",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
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
    fn hnsw_index_kind_survives_round_trip() {
        let path = temp_path("hnsw");
        let mut db = VectorDb::with_index(3, IndexKind::Hnsw(AnnParams::default())).unwrap();
        db.insert(record(1, vec![1.0, 0.0, 0.0])).unwrap();
        db.insert(record(2, vec![0.0, 0.0, 1.0])).unwrap();
        db.save_to_path(&path).unwrap();

        let loaded = VectorDb::load_from_path(&path).unwrap();
        assert_eq!(loaded.index_kind(), IndexKind::Hnsw(AnnParams::default()));
        assert_eq!(loaded.search(&[0.9, 0.0, 0.1], 1).unwrap()[0].record.id, 1);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn lvdb_binary_round_trip() {
        let path = temp_path("bin");
        let mut db = VectorDb::with_index(3, IndexKind::Hnsw(AnnParams::default())).unwrap();
        db.insert(record(1, vec![1.0, 0.0, 0.0]).with_metadata("topic", "rust"))
            .unwrap();
        db.insert(record(2, vec![0.0, 0.0, 1.0]).with_metadata("topic", "db"))
            .unwrap();
        db.save_lvdb(&path).unwrap();

        let loaded = VectorDb::load_lvdb(&path).unwrap();
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
    fn lvdb_rejects_corrupt_file() {
        let path = temp_path("corrupt");
        let mut db = VectorDb::with_dimension(2).unwrap();
        db.insert(record(1, vec![1.0, 0.0])).unwrap();
        db.save_lvdb(&path).unwrap();

        let mut bytes = fs::read(&path).unwrap();
        bytes[0] = b'X'; // clobber the magic
        fs::write(&path, bytes).unwrap();
        assert!(matches!(
            VectorDb::load_lvdb(&path),
            Err(VectorDbError::Corrupt(_))
        ));
        fs::remove_file(path).unwrap();
    }
}
