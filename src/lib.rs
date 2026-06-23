//! A local-first vector database library.
//!
//! `light_vector_db` intentionally does not generate embeddings. Your
//! application supplies a `Vec<f32>` from any embedding model; this crate
//! validates, stores, persists, and searches those vectors.

use std::{collections::BTreeMap, error::Error, fmt, fs, path::Path};

use serde::{Deserialize, Serialize};

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
    DimensionMismatch { expected: usize, actual: usize },
    DuplicateId(u64),
    InvalidVector(&'static str),
    NotFound(u64),
    Io(std::io::Error),
    Serialization(serde_json::Error),
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

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VectorDb {
    dimension: Option<usize>,
    #[serde(default)]
    records: BTreeMap<u64, Record>,
}

impl VectorDb {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_dimension(dimension: usize) -> Result<Self, VectorDbError> {
        if dimension == 0 {
            return Err(VectorDbError::InvalidVector(
                "dimension must be greater than zero",
            ));
        }
        Ok(Self {
            dimension: Some(dimension),
            records: BTreeMap::new(),
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
        self.records.insert(record.id, record);
        Ok(())
    }

    pub fn upsert(&mut self, record: Record) -> Result<(), VectorDbError> {
        self.validate_vector(&record.vector)?;
        self.records.insert(record.id, record);
        Ok(())
    }

    pub fn get(&self, id: u64) -> Option<&Record> {
        self.records.get(&id)
    }

    pub fn delete(&mut self, id: u64) -> Result<Record, VectorDbError> {
        self.records.remove(&id).ok_or(VectorDbError::NotFound(id))
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
        scored.sort_by(|left, right| right.0.total_cmp(&left.0));
        Ok(scored
            .into_iter()
            .take(limit)
            .map(|(score, record)| SearchResult {
                score,
                record: record.clone(),
            })
            .collect())
    }

    pub fn save_to_path(&self, path: impl AsRef<Path>) -> Result<(), VectorDbError> {
        let path = path.as_ref();
        let parent = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        fs::create_dir_all(parent)?;
        let temporary = path.with_extension("tmp");
        fs::write(&temporary, serde_json::to_vec_pretty(self)?)?;
        // Rename is atomic, so an interrupted save never corrupts the existing file.
        fs::rename(temporary, path)?;
        Ok(())
    }

    pub fn load_from_path(path: impl AsRef<Path>) -> Result<Self, VectorDbError> {
        let db: Self = serde_json::from_slice(&fs::read(path)?)?;
        db.validate_database()?;
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
}
