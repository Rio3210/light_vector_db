//! Records, metadata, search results, and vector validation.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::VectorDbError;

/// String key/value metadata attached to a [`Record`].
pub type Metadata = BTreeMap<String, String>;

/// A stored item: an id, its embedding vector, its text, and metadata.
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

/// A search hit: a similarity score and the matching record.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    pub score: f32,
    pub record: Record,
}

/// Reject empty or non-finite vectors. Shared by the collection and indexes.
pub(crate) fn validate_numbers(vector: &[f32]) -> Result<(), VectorDbError> {
    if vector.is_empty() {
        return Err(VectorDbError::InvalidVector("vector cannot be empty"));
    }
    if vector.iter().any(|value| !value.is_finite()) {
        return Err(VectorDbError::InvalidVector("all values must be finite"));
    }
    Ok(())
}
