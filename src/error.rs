//! The crate's error type.

use std::error::Error;
use std::fmt;

/// Errors returned by [`VectorDb`](crate::VectorDb) operations.
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
