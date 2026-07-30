//! Integration tests for `VectorDb` public-API validation edge cases.
//!
//! These exercise the crate's exported surface (`VectorDb`, `Record`,
//! `VectorDbError`) rather than private helpers, which stay unit-tested in
//! `src/lib.rs`.

use light_vector_db::{Record, VectorDb, VectorDbError};

fn record(id: u64, vector: Vec<f32>) -> Record {
    Record::new(id, vector, format!("record {id}"))
}

#[test]
fn insert_rejects_empty_vector() {
    let mut db = VectorDb::new();
    assert!(matches!(
        db.insert(record(1, vec![])),
        Err(VectorDbError::InvalidVector(_))
    ));
    assert!(db.is_empty());
}

#[test]
fn upsert_rejects_empty_vector() {
    let mut db = VectorDb::new();
    assert!(matches!(
        db.upsert(record(1, vec![])),
        Err(VectorDbError::InvalidVector(_))
    ));
    assert!(db.is_empty());
}

#[test]
fn insert_rejects_non_finite_values() {
    let mut db = VectorDb::with_dimension(2).unwrap();
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(matches!(
            db.insert(record(1, vec![value, 1.0])),
            Err(VectorDbError::InvalidVector(_))
        ));
    }
    assert!(db.is_empty());
}

#[test]
fn upsert_rejects_non_finite_values() {
    let mut db = VectorDb::with_dimension(2).unwrap();
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(matches!(
            db.upsert(record(1, vec![1.0, value])),
            Err(VectorDbError::InvalidVector(_))
        ));
    }
    assert!(db.is_empty());
}

#[test]
fn insert_detects_dimension_mismatch() {
    let mut db = VectorDb::with_dimension(3).unwrap();
    assert!(matches!(
        db.insert(record(1, vec![1.0, 2.0])),
        Err(VectorDbError::DimensionMismatch {
            expected: 3,
            actual: 2
        })
    ));
    assert!(db.is_empty());
}

#[test]
fn insert_dimension_is_inferred_then_enforced() {
    // A dimensionless db locks its dimension to the first inserted vector.
    let mut db = VectorDb::new();
    db.insert(record(1, vec![1.0, 0.0])).unwrap();
    assert_eq!(db.dimension(), Some(2));
    assert!(matches!(
        db.insert(record(2, vec![1.0, 0.0, 0.0])),
        Err(VectorDbError::DimensionMismatch {
            expected: 2,
            actual: 3
        })
    ));
    assert_eq!(db.len(), 1);
}

#[test]
fn search_detects_dimension_mismatch() {
    let mut db = VectorDb::with_dimension(3).unwrap();
    db.insert(record(1, vec![1.0, 0.0, 0.0])).unwrap();
    assert!(matches!(
        db.search(&[1.0, 0.0], 5),
        Err(VectorDbError::DimensionMismatch {
            expected: 3,
            actual: 2
        })
    ));
}

#[test]
fn search_rejects_non_finite_query() {
    let mut db = VectorDb::with_dimension(2).unwrap();
    db.insert(record(1, vec![1.0, 0.0])).unwrap();
    assert!(matches!(
        db.search(&[f32::NAN, 1.0], 5),
        Err(VectorDbError::InvalidVector(_))
    ));
}

#[test]
fn search_rejects_empty_query() {
    let mut db = VectorDb::with_dimension(2).unwrap();
    db.insert(record(1, vec![1.0, 0.0])).unwrap();
    assert!(matches!(
        db.search(&[], 5),
        Err(VectorDbError::InvalidVector(_))
    ));
}
