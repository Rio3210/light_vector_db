//! Minimal, dependency-free benchmarks for `VectorDb` insert and brute-force
//! search.
//!
//! This is a scaffold, not a tuned benchmark suite. It measures the two core
//! operations on a fixed dimension with deterministic sample data so results
//! are reproducible across runs. It deliberately avoids external benchmarking
//! crates (e.g. criterion) to keep the dependency footprint small; swap in a
//! richer harness later if statistical rigor is needed.
//!
//! Run with:
//!
//! ```text
//! cargo bench
//! ```
//!
//! Configured as a `harness = false` benchmark in `Cargo.toml`, so `cargo bench`
//! executes `main` directly (compiled in release mode).

use std::hint::black_box;
use std::time::Instant;

use light_vector_db::{Record, VectorDb};

/// Vector dimension used for every benchmark. Fixed so runs are comparable.
const DIMENSION: usize = 128;
/// Number of records inserted into the collection.
const RECORD_COUNT: usize = 10_000;
/// Number of brute-force searches timed over the populated collection.
const SEARCH_ITERATIONS: usize = 100;

/// Deterministic pseudo-random vector generator.
///
/// Uses a small splitmix64-style mix so sample data is reproducible without an
/// external RNG dependency. Values land in the range `[-1.0, 1.0)`.
fn sample_vector(seed: u64, dimension: usize) -> Vec<f32> {
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..dimension)
        .map(|_| {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            // Map the top 24 bits to [0, 1), then shift to [-1, 1).
            let unit = (z >> 40) as f32 / (1u32 << 24) as f32;
            unit * 2.0 - 1.0
        })
        .collect()
}

/// Build the deterministic sample records shared by the benchmarks.
fn sample_records(count: usize, dimension: usize) -> Vec<Record> {
    (0..count)
        .map(|i| {
            Record::new(
                i as u64,
                sample_vector(i as u64, dimension),
                format!("record {i}"),
            )
        })
        .collect()
}

/// Time inserting every record into a fresh collection.
fn bench_insert(records: &[Record]) {
    let start = Instant::now();
    let mut db = VectorDb::with_dimension(DIMENSION).expect("dimension is non-zero");
    for record in records {
        db.insert(record.clone()).expect("sample records are valid");
    }
    let elapsed = start.elapsed();
    black_box(&db);
    report("insert", records.len(), elapsed);
}

/// Time brute-force searches over a pre-populated collection.
fn bench_search(records: &[Record]) {
    let mut db = VectorDb::with_dimension(DIMENSION).expect("dimension is non-zero");
    for record in records {
        db.insert(record.clone()).expect("sample records are valid");
    }

    let start = Instant::now();
    for i in 0..SEARCH_ITERATIONS {
        let query = sample_vector(u64::MAX - i as u64, DIMENSION);
        let hits = db.search(&query, 10).expect("query is valid");
        black_box(hits);
    }
    let elapsed = start.elapsed();
    report("search", SEARCH_ITERATIONS, elapsed);
}

/// Print a one-line summary: total time and average per operation.
fn report(name: &str, operations: usize, elapsed: std::time::Duration) {
    let per_op = elapsed.as_secs_f64() / operations as f64;
    println!(
        "{name:<8} {operations:>7} ops in {elapsed:>10.3?}  ({:>10.3} us/op)",
        per_op * 1e6
    );
}

fn main() {
    println!(
        "light_vector_db benchmark scaffold (dimension = {DIMENSION}, records = {RECORD_COUNT})"
    );
    let records = sample_records(RECORD_COUNT, DIMENSION);
    bench_insert(&records);
    bench_search(&records);
}
