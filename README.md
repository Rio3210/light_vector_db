# light_vector_db

[![Status](https://img.shields.io/badge/status-work%20in%20progress-orange.svg)](#-project-status)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust 2024](https://img.shields.io/badge/rust-2024%20edition-dea584.svg)](https://www.rust-lang.org/)

`light_vector_db` is a local-first Rust library for storing and searching embedding vectors. It does **not** generate embeddings: your application supplies `Vec<f32>` values from any model or provider. That keeps the database small, portable, and model-independent.

> ## 🚧 Project status
>
> **Work in progress.** This is an actively evolving learning project, not a
> production database. The public API may change without notice between versions,
> and it is not yet published to crates.io. Issues, ideas, and feedback are welcome.

## What it provides

- Fixed-dimension vector collections
- Cosine-similarity search
- Insert, upsert, get, and delete operations
- String metadata and metadata-filtered search
- JSON persistence for local use (atomic, crash-safe writes)
- Validation for empty, non-finite, and mismatched vectors

## Roadmap

- [x] In-memory, fixed-dimension collections
- [x] Brute-force cosine-similarity search
- [x] Insert / upsert / get / delete
- [x] Metadata + metadata-filtered search
- [x] JSON persistence with atomic saves
- [x] Input validation (empty / non-finite / dimension mismatch)
- [ ] Benchmarks on realistic vector sizes
- [ ] Approximate nearest-neighbour index (HNSW) for large collections
- [ ] Optional embedding-provider integrations (e.g. a local Ollama helper)

## Use it from another Rust project

This crate is not on crates.io yet, so depend on the Git repository or a local path:

```toml
[dependencies]
# From GitHub:
light_vector_db = { git = "https://github.com/Rio3210/light_vector_db" }

# Or from a local checkout:
# light_vector_db = { path = "../light_vector_db" }
```

```rust
use light_vector_db::{Metadata, Record, VectorDb};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // A collection must use one vector dimension consistently. The source of
    // these vectors is up to your application: Ollama, an API, or your own model.
    let mut notes = VectorDb::with_dimension(3)?;

    notes.insert(
        Record::new(1, vec![0.90, 0.10, 0.00], "Rust ownership prevents data races.")
            .with_metadata("topic", "rust"),
    )?;

    notes.insert(
        Record::new(2, vec![0.05, 0.20, 0.95], "Databases persist application data.")
            .with_metadata("topic", "databases"),
    )?;

    let hits = notes.search(&[0.85, 0.15, 0.05], 5)?;
    println!("{}", hits[0].record.text);

    notes.save_to_path("data/notes.json")?;
    let restored = VectorDb::load_from_path("data/notes.json")?;
    assert_eq!(restored.len(), 2);

    let mut filter = Metadata::new();
    filter.insert("topic".into(), "rust".into());
    let rust_hits = restored.search_filtered(&[0.85, 0.15, 0.05], 5, &filter)?;
    assert_eq!(rust_hits.len(), 1);
    Ok(())
}
```

## Rules for embeddings

Every record and query in one `VectorDb` must have the same dimension and should come from the same embedding model. Do not mix a 384-dimensional model with a 768-dimensional model, or vectors produced by unrelated models, in one collection.

## Run the included demo

```powershell
cargo run     # runs the demo in src/main.rs
cargo test    # runs the test suite
```

## Benchmarks

A minimal, dependency-free benchmark scaffold measures insert and brute-force
search on a fixed dimension with deterministic sample data:

```powershell
cargo bench            # runs benches/vector_ops.rs in release mode
```

It reports total time and average time per operation for inserting records and
running searches over an in-memory collection. The scaffold is intentionally
simple and easy to extend — adjust `DIMENSION`, `RECORD_COUNT`, and
`SEARCH_ITERATIONS` in `benches/vector_ops.rs`, or add new benchmark functions
alongside the existing ones. It does not implement or measure any approximate
nearest-neighbour index.

## Contributing

This is a personal learning project, but suggestions and bug reports via GitHub
issues are appreciated. If you open a pull request, please run `cargo fmt`,
`cargo clippy --all-targets -- -D warnings`, and `cargo test` first.

## License

Licensed under the [MIT License](LICENSE).

> Note: the Rust ecosystem also commonly dual-licenses as `MIT OR Apache-2.0`.
> If you later prefer that, add an `Apache-2.0` license file and update the
> `license` field in `Cargo.toml` to `"MIT OR Apache-2.0"`.
