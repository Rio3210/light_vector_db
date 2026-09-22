# light_vector_db

[![CI](https://github.com/Rio3210/light_vector_db/actions/workflows/ci.yml/badge.svg)](https://github.com/Rio3210/light_vector_db/actions/workflows/ci.yml)
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
- Pluggable search index (exact brute force, or an approximate HNSW/NSW graph)
- Single-file `.lvdb` binary persistence (atomic, crash-safe) — plus JSON export
  for inspection and fixtures
- Validation for empty, non-finite, and mismatched vectors

## Roadmap

- [x] In-memory, fixed-dimension collections
- [x] Brute-force cosine-similarity search
- [x] Insert / upsert / get / delete
- [x] Metadata + metadata-filtered search
- [x] Pluggable index behind a `VectorIndex` trait (exact + approximate)
- [x] Single-file `.lvdb` binary format (versioned, checksummed) + JSON export
- [x] Input validation (empty / non-finite / dimension mismatch)
- [x] Benchmarks on realistic vector sizes
- [x] `lvdb` command-line tool
- [x] On-disk (persisted) index — the graph is stored in the `.lvdb` file, so
  loading never rebuilds it
- [x] Memory-mapped reads — scan a file larger than RAM without loading its
  vectors (`MmapDb`, or `lvdb search --mmap`)
- [x] Tombstoned deletes + `compact` to reclaim space
- [x] Scalar (int8) quantization — 4× smaller vectors on disk (`--encoding int8`)
- [ ] Optional embedding-provider integrations (e.g. a local Ollama helper)

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full design and milestone plan.

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

    // The whole database is one portable file you can commit and share.
    notes.save_to_path("data/notes.lvdb")?;
    let restored = VectorDb::load_from_path("data/notes.lvdb")?;
    assert_eq!(restored.len(), 2);

    // Or export human-readable JSON for inspection / fixtures:
    notes.export_json("data/notes.json")?;

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

## Command-line tool (`lvdb`)

The crate ships an `lvdb` binary — one database per file, driven from the
terminal (in the spirit of the `sqlite3` shell):

```powershell
cargo run --bin lvdb -- create  notes.lvdb --dim 3 --index hnsw
cargo run --bin lvdb -- create  small.lvdb --dim 3 --encoding int8   # 4x smaller vectors
cargo run --bin lvdb -- insert  notes.lvdb --id 1 --vector 0.9,0.1,0 --text "the cat sat" --meta topic=animals
cargo run --bin lvdb -- insert  notes.lvdb --id 2 --vector 0.1,0.1,0.9 --text "rust is fast" --meta topic=rust
cargo run --bin lvdb -- search  notes.lvdb --vector 0.85,0.15,0.05 -k 3
cargo run --bin lvdb -- search  notes.lvdb --vector 0.85,0.15,0.05 --filter topic=rust
cargo run --bin lvdb -- search  notes.lvdb --vector 0.85,0.15,0.05 --mmap   # memory-mapped scan
cargo run --bin lvdb -- delete  notes.lvdb --id 2
cargo run --bin lvdb -- compact notes.lvdb
cargo run --bin lvdb -- stats   notes.lvdb
cargo run --bin lvdb -- export  notes.lvdb notes.json   # human-readable copy
cargo run --bin lvdb -- import  notes.json notes.lvdb
```

A `--vector` value of the form `@path` is read from a file. Run
`cargo run --bin lvdb -- help` for the full reference.

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
