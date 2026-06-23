# light_vector_db

`light_vector_db` is a local-first Rust library for storing and searching embedding vectors. It does **not** generate embeddings: your application supplies `Vec<f32>` values from any model or provider. That keeps the database small, portable, and model-independent.

## What it provides

- Fixed-dimension vector collections
- Cosine-similarity search
- Insert, upsert, get, and delete operations
- String metadata and metadata-filtered search
- JSON persistence for local use
- Validation for empty, non-finite, and mismatched vectors

## Use it from another Rust project

Until this crate is published on crates.io, depend on a Git revision or local path:

```toml
[dependencies]
light_vector_db = { path = "../vector_db" }
```

The Cargo package name uses an underscore, while Rust imports use the same name:

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
cargo run
cargo test
```

## Publishing later

Before publishing to crates.io, add your repository URL, choose a license, write a changelog, and run `cargo publish --dry-run`. The public library API is intentionally independent from any embedding provider, so integrations can be added later as optional features.
