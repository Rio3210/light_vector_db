use light_vector_db::{Record, VectorDb};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut db = VectorDb::with_dimension(3)?;
    db.insert(
        Record::new(1, vec![0.9, 0.1, 0.0], "The cat sat on the mat.")
            .with_metadata("topic", "animals"),
    )?;
    db.insert(
        Record::new(2, vec![0.8, 0.2, 0.1], "A kitten napped on the rug.")
            .with_metadata("topic", "animals"),
    )?;
    db.insert(
        Record::new(
            3,
            vec![0.1, 0.1, 0.9],
            "Rust is a systems programming language.",
        )
        .with_metadata("topic", "rust"),
    )?;

    let query = vec![0.85, 0.15, 0.05];
    println!(
        "Stored {} records with {} dimensions.\n",
        db.len(),
        db.dimension().unwrap()
    );
    for (rank, hit) in db.search(&query, 3)?.iter().enumerate() {
        println!(
            "{}. [{:.3}] (id {}) {}",
            rank + 1,
            hit.score,
            hit.record.id,
            hit.record.text
        );
    }
    Ok(())
}
