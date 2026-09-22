//! `lvdb` — a command-line tool for `light_vector_db` databases.
//!
//! One file per database, operated from the terminal — the `sqlite3` shell's
//! spirit for vectors. Arguments are parsed by hand to keep the crate
//! dependency-free.
//!
//! ```text
//! lvdb create <file> --dim N [--index exact|hnsw]
//! lvdb insert <file> --id N --vector 0.1,0.2,... [--text "..."] [--meta k=v]... [--upsert]
//! lvdb search <file> --vector 0.1,0.2,... [-k N] [--filter k=v]... [--mmap]
//! lvdb stats  <file>
//! lvdb export <file> <out.json>
//! lvdb import <in.json> <file>
//! ```
//!
//! A `--vector`/`--filter` value of `@path` reads the value from a file.

use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::process::ExitCode;

use light_vector_db::{AnnParams, IndexKind, Metadata, MmapDb, Record, SearchResult, VectorDb};

type CliResult = Result<(), Box<dyn Error>>;

const USAGE: &str = "\
lvdb — a command-line tool for light_vector_db

USAGE:
    lvdb <command> [args]

COMMANDS:
    create <file> --dim N [--index exact|hnsw]
        Create a new empty database file.

    insert <file> --id N --vector 0.1,0.2,... [--text \"...\"] [--meta k=v]... [--upsert]
        Add (or, with --upsert, replace) a record.

    search <file> --vector 0.1,0.2,... [-k N] [--filter k=v]... [--mmap]
        Search for the nearest records; prints ranked hits.
        --mmap scans the file in place without loading vectors into memory.

    stats <file>
        Show record count, dimension, index kind, and file size.

    export <file> <out.json>      Write a human-readable JSON copy.
    import <in.json> <file>       Build a database file from JSON.

    help                          Show this message.

NOTES:
    A --vector or --filter value of the form @path reads it from a file.
    Vectors are comma- or whitespace-separated floats (brackets ok).";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: &[String]) -> CliResult {
    let Some((command, rest)) = args.split_first() else {
        println!("{USAGE}");
        return Ok(());
    };
    match command.as_str() {
        "create" => create(rest),
        "insert" => insert(rest),
        "search" => search(rest),
        "stats" => stats(rest),
        "export" => export(rest),
        "import" => import(rest),
        "help" | "--help" | "-h" => {
            println!("{USAGE}");
            Ok(())
        }
        other => Err(format!("unknown command '{other}'\n\n{USAGE}").into()),
    }
}

fn create(rest: &[String]) -> CliResult {
    let args = Parsed::parse(rest, &["upsert"])?;
    let file = args.positional(0, "file")?;
    let dim: usize = args.required("dim")?.parse()?;
    let index_kind = parse_index_kind(args.get("index").unwrap_or("exact"))?;

    let db = VectorDb::with_index(dim, index_kind)?;
    db.save_to_path(file)?;
    println!(
        "Created {file}: dim={dim}, index={}",
        index_kind_label(index_kind)
    );
    Ok(())
}

fn insert(rest: &[String]) -> CliResult {
    let args = Parsed::parse(rest, &["upsert"])?;
    let file = args.positional(0, "file")?;
    let id: u64 = args.required("id")?.parse()?;
    let vector = parse_vector(args.required("vector")?)?;
    let text = args.get("text").unwrap_or("").to_string();

    let mut record = Record::new(id, vector, text);
    for pair in args.get_all("meta") {
        let (key, value) = pair
            .split_once('=')
            .ok_or_else(|| format!("--meta expects key=value, got '{pair}'"))?;
        record.metadata.insert(key.to_string(), value.to_string());
    }

    let mut db = VectorDb::load_from_path(file)?;
    if args.flag("upsert") {
        db.upsert(record)?;
    } else {
        db.insert(record)?;
    }
    db.save_to_path(file)?;
    println!("Inserted id={id} into {file} ({} records)", db.len());
    Ok(())
}

fn search(rest: &[String]) -> CliResult {
    let args = Parsed::parse(rest, &["mmap"])?;
    let file = args.positional(0, "file")?;
    let query = parse_vector(args.required("vector")?)?;
    let limit: usize = args.get("limit").unwrap_or("10").parse()?;

    let mut filter = Metadata::new();
    for pair in args.get_all("filter") {
        let (key, value) = pair
            .split_once('=')
            .ok_or_else(|| format!("--filter expects key=value, got '{pair}'"))?;
        filter.insert(key.to_string(), value.to_string());
    }

    // --mmap searches the file in place (exact brute force) without loading the
    // vectors into memory; otherwise load the database and use its index.
    let hits = if args.flag("mmap") {
        MmapDb::open(file)?.search_filtered(&query, limit, &filter)?
    } else {
        VectorDb::load_from_path(file)?.search_filtered(&query, limit, &filter)?
    };
    print_hits(&hits);
    Ok(())
}

fn print_hits(hits: &[SearchResult]) {
    if hits.is_empty() {
        println!("no results");
        return;
    }
    for (rank, hit) in hits.iter().enumerate() {
        println!(
            "{:>2}. [{:.4}] id={} {}",
            rank + 1,
            hit.score,
            hit.record.id,
            hit.record.text
        );
    }
}

fn stats(rest: &[String]) -> CliResult {
    let args = Parsed::parse(rest, &[])?;
    let file = args.positional(0, "file")?;
    let db = VectorDb::load_from_path(file)?;
    let bytes = fs::metadata(file).map(|m| m.len()).unwrap_or(0);
    println!("file:      {file}");
    println!("records:   {}", db.len());
    println!(
        "dimension: {}",
        db.dimension()
            .map_or_else(|| "unset".to_string(), |d| d.to_string())
    );
    println!("index:     {}", index_kind_label(db.index_kind()));
    println!("size:      {bytes} bytes");
    Ok(())
}

fn export(rest: &[String]) -> CliResult {
    let args = Parsed::parse(rest, &[])?;
    let file = args.positional(0, "file")?;
    let out = args.positional(1, "out.json")?;
    let db = VectorDb::load_from_path(file)?;
    db.export_json(out)?;
    println!("Exported {file} -> {out} ({} records)", db.len());
    Ok(())
}

fn import(rest: &[String]) -> CliResult {
    let args = Parsed::parse(rest, &[])?;
    let input = args.positional(0, "in.json")?;
    let file = args.positional(1, "file")?;
    let db = VectorDb::import_json(input)?;
    db.save_to_path(file)?;
    println!("Imported {input} -> {file} ({} records)", db.len());
    Ok(())
}

fn parse_index_kind(value: &str) -> Result<IndexKind, Box<dyn Error>> {
    match value {
        "exact" => Ok(IndexKind::Exact),
        "hnsw" => Ok(IndexKind::Hnsw(AnnParams::default())),
        other => Err(format!("unknown index kind '{other}' (expected exact or hnsw)").into()),
    }
}

fn index_kind_label(kind: IndexKind) -> String {
    match kind {
        IndexKind::Exact => "exact".to_string(),
        IndexKind::Hnsw(p) => format!(
            "hnsw(M={}, ef_construction={}, ef_search={})",
            p.max_neighbors, p.ef_construction, p.ef_search
        ),
    }
}

/// Parse a vector from a string, or from a file if the string starts with `@`.
/// Accepts comma- or whitespace-separated floats; square brackets are ignored.
fn parse_vector(value: &str) -> Result<Vec<f32>, Box<dyn Error>> {
    let content = match value.strip_prefix('@') {
        Some(path) => fs::read_to_string(path)?,
        None => value.to_string(),
    };
    let cleaned: String = content
        .chars()
        .map(|c| if c == '[' || c == ']' { ' ' } else { c })
        .collect();
    let mut vector = Vec::new();
    for token in cleaned.split(|c: char| c == ',' || c.is_whitespace()) {
        if token.is_empty() {
            continue;
        }
        vector.push(
            token
                .parse::<f32>()
                .map_err(|_| format!("invalid vector value '{token}'"))?,
        );
    }
    Ok(vector)
}

/// Minimal hand-rolled argument parser: positionals, `--flag value`,
/// `--flag=value`, repeatable flags, boolean flags, and the `-k` alias.
struct Parsed {
    positionals: Vec<String>,
    flags: BTreeMap<String, Vec<String>>,
}

impl Parsed {
    fn parse(tokens: &[String], booleans: &[&str]) -> Result<Self, Box<dyn Error>> {
        let mut positionals = Vec::new();
        let mut flags: BTreeMap<String, Vec<String>> = BTreeMap::new();
        let mut i = 0;
        while i < tokens.len() {
            let token = &tokens[i];
            if let Some(name) = token.strip_prefix("--") {
                if let Some((name, value)) = name.split_once('=') {
                    flags
                        .entry(name.to_string())
                        .or_default()
                        .push(value.to_string());
                } else if booleans.contains(&name) {
                    flags
                        .entry(name.to_string())
                        .or_default()
                        .push("true".to_string());
                } else {
                    let value = tokens
                        .get(i + 1)
                        .ok_or_else(|| format!("flag --{name} needs a value"))?;
                    flags
                        .entry(name.to_string())
                        .or_default()
                        .push(value.clone());
                    i += 1;
                }
            } else if token == "-k" {
                let value = tokens.get(i + 1).ok_or("flag -k needs a value")?;
                flags
                    .entry("limit".to_string())
                    .or_default()
                    .push(value.clone());
                i += 1;
            } else {
                positionals.push(token.clone());
            }
            i += 1;
        }
        Ok(Self { positionals, flags })
    }

    fn positional(&self, index: usize, name: &str) -> Result<&str, Box<dyn Error>> {
        self.positionals
            .get(index)
            .map(String::as_str)
            .ok_or_else(|| format!("missing required argument <{name}>").into())
    }

    /// The last value given for `name`, if any.
    fn get(&self, name: &str) -> Option<&str> {
        self.flags
            .get(name)
            .and_then(|v| v.last())
            .map(String::as_str)
    }

    fn required(&self, name: &str) -> Result<&str, Box<dyn Error>> {
        self.get(name)
            .ok_or_else(|| format!("missing required flag --{name}").into())
    }

    /// All values given for a repeatable flag.
    fn get_all(&self, name: &str) -> &[String] {
        self.flags.get(name).map_or(&[], Vec::as_slice)
    }

    fn flag(&self, name: &str) -> bool {
        self.flags.contains_key(name)
    }
}
