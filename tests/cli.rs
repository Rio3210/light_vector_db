//! End-to-end tests for the `lvdb` command-line tool.
//!
//! Cargo builds the binary and exposes its path as `CARGO_BIN_EXE_lvdb`, so
//! these drive the real executable the way a user would.

use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn lvdb() -> Command {
    Command::new(env!("CARGO_BIN_EXE_lvdb"))
}

fn temp_db(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "lvdb-cli-{tag}-{}.lvdb",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ))
}

fn run(args: &[&str]) -> (bool, String, String) {
    let output = lvdb().args(args).output().expect("failed to run lvdb");
    (
        output.status.success(),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

#[test]
fn create_insert_search_stats_flow() {
    let db = temp_db("flow");
    let path = db.to_str().unwrap();

    let (ok, _, err) = run(&["create", path, "--dim", "3"]);
    assert!(ok, "create failed: {err}");

    let (ok, _, err) = run(&[
        "insert",
        path,
        "--id",
        "1",
        "--vector",
        "1,0,0",
        "--text",
        "first",
        "--meta",
        "topic=rust",
    ]);
    assert!(ok, "insert 1 failed: {err}");
    let (ok, _, err) = run(&[
        "insert", path, "--id", "2", "--vector", "0,0,1", "--text", "second",
    ]);
    assert!(ok, "insert 2 failed: {err}");

    // Nearest to [0.9, 0, 0.1] is record 1.
    let (ok, out, err) = run(&["search", path, "--vector", "0.9,0,0.1", "-k", "1"]);
    assert!(ok, "search failed: {err}");
    assert!(out.contains("id=1"), "unexpected search output: {out}");

    // Metadata filter narrows to the matching record.
    let (ok, out, _) = run(&[
        "search",
        path,
        "--vector",
        "0,0,1",
        "--filter",
        "topic=rust",
    ]);
    assert!(ok);
    assert!(
        out.contains("id=1") && !out.contains("id=2"),
        "filter output: {out}"
    );

    let (ok, out, err) = run(&["stats", path]);
    assert!(ok, "stats failed: {err}");
    assert!(out.contains("records:   2"), "stats output: {out}");
    assert!(out.contains("dimension: 3"), "stats output: {out}");

    let _ = std::fs::remove_file(&db);
}

#[test]
fn duplicate_insert_fails_without_upsert() {
    let db = temp_db("dup");
    let path = db.to_str().unwrap();
    run(&["create", path, "--dim", "2"]);
    run(&["insert", path, "--id", "1", "--vector", "1,0"]);

    let (ok, _, err) = run(&["insert", path, "--id", "1", "--vector", "0,1"]);
    assert!(!ok, "duplicate insert should fail");
    assert!(err.contains("already exists"), "stderr: {err}");

    // With --upsert it succeeds.
    let (ok, _, err) = run(&["insert", path, "--id", "1", "--vector", "0,1", "--upsert"]);
    assert!(ok, "upsert failed: {err}");

    let _ = std::fs::remove_file(&db);
}

#[test]
fn export_then_import_round_trips() {
    let db = temp_db("exp");
    let path = db.to_str().unwrap();
    let json = db.with_extension("json");
    let json_path = json.to_str().unwrap();
    let db2 = db.with_extension("copy.lvdb");
    let db2_path = db2.to_str().unwrap();

    run(&["create", path, "--dim", "2", "--index", "hnsw"]);
    run(&[
        "insert", path, "--id", "7", "--vector", "0.4,0.8", "--meta", "src=test",
    ]);

    let (ok, _, err) = run(&["export", path, json_path]);
    assert!(ok, "export failed: {err}");
    let (ok, _, err) = run(&["import", json_path, db2_path]);
    assert!(ok, "import failed: {err}");

    let (ok, out, _) = run(&["stats", db2_path]);
    assert!(ok);
    assert!(out.contains("records:   1"), "stats: {out}");
    assert!(out.contains("hnsw"), "index kind should survive: {out}");

    for p in [&db, &json, &db2] {
        let _ = std::fs::remove_file(p);
    }
}

#[test]
fn delete_and_compact_flow() {
    let db = temp_db("del");
    let path = db.to_str().unwrap();
    run(&["create", path, "--dim", "2", "--index", "hnsw"]);
    run(&[
        "insert", path, "--id", "1", "--vector", "1,0", "--text", "keep",
    ]);
    run(&[
        "insert", path, "--id", "2", "--vector", "0,1", "--text", "drop",
    ]);

    let (ok, out, err) = run(&["delete", path, "--id", "2"]);
    assert!(ok, "delete failed: {err}");
    assert!(out.contains("1 records"), "delete output: {out}");

    // The deleted record is gone from search and stats.
    let (_, out, _) = run(&["search", path, "--vector", "0,1", "-k", "5"]);
    assert!(!out.contains("id=2"), "deleted record still found: {out}");
    let (_, out, _) = run(&["stats", path]);
    assert!(out.contains("records:   1"), "stats after delete: {out}");

    let (ok, _, err) = run(&["compact", path]);
    assert!(ok, "compact failed: {err}");

    let _ = std::fs::remove_file(&db);
}

#[test]
fn search_mmap_flag_works() {
    let db = temp_db("mmap");
    let path = db.to_str().unwrap();
    run(&["create", path, "--dim", "3"]);
    run(&[
        "insert", path, "--id", "1", "--vector", "1,0,0", "--text", "near",
    ]);
    run(&[
        "insert", path, "--id", "2", "--vector", "0,0,1", "--text", "far",
    ]);

    let (ok, out, err) = run(&["search", path, "--vector", "0.9,0,0.1", "-k", "1", "--mmap"]);
    assert!(ok, "mmap search failed: {err}");
    assert!(out.contains("id=1"), "mmap search output: {out}");

    let _ = std::fs::remove_file(&db);
}

#[test]
fn missing_args_fail_cleanly() {
    let (ok, _, err) = run(&["create", "/tmp/whatever.lvdb"]);
    assert!(!ok);
    assert!(err.contains("--dim"), "stderr: {err}");

    let (ok, _, err) = run(&["frobnicate"]);
    assert!(!ok);
    assert!(err.contains("unknown command"), "stderr: {err}");
}
