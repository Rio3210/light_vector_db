//! Memory-mapped, read-only search over a `.lvdb` file.
//!
//! [`MmapDb`] opens a database by memory-mapping it and searching the vectors
//! *in place* — the big vectors block is never copied into the heap, so the OS
//! pages it in on demand and a file far larger than RAM can still be scanned.
//! Only the small parts (ids and payloads) are read into memory up front.
//!
//! It is read-only and always searches by exact brute force, ignoring any
//! persisted graph index — the classic memory-mapped scan. For inserts, updates,
//! or approximate search, use [`VectorDb`](crate::VectorDb).

use std::fs::File;
use std::path::Path;

use memmap2::Mmap;

use crate::{
    Encoding, Metadata, Record, SearchResult, VectorDbError, cosine_similarity, validate_numbers,
};

/// A read-only, memory-mapped view of a `.lvdb` database.
#[derive(Debug)]
pub struct MmapDb {
    mmap: Mmap,
    dimension: Option<usize>,
    ids: Vec<u64>,
    payloads: Vec<(String, Metadata)>,
    vectors_offset: usize,
    stride: usize,
    encoding: Encoding,
    bounds: (f32, f32),
}

impl MmapDb {
    /// Memory-map the `.lvdb` file at `path` for read-only search.
    ///
    /// Returns [`VectorDbError::Corrupt`] for a malformed file and
    /// [`VectorDbError::UnsupportedVersion`] for a newer major format version.
    /// Unlike [`VectorDb::load_from_path`](crate::VectorDb::load_from_path), the
    /// vectors are not read here — only the header, ids, and payloads.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, VectorDbError> {
        let file = File::open(path)?;
        // SAFETY: memory-mapping is unsafe because the mapped file must not be
        // modified by another process while the map is alive (that would be a
        // data race). Callers open their own `.lvdb` files for reading; we hold
        // the map for the lifetime of `MmapDb` and never write through it.
        let mmap = unsafe { Mmap::map(&file)? };
        let layout = crate::storage::parse_mmap(&mmap)?;
        Ok(Self {
            mmap,
            dimension: layout.dimension,
            ids: layout.ids,
            payloads: layout.payloads,
            vectors_offset: layout.vectors_offset,
            stride: layout.stride,
            encoding: layout.encoding,
            bounds: layout.bounds,
        })
    }

    /// Number of records in the mapped database.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Whether the database holds no records.
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// The fixed vector dimension, if the file records one.
    pub fn dimension(&self) -> Option<usize> {
        self.dimension
    }

    /// Search for the `k` most similar records (exact brute force).
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>, VectorDbError> {
        self.search_filtered(query, k, &Metadata::new())
    }

    /// Search with a metadata filter. Filtering uses the in-memory payloads;
    /// scoring reads each candidate vector straight from the memory map.
    pub fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        filter: &Metadata,
    ) -> Result<Vec<SearchResult>, VectorDbError> {
        self.validate_query(query)?;
        if k == 0 {
            return Ok(Vec::new());
        }

        let mut buffer = vec![0f32; self.stride];
        let mut scored: Vec<(f32, usize)> = Vec::new();
        for i in 0..self.ids.len() {
            if !filter.is_empty() {
                let (_, metadata) = &self.payloads[i];
                if !filter
                    .iter()
                    .all(|(key, value)| metadata.get(key) == Some(value))
                {
                    continue;
                }
            }
            self.read_vector(i, &mut buffer);
            scored.push((cosine_similarity(query, &buffer), i));
        }

        scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(self.ids[a.1].cmp(&self.ids[b.1])));
        scored.truncate(k);
        Ok(scored
            .into_iter()
            .map(|(score, i)| {
                let mut vector = vec![0f32; self.stride];
                self.read_vector(i, &mut vector);
                let (text, metadata) = self.payloads[i].clone();
                SearchResult {
                    score,
                    record: Record {
                        id: self.ids[i],
                        vector,
                        text,
                        metadata,
                    },
                }
            })
            .collect())
    }

    /// Read vector `i` from the memory map into `buffer` (length == `stride`),
    /// decoding per the file's encoding. Touches only that vector's bytes, so
    /// the OS pages in just what is scored.
    fn read_vector(&self, i: usize, buffer: &mut [f32]) {
        match self.encoding {
            Encoding::Float32 => {
                let start = self.vectors_offset + i * self.stride * 4;
                for (j, slot) in buffer.iter_mut().enumerate() {
                    let offset = start + j * 4;
                    *slot = f32::from_le_bytes(self.mmap[offset..offset + 4].try_into().unwrap());
                }
            }
            Encoding::ScalarU8 => {
                let (min, max) = self.bounds;
                let start = self.vectors_offset + i * self.stride;
                for (j, slot) in buffer.iter_mut().enumerate() {
                    *slot = crate::storage::dequantize(self.mmap[start + j], min, max);
                }
            }
        }
    }

    fn validate_query(&self, query: &[f32]) -> Result<(), VectorDbError> {
        validate_numbers(query)?;
        if let Some(expected) = self.dimension
            && query.len() != expected
        {
            return Err(VectorDbError::DimensionMismatch {
                expected,
                actual: query.len(),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{IndexKind, VectorDb};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_path(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "lvdb-mmap-{tag}-{}.lvdb",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    fn sample_vector(seed: u64, dimension: usize) -> Vec<f32> {
        let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        (0..dimension)
            .map(|_| {
                state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                (z >> 40) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
            })
            .collect()
    }

    /// A memory-mapped scan must match an exact in-memory search over the same
    /// file (both are exact brute force).
    #[test]
    fn mmap_search_matches_vectordb() {
        let dimension = 12;
        let mut db = VectorDb::with_index(dimension, IndexKind::Exact).unwrap();
        for id in 0..40u64 {
            db.insert(Record::new(
                id,
                sample_vector(id, dimension),
                format!("r{id}"),
            ))
            .unwrap();
        }
        let path = temp_path("match");
        db.save_to_path(&path).unwrap();

        let mapped = MmapDb::open(&path).unwrap();
        assert_eq!(mapped.len(), 40);
        assert_eq!(mapped.dimension(), Some(dimension));

        let query = sample_vector(999, dimension);
        let expected = db.search(&query, 5).unwrap();
        let actual = mapped.search(&query, 5).unwrap();
        assert_eq!(expected.len(), actual.len());
        for (e, a) in expected.iter().zip(&actual) {
            assert_eq!(e.record.id, a.record.id);
            assert!((e.score - a.score).abs() < 1e-6);
            assert_eq!(e.record.vector, a.record.vector);
            assert_eq!(e.record.text, a.record.text);
        }
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn mmap_search_respects_filter_and_validates() {
        let mut db = VectorDb::with_dimension(2).unwrap();
        db.insert(Record::new(1, vec![1.0, 0.0], "a").with_metadata("t", "x"))
            .unwrap();
        db.insert(Record::new(2, vec![0.0, 1.0], "b").with_metadata("t", "y"))
            .unwrap();
        let path = temp_path("filter");
        db.save_to_path(&path).unwrap();

        let mapped = MmapDb::open(&path).unwrap();
        let mut filter = Metadata::new();
        filter.insert("t".into(), "y".into());
        let hits = mapped.search_filtered(&[1.0, 0.0], 5, &filter).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].record.id, 2);

        // Query validation still applies.
        assert!(matches!(
            mapped.search(&[1.0, 0.0, 0.0], 5),
            Err(VectorDbError::DimensionMismatch { .. })
        ));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn open_rejects_corrupt_header() {
        let mut db = VectorDb::with_dimension(2).unwrap();
        db.insert(Record::new(1, vec![1.0, 0.0], "a")).unwrap();
        let path = temp_path("corrupt");
        db.save_to_path(&path).unwrap();

        let mut bytes = std::fs::read(&path).unwrap();
        bytes[0] = b'X';
        std::fs::write(&path, bytes).unwrap();
        assert!(matches!(
            MmapDb::open(&path),
            Err(VectorDbError::Corrupt(_))
        ));
        std::fs::remove_file(path).unwrap();
    }
}
