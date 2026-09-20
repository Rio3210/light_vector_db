//! The `.lvdb` binary file format (v1).
//!
//! A single, portable, little-endian file. The layout is a fixed 64-byte header
//! followed by three sections written in ascending id order:
//!
//! ```text
//! [ HEADER 64B ]              magic, version, dimension, counts, checksums
//! [ IDS       count x u64 ]   the id of each record, in order
//! [ VECTORS   count x dim x f32 ]  a contiguous, fixed-stride matrix (mmap-ready)
//! [ PAYLOADS  count x (text + metadata) ]  variable-length, in the same order
//! ```
//!
//! The vectors form one contiguous fixed-stride block so a future milestone can
//! memory-map them as a matrix (roadmap M4). This v1 reads sequentially; random
//! access via an offset table and mmap comes later. The format is versioned:
//! readers refuse a newer *major* version.
//!
//! Integrity is checked twice — a CRC-32 over the header and a CRC-32 over the
//! body — so truncation or corruption is caught at load rather than surfacing as
//! garbage results.

use std::collections::BTreeMap;

use crate::{IndexKind, Metadata, Record, VectorDbError};

/// The database state carried in a `.lvdb` file: dimension, index kind, records.
type DecodedDb = (Option<usize>, IndexKind, BTreeMap<u64, Record>);

const MAGIC: &[u8; 4] = b"LVDB";
const VERSION_MAJOR: u16 = 1;
const VERSION_MINOR: u16 = 0;
const HEADER_LEN: usize = 64;
/// The header CRC covers everything before it: bytes `[0, HEADER_CRC_OFFSET)`.
const HEADER_CRC_OFFSET: usize = 60;

const FLAG_DIMENSION_PRESENT: u32 = 1;

const METRIC_COSINE: u8 = 0;
const ENCODING_F32LE: u8 = 0;
const INDEX_TAG_EXACT: u8 = 0;
const INDEX_TAG_HNSW: u8 = 1;

/// Serialize a database's state to `.lvdb` bytes.
pub(crate) fn encode(
    dimension: Option<usize>,
    index_kind: IndexKind,
    records: &BTreeMap<u64, Record>,
) -> Vec<u8> {
    // The vector stride: the fixed dimension, or the first record's length when
    // the dimension is not yet fixed but records exist.
    let stride = dimension
        .or_else(|| records.values().next().map(|record| record.vector.len()))
        .unwrap_or(0);

    let mut body = Vec::new();
    for record in records.values() {
        body.extend_from_slice(&record.id.to_le_bytes());
    }
    for record in records.values() {
        for value in &record.vector {
            body.extend_from_slice(&value.to_le_bytes());
        }
    }
    for record in records.values() {
        put_str(&mut body, &record.text);
        body.extend_from_slice(&(record.metadata.len() as u32).to_le_bytes());
        for (key, value) in &record.metadata {
            put_str(&mut body, key);
            put_str(&mut body, value);
        }
    }

    let (index_tag, params) = match index_kind {
        IndexKind::Exact => (INDEX_TAG_EXACT, None),
        IndexKind::Hnsw(params) => (INDEX_TAG_HNSW, Some(params)),
    };

    let mut header = vec![0u8; HEADER_LEN];
    header[0..4].copy_from_slice(MAGIC);
    header[4..6].copy_from_slice(&VERSION_MAJOR.to_le_bytes());
    header[6..8].copy_from_slice(&VERSION_MINOR.to_le_bytes());
    let flags = if dimension.is_some() {
        FLAG_DIMENSION_PRESENT
    } else {
        0
    };
    header[8..12].copy_from_slice(&flags.to_le_bytes());
    header[12..16].copy_from_slice(&(dimension.unwrap_or(0) as u32).to_le_bytes());
    header[16] = METRIC_COSINE;
    header[17] = ENCODING_F32LE;
    header[18] = index_tag;
    // header[19] reserved
    header[20..28].copy_from_slice(&(records.len() as u64).to_le_bytes());
    header[28..32].copy_from_slice(&(stride as u32).to_le_bytes());
    if let Some(params) = params {
        header[32..36].copy_from_slice(&(params.max_neighbors as u32).to_le_bytes());
        header[36..40].copy_from_slice(&(params.ef_construction as u32).to_le_bytes());
        header[40..44].copy_from_slice(&(params.ef_search as u32).to_le_bytes());
    }
    header[44..48].copy_from_slice(&crc32(&body).to_le_bytes());
    // header[48..60] reserved
    let header_crc = crc32(&header[..HEADER_CRC_OFFSET]);
    header[HEADER_CRC_OFFSET..HEADER_LEN].copy_from_slice(&header_crc.to_le_bytes());

    header.extend_from_slice(&body);
    header
}

/// Parse `.lvdb` bytes back into a database's state.
pub(crate) fn decode(bytes: &[u8]) -> Result<DecodedDb, VectorDbError> {
    if bytes.len() < HEADER_LEN {
        return Err(VectorDbError::Corrupt("file shorter than header"));
    }
    let header = &bytes[..HEADER_LEN];
    if &header[0..4] != MAGIC {
        return Err(VectorDbError::Corrupt("bad magic: not an .lvdb file"));
    }
    let stored_header_crc =
        u32::from_le_bytes(header[HEADER_CRC_OFFSET..HEADER_LEN].try_into().unwrap());
    if crc32(&header[..HEADER_CRC_OFFSET]) != stored_header_crc {
        return Err(VectorDbError::Corrupt("header checksum mismatch"));
    }

    let major = u16::from_le_bytes(header[4..6].try_into().unwrap());
    let minor = u16::from_le_bytes(header[6..8].try_into().unwrap());
    if major != VERSION_MAJOR {
        return Err(VectorDbError::UnsupportedVersion { major, minor });
    }

    let flags = u32::from_le_bytes(header[8..12].try_into().unwrap());
    let dimension = if flags & FLAG_DIMENSION_PRESENT != 0 {
        Some(u32::from_le_bytes(header[12..16].try_into().unwrap()) as usize)
    } else {
        None
    };
    let index_tag = header[18];
    let count = u64::from_le_bytes(header[20..28].try_into().unwrap()) as usize;
    let stride = u32::from_le_bytes(header[28..32].try_into().unwrap()) as usize;
    let index_kind = match index_tag {
        INDEX_TAG_EXACT => IndexKind::Exact,
        INDEX_TAG_HNSW => IndexKind::Hnsw(crate::AnnParams {
            max_neighbors: u32::from_le_bytes(header[32..36].try_into().unwrap()) as usize,
            ef_construction: u32::from_le_bytes(header[36..40].try_into().unwrap()) as usize,
            ef_search: u32::from_le_bytes(header[40..44].try_into().unwrap()) as usize,
        }),
        _ => return Err(VectorDbError::Corrupt("unknown index kind")),
    };

    let stored_body_crc = u32::from_le_bytes(header[44..48].try_into().unwrap());
    let body = &bytes[HEADER_LEN..];
    if crc32(body) != stored_body_crc {
        return Err(VectorDbError::Corrupt("body checksum mismatch"));
    }
    if count > 0 && stride == 0 {
        return Err(VectorDbError::Corrupt("records present but zero stride"));
    }

    let mut reader = Reader::new(body);
    let mut ids = Vec::with_capacity(count);
    for _ in 0..count {
        ids.push(reader.u64()?);
    }
    let mut vectors = Vec::with_capacity(count);
    for _ in 0..count {
        let mut vector = Vec::with_capacity(stride);
        for _ in 0..stride {
            vector.push(reader.f32()?);
        }
        vectors.push(vector);
    }

    let mut records = BTreeMap::new();
    for (id, vector) in ids.into_iter().zip(vectors) {
        let text = reader.string()?;
        let meta_count = reader.u32()? as usize;
        let mut metadata = Metadata::new();
        for _ in 0..meta_count {
            let key = reader.string()?;
            let value = reader.string()?;
            metadata.insert(key, value);
        }
        records.insert(
            id,
            Record {
                id,
                vector,
                text,
                metadata,
            },
        );
    }

    Ok((dimension, index_kind, records))
}

/// Append a length-prefixed UTF-8 string to `buf`.
fn put_str(buf: &mut Vec<u8>, value: &str) {
    buf.extend_from_slice(&(value.len() as u32).to_le_bytes());
    buf.extend_from_slice(value.as_bytes());
}

/// A bounds-checked cursor over the body bytes.
struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], VectorDbError> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or(VectorDbError::Corrupt("length overflow"))?;
        if end > self.buf.len() {
            return Err(VectorDbError::Corrupt("unexpected end of file"));
        }
        let slice = &self.buf[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn u32(&mut self) -> Result<u32, VectorDbError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64, VectorDbError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn f32(&mut self) -> Result<f32, VectorDbError> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn string(&mut self) -> Result<String, VectorDbError> {
        let len = self.u32()? as usize;
        let bytes = self.take(len)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| VectorDbError::Corrupt("invalid utf-8 text"))
    }
}

/// CRC-32 (IEEE 802.3, reflected). Table-free to keep the crate dependency-free.
pub(crate) fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AnnParams;

    fn record(id: u64, vector: Vec<f32>) -> Record {
        Record::new(id, vector, format!("record {id}")).with_metadata("k", format!("v{id}"))
    }

    fn sample() -> BTreeMap<u64, Record> {
        let mut records = BTreeMap::new();
        for r in [record(1, vec![1.0, 0.0]), record(2, vec![0.0, 1.0])] {
            records.insert(r.id, r);
        }
        records
    }

    #[test]
    fn crc32_matches_known_vector() {
        // The standard CRC-32 of "123456789" is 0xCBF43926.
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn round_trip_exact() {
        let records = sample();
        let bytes = encode(Some(2), IndexKind::Exact, &records);
        let (dim, kind, decoded) = decode(&bytes).unwrap();
        assert_eq!(dim, Some(2));
        assert_eq!(kind, IndexKind::Exact);
        assert_eq!(decoded, records);
    }

    #[test]
    fn round_trip_hnsw_params() {
        let records = sample();
        let params = AnnParams {
            max_neighbors: 12,
            ef_construction: 48,
            ef_search: 24,
        };
        let bytes = encode(Some(2), IndexKind::Hnsw(params), &records);
        let (_, kind, _) = decode(&bytes).unwrap();
        assert_eq!(kind, IndexKind::Hnsw(params));
    }

    #[test]
    fn round_trip_empty() {
        let records = BTreeMap::new();
        let bytes = encode(None, IndexKind::Exact, &records);
        let (dim, kind, decoded) = decode(&bytes).unwrap();
        assert_eq!(dim, None);
        assert_eq!(kind, IndexKind::Exact);
        assert!(decoded.is_empty());
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = encode(Some(2), IndexKind::Exact, &sample());
        bytes[0] = b'X';
        assert!(matches!(decode(&bytes), Err(VectorDbError::Corrupt(_))));
    }

    #[test]
    fn rejects_truncation() {
        let bytes = encode(Some(2), IndexKind::Exact, &sample());
        let truncated = &bytes[..bytes.len() - 3];
        assert!(matches!(decode(truncated), Err(VectorDbError::Corrupt(_))));
    }

    #[test]
    fn rejects_body_corruption() {
        let mut bytes = encode(Some(2), IndexKind::Exact, &sample());
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;
        assert!(matches!(decode(&bytes), Err(VectorDbError::Corrupt(_))));
    }

    #[test]
    fn rejects_future_major_version() {
        let mut bytes = encode(Some(2), IndexKind::Exact, &sample());
        // Bump the major version and repair the header CRC so only the version
        // check can reject it.
        bytes[4..6].copy_from_slice(&2u16.to_le_bytes());
        let header_crc = crc32(&bytes[..HEADER_CRC_OFFSET]);
        bytes[HEADER_CRC_OFFSET..HEADER_LEN].copy_from_slice(&header_crc.to_le_bytes());
        assert!(matches!(
            decode(&bytes),
            Err(VectorDbError::UnsupportedVersion { major: 2, minor: 0 })
        ));
    }
}
