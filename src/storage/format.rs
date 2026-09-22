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

use crate::index::IndexData;
use crate::{Encoding, IndexKind, Metadata, Record, VectorDbError};

/// The database state carried in a `.lvdb` file: dimension, index kind, vector
/// encoding, records, and an optional persisted graph index.
type DecodedDb = (
    Option<usize>,
    IndexKind,
    Encoding,
    BTreeMap<u64, Record>,
    Option<IndexData>,
);

const MAGIC: &[u8; 4] = b"LVDB";
const VERSION_MAJOR: u16 = 1;
// Minor 1 added the optional persisted index section; minor 2 added scalar
// quantization. Backward compatible: an older reader hitting a newer feature
// (an index section it ignores, or a byte count it doesn't expect) either skips
// it or fails cleanly rather than misreading.
const VERSION_MINOR: u16 = 2;
const HEADER_LEN: usize = 64;
/// The header CRC covers everything before it: bytes `[0, HEADER_CRC_OFFSET)`.
const HEADER_CRC_OFFSET: usize = 60;

const FLAG_DIMENSION_PRESENT: u32 = 1;
const FLAG_HAS_INDEX: u32 = 2;

const METRIC_COSINE: u8 = 0;
const ENCODING_F32LE: u8 = 0;
const ENCODING_SCALAR_U8: u8 = 1;
const INDEX_TAG_EXACT: u8 = 0;
const INDEX_TAG_HNSW: u8 = 1;

/// Bytes per vector component for a given encoding.
fn component_bytes(encoding: Encoding) -> usize {
    match encoding {
        Encoding::Float32 => 4,
        Encoding::ScalarU8 => 1,
    }
}

/// Global `(min, max)` across every component of every record's vector, used as
/// the scalar-quantization range. `(0, 0)` when there are no components.
fn scalar_bounds(records: &BTreeMap<u64, Record>) -> (f32, f32) {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for record in records.values() {
        for &value in &record.vector {
            min = min.min(value);
            max = max.max(value);
        }
    }
    if min.is_finite() && max.is_finite() {
        (min, max)
    } else {
        (0.0, 0.0)
    }
}

/// Quantize one component to a byte against the `[min, max]` range.
fn quantize(value: f32, min: f32, max: f32) -> u8 {
    let span = max - min;
    if span <= 0.0 {
        return 0;
    }
    (((value - min) / span) * 255.0).round().clamp(0.0, 255.0) as u8
}

/// Reconstruct a component from its quantized byte (lossy).
pub(crate) fn dequantize(code: u8, min: f32, max: f32) -> f32 {
    min + (f32::from(code) / 255.0) * (max - min)
}

/// Serialize a database's state to `.lvdb` bytes.
pub(crate) fn encode(
    dimension: Option<usize>,
    index_kind: IndexKind,
    encoding: Encoding,
    records: &BTreeMap<u64, Record>,
    index: Option<&IndexData>,
) -> Vec<u8> {
    // The vector stride: the fixed dimension, or the first record's length when
    // the dimension is not yet fixed but records exist.
    let stride = dimension
        .or_else(|| records.values().next().map(|record| record.vector.len()))
        .unwrap_or(0);

    // Scalar quantization derives a global range from the data.
    let (encoding_tag, qmin, qmax) = match encoding {
        Encoding::Float32 => (ENCODING_F32LE, 0.0, 0.0),
        Encoding::ScalarU8 => {
            let (min, max) = scalar_bounds(records);
            (ENCODING_SCALAR_U8, min, max)
        }
    };

    let mut body = Vec::new();
    for record in records.values() {
        body.extend_from_slice(&record.id.to_le_bytes());
    }
    for record in records.values() {
        match encoding {
            Encoding::Float32 => {
                for value in &record.vector {
                    body.extend_from_slice(&value.to_le_bytes());
                }
            }
            Encoding::ScalarU8 => {
                for &value in &record.vector {
                    body.push(quantize(value, qmin, qmax));
                }
            }
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

    // Optional index section: the graph's structure (node ids + adjacency +
    // entry). Vectors are not repeated here — they come from the records above.
    if let Some(index) = index {
        body.push(u8::from(index.entry.is_some()));
        body.extend_from_slice(&index.entry.unwrap_or(0).to_le_bytes());
        body.extend_from_slice(&(index.node_ids.len() as u32).to_le_bytes());
        for id in &index.node_ids {
            body.extend_from_slice(&id.to_le_bytes());
        }
        for adjacency in &index.neighbors {
            body.extend_from_slice(&(adjacency.len() as u32).to_le_bytes());
            for neighbor in adjacency {
                body.extend_from_slice(&neighbor.to_le_bytes());
            }
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
    let mut flags = 0u32;
    if dimension.is_some() {
        flags |= FLAG_DIMENSION_PRESENT;
    }
    if index.is_some() {
        flags |= FLAG_HAS_INDEX;
    }
    header[8..12].copy_from_slice(&flags.to_le_bytes());
    header[12..16].copy_from_slice(&(dimension.unwrap_or(0) as u32).to_le_bytes());
    header[16] = METRIC_COSINE;
    header[17] = encoding_tag;
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
    // Scalar-quantization range lives in the otherwise-reserved [48, 56).
    if encoding_tag == ENCODING_SCALAR_U8 {
        header[48..52].copy_from_slice(&qmin.to_le_bytes());
        header[52..56].copy_from_slice(&qmax.to_le_bytes());
    }
    // header[56..60] reserved
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
    let (encoding, qmin, qmax) = match header[17] {
        ENCODING_F32LE => (Encoding::Float32, 0.0, 0.0),
        ENCODING_SCALAR_U8 => (
            Encoding::ScalarU8,
            f32::from_le_bytes(header[48..52].try_into().unwrap()),
            f32::from_le_bytes(header[52..56].try_into().unwrap()),
        ),
        _ => return Err(VectorDbError::Corrupt("unknown vector encoding")),
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
            let value = match encoding {
                Encoding::Float32 => reader.f32()?,
                Encoding::ScalarU8 => dequantize(reader.u8()?, qmin, qmax),
            };
            vector.push(value);
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

    let index_data = if flags & FLAG_HAS_INDEX != 0 {
        let has_entry = reader.u8()? != 0;
        let entry_raw = reader.u32()?;
        let entry = has_entry.then_some(entry_raw);
        let node_count = reader.u32()? as usize;
        let mut node_ids = Vec::new();
        for _ in 0..node_count {
            node_ids.push(reader.u64()?);
        }
        let mut neighbors = Vec::new();
        for _ in 0..node_count {
            let neighbor_count = reader.u32()? as usize;
            let mut adjacency = Vec::new();
            for _ in 0..neighbor_count {
                adjacency.push(reader.u32()?);
            }
            neighbors.push(adjacency);
        }
        Some(IndexData {
            node_ids,
            neighbors,
            entry,
        })
    } else {
        None
    };

    Ok((dimension, index_kind, encoding, records, index_data))
}

/// The parts of a `.lvdb` file needed to search it via memory-mapping: the
/// small owned sections (ids, payloads) plus where the big vectors block lives.
///
/// The vectors themselves stay in the memory map and are read on demand — never
/// copied into the heap here — so a large file can be searched without loading
/// it all into RAM.
pub(crate) struct MmapLayout {
    pub dimension: Option<usize>,
    pub ids: Vec<u64>,
    pub payloads: Vec<(String, Metadata)>,
    /// Byte offset of the contiguous vectors block within the mapped file.
    pub vectors_offset: usize,
    /// Vector stride (number of components per vector).
    pub stride: usize,
    /// How each component is encoded in the map.
    pub encoding: Encoding,
    /// Scalar-quantization `(min, max)` range (unused for `Float32`).
    pub bounds: (f32, f32),
}

/// Parse the header, ids, and payloads of mapped `.lvdb` bytes, leaving the
/// vectors block untouched in the map.
///
/// Verifies the header checksum but deliberately **not** the body checksum:
/// that would read every vector page and defeat the point of memory-mapping.
/// The vectors region is bounds-checked so on-demand reads stay in range.
pub(crate) fn parse_mmap(bytes: &[u8]) -> Result<MmapLayout, VectorDbError> {
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
    let (encoding, bounds) = match header[17] {
        ENCODING_F32LE => (Encoding::Float32, (0.0, 0.0)),
        ENCODING_SCALAR_U8 => (
            Encoding::ScalarU8,
            (
                f32::from_le_bytes(header[48..52].try_into().unwrap()),
                f32::from_le_bytes(header[52..56].try_into().unwrap()),
            ),
        ),
        _ => return Err(VectorDbError::Corrupt("unknown vector encoding")),
    };
    let count = u64::from_le_bytes(header[20..28].try_into().unwrap()) as usize;
    let stride = u32::from_le_bytes(header[28..32].try_into().unwrap()) as usize;
    if count > 0 && stride == 0 {
        return Err(VectorDbError::Corrupt("records present but zero stride"));
    }

    // Ids come first, right after the header.
    let mut ids = Vec::new();
    {
        let mut reader = Reader::new(&bytes[HEADER_LEN..]);
        for _ in 0..count {
            ids.push(reader.u64()?);
        }
    }

    // The vectors block follows the ids; bounds-check it, then skip over it.
    let vectors_offset = HEADER_LEN
        .checked_add(
            count
                .checked_mul(8)
                .ok_or(VectorDbError::Corrupt("size overflow"))?,
        )
        .ok_or(VectorDbError::Corrupt("size overflow"))?;
    let vectors_len = count
        .checked_mul(stride)
        .and_then(|n| n.checked_mul(component_bytes(encoding)))
        .ok_or(VectorDbError::Corrupt("size overflow"))?;
    let payloads_offset = vectors_offset
        .checked_add(vectors_len)
        .ok_or(VectorDbError::Corrupt("size overflow"))?;
    if payloads_offset > bytes.len() {
        return Err(VectorDbError::Corrupt("vectors section out of range"));
    }

    // Payloads follow the vectors block.
    let mut payloads = Vec::new();
    {
        let mut reader = Reader::new(&bytes[payloads_offset..]);
        for _ in 0..count {
            let text = reader.string()?;
            let meta_count = reader.u32()? as usize;
            let mut metadata = Metadata::new();
            for _ in 0..meta_count {
                let key = reader.string()?;
                let value = reader.string()?;
                metadata.insert(key, value);
            }
            payloads.push((text, metadata));
        }
    }

    Ok(MmapLayout {
        dimension,
        ids,
        payloads,
        vectors_offset,
        stride,
        encoding,
        bounds,
    })
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

    fn u8(&mut self) -> Result<u8, VectorDbError> {
        Ok(self.take(1)?[0])
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
        let bytes = encode(Some(2), IndexKind::Exact, Encoding::Float32, &records, None);
        let (dim, kind, _encoding, decoded, index) = decode(&bytes).unwrap();
        assert_eq!(dim, Some(2));
        assert_eq!(kind, IndexKind::Exact);
        assert_eq!(decoded, records);
        assert!(index.is_none());
    }

    #[test]
    fn round_trip_hnsw_params() {
        let records = sample();
        let params = AnnParams {
            max_neighbors: 12,
            ef_construction: 48,
            ef_search: 24,
        };
        let bytes = encode(
            Some(2),
            IndexKind::Hnsw(params),
            Encoding::Float32,
            &records,
            None,
        );
        let (_, kind, _, _, _) = decode(&bytes).unwrap();
        assert_eq!(kind, IndexKind::Hnsw(params));
    }

    #[test]
    fn round_trip_index_section() {
        let records = sample();
        let index = IndexData {
            node_ids: vec![2, 1],
            neighbors: vec![vec![1], vec![0]],
            entry: Some(0),
        };
        let bytes = encode(
            Some(2),
            IndexKind::Hnsw(AnnParams::default()),
            Encoding::Float32,
            &records,
            Some(&index),
        );
        let (_, _, _, decoded, decoded_index) = decode(&bytes).unwrap();
        assert_eq!(decoded, records);
        assert_eq!(decoded_index, Some(index));
    }

    #[test]
    fn parse_mmap_locates_sections() {
        let records = sample(); // ids 1 and 2, dim 2
        let bytes = encode(Some(2), IndexKind::Exact, Encoding::Float32, &records, None);
        let layout = parse_mmap(&bytes).unwrap();

        assert_eq!(layout.dimension, Some(2));
        assert_eq!(layout.stride, 2);
        assert_eq!(layout.ids, vec![1, 2]);
        assert_eq!(layout.payloads.len(), 2);
        assert_eq!(layout.payloads[0].0, "record 1");
        assert_eq!(layout.payloads[0].1["k"], "v1");

        // Reading the first vector from the reported offset matches the record.
        let start = layout.vectors_offset;
        let x = f32::from_le_bytes(bytes[start..start + 4].try_into().unwrap());
        let y = f32::from_le_bytes(bytes[start + 4..start + 8].try_into().unwrap());
        assert_eq!(vec![x, y], records[&1].vector);
    }

    #[test]
    fn scalar_quantization_round_trip_is_smaller_and_close() {
        let mut records = BTreeMap::new();
        for r in [
            record(1, vec![0.1, 0.9, 0.5]),
            record(2, vec![0.8, 0.2, 0.0]),
        ] {
            records.insert(r.id, r);
        }
        let f32_bytes = encode(Some(3), IndexKind::Exact, Encoding::Float32, &records, None);
        let q_bytes = encode(
            Some(3),
            IndexKind::Exact,
            Encoding::ScalarU8,
            &records,
            None,
        );
        // 3 f32 (12 bytes) vs 3 u8 (3 bytes) per vector.
        assert!(
            q_bytes.len() < f32_bytes.len(),
            "quantized should be smaller"
        );

        let (_, _, encoding, decoded, _) = decode(&q_bytes).unwrap();
        assert_eq!(encoding, Encoding::ScalarU8);
        // Dequantized values are close to the originals (bounded by range/255).
        for (id, original) in &records {
            let restored = &decoded[id].vector;
            for (o, r) in original.vector.iter().zip(restored) {
                assert!((o - r).abs() < 0.01, "component drifted: {o} vs {r}");
            }
        }
    }

    #[test]
    fn round_trip_empty() {
        let records = BTreeMap::new();
        let bytes = encode(None, IndexKind::Exact, Encoding::Float32, &records, None);
        let (dim, kind, _, decoded, _) = decode(&bytes).unwrap();
        assert_eq!(dim, None);
        assert_eq!(kind, IndexKind::Exact);
        assert!(decoded.is_empty());
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = encode(
            Some(2),
            IndexKind::Exact,
            Encoding::Float32,
            &sample(),
            None,
        );
        bytes[0] = b'X';
        assert!(matches!(decode(&bytes), Err(VectorDbError::Corrupt(_))));
    }

    #[test]
    fn rejects_truncation() {
        let bytes = encode(
            Some(2),
            IndexKind::Exact,
            Encoding::Float32,
            &sample(),
            None,
        );
        let truncated = &bytes[..bytes.len() - 3];
        assert!(matches!(decode(truncated), Err(VectorDbError::Corrupt(_))));
    }

    #[test]
    fn rejects_body_corruption() {
        let mut bytes = encode(
            Some(2),
            IndexKind::Exact,
            Encoding::Float32,
            &sample(),
            None,
        );
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;
        assert!(matches!(decode(&bytes), Err(VectorDbError::Corrupt(_))));
    }

    #[test]
    fn rejects_future_major_version() {
        let mut bytes = encode(
            Some(2),
            IndexKind::Exact,
            Encoding::Float32,
            &sample(),
            None,
        );
        // Bump the major version and repair the header CRC so only the version
        // check can reject it.
        bytes[4..6].copy_from_slice(&2u16.to_le_bytes());
        let header_crc = crc32(&bytes[..HEADER_CRC_OFFSET]);
        bytes[HEADER_CRC_OFFSET..HEADER_LEN].copy_from_slice(&header_crc.to_le_bytes());
        assert!(matches!(
            decode(&bytes),
            Err(VectorDbError::UnsupportedVersion { major: 2, .. })
        ));
    }
}
