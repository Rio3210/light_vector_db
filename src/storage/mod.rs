//! The `.lvdb` on-disk format and storage engine.
//!
//! Today this is just the [`format`] codec (binary encode/decode with CRC-32).
//! Later milestones add a pager and memory-mapped reads alongside it.

mod format;

pub(crate) use format::{decode, encode};
