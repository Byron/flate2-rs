//! This module contains backend-specific code.

use crate::mem::{CompressError, DecompressError, FlushCompress, FlushDecompress, Status};
use crate::Compression;

/// Traits specifying the interface of the backends.
///
/// Sync + Send are added as a condition to ensure they are available
/// for the frontend.
pub trait Backend: Sync + Send {
    fn total_in(&self) -> u64;
    fn total_out(&self) -> u64;
}

pub trait InflateBackend: Backend {
    fn make(zlib_header: bool, window_bits: u8) -> Self;
    fn decompress(
        &mut self,
        input: &[u8],
        output: &mut [u8],
        flush: FlushDecompress,
    ) -> Result<Status, DecompressError>;
    fn reset(&mut self, zlib_header: bool);
}

pub trait DeflateBackend: Backend {
    fn make(level: Compression, zlib_header: bool, window_bits: u8) -> Self;
    fn compress(
        &mut self,
        input: &[u8],
        output: &mut [u8],
        flush: FlushCompress,
    ) -> Result<Status, CompressError>;
    fn reset(&mut self);
}

#[cfg(feature = "any_c_zlib")]
mod c;
#[cfg(feature = "any_c_zlib")]
pub use self::c::*;

#[cfg(all(not(feature = "any_c_zlib"), feature = "zlib-rs"))]
mod zlib_rs;
#[cfg(all(not(feature = "any_c_zlib"), feature = "zlib-rs"))]
pub use self::zlib_rs::*;

#[cfg(all(not(feature = "any_zlib"), feature = "rust_backend"))]
mod rust;
#[cfg(all(not(feature = "any_zlib"), feature = "rust_backend"))]
pub use self::rust::*;

#[cfg(not(feature = "any_impl"))]
compile_error!(
    "No compression backend selected; enable `zlib`, `zlib-ng`, `zlib-rs`, or use the default `rust_backend` feature."
);

impl std::fmt::Debug for ErrorMessage {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.get().fmt(f)
    }
}
