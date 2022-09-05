//! SIMD-optimized methods for [`ndarray`](ndarray) arrays.

mod activation;

mod array;
pub use array::{SimdArrayError, SimdArrayMut};

mod distribution;

mod elementary;

mod slice;

mod util;

mod vector;
