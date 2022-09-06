//! SIMD-optimized methods for [`ndarray`](ndarray) arrays.

mod activation;

mod array;
pub use array::{SimdArrayError, SimdArrayMut};

mod distribution;

mod elementary;

mod slice;

#[doc(hidden)]
pub use slice::{PlatformSimdSlice, SimdSlice};

mod util;

mod vector;
