pub(crate) mod activation;

mod distribution;

mod elementary;

mod slice;
pub use slice::{all_platform_arrays, platform_arrays, SimdSlice};

pub(crate) mod util;

pub(crate) mod vector;