pub(crate) mod activation;

mod array;
pub use array::{SimdArrayError, SimdArrayMut};

mod distribution;

mod elementary;

pub(crate) mod slice;
pub use slice::{PlatformSimdSlice, SimdSlice};

pub(crate) mod util;

pub(crate) mod vector;
