#[cfg(target_os = "macos")]
extern crate accelerate_src;

#[cfg(target_os = "windows")]
extern crate intel_mkl_src;

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
extern crate blis_src;

use pyo3::prelude::PyModule;
use pyo3::{pymodule, PyResult, Python};

mod ops;
use ops::RustOps;

mod elementary;

pub(crate) mod simd;

#[pymodule]
fn rust_ops(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<RustOps>()?;
    Ok(())
}
