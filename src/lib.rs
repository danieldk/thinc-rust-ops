#[cfg(target_os = "macos")]
extern crate accelerate_src;

use pyo3::prelude::PyModule;
use pyo3::{pymodule, PyResult, Python};

mod ops;
use ops::RustOps;

#[pymodule]
fn rust_ops(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<RustOps>()?;
    Ok(())
}
