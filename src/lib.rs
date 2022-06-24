use pyo3::prelude::PyModule;
use pyo3::{pymodule, PyResult, Python};

#[pymodule]
fn rust_ops(_py: Python<'_>, _m: &PyModule) -> PyResult<()> {
    Ok(())
}
