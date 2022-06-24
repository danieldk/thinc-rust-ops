use ndarray::linalg::general_mat_mul;
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::{pyclass, pymethods, PyResult, Python};

#[pyclass(subclass)]
pub struct RustOps {}

#[pymethods]
impl RustOps {
    #[new]
    fn new() -> Self {
        RustOps {}
    }

    #[args("*", out = "None", trans1 = "false", trans2 = "false")]
    fn gemm<'p>(
        &self,
        py: Python<'p>,
        a: PyReadonlyArray2<f32>,
        b: PyReadonlyArray2<f32>,
        out: Option<&'p PyArray2<f32>>,
        trans1: bool,
        trans2: bool,
    ) -> PyResult<&'p PyArray2<f32>> {
        let a = a.as_array();
        let b = b.as_array();
        let at = if trans1 { a.t() } else { a };
        let bt = if trans2 { b.t() } else { b };

        if at.shape()[1] != bt.shape()[0] {
            return Err(PyValueError::new_err(format!(
                "Matrix shapes do not align: {:?} {:?}",
                at.shape(),
                bt.shape()
            )));
        }

        let out = match out {
            Some(out) => out,
            None => unsafe { PyArray2::new(py, (at.dim().0, bt.dim().1), false) },
        };

        let mut c = unsafe { out.as_array_mut() };

        general_mat_mul(1.0, &at, &bt, 0.0, &mut c);

        Ok(out)
    }
}
