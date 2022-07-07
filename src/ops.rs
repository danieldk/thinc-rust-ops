use ndarray::linalg::general_mat_mul;
use numpy::{PyArray2, PyArrayDyn, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::{pyclass, pymethods, FromPyObject, IntoPy, PyObject, PyResult, Python};

use crate::simd::{platform_arrays, Array};

#[derive(FromPyObject)]
enum PyArrayDynFloat<'a> {
    F32(&'a PyArrayDyn<f32>),
    F64(&'a PyArrayDyn<f64>),
}

impl<'a> PyArrayDynFloat<'a> {
    fn copy_array<'py>(&'a self, py: Python<'py>) -> PyArrayDynFloat<'py> {
        match self {
            PyArrayDynFloat::F32(a) => {
                PyArrayDynFloat::F32(PyArrayDyn::from_array(py, &a.readonly().as_array()))
            }
            PyArrayDynFloat::F64(a) => {
                PyArrayDynFloat::F64(PyArrayDyn::from_array(py, &a.readonly().as_array()))
            }
        }
    }
}

impl<'a> IntoPy<PyObject> for PyArrayDynFloat<'a> {
    fn into_py(self, py: Python<'_>) -> PyObject {
        match self {
            PyArrayDynFloat::F32(a) => a.into_py(py),
            PyArrayDynFloat::F64(a) => a.into_py(py),
        }
    }
}

#[pyclass(subclass)]
pub struct RustOps {
    array_f32: Box<dyn Array<Scalar = f32>>,
    array_f64: Box<dyn Array<Scalar = f64>>,
}

#[pymethods]
impl RustOps {
    #[new]
    fn new() -> Self {
        let (array_f32, array_f64) = platform_arrays();
        RustOps {
            array_f32,
            array_f64,
        }
    }

    #[args(
        slope = "1.0",
        offset = "0.0",
        min_val = "0.0",
        max_val = "1.0",
        inplace = "false"
    )]
    fn clipped_linear<'py>(
        &self,
        py: Python<'py>,
        x: PyArrayDynFloat<'py>,
        slope: f64,
        offset: f64,
        min_val: f64,
        max_val: f64,
        inplace: bool,
    ) -> PyResult<PyArrayDynFloat<'py>> {
        Self::elementwise_op(
            py,
            x,
            inplace,
            |s| {
                self.array_f32.clipped_linear(
                    s,
                    slope as f32,
                    offset as f32,
                    min_val as f32,
                    max_val as f32,
                )
            },
            |s| {
                self.array_f64
                    .clipped_linear(s, slope, offset, min_val, max_val)
            },
        )
    }

    #[args(inplace = "false")]
    fn hard_sigmoid<'py>(
        &self,
        py: Python<'py>,
        x: PyArrayDynFloat<'py>,
        inplace: bool,
    ) -> PyResult<PyArrayDynFloat<'py>> {
        Self::elementwise_op(
            py,
            x,
            inplace,
            |s| self.array_f32.hard_sigmoid(s),
            |s| self.array_f64.hard_sigmoid(s),
        )
    }

    #[args(inplace = "false")]
    fn hard_tanh<'py>(
        &self,
        py: Python<'py>,
        x: PyArrayDynFloat<'py>,
        inplace: bool,
    ) -> PyResult<PyArrayDynFloat<'py>> {
        Self::elementwise_op(
            py,
            x,
            inplace,
            |s| self.array_f32.hard_tanh(s),
            |s| self.array_f64.hard_tanh(s),
        )
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

    #[args(inplace = "false")]
    fn relu<'py>(
        &self,
        py: Python<'py>,
        x: PyArrayDynFloat<'py>,
        inplace: bool,
    ) -> PyResult<PyArrayDynFloat<'py>> {
        Self::elementwise_op(
            py,
            x,
            inplace,
            |s| self.array_f32.relu(s),
            |s| self.array_f64.relu(s),
        )
    }

    #[args(inplace = "false")]
    fn sigmoid<'py>(
        &self,
        py: Python<'py>,
        x: PyArrayDynFloat<'py>,
        inplace: bool,
    ) -> PyResult<PyArrayDynFloat<'py>> {
        Self::elementwise_op(
            py,
            x,
            inplace,
            |s| self.array_f32.logistic_cdf(s),
            |s| self.array_f64.logistic_cdf(s),
        )
    }

    #[args(inplace = "false")]
    fn swish<'py>(
        &self,
        py: Python<'py>,
        x: PyArrayDynFloat<'py>,
        inplace: bool,
    ) -> PyResult<PyArrayDynFloat<'py>> {
        Self::elementwise_op(
            py,
            x,
            inplace,
            |s| self.array_f32.swish(s),
            |s| self.array_f64.swish(s),
        )
    }
}

impl RustOps {
    fn elementwise_op<'py>(
        py: Python<'py>,
        mut x: PyArrayDynFloat<'py>,
        inplace: bool,
        apply_f32: impl Fn(&mut [f32]) + Sync,
        apply_f64: impl Fn(&mut [f64]) + Sync,
    ) -> PyResult<PyArrayDynFloat<'py>> {
        if !inplace {
            x = x.copy_array(py)
        }

        match x {
            PyArrayDynFloat::F32(x) => {
                let s = unsafe { x.as_slice_mut()? };
                py.allow_threads(|| apply_f32(s));
            }
            PyArrayDynFloat::F64(x) => {
                let s = unsafe { x.as_slice_mut()? };
                py.allow_threads(|| apply_f64(s));
            }
        };

        Ok(x)
    }
}
