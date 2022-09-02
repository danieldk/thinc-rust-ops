use ndarray::linalg::general_mat_mul;
use ndarray::ArrayViewMutD;
use numpy::{PyArray2, PyArrayDyn, PyReadonlyArray2};
use pyo3::exceptions::{self, PyValueError};
use pyo3::{pyclass, pymethods, FromPyObject, IntoPy, PyObject, PyResult, Python};
use simd_array::{SimdArrayError, SimdArrayMut};

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

    fn shape(&self) -> &[usize] {
        match self {
            PyArrayDynFloat::F32(a) => a.shape(),
            PyArrayDynFloat::F64(a) => a.shape(),
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
pub struct RustOps;

#[pymethods]
impl RustOps {
    #[new]
    fn new() -> Self {
        RustOps
    }

    #[allow(clippy::too_many_arguments)]
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
            |mut a| a.clipped_linear(slope as f32, offset as f32, min_val as f32, max_val as f32),
            |mut a| a.clipped_linear(slope, offset, min_val, max_val),
        )
    }

    #[args(inplace = "false")]
    fn gelu<'py>(
        &self,
        py: Python<'py>,
        x: PyArrayDynFloat<'py>,
        inplace: bool,
    ) -> PyResult<PyArrayDynFloat<'py>> {
        Self::elementwise_op(py, x, inplace, |mut a| a.gelu(), |mut a| a.gelu())
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
            |mut a| a.hard_sigmoid(),
            |mut a| a.hard_sigmoid(),
        )
    }

    #[args(inplace = "false")]
    fn hard_tanh<'py>(
        &self,
        py: Python<'py>,
        x: PyArrayDynFloat<'py>,
        inplace: bool,
    ) -> PyResult<PyArrayDynFloat<'py>> {
        Self::elementwise_op(py, x, inplace, |mut a| a.hard_tanh(), |mut a| a.hard_tanh())
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
        Self::elementwise_op(py, x, inplace, |mut a| a.relu(), |mut a| a.relu())
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
            |mut a| a.logistic_cdf(),
            |mut a| a.logistic_cdf(),
        )
    }

    #[args(axis = -1, inplace = "false", temperature = "1.0")]
    fn softmax<'py>(
        &self,
        py: Python<'py>,
        mut x: PyArrayDynFloat<'py>,
        inplace: bool,
        axis: isize,
        temperature: f64,
    ) -> PyResult<PyArrayDynFloat<'py>> {
        let shape = x.shape().to_owned();

        // FIXME: add support for other axes.
        assert!(axis == -1 || axis == shape.len() as isize - 1);

        if !inplace {
            x = x.copy_array(py)
        }

        let temperature = if temperature == 1.0 {
            None
        } else {
            Some(temperature)
        };

        let r = match x {
            PyArrayDynFloat::F32(a) => {
                let mut a = unsafe { a.as_array_mut() };
                a.softmax(*shape.last().unwrap(), temperature.map(|v| v as f32))
            }
            PyArrayDynFloat::F64(a) => {
                let mut a = unsafe { a.as_array_mut() };
                a.softmax(*shape.last().unwrap(), temperature)
            }
        };

        simd_array_error_to_py(r)?;

        Ok(x)
    }

    #[args(inplace = "false")]
    fn swish<'py>(
        &self,
        py: Python<'py>,
        x: PyArrayDynFloat<'py>,
        inplace: bool,
    ) -> PyResult<PyArrayDynFloat<'py>> {
        Self::elementwise_op(py, x, inplace, |mut a| a.swish(), |mut a| a.swish())
    }
}

impl RustOps {
    fn elementwise_op<'py>(
        py: Python<'py>,
        mut x: PyArrayDynFloat<'py>,
        inplace: bool,
        apply_f32: impl Fn(ArrayViewMutD<f32>) -> Result<(), SimdArrayError> + Sync,
        apply_f64: impl Fn(ArrayViewMutD<f64>) -> Result<(), SimdArrayError> + Sync,
    ) -> PyResult<PyArrayDynFloat<'py>> {
        if !inplace {
            x = x.copy_array(py)
        }

        let r = match x {
            PyArrayDynFloat::F32(x) => {
                let mut a = unsafe { x.as_array_mut() };
                py.allow_threads(|| apply_f32(a.view_mut()))
            }
            PyArrayDynFloat::F64(x) => {
                let mut a = unsafe { x.as_array_mut() };
                py.allow_threads(|| apply_f64(a.view_mut()))
            }
        };

        simd_array_error_to_py(r)?;

        Ok(x)
    }
}

// We could implement `From` trait, but we need a wrapper for the Rust
// type. Keep it simple for now and just make a small conversion function.
fn simd_array_error_to_py<T>(r: Result<T, SimdArrayError>) -> PyResult<T> {
    r.map_err(|err| match err {
        SimdArrayError::NonContiguous => {
            exceptions::PyRuntimeError::new_err("array is not contiguous")
        }
    })
}
