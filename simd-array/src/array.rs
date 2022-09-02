use ndarray::{ArrayBase, DataMut, Dimension};
use num_traits::Float;
use thiserror::Error;

use crate::slice::PlatformSimdSlice;

#[derive(Debug, Error)]
pub enum SimdArrayError {
    #[error("array is not contiguous")]
    NonContiguous,
}

pub trait SimdArrayMut {
    type Elem;

    /// Clipped linear function.
    fn clipped_linear(
        &mut self,
        slope: Self::Elem,
        offset: Self::Elem,
        min_val: Self::Elem,
        max_val: Self::Elem,
    ) -> Result<(), SimdArrayError>;

    /// Gaussian error linear unit (GELU).
    fn gelu(&mut self) -> Result<(), SimdArrayError>;

    /// Hard sigmoid.
    fn hard_sigmoid(&mut self) -> Result<(), SimdArrayError>;

    /// Hard hyperbolic tangent.
    fn hard_tanh(&mut self) -> Result<(), SimdArrayError>;

    /// Logistic cumulative density function.
    fn logistic_cdf(&mut self) -> Result<(), SimdArrayError>;

    /// Rectified linear unit (ReLU).
    fn relu(&mut self) -> Result<(), SimdArrayError>;

    /// Softmax.
    fn softmax(
        &mut self,
        n_class: usize,
        temperature: Option<Self::Elem>,
    ) -> Result<(), SimdArrayError>;

    /// Swish activation.
    fn swish(&mut self) -> Result<(), SimdArrayError>;

    /// Dish™ activation.
    fn dish(&mut self) -> Result<(), SimdArrayError>;
}

impl<S, D, A> SimdArrayMut for ArrayBase<S, D>
where
    S: DataMut<Elem = A>,
    D: Dimension,
    A: Float + PlatformSimdSlice,
{
    type Elem = A;

    fn clipped_linear(
        &mut self,
        slope: Self::Elem,
        offset: Self::Elem,
        min_val: Self::Elem,
        max_val: Self::Elem,
    ) -> Result<(), SimdArrayError> {
        A::simd_slice().clipped_linear(
            self.as_slice_memory_order_mut()
                .ok_or(SimdArrayError::NonContiguous)?,
            slope,
            offset,
            min_val,
            max_val,
        );
        Ok(())
    }

    fn gelu(&mut self) -> Result<(), SimdArrayError> {
        A::simd_slice().gelu(
            self.as_slice_memory_order_mut()
                .ok_or(SimdArrayError::NonContiguous)?,
        );
        Ok(())
    }

    fn hard_sigmoid(&mut self) -> Result<(), SimdArrayError> {
        A::simd_slice().hard_sigmoid(
            self.as_slice_memory_order_mut()
                .ok_or(SimdArrayError::NonContiguous)?,
        );
        Ok(())
    }

    fn hard_tanh(&mut self) -> Result<(), SimdArrayError> {
        A::simd_slice().hard_tanh(
            self.as_slice_memory_order_mut()
                .ok_or(SimdArrayError::NonContiguous)?,
        );
        Ok(())
    }

    fn logistic_cdf(&mut self) -> Result<(), SimdArrayError> {
        A::simd_slice().logistic_cdf(
            self.as_slice_memory_order_mut()
                .ok_or(SimdArrayError::NonContiguous)?,
        );
        Ok(())
    }

    fn relu(&mut self) -> Result<(), SimdArrayError> {
        A::simd_slice().relu(
            self.as_slice_memory_order_mut()
                .ok_or(SimdArrayError::NonContiguous)?,
        );
        Ok(())
    }

    /// Softmax.
    fn softmax(
        &mut self,
        n_class: usize,
        temperature: Option<Self::Elem>,
    ) -> Result<(), SimdArrayError> {
        A::simd_slice().softmax(
            self.as_slice_memory_order_mut()
                .ok_or(SimdArrayError::NonContiguous)?,
            n_class,
            temperature,
        );
        Ok(())
    }

    fn swish(&mut self) -> Result<(), SimdArrayError> {
        A::simd_slice().swish(
            self.as_slice_memory_order_mut()
                .ok_or(SimdArrayError::NonContiguous)?,
        );
        Ok(())
    }

    fn dish(&mut self) -> Result<(), SimdArrayError> {
        A::simd_slice().dish(
            self.as_slice_memory_order_mut()
                .ok_or(SimdArrayError::NonContiguous)?,
        );
        Ok(())
    }
}
