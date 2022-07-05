use num_traits::{NumCast, One, Zero};
use std::ops::Neg;

use crate::simd::elementary::Elementary;
use crate::simd::vector::SimdVector;

pub trait Activation {
    type Float;
    type FloatScalar;

    unsafe fn clipped_linear(
        x: Self::Float,
        slope: Self::FloatScalar,
        offset: Self::FloatScalar,
        min_val: Self::FloatScalar,
        max_val: Self::FloatScalar,
    ) -> Self::Float;

    unsafe fn hard_sigmoid(x: Self::Float) -> Self::Float;

    unsafe fn hard_tanh(x: Self::Float) -> Self::Float;

    unsafe fn logistic_function(x: Self::Float) -> Self::Float;

    unsafe fn relu(x: Self::Float) -> Self::Float;

    unsafe fn swish(x: Self::Float) -> Self::Float;
}

impl<V, T> Activation for V
where
    T: Copy,
    V: SimdVector<Float = T> + Elementary<Float = T>,
{
    type Float = <V as SimdVector>::Float;
    type FloatScalar = <V as SimdVector>::FloatScalar;

    unsafe fn clipped_linear(
        x: Self::Float,
        slope: Self::FloatScalar,
        offset: Self::FloatScalar,
        min_val: Self::FloatScalar,
        max_val: Self::FloatScalar,
    ) -> Self::Float {
        let x_min_val = V::splat(min_val);
        let x_max_val = V::splat(max_val);
        let x = V::add_scalar(V::mul_scalar(x, slope), offset);
        V::vmin(V::vmax(x, x_min_val), x_max_val)
    }

    unsafe fn hard_sigmoid(x: Self::Float) -> Self::Float {
        Self::clipped_linear(
            x,
            <Self::FloatScalar as NumCast>::from(0.2).unwrap(),
            <Self::FloatScalar as NumCast>::from(0.5).unwrap(),
            Self::FloatScalar::zero(),
            Self::FloatScalar::one(),
        )
    }

    unsafe fn hard_tanh(x: Self::Float) -> Self::Float {
        Self::clipped_linear(
            x,
            <Self::FloatScalar as NumCast>::from(1.).unwrap(),
            <Self::FloatScalar as NumCast>::from(0.).unwrap(),
            Self::FloatScalar::one().neg(),
            Self::FloatScalar::one(),
        )
    }

    unsafe fn logistic_function(x: Self::Float) -> Self::Float {
        let one = V::splat(<V::FloatScalar as NumCast>::from(1.0).unwrap());
        V::div(one, V::add(V::exp(V::neg(x)), one))
    }

    unsafe fn relu(x: Self::Float) -> Self::Float {
        let zero = V::splat(V::FloatScalar::zero());
        V::vmax(x, zero)
    }

    unsafe fn swish(x: Self::Float) -> Self::Float {
        V::mul(x, Self::logistic_function(x))
    }
}
