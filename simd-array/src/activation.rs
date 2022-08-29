use num_traits::{NumCast, One, Zero};
use std::ops::Neg;

use crate::distribution::Distribution;
use crate::elementary::Elementary;
use crate::vector::SimdVector;

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

    unsafe fn gelu(x: Self::Float) -> Self::Float;

    unsafe fn hard_sigmoid(x: Self::Float) -> Self::Float;

    unsafe fn hard_tanh(x: Self::Float) -> Self::Float;

    unsafe fn relu(x: Self::Float) -> Self::Float;

    unsafe fn swish(x: Self::Float) -> Self::Float;
}

impl<V, T> Activation for V
where
    T: Copy,
    V: Distribution<Float = T> + Elementary<Float = T> + SimdVector<Float = T>,
{
    type Float = <V as SimdVector>::Float;
    type FloatScalar = <V as SimdVector>::FloatScalar;

    #[inline(always)]
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
        V::clamp_max(V::clamp_min(x, x_min_val), x_max_val)
    }

    #[inline(always)]
    unsafe fn gelu(x: Self::Float) -> Self::Float {
        V::mul(x, V::normal_cdf(x))
    }

    #[inline(always)]
    unsafe fn hard_sigmoid(x: Self::Float) -> Self::Float {
        Self::clipped_linear(
            x,
            <Self::FloatScalar as NumCast>::from(0.2).unwrap(),
            <Self::FloatScalar as NumCast>::from(0.5).unwrap(),
            Self::FloatScalar::zero(),
            Self::FloatScalar::one(),
        )
    }

    #[inline(always)]
    unsafe fn hard_tanh(x: Self::Float) -> Self::Float {
        Self::clipped_linear(
            x,
            <Self::FloatScalar as NumCast>::from(1.).unwrap(),
            <Self::FloatScalar as NumCast>::from(0.).unwrap(),
            Self::FloatScalar::one().neg(),
            Self::FloatScalar::one(),
        )
    }

    #[inline(always)]
    unsafe fn relu(x: Self::Float) -> Self::Float {
        let zero = V::splat(V::FloatScalar::zero());
        V::clamp_min(x, zero)
    }

    #[inline(always)]
    unsafe fn swish(x: Self::Float) -> Self::Float {
        V::mul(x, Self::logistic_cdf(x))
    }
}
