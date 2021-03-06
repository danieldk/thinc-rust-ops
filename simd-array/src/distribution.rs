use num_traits::{FloatConst, One};

use crate::elementary::Elementary;
use crate::vector::SimdVector;

pub trait Distribution {
    type Float;

    /// CDF of the normal distribution.
    unsafe fn normal_cdf(x: Self::Float) -> Self::Float;

    /// CDF of the logistic distribution.
    unsafe fn logistic_cdf(x: Self::Float) -> Self::Float;
}

impl<V, T> Distribution for V
where
    T: Copy,
    V: SimdVector<Float = T> + Elementary<Float = T>,
{
    type Float = <V as SimdVector>::Float;

    #[inline(always)]
    unsafe fn normal_cdf(x: Self::Float) -> Self::Float {
        let half = V::from_f64(0.5);
        let one = V::splat(V::FloatScalar::one());
        let sqrt_2_inv = V::splat(V::FloatScalar::one() / V::FloatScalar::SQRT_2());
        // 1/2 (1 + erf(x/sqrt(2)))
        V::mul(half, V::add(V::erf(V::mul(sqrt_2_inv, x)), one))
    }

    #[inline(always)]
    unsafe fn logistic_cdf(x: Self::Float) -> Self::Float {
        let one = V::splat(V::FloatScalar::one());
        V::div(one, V::add(V::exp(V::neg(x)), one))
    }
}
