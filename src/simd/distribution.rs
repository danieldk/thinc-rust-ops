use crate::simd::elementary::Elementary;
use crate::simd::vector::SimdVector;
use num_traits::NumCast;

pub trait Distribution {
    type Float;

    /// CDF of the logistic distribution.
    unsafe fn logistic_cdf(x: Self::Float) -> Self::Float;
}

impl<V, T> Distribution for V
where
    T: Copy,
    V: SimdVector<Float = T> + Elementary<Float = T>,
{
    type Float = <V as SimdVector>::Float;

    unsafe fn logistic_cdf(x: Self::Float) -> Self::Float {
        let one = V::splat(<V::FloatScalar as NumCast>::from(1.0).unwrap());
        V::div(one, V::add(V::exp(V::neg(x)), one))
    }
}
