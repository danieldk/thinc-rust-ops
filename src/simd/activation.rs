use num_traits::NumCast;

use crate::simd::elementary::Elementary;
use crate::simd::vector::SimdVector;

pub trait Activation {
    type Float;

    unsafe fn logistic_function(x: Self::Float) -> Self::Float;

    unsafe fn swish(x: Self::Float) -> Self::Float;
}

impl<V, T> Activation for V
where
    T: Copy,
    V: SimdVector<Float = T> + Elementary<Float = T>,
{
    type Float = <V as SimdVector>::Float;

    unsafe fn logistic_function(x: Self::Float) -> Self::Float {
        let one = V::splat(<V::FloatScalar as NumCast>::from(1.0).unwrap());
        V::div(one, V::add(V::exp(V::neg(x)), one))
    }

    unsafe fn swish(x: Self::Float) -> Self::Float {
        V::mul(x, Self::logistic_function(x))
    }
}
