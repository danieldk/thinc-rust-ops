use as_slice::AsSlice;
use std::fmt::Debug;
use std::mem;

use num_traits::{Float, FloatConst, NumCast, PrimInt};

pub trait FloatingPointProps {
    fn bias() -> usize;

    fn mantissa_bits() -> usize;

    fn max_ln() -> Self;
}

impl FloatingPointProps for f32 {
    fn bias() -> usize {
        127
    }

    fn mantissa_bits() -> usize {
        23
    }

    fn max_ln() -> Self {
        2f32.powi(f32::MAX_EXP - 1).ln()
    }
}

impl FloatingPointProps for f64 {
    fn bias() -> usize {
        1023
    }

    fn mantissa_bits() -> usize {
        52
    }

    fn max_ln() -> Self {
        2f64.powi(f64::MAX_EXP - 1).ln()
    }
}

pub trait SimdVector: Default + Send + Sync {
    type Lower: SimdVector<FloatScalar = Self::FloatScalar>;
    type Float: Copy;
    type FloatScalar: Debug + Float + FloatConst + FloatingPointProps;
    type FloatScalarArray: AsSlice<Element = Self::FloatScalar>;
    type Int: Copy;
    type IntScalar: PrimInt;
    type Mask: Copy;

    /// Absolute value.
    unsafe fn abs(a: Self::Float) -> Self::Float;

    unsafe fn add(a: Self::Float, b: Self::Float) -> Self::Float;

    // Add across lanes
    unsafe fn add_lanes(a: Self::Float) -> Self::FloatScalar;

    /// Add a scalar to every vector element.
    unsafe fn add_scalar(a: Self::Float, b: Self::FloatScalar) -> Self::Float;

    /// Select bits which are set in `a` in `b` or otherwise `c`.
    unsafe fn bitwise_select(a: Self::Mask, b: Self::Float, c: Self::Float) -> Self::Float;

    /// Clamp values to the range `[,max]`.
    unsafe fn clamp_max(a: Self::Float, max: Self::Float) -> Self::Float;

    /// Clamp values to the range `[min,]`.
    unsafe fn clamp_min(a: Self::Float, min: Self::Float) -> Self::Float;

    unsafe fn copy_sign(sign_src: Self::Float, dest: Self::Float) -> Self::Float;

    unsafe fn div(a: Self::Float, b: Self::Float) -> Self::Float;

    unsafe fn div_scalar(a: Self::Float, b: Self::FloatScalar) -> Self::Float {
        let b = Self::splat(b);
        Self::div(a, b)
    }

    /// Fused mutiply-add, a * b + c
    unsafe fn fma(a: Self::Float, b: Self::Float, c: Self::Float) -> Self::Float;

    /// Round to largest integers lower than or equal to the given numbers.
    unsafe fn floor(a: Self::Float) -> Self::Float;

    /// Convert from f64 into the scalar float type and then splat.
    unsafe fn from_f64(a: f64) -> Self::Float {
        Self::splat(<Self::FloatScalar as NumCast>::from(a).unwrap())
    }

    unsafe fn eq(a: Self::Float, b: Self::Float) -> Self::Mask;

    /// If a is greater than b, set all corresponding lanes to 1.
    unsafe fn gt(a: Self::Float, b: Self::Float) -> Self::Mask;

    unsafe fn load(a: &[Self::FloatScalar]) -> Self::Float;

    /// If a is less than b, set all corresponding lanes to 1.
    unsafe fn lt(a: Self::Float, b: Self::Float) -> Self::Mask;

    unsafe fn max_lanes(a: Self::Float) -> Self::FloatScalar;

    /// Vector element-wise multiplication.
    unsafe fn mul(a: Self::Float, b: Self::Float) -> Self::Float;

    /// Multiply every vector element by a scalar.
    unsafe fn mul_scalar(a: Self::Float, b: Self::FloatScalar) -> Self::Float;

    unsafe fn neg(a: Self::Float) -> Self::Float;

    unsafe fn sub(a: Self::Float, b: Self::Float) -> Self::Float;

    /// Vector element-wise maximum.
    ///
    /// All implementations must be NaN-propagating.
    unsafe fn max(a: Self::Float, b: Self::Float) -> Self::Float;

    /// Vector element-wise minimum.
    unsafe fn vmin(a: Self::Float, b: Self::Float) -> Self::Float;

    unsafe fn splat(v: Self::FloatScalar) -> Self::Float;

    unsafe fn sub_scalar(a: Self::Float, b: Self::FloatScalar) -> Self::Float {
        let b = Self::splat(b);
        Self::sub(a, b)
    }

    unsafe fn reinterpret_float_signed(v: Self::Int) -> Self::Float;

    unsafe fn to_int(v: Self::Float) -> Self::Int;

    unsafe fn to_float_scalar_array(v: Self::Float) -> Self::FloatScalarArray;

    unsafe fn with_load_store(f: &impl Fn(Self::Float) -> Self::Float, a: &mut [Self::FloatScalar]);

    unsafe fn apply_elementwise(
        f: impl Fn(Self::Float) -> Self::Float,
        f_rest: impl Fn(&mut [Self::FloatScalar]),
        a: &mut [Self::FloatScalar],
    );

    /// Array reduction.
    ///
    /// Reduce an array to a scalar using the provided function(s). The
    /// function `f` first applies the reduction at the SIMD level.
    ///
    /// * `f` is the main reduction function to apply. For instance,
    ///   if the SIMD addition function is used then the result of
    ///   the main reduction loop will be a SIMD float containing the
    ///   (partial) sums.
    /// * `f_lanes` applies the reduction to the lanes of the SIMD
    ///    floats that are the result of the reduction. For example,
    ///   if `f` uses addition, then this function could apply lane
    ///   addition to get the sum from the partial sums.
    /// * `f_rest` should apply the reduction to the remainder of
    ///   the array if the array size is not a multiple of the SIMD
    ///   size. The result of the reduction using `f` and `f_lanes`
    ///   is passed as an initialization value to `f_rest`.
    /// * `init` is the initial value for the reduction. E.g. for
    ///   addition this should be 0.0.
    unsafe fn reduce(
        f: impl Fn(Self::Float, Self::Float) -> Self::Float,
        f_lanes: impl Fn(Self::Float) -> Self::FloatScalar,
        f_rest: impl Fn(Self::FloatScalar, &[Self::FloatScalar]) -> Self::FloatScalar,
        init: Self::FloatScalar,
        a: &[Self::FloatScalar],
    ) -> Self::FloatScalar;
}

// TODO: get rid of the first argument. Needed so far to help type inference.
unsafe fn apply_elementwise_generic<V>(
    _v: V,
    f: impl Fn(V::Float) -> V::Float,
    f_rest: impl Fn(&mut [V::FloatScalar]),
    mut a: &mut [V::FloatScalar],
) where
    V: SimdVector,
{
    let elem_size = mem::size_of::<V::Float>() / mem::size_of::<V::FloatScalar>();

    while a.len() >= elem_size {
        V::with_load_store(&f, a);
        a = &mut a[elem_size..];
    }

    if a.len() > 0 {
        f_rest(a);
    }
}

macro_rules! reduce_generic {
    ($v:ty,$f:ident,$f_lanes:ident,$f_rest:ident,$init:ident,$a:ident) => {{
        let elem_size = mem::size_of::<<$v>::Float>() / mem::size_of::<<$v>::FloatScalar>();

        let mut a = $a;
        let mut acc = <$v>::splat($init);

        while a.len() >= elem_size {
            let val = <$v>::load(a);
            acc = $f(acc, val);
            a = &a[elem_size..];
        }

        let scalar = $f_lanes(acc);

        if a.is_empty() {
            scalar
        } else {
            $f_rest(scalar, a)
        }
    }};
}

pub mod scalar;

#[cfg(all(target_arch = "x86_64"))]
pub mod avx;

#[cfg(all(target_arch = "x86_64"))]
pub mod avx2;

#[cfg(all(target_arch = "x86_64"))]
pub mod sse2;

#[cfg(all(target_arch = "x86_64"))]
pub mod sse41;

#[cfg(all(target_arch = "aarch64"))]
pub mod neon;

#[cfg(test)]
mod tests {
    use std::fmt;

    use as_slice::AsSlice;
    use num_traits::Float;

    use super::scalar::{ScalarVector32, ScalarVector64};
    use super::SimdVector;

    fn max_nan_is_nan<V, S>()
    where
        V: SimdVector<FloatScalar = S>,
        S: fmt::Debug + Float,
    {
        let nan = unsafe { V::splat(S::nan()) };
        let zero = unsafe { V::splat(S::zero()) };

        let max_zero_nan = unsafe { V::to_float_scalar_array(V::max(zero, nan)) }.as_slice()[0];
        assert!(max_zero_nan.is_nan());

        let max_zero_nan = unsafe { V::to_float_scalar_array(V::max(nan, zero)) }.as_slice()[0];
        assert!(max_zero_nan.is_nan());
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn max_nan_is_nan_neon() {
        use super::neon::{NeonVector32, NeonVector64};

        max_nan_is_nan::<NeonVector32, f32>();
        max_nan_is_nan::<NeonVector64, f64>();
    }

    #[test]
    fn max_nan_is_nan_scalar() {
        max_nan_is_nan::<ScalarVector32, f32>();
        max_nan_is_nan::<ScalarVector64, f64>();
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn max_nan_is_nan_x86_64() {
        use super::avx::{AVXVector32, AVXVector64};
        use super::avx2::{AVX2Vector32, AVX2Vector64};
        use super::sse2::{SSE2Vector32, SSE2Vector64};
        use super::sse41::{SSE41Vector32, SSE41Vector64};

        if is_x86_feature_detected!("sse2") {
            max_nan_is_nan::<SSE2Vector32, f32>();
            max_nan_is_nan::<SSE2Vector64, f64>();
        }

        if is_x86_feature_detected!("sse4.1") {
            max_nan_is_nan::<SSE41Vector32, f32>();
            max_nan_is_nan::<SSE41Vector64, f64>();
        }

        if is_x86_feature_detected!("avx") {
            max_nan_is_nan::<AVXVector32, f32>();
            max_nan_is_nan::<AVXVector64, f64>();
        }

        if is_x86_feature_detected!("avx2") {
            max_nan_is_nan::<AVX2Vector32, f32>();
            max_nan_is_nan::<AVX2Vector64, f64>();
        }
    }
}
