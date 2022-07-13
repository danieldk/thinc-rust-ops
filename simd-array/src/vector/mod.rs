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

    /// Add a scalar to every vector element.
    unsafe fn add_scalar(a: Self::Float, b: Self::FloatScalar) -> Self::Float;

    /// Select bits which are set in `a` in `b` or otherwise `c`.
    unsafe fn bitwise_select(a: Self::Mask, b: Self::Float, c: Self::Float) -> Self::Float;

    unsafe fn copy_sign(sign_src: Self::Float, dest: Self::Float) -> Self::Float;

    unsafe fn div(a: Self::Float, b: Self::Float) -> Self::Float;

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

    /// If a is less than b, set all corresponding lanes to 1.
    unsafe fn lt(a: Self::Float, b: Self::Float) -> Self::Mask;

    /// Vector element-wise multiplication.
    unsafe fn mul(a: Self::Float, b: Self::Float) -> Self::Float;

    /// Multiply every vector element by a scalar.
    unsafe fn mul_scalar(a: Self::Float, b: Self::FloatScalar) -> Self::Float;

    unsafe fn neg(a: Self::Float) -> Self::Float;

    unsafe fn sub(a: Self::Float, b: Self::Float) -> Self::Float;

    /// Vector element-wise maximum.
    unsafe fn vmax(a: Self::Float, b: Self::Float) -> Self::Float;

    /// Vector element-wise minimum.
    unsafe fn vmin(a: Self::Float, b: Self::Float) -> Self::Float;

    unsafe fn splat(v: Self::FloatScalar) -> Self::Float;

    unsafe fn reinterpret_float_signed(v: Self::Int) -> Self::Float;

    unsafe fn to_int(v: Self::Float) -> Self::Int;

    unsafe fn to_float_scalar_array(v: Self::Float) -> Self::FloatScalarArray;

    unsafe fn with_load_store(f: &impl Fn(Self::Float) -> Self::Float, a: &mut [Self::FloatScalar]);

    unsafe fn apply_elementwise(
        f: impl Fn(Self::Float) -> Self::Float,
        f_rest: impl Fn(&mut [Self::FloatScalar]),
        a: &mut [Self::FloatScalar],
    );
}

// TODO: get rid of the first argument. Needed so far to help type inference.
unsafe fn apply_elementwise_generic<V>(
    _v: &V,
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
