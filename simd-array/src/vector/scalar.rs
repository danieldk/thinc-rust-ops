use std::mem;

use crate::util::{maximum, minimum};
use crate::vector::{apply_elementwise_generic, SimdVector};

#[derive(Default)]
pub struct ScalarVector32;

impl SimdVector for ScalarVector32 {
    type Lower = ScalarVector32;
    type Float = f32;
    type FloatScalar = f32;
    type FloatScalarArray =
        [Self::FloatScalar; mem::size_of::<Self::Float>() / mem::size_of::<Self::FloatScalar>()];
    type Int = i32;
    type IntScalar = i32;
    type Mask = u32;

    unsafe fn abs(a: Self::Float) -> Self::Float {
        a.abs()
    }

    unsafe fn add(a: Self::Float, b: Self::Float) -> Self::Float {
        a + b
    }

    unsafe fn add_lanes(a: Self::Float) -> Self::FloatScalar {
        a
    }

    unsafe fn add_scalar(a: Self::Float, b: f32) -> Self::Float {
        a + b
    }

    unsafe fn bitwise_select(a: Self::Mask, b: Self::Float, c: Self::Float) -> Self::Float {
        Self::Float::from_bits((a & b.to_bits()) | ((!a) & c.to_bits()))
    }

    unsafe fn clamp_max(a: Self::Float, max: Self::Float) -> Self::Float {
        max.min(a)
    }

    unsafe fn clamp_min(a: Self::Float, min: Self::Float) -> Self::Float {
        min.max(a)
    }

    unsafe fn copy_sign(sign: Self::Float, dest: Self::Float) -> Self::Float {
        dest.copysign(sign)
    }

    unsafe fn div(a: Self::Float, b: Self::Float) -> Self::Float {
        a / b
    }

    unsafe fn fma(a: Self::Float, b: Self::Float, c: Self::Float) -> Self::Float {
        a * b + c
    }

    unsafe fn floor(a: Self::Float) -> Self::Float {
        a.floor()
    }

    unsafe fn eq(a: Self::Float, b: Self::Float) -> Self::Mask {
        if a == b {
            !0
        } else {
            0
        }
    }

    unsafe fn gt(a: Self::Float, b: Self::Float) -> Self::Mask {
        if a > b {
            !0
        } else {
            0
        }
    }

    unsafe fn load(a: &[Self::FloatScalar]) -> Self::Float {
        a[0]
    }

    unsafe fn lt(a: Self::Float, b: Self::Float) -> Self::Mask {
        if a < b {
            !0
        } else {
            0
        }
    }

    unsafe fn max_lanes(a: Self::Float) -> Self::FloatScalar {
        a
    }

    unsafe fn min_lanes(a: Self::Float) -> Self::FloatScalar {
        a
    }

    unsafe fn mul(a: Self::Float, b: Self::Float) -> Self::Float {
        a * b
    }

    unsafe fn mul_scalar(a: Self::Float, b: f32) -> Self::Float {
        a * b
    }

    unsafe fn neg(a: Self::Float) -> Self::Float {
        -a
    }

    unsafe fn sub(a: Self::Float, b: Self::Float) -> Self::Float {
        a - b
    }

    unsafe fn max(a: Self::Float, b: Self::Float) -> Self::Float {
        maximum(a, b)
    }

    unsafe fn min(a: Self::Float, b: Self::Float) -> Self::Float {
        minimum(a, b)
    }

    unsafe fn splat(v: f32) -> Self::Float {
        v
    }

    unsafe fn sqrt(v: Self::Float) -> Self::Float {
        v.sqrt()
    }

    unsafe fn reinterpret_float_signed(v: Self::Int) -> Self::Float {
        Self::Float::from_bits(v as u32)
    }

    unsafe fn to_int(v: Self::Float) -> Self::Int {
        v as Self::Int
    }

    unsafe fn to_float_scalar_array(v: Self::Float) -> Self::FloatScalarArray {
        [v]
    }

    unsafe fn with_load_store(f: &impl Fn(Self::Float) -> Self::Float, a: &mut [f32]) {
        a[0] = f(a[0])
    }

    unsafe fn apply_elementwise(
        f: impl Fn(Self::Float) -> Self::Float,
        f_rest: impl Fn(&mut [f32]),
        a: &mut [f32],
    ) {
        apply_elementwise_generic(Self, f, f_rest, a);
    }

    unsafe fn reduce(
        f: impl Fn(Self::Float, Self::Float) -> Self::Float,
        f_lanes: impl Fn(Self::FloatScalar) -> Self::FloatScalar,
        f_rest: impl Fn(Self::FloatScalar, &[Self::FloatScalar]) -> Self::FloatScalar,
        init: Self::FloatScalar,
        a: &[Self::FloatScalar],
    ) -> Self::FloatScalar {
        reduce_generic!(Self, f, f_lanes, f_rest, init, a)
    }
}

#[derive(Default)]
pub struct ScalarVector64;

impl SimdVector for ScalarVector64 {
    type Lower = ScalarVector64;
    type Float = f64;
    type FloatScalar = f64;
    type FloatScalarArray =
        [Self::FloatScalar; mem::size_of::<Self::Float>() / mem::size_of::<Self::FloatScalar>()];
    type Int = i64;
    type IntScalar = i64;
    type Mask = u64;

    unsafe fn abs(a: Self::Float) -> Self::Float {
        a.abs()
    }

    unsafe fn add(a: Self::Float, b: Self::Float) -> Self::Float {
        a + b
    }

    unsafe fn add_lanes(a: Self::Float) -> Self::FloatScalar {
        a
    }

    unsafe fn add_scalar(a: Self::Float, b: f64) -> Self::Float {
        a + b
    }

    unsafe fn bitwise_select(a: Self::Mask, b: Self::Float, c: Self::Float) -> Self::Float {
        Self::Float::from_bits((a & b.to_bits()) | ((!a) & c.to_bits()))
    }

    unsafe fn clamp_max(a: Self::Float, max: Self::Float) -> Self::Float {
        max.min(a)
    }

    unsafe fn clamp_min(a: Self::Float, min: Self::Float) -> Self::Float {
        min.max(a)
    }

    unsafe fn copy_sign(sign_src: Self::Float, dest: Self::Float) -> Self::Float {
        dest.copysign(sign_src)
    }

    unsafe fn div(a: Self::Float, b: Self::Float) -> Self::Float {
        a / b
    }

    unsafe fn fma(a: Self::Float, b: Self::Float, c: Self::Float) -> Self::Float {
        a * b + c
    }

    unsafe fn floor(a: Self::Float) -> Self::Float {
        a.floor()
    }

    unsafe fn eq(a: Self::Float, b: Self::Float) -> Self::Mask {
        if a == b {
            !0
        } else {
            0
        }
    }

    unsafe fn gt(a: Self::Float, b: Self::Float) -> Self::Mask {
        if a > b {
            !0
        } else {
            0
        }
    }

    unsafe fn load(a: &[Self::FloatScalar]) -> Self::Float {
        a[0]
    }

    unsafe fn lt(a: Self::Float, b: Self::Float) -> Self::Mask {
        if a < b {
            !0
        } else {
            0
        }
    }

    unsafe fn max_lanes(a: Self::Float) -> Self::FloatScalar {
        a
    }

    unsafe fn min_lanes(a: Self::Float) -> Self::FloatScalar {
        a
    }

    unsafe fn mul(a: Self::Float, b: Self::Float) -> Self::Float {
        a * b
    }

    unsafe fn mul_scalar(a: Self::Float, b: f64) -> Self::Float {
        a * b
    }

    unsafe fn neg(a: Self::Float) -> Self::Float {
        -a
    }
    unsafe fn sub(a: Self::Float, b: Self::Float) -> Self::Float {
        a - b
    }

    unsafe fn max(a: Self::Float, b: Self::Float) -> Self::Float {
        maximum(a, b)
    }

    unsafe fn min(a: Self::Float, b: Self::Float) -> Self::Float {
        minimum(a, b)
    }

    unsafe fn splat(v: f64) -> Self::Float {
        v
    }
    unsafe fn sqrt(v: Self::Float) -> Self::Float {
        v.sqrt()
    }

    unsafe fn reinterpret_float_signed(v: Self::Int) -> Self::Float {
        f64::from_bits(v as u64)
    }

    unsafe fn to_int(v: Self::Float) -> Self::Int {
        v as Self::Int
    }

    unsafe fn to_float_scalar_array(v: Self::Float) -> Self::FloatScalarArray {
        [v]
    }

    unsafe fn with_load_store(f: &impl Fn(Self::Float) -> Self::Float, a: &mut [f64]) {
        a[0] = f(a[0])
    }

    unsafe fn apply_elementwise(
        f: impl Fn(Self::Float) -> Self::Float,
        f_rest: impl Fn(&mut [f64]),
        a: &mut [f64],
    ) {
        apply_elementwise_generic(Self, f, f_rest, a);
    }

    unsafe fn reduce(
        f: impl Fn(Self::Float, Self::Float) -> Self::Float,
        f_lanes: impl Fn(Self::Float) -> Self::FloatScalar,
        f_rest: impl Fn(Self::FloatScalar, &[Self::FloatScalar]) -> Self::FloatScalar,
        init: Self::FloatScalar,
        a: &[Self::FloatScalar],
    ) -> Self::FloatScalar {
        reduce_generic!(Self, f, f_lanes, f_rest, init, a)
    }
}
