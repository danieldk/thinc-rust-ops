use std::mem;

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

    unsafe fn add_scalar(a: Self::Float, b: f32) -> Self::Float {
        a + b
    }

    unsafe fn bitwise_select(a: Self::Mask, b: Self::Float, c: Self::Float) -> Self::Float {
        Self::Float::from_bits((a & b.to_bits()) | ((!a) & c.to_bits()))
    }

    unsafe fn copy_sign(sign: Self::Float, dest: Self::Float) -> Self::Float {
        dest.copysign(sign)
    }

    unsafe fn div(a: Self::Float, b: Self::Float) -> Self::Float {
        a / b
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

    unsafe fn fma(a: Self::Float, b: Self::Float, c: Self::Float) -> Self::Float {
        a * b + c
    }

    unsafe fn gt(a: Self::Float, b: Self::Float) -> Self::Mask {
        if a > b {
            !0
        } else {
            0
        }
    }

    unsafe fn lt(a: Self::Float, b: Self::Float) -> Self::Mask {
        if a < b {
            !0
        } else {
            0
        }
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

    unsafe fn vmax(a: Self::Float, b: Self::Float) -> Self::Float {
        if a > b {
            a
        } else {
            b
        }
    }

    unsafe fn vmin(a: Self::Float, b: Self::Float) -> Self::Float {
        if a > b {
            b
        } else {
            a
        }
    }

    unsafe fn splat(v: f32) -> Self::Float {
        v
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

    unsafe fn add_scalar(a: Self::Float, b: f64) -> Self::Float {
        a + b
    }

    unsafe fn bitwise_select(a: Self::Mask, b: Self::Float, c: Self::Float) -> Self::Float {
        Self::Float::from_bits((a & b.to_bits()) | ((!a) & c.to_bits()))
    }

    unsafe fn copy_sign(sign_src: Self::Float, dest: Self::Float) -> Self::Float {
        dest.copysign(sign_src)
    }

    unsafe fn div(a: Self::Float, b: Self::Float) -> Self::Float {
        a / b
    }

    unsafe fn floor(a: Self::Float) -> Self::Float {
        a.floor()
    }

    unsafe fn fma(a: Self::Float, b: Self::Float, c: Self::Float) -> Self::Float {
        a * b + c
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

    unsafe fn lt(a: Self::Float, b: Self::Float) -> Self::Mask {
        if a < b {
            !0
        } else {
            0
        }
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

    unsafe fn vmax(a: Self::Float, b: Self::Float) -> Self::Float {
        if a > b {
            a
        } else {
            b
        }
    }
    unsafe fn vmin(a: Self::Float, b: Self::Float) -> Self::Float {
        if a > b {
            b
        } else {
            a
        }
    }

    unsafe fn splat(v: f64) -> Self::Float {
        v
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
}
