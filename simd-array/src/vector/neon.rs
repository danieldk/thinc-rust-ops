use std::mem;
use std::ops::Neg;
use std::arch::aarch64::{
    float32x4_t, float64x2_t, int32x4_t, int64x2_t, uint32x4_t, uint64x2_t, vabsq_f32,
    vabsq_f64, vaddq_f32, vaddq_f64, vandq_u32, vandq_u64, vbicq_u32, vbicq_u64, vceqq_f32,
    vceqq_f64, vcgtq_f32, vcgtq_f64, vcltq_f32, vcltq_f64, vcvtq_s32_f32, vcvtq_s64_f64,
    vdivq_f32, vdivq_f64, vdupq_n_f32, vdupq_n_f64, vfmaq_f32, vfmaq_f64, vld1q_f32, vld1q_f64,
    vmaxq_f32, vmaxq_f64, vminq_f32, vminq_f64, vmulq_f32, vmulq_f64, vnegq_f32, vnegq_f64,
    vorrq_u32, vorrq_u64, vreinterpretq_f32_s32, vreinterpretq_f32_u32, vreinterpretq_f64_s64,
    vreinterpretq_f64_u64, vreinterpretq_u32_f32, vreinterpretq_u64_f64, vrndmq_f32,
    vrndmq_f64, vst1q_f32, vst1q_f64, vsubq_f32, vsubq_f64,
};

use num_traits::Zero;

use crate::vector::scalar::{ScalarVector32, ScalarVector64};

use super::{SimdVector};

#[derive(Default)]
pub struct NeonVector32;

impl SimdVector for NeonVector32 {
    type Lower = ScalarVector32;
    type Float = float32x4_t;
    type FloatScalar = f32;
    type FloatScalarArray = [Self::FloatScalar;
        mem::size_of::<Self::Float>() / mem::size_of::<Self::FloatScalar>()];
    type Int = int32x4_t;
    type IntScalar = i32;
    type Mask = uint32x4_t;

    #[target_feature(enable = "neon")]
    unsafe fn abs(a: Self::Float) -> Self::Float {
        vabsq_f32(a)
    }

    #[target_feature(enable = "neon")]
    unsafe fn add(a: Self::Float, b: Self::Float) -> Self::Float {
        vaddq_f32(a, b)
    }

    #[target_feature(enable = "neon")]
    unsafe fn add_scalar(a: Self::Float, b: f32) -> Self::Float {
        let b_simd = vdupq_n_f32(b);
        vaddq_f32(a, b_simd)
    }

    #[target_feature(enable = "neon")]
    unsafe fn bitwise_select(a: Self::Mask, b: Self::Float, c: Self::Float) -> Self::Float {
        // We want to use the bit selection intrinsic, however it is currently broken:
        // https://github.com/rust-lang/stdarch/issues/1191
        // vbslq_f32(a, b, c)

        let b = vreinterpretq_u32_f32(b);
        let c = vreinterpretq_u32_f32(c);
        let r = vorrq_u32(vandq_u32(a, b), vbicq_u32(c, a));
        vreinterpretq_f32_u32(r)
    }

    #[target_feature(enable = "neon")]
    unsafe fn copy_sign(sign_src: Self::Float, dest: Self::Float) -> Self::Float {
        // Negative zero has all bits unset, except the sign bit.
        let sign_bit_mask = vreinterpretq_u32_f32(Self::splat(Self::FloatScalar::zero().neg()));
        Self::bitwise_select(sign_bit_mask, sign_src, dest)
    }

    #[target_feature(enable = "neon")]
    unsafe fn div(a: Self::Float, b: Self::Float) -> Self::Float {
        vdivq_f32(a, b)
    }

    #[target_feature(enable = "neon")]
    unsafe fn floor(a: Self::Float) -> Self::Float {
        vrndmq_f32(a)
    }

    #[target_feature(enable = "neon")]
    unsafe fn fma(a: Self::Float, b: Self::Float, c: Self::Float) -> Self::Float {
        vfmaq_f32(c, a, b)
    }

    #[target_feature(enable = "neon")]
    unsafe fn eq(a: Self::Float, b: Self::Float) -> Self::Mask {
        vceqq_f32(a, b)
    }

    #[target_feature(enable = "neon")]
    unsafe fn gt(a: Self::Float, b: Self::Float) -> Self::Mask {
        vcgtq_f32(a, b)
    }

    #[target_feature(enable = "neon")]
    unsafe fn lt(a: Self::Float, b: Self::Float) -> Self::Mask {
        vcltq_f32(a, b)
    }

    #[target_feature(enable = "neon")]
    unsafe fn mul(a: Self::Float, b: Self::Float) -> Self::Float {
        vmulq_f32(a, b)
    }

    #[target_feature(enable = "neon")]
    unsafe fn mul_scalar(a: Self::Float, b: f32) -> Self::Float {
        let b_simd = vdupq_n_f32(b);
        vmulq_f32(a, b_simd)
    }

    #[target_feature(enable = "neon")]
    unsafe fn neg(a: Self::Float) -> Self::Float {
        vnegq_f32(a)
    }

    #[target_feature(enable = "neon")]
    unsafe fn sub(a: Self::Float, b: Self::Float) -> Self::Float {
        vsubq_f32(a, b)
    }

    #[target_feature(enable = "neon")]
    unsafe fn vmax(a: Self::Float, b: Self::Float) -> Self::Float {
        vmaxq_f32(a, b)
    }

    #[target_feature(enable = "neon")]
    unsafe fn vmin(a: Self::Float, b: Self::Float) -> Self::Float {
        vminq_f32(a, b)
    }

    #[target_feature(enable = "neon")]
    unsafe fn splat(v: f32) -> Self::Float {
        vdupq_n_f32(v)
    }

    #[target_feature(enable = "neon")]
    unsafe fn reinterpret_float_signed(v: Self::Int) -> Self::Float {
        vreinterpretq_f32_s32(v)
    }

    #[target_feature(enable = "neon")]
    unsafe fn to_int(v: Self::Float) -> Self::Int {
        vcvtq_s32_f32(v)
    }

    #[target_feature(enable = "neon")]
    unsafe fn to_float_scalar_array(v: Self::Float) -> Self::FloatScalarArray {
        let mut a = [0f32; 4];
        vst1q_f32(a.as_mut_ptr(), v);
        a
    }

    #[target_feature(enable = "neon")]
    unsafe fn with_load_store(f: &impl Fn(Self::Float) -> Self::Float, a: &mut [f32]) {
        let mut val = vld1q_f32(a.as_ptr());
        val = f(val);
        vst1q_f32(a.as_mut_ptr(), val)
    }

    #[target_feature(enable = "neon")]
    unsafe fn apply_elementwise(
        f: impl Fn(Self::Float) -> Self::Float,
        f_rest: impl Fn(&mut [f32]),
        a: &mut [f32],
    ) {
        let v = Self;
        super::apply_elementwise_generic(&v, f, f_rest, a);
    }
}

#[derive(Default)]
pub struct NeonVector64;

impl SimdVector for NeonVector64 {
    type Lower = ScalarVector64;
    type Float = float64x2_t;
    type FloatScalar = f64;
    type FloatScalarArray = [Self::FloatScalar;
        mem::size_of::<Self::Float>() / mem::size_of::<Self::FloatScalar>()];
    type Int = int64x2_t;
    type IntScalar = i64;
    type Mask = uint64x2_t;

    #[target_feature(enable = "neon")]
    unsafe fn abs(a: Self::Float) -> Self::Float {
        vabsq_f64(a)
    }

    #[target_feature(enable = "neon")]
    unsafe fn add(a: Self::Float, b: Self::Float) -> Self::Float {
        vaddq_f64(a, b)
    }

    #[target_feature(enable = "neon")]
    unsafe fn add_scalar(a: Self::Float, b: f64) -> Self::Float {
        let b_simd = vdupq_n_f64(b);
        vaddq_f64(a, b_simd)
    }

    #[target_feature(enable = "neon")]
    unsafe fn bitwise_select(a: Self::Mask, b: Self::Float, c: Self::Float) -> Self::Float {
        // We want to use the bit selection intrinsic, however it is currently broken:
        // https://github.com/rust-lang/stdarch/issues/1191
        // vbslq_f64(a, b, c)

        let b = vreinterpretq_u64_f64(b);
        let c = vreinterpretq_u64_f64(c);
        let r = vorrq_u64(vandq_u64(a, b), vbicq_u64(c, a));
        vreinterpretq_f64_u64(r)
    }

    #[target_feature(enable = "neon")]
    unsafe fn copy_sign(sign_src: Self::Float, dest: Self::Float) -> Self::Float {
        // Negative zero has all bits unset, except the sign bit.
        let sign_bit_mask = vreinterpretq_u64_f64(Self::splat(Self::FloatScalar::zero().neg()));
        Self::bitwise_select(sign_bit_mask, sign_src, dest)
    }

    #[target_feature(enable = "neon")]
    unsafe fn div(a: Self::Float, b: Self::Float) -> Self::Float {
        vdivq_f64(a, b)
    }

    #[target_feature(enable = "neon")]
    unsafe fn floor(a: Self::Float) -> Self::Float {
        vrndmq_f64(a)
    }

    #[target_feature(enable = "neon")]
    unsafe fn fma(a: Self::Float, b: Self::Float, c: Self::Float) -> Self::Float {
        vfmaq_f64(c, a, b)
    }

    #[target_feature(enable = "neon")]
    unsafe fn eq(a: Self::Float, b: Self::Float) -> Self::Mask {
        vceqq_f64(a, b)
    }

    #[target_feature(enable = "neon")]
    unsafe fn gt(a: Self::Float, b: Self::Float) -> Self::Mask {
        vcgtq_f64(a, b)
    }

    #[target_feature(enable = "neon")]
    unsafe fn lt(a: Self::Float, b: Self::Float) -> Self::Mask {
        vcltq_f64(a, b)
    }

    #[target_feature(enable = "neon")]
    unsafe fn mul(a: Self::Float, b: Self::Float) -> Self::Float {
        vmulq_f64(a, b)
    }

    #[target_feature(enable = "neon")]
    unsafe fn mul_scalar(a: Self::Float, b: f64) -> Self::Float {
        let b_simd = vdupq_n_f64(b);
        vmulq_f64(a, b_simd)
    }

    #[target_feature(enable = "neon")]
    unsafe fn neg(a: Self::Float) -> Self::Float {
        vnegq_f64(a)
    }

    #[target_feature(enable = "neon")]
    unsafe fn sub(a: Self::Float, b: Self::Float) -> Self::Float {
        vsubq_f64(a, b)
    }

    #[target_feature(enable = "neon")]
    unsafe fn vmax(a: Self::Float, b: Self::Float) -> Self::Float {
        vmaxq_f64(a, b)
    }

    #[target_feature(enable = "neon")]
    unsafe fn vmin(a: Self::Float, b: Self::Float) -> Self::Float {
        vminq_f64(a, b)
    }

    #[target_feature(enable = "neon")]
    unsafe fn splat(v: f64) -> Self::Float {
        vdupq_n_f64(v)
    }

    #[target_feature(enable = "neon")]
    unsafe fn reinterpret_float_signed(v: Self::Int) -> Self::Float {
        vreinterpretq_f64_s64(v)
    }

    #[target_feature(enable = "neon")]
    unsafe fn to_int(v: Self::Float) -> Self::Int {
        vcvtq_s64_f64(v)
    }

    #[target_feature(enable = "neon")]
    unsafe fn to_float_scalar_array(v: Self::Float) -> Self::FloatScalarArray {
        let mut a = [0f64; 2];
        vst1q_f64(a.as_mut_ptr(), v);
        a
    }

    #[target_feature(enable = "neon")]
    unsafe fn with_load_store(f: &impl Fn(Self::Float) -> Self::Float, a: &mut [f64]) {
        let mut val = vld1q_f64(a.as_ptr());
        val = f(val);
        vst1q_f64(a.as_mut_ptr(), val)
    }

    #[target_feature(enable = "neon")]
    unsafe fn apply_elementwise(
        f: impl Fn(Self::Float) -> Self::Float,
        f_rest: impl Fn(&mut [f64]),
        a: &mut [f64],
    ) {
        let v = Self;
        super::apply_elementwise_generic(&v, f, f_rest, a);
    }
}