use std::arch::x86_64::{
    __m128, __m128d, __m128i, _mm_floor_pd, _mm_floor_ps, _mm_loadu_pd, _mm_loadu_ps,
    _mm_storeu_pd, _mm_storeu_ps,
};
use std::mem;

use aligned::{Aligned, A16};

use crate::vector::scalar::{ScalarVector32, ScalarVector64};
use crate::vector::SimdVector;

use super::sse2::{SSE2Vector32, SSE2Vector64};

#[derive(Default)]
pub struct SSE41Vector32;

impl SimdVector for SSE41Vector32 {
    type Lower = ScalarVector32;
    type Float = __m128;
    type FloatScalar = f32;
    type FloatScalarArray = Aligned<
        A16,
        [Self::FloatScalar; mem::size_of::<Self::Float>() / mem::size_of::<Self::FloatScalar>()],
    >;
    type Int = __m128i;
    type IntScalar = i32;
    type Mask = __m128;

    #[target_feature(enable = "sse2")]
    unsafe fn abs(a: Self::Float) -> Self::Float {
        SSE2Vector32::abs(a)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn add(a: Self::Float, b: Self::Float) -> Self::Float {
        SSE2Vector32::add(a, b)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn add_lanes(a: Self::Float) -> Self::FloatScalar {
        SSE2Vector32::add_lanes(a)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn add_scalar(a: Self::Float, b: Self::FloatScalar) -> Self::Float {
        SSE2Vector32::add_scalar(a, b)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn bitwise_select(a: Self::Mask, b: Self::Float, c: Self::Float) -> Self::Float {
        SSE2Vector32::bitwise_select(a, b, c)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn clamp_max(a: Self::Float, max: Self::Float) -> Self::Float {
        SSE2Vector32::clamp_max(a, max)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn clamp_min(a: Self::Float, min: Self::Float) -> Self::Float {
        SSE2Vector32::clamp_min(a, min)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn copy_sign(sign_src: Self::Float, dest: Self::Float) -> Self::Float {
        SSE2Vector32::copy_sign(sign_src, dest)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn div(a: Self::Float, b: Self::Float) -> Self::Float {
        SSE2Vector32::div(a, b)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn fma(a: Self::Float, b: Self::Float, c: Self::Float) -> Self::Float {
        SSE2Vector32::fma(a, b, c)
    }

    #[target_feature(enable = "sse4.1")]
    unsafe fn floor(a: Self::Float) -> Self::Float {
        _mm_floor_ps(a)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn eq(a: Self::Float, b: Self::Float) -> Self::Mask {
        SSE2Vector32::eq(a, b)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn gt(a: Self::Float, b: Self::Float) -> Self::Mask {
        SSE2Vector32::gt(a, b)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn load(a: &[Self::FloatScalar]) -> Self::Float {
        SSE2Vector32::load(a)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn lt(a: Self::Float, b: Self::Float) -> Self::Mask {
        SSE2Vector32::lt(a, b)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn max_lanes(a: Self::Float) -> Self::FloatScalar {
        SSE2Vector32::max_lanes(a)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn min_lanes(a: Self::Float) -> Self::FloatScalar {
        SSE2Vector32::min_lanes(a)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn mul(a: Self::Float, b: Self::Float) -> Self::Float {
        SSE2Vector32::mul(a, b)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn mul_scalar(a: Self::Float, b: Self::FloatScalar) -> Self::Float {
        SSE2Vector32::mul_scalar(a, b)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn neg(a: Self::Float) -> Self::Float {
        SSE2Vector32::neg(a)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn sub(a: Self::Float, b: Self::Float) -> Self::Float {
        SSE2Vector32::sub(a, b)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn max(a: Self::Float, b: Self::Float) -> Self::Float {
        SSE2Vector32::max(a, b)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn min(a: Self::Float, b: Self::Float) -> Self::Float {
        SSE2Vector32::min(a, b)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn splat(v: Self::FloatScalar) -> Self::Float {
        SSE2Vector32::splat(v)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn sqrt(v: Self::Float) -> Self::Float {
        SSE2Vector32::sqrt(v)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn reinterpret_float_signed(v: Self::Int) -> Self::Float {
        SSE2Vector32::reinterpret_float_signed(v)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn to_int(v: Self::Float) -> Self::Int {
        SSE2Vector32::to_int(v)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn to_float_scalar_array(v: Self::Float) -> Self::FloatScalarArray {
        SSE2Vector32::to_float_scalar_array(v)
    }

    #[target_feature(enable = "sse4.1")]
    unsafe fn with_load_store(
        f: &impl Fn(Self::Float) -> Self::Float,
        a: &mut [Self::FloatScalar],
    ) {
        let mut val = _mm_loadu_ps(a.as_ptr());
        val = f(val);
        _mm_storeu_ps(a.as_mut_ptr(), val);
    }

    #[target_feature(enable = "sse4.1")]
    unsafe fn apply_elementwise(
        f: impl Fn(Self::Float) -> Self::Float,
        f_rest: impl Fn(&mut [Self::FloatScalar]),
        a: &mut [Self::FloatScalar],
    ) {
        super::apply_elementwise_generic(Self, f, f_rest, a);
    }

    #[target_feature(enable = "sse4.1")]
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

#[derive(Default)]
pub struct SSE41Vector64;

impl SimdVector for SSE41Vector64 {
    type Lower = ScalarVector64;
    type Float = __m128d;
    type FloatScalar = f64;
    type FloatScalarArray = Aligned<
        A16,
        [Self::FloatScalar; mem::size_of::<Self::Float>() / mem::size_of::<Self::FloatScalar>()],
    >;
    type Int = __m128i;
    type IntScalar = i64;
    type Mask = __m128d;

    #[target_feature(enable = "sse2")]
    unsafe fn abs(a: Self::Float) -> Self::Float {
        SSE2Vector64::abs(a)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn add(a: Self::Float, b: Self::Float) -> Self::Float {
        SSE2Vector64::add(a, b)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn add_lanes(a: Self::Float) -> Self::FloatScalar {
        SSE2Vector64::add_lanes(a)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn add_scalar(a: Self::Float, b: Self::FloatScalar) -> Self::Float {
        SSE2Vector64::add_scalar(a, b)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn bitwise_select(a: Self::Mask, b: Self::Float, c: Self::Float) -> Self::Float {
        SSE2Vector64::bitwise_select(a, b, c)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn clamp_max(a: Self::Float, max: Self::Float) -> Self::Float {
        SSE2Vector64::clamp_max(a, max)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn clamp_min(a: Self::Float, min: Self::Float) -> Self::Float {
        SSE2Vector64::clamp_min(a, min)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn copy_sign(sign_src: Self::Float, dest: Self::Float) -> Self::Float {
        SSE2Vector64::copy_sign(sign_src, dest)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn div(a: Self::Float, b: Self::Float) -> Self::Float {
        SSE2Vector64::div(a, b)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn fma(a: Self::Float, b: Self::Float, c: Self::Float) -> Self::Float {
        SSE2Vector64::fma(a, b, c)
    }

    #[target_feature(enable = "sse4.1")]
    unsafe fn floor(a: Self::Float) -> Self::Float {
        _mm_floor_pd(a)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn eq(a: Self::Float, b: Self::Float) -> Self::Mask {
        SSE2Vector64::eq(a, b)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn gt(a: Self::Float, b: Self::Float) -> Self::Mask {
        SSE2Vector64::gt(a, b)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn load(a: &[Self::FloatScalar]) -> Self::Float {
        SSE2Vector64::load(a)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn lt(a: Self::Float, b: Self::Float) -> Self::Mask {
        SSE2Vector64::lt(a, b)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn max_lanes(a: Self::Float) -> Self::FloatScalar {
        SSE2Vector64::max_lanes(a)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn min_lanes(a: Self::Float) -> Self::FloatScalar {
        SSE2Vector64::min_lanes(a)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn mul(a: Self::Float, b: Self::Float) -> Self::Float {
        SSE2Vector64::mul(a, b)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn mul_scalar(a: Self::Float, b: Self::FloatScalar) -> Self::Float {
        SSE2Vector64::mul_scalar(a, b)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn neg(a: Self::Float) -> Self::Float {
        SSE2Vector64::neg(a)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn sub(a: Self::Float, b: Self::Float) -> Self::Float {
        SSE2Vector64::sub(a, b)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn max(a: Self::Float, b: Self::Float) -> Self::Float {
        SSE2Vector64::max(a, b)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn min(a: Self::Float, b: Self::Float) -> Self::Float {
        SSE2Vector64::min(a, b)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn splat(v: Self::FloatScalar) -> Self::Float {
        SSE2Vector64::splat(v)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn sqrt(v: Self::Float) -> Self::Float {
        SSE2Vector64::sqrt(v)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn reinterpret_float_signed(v: Self::Int) -> Self::Float {
        SSE2Vector64::reinterpret_float_signed(v)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn to_int(v: Self::Float) -> Self::Int {
        SSE2Vector64::to_int(v)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn to_float_scalar_array(v: Self::Float) -> Self::FloatScalarArray {
        SSE2Vector64::to_float_scalar_array(v)
    }

    #[target_feature(enable = "sse4.1")]
    unsafe fn with_load_store(
        f: &impl Fn(Self::Float) -> Self::Float,
        a: &mut [Self::FloatScalar],
    ) {
        let mut val = _mm_loadu_pd(a.as_ptr());
        val = f(val);
        _mm_storeu_pd(a.as_mut_ptr(), val);
    }

    #[target_feature(enable = "sse4.1")]
    unsafe fn apply_elementwise(
        f: impl Fn(Self::Float) -> Self::Float,
        f_rest: impl Fn(&mut [Self::FloatScalar]),
        a: &mut [Self::FloatScalar],
    ) {
        super::apply_elementwise_generic(Self, f, f_rest, a);
    }

    #[target_feature(enable = "sse4.1")]
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
