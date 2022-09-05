use std::arch::x86_64::{
    __m256, __m256d, __m256i, _mm256_fmadd_pd, _mm256_fmadd_ps, _mm256_loadu_pd, _mm256_loadu_ps,
    _mm256_storeu_pd, _mm256_storeu_ps,
};
use std::mem;

use aligned::{Aligned, A32};

use super::avx::{AVXVector32, AVXVector64};
use super::SimdVector;
use crate::vector::sse41::{SSE41Vector32, SSE41Vector64};

#[derive(Default)]
pub struct AVX2Vector32;

impl SimdVector for AVX2Vector32 {
    type Lower = SSE41Vector32;
    type Float = __m256;
    type FloatScalar = f32;
    type FloatScalarArray = Aligned<
        A32,
        [Self::FloatScalar; mem::size_of::<Self::Float>() / mem::size_of::<Self::FloatScalar>()],
    >;
    type Int = __m256i;
    type IntScalar = i32;
    type Mask = __m256;

    #[target_feature(enable = "avx")]
    unsafe fn abs(a: Self::Float) -> Self::Float {
        AVXVector32::abs(a)
    }

    #[target_feature(enable = "avx")]
    unsafe fn add(a: Self::Float, b: Self::Float) -> Self::Float {
        AVXVector32::add(a, b)
    }

    #[target_feature(enable = "avx")]
    unsafe fn add_lanes(a: Self::Float) -> Self::FloatScalar {
        AVXVector32::add_lanes(a)
    }

    #[target_feature(enable = "avx")]
    unsafe fn add_scalar(a: Self::Float, b: f32) -> Self::Float {
        AVXVector32::add_scalar(a, b)
    }

    #[target_feature(enable = "avx")]
    unsafe fn bitwise_select(a: Self::Mask, b: Self::Float, c: Self::Float) -> Self::Float {
        AVXVector32::bitwise_select(a, b, c)
    }

    #[target_feature(enable = "avx")]
    unsafe fn clamp_max(a: Self::Float, max: Self::Float) -> Self::Float {
        AVXVector32::clamp_max(a, max)
    }

    #[target_feature(enable = "avx")]
    unsafe fn clamp_min(a: Self::Float, min: Self::Float) -> Self::Float {
        AVXVector32::clamp_min(a, min)
    }

    #[target_feature(enable = "avx")]
    unsafe fn copy_sign(sign_src: Self::Float, dest: Self::Float) -> Self::Float {
        AVXVector32::copy_sign(sign_src, dest)
    }

    #[target_feature(enable = "avx")]
    unsafe fn div(a: Self::Float, b: Self::Float) -> Self::Float {
        AVXVector32::div(a, b)
    }

    #[target_feature(enable = "avx")]
    unsafe fn floor(a: Self::Float) -> Self::Float {
        AVXVector32::floor(a)
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn fma(a: Self::Float, b: Self::Float, c: Self::Float) -> Self::Float {
        _mm256_fmadd_ps(a, b, c)
    }

    #[target_feature(enable = "avx")]
    unsafe fn eq(a: Self::Float, b: Self::Float) -> Self::Mask {
        AVXVector32::eq(a, b)
    }

    #[target_feature(enable = "avx")]
    unsafe fn gt(a: Self::Float, b: Self::Float) -> Self::Mask {
        AVXVector32::gt(a, b)
    }

    #[target_feature(enable = "avx")]
    unsafe fn load(a: &[Self::FloatScalar]) -> Self::Float {
        AVXVector32::load(a)
    }

    #[target_feature(enable = "avx")]
    unsafe fn lt(a: Self::Float, b: Self::Float) -> Self::Mask {
        AVXVector32::lt(a, b)
    }

    #[target_feature(enable = "avx")]
    unsafe fn max_lanes(a: Self::Float) -> Self::FloatScalar {
        AVXVector32::max_lanes(a)
    }

    #[target_feature(enable = "avx")]
    unsafe fn min_lanes(a: Self::Float) -> Self::FloatScalar {
        AVXVector32::min_lanes(a)
    }

    #[target_feature(enable = "avx")]
    unsafe fn mul(a: Self::Float, b: Self::Float) -> Self::Float {
        AVXVector32::mul(a, b)
    }

    #[target_feature(enable = "avx")]
    unsafe fn mul_scalar(a: Self::Float, b: f32) -> Self::Float {
        AVXVector32::mul_scalar(a, b)
    }

    #[target_feature(enable = "avx")]
    unsafe fn neg(a: Self::Float) -> Self::Float {
        AVXVector32::neg(a)
    }

    #[target_feature(enable = "avx")]
    unsafe fn sub(a: Self::Float, b: Self::Float) -> Self::Float {
        AVXVector32::sub(a, b)
    }

    #[target_feature(enable = "avx")]
    unsafe fn max(a: Self::Float, b: Self::Float) -> Self::Float {
        AVXVector32::max(a, b)
    }

    #[target_feature(enable = "avx")]
    unsafe fn min(a: Self::Float, b: Self::Float) -> Self::Float {
        AVXVector32::min(a, b)
    }

    #[target_feature(enable = "avx")]
    unsafe fn splat(v: f32) -> Self::Float {
        AVXVector32::splat(v)
    }

    #[target_feature(enable = "avx")]
    unsafe fn sqrt(v: Self::Float) -> Self::Float {
        AVXVector32::sqrt(v)
    }

    #[target_feature(enable = "avx")]
    unsafe fn reinterpret_float_signed(v: Self::Int) -> Self::Float {
        AVXVector32::reinterpret_float_signed(v)
    }

    #[target_feature(enable = "avx")]
    unsafe fn to_int(v: Self::Float) -> Self::Int {
        AVXVector32::to_int(v)
    }

    #[target_feature(enable = "avx")]
    unsafe fn to_float_scalar_array(v: Self::Float) -> Self::FloatScalarArray {
        AVXVector32::to_float_scalar_array(v)
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn with_load_store(f: &impl Fn(Self::Float) -> Self::Float, a: &mut [f32]) {
        let mut val = _mm256_loadu_ps(a.as_ptr());
        val = f(val);
        _mm256_storeu_ps(a.as_mut_ptr(), val);
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn apply_elementwise(
        f: impl Fn(Self::Float) -> Self::Float,
        f_rest: impl Fn(&mut [f32]),
        a: &mut [f32],
    ) {
        super::apply_elementwise_generic(Self, f, f_rest, a);
    }

    #[target_feature(enable = "avx2")]
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
pub struct AVX2Vector64;

impl SimdVector for AVX2Vector64 {
    type Lower = SSE41Vector64;
    type Float = __m256d;
    type FloatScalar = f64;
    type FloatScalarArray = Aligned<
        A32,
        [Self::FloatScalar; mem::size_of::<Self::Float>() / mem::size_of::<Self::FloatScalar>()],
    >;
    type Int = __m256i;
    type IntScalar = i64;
    type Mask = __m256d;

    #[target_feature(enable = "avx")]
    unsafe fn abs(a: Self::Float) -> Self::Float {
        AVXVector64::abs(a)
    }

    #[target_feature(enable = "avx")]
    unsafe fn add(a: Self::Float, b: Self::Float) -> Self::Float {
        AVXVector64::add(a, b)
    }

    #[target_feature(enable = "avx")]
    unsafe fn add_lanes(a: Self::Float) -> Self::FloatScalar {
        AVXVector64::add_lanes(a)
    }

    #[target_feature(enable = "avx")]
    unsafe fn add_scalar(a: Self::Float, b: f64) -> Self::Float {
        AVXVector64::add_scalar(a, b)
    }

    #[target_feature(enable = "avx")]
    unsafe fn bitwise_select(a: Self::Mask, b: Self::Float, c: Self::Float) -> Self::Float {
        AVXVector64::bitwise_select(a, b, c)
    }

    #[target_feature(enable = "avx")]
    unsafe fn clamp_max(a: Self::Float, max: Self::Float) -> Self::Float {
        AVXVector64::clamp_max(a, max)
    }

    #[target_feature(enable = "avx")]
    unsafe fn clamp_min(a: Self::Float, min: Self::Float) -> Self::Float {
        AVXVector64::clamp_min(a, min)
    }

    #[target_feature(enable = "avx")]
    unsafe fn copy_sign(sign_src: Self::Float, dest: Self::Float) -> Self::Float {
        AVXVector64::copy_sign(sign_src, dest)
    }

    #[target_feature(enable = "avx")]
    unsafe fn div(a: Self::Float, b: Self::Float) -> Self::Float {
        AVXVector64::div(a, b)
    }

    #[target_feature(enable = "avx")]
    unsafe fn floor(a: Self::Float) -> Self::Float {
        AVXVector64::floor(a)
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn fma(a: Self::Float, b: Self::Float, c: Self::Float) -> Self::Float {
        _mm256_fmadd_pd(a, b, c)
    }

    #[target_feature(enable = "avx")]
    unsafe fn eq(a: Self::Float, b: Self::Float) -> Self::Mask {
        AVXVector64::eq(a, b)
    }

    #[target_feature(enable = "avx")]
    unsafe fn gt(a: Self::Float, b: Self::Float) -> Self::Mask {
        AVXVector64::gt(a, b)
    }

    #[target_feature(enable = "avx")]
    unsafe fn load(a: &[Self::FloatScalar]) -> Self::Float {
        AVXVector64::load(a)
    }

    #[target_feature(enable = "avx")]
    unsafe fn lt(a: Self::Float, b: Self::Float) -> Self::Mask {
        AVXVector64::lt(a, b)
    }

    #[target_feature(enable = "avx")]
    unsafe fn max_lanes(a: Self::Float) -> Self::FloatScalar {
        AVXVector64::max_lanes(a)
    }

    #[target_feature(enable = "avx")]
    unsafe fn min_lanes(a: Self::Float) -> Self::FloatScalar {
        AVXVector64::min_lanes(a)
    }

    #[target_feature(enable = "avx")]
    unsafe fn mul(a: Self::Float, b: Self::Float) -> Self::Float {
        AVXVector64::mul(a, b)
    }

    #[target_feature(enable = "avx")]
    unsafe fn mul_scalar(a: Self::Float, b: f64) -> Self::Float {
        AVXVector64::mul_scalar(a, b)
    }

    #[target_feature(enable = "avx")]
    unsafe fn neg(a: Self::Float) -> Self::Float {
        AVXVector64::neg(a)
    }

    #[target_feature(enable = "avx")]
    unsafe fn sub(a: Self::Float, b: Self::Float) -> Self::Float {
        AVXVector64::sub(a, b)
    }

    #[target_feature(enable = "avx")]
    unsafe fn max(a: Self::Float, b: Self::Float) -> Self::Float {
        AVXVector64::max(a, b)
    }

    #[target_feature(enable = "avx")]
    unsafe fn min(a: Self::Float, b: Self::Float) -> Self::Float {
        AVXVector64::min(a, b)
    }

    #[target_feature(enable = "avx")]
    unsafe fn splat(v: f64) -> Self::Float {
        AVXVector64::splat(v)
    }

    #[target_feature(enable = "avx")]
    unsafe fn sqrt(v: Self::Float) -> Self::Float {
        AVXVector64::sqrt(v)
    }

    #[target_feature(enable = "avx")]
    unsafe fn reinterpret_float_signed(v: Self::Int) -> Self::Float {
        AVXVector64::reinterpret_float_signed(v)
    }

    #[target_feature(enable = "avx")]
    unsafe fn to_int(v: Self::Float) -> Self::Int {
        AVXVector64::to_int(v)
    }

    #[target_feature(enable = "avx")]
    unsafe fn to_float_scalar_array(v: Self::Float) -> Self::FloatScalarArray {
        AVXVector64::to_float_scalar_array(v)
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn with_load_store(f: &impl Fn(Self::Float) -> Self::Float, a: &mut [f64]) {
        let mut val = _mm256_loadu_pd(a.as_ptr());
        val = f(val);
        _mm256_storeu_pd(a.as_mut_ptr(), val);
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn apply_elementwise(
        f: impl Fn(Self::Float) -> Self::Float,
        f_rest: impl Fn(&mut [f64]),
        a: &mut [f64],
    ) {
        super::apply_elementwise_generic(Self, f, f_rest, a);
    }

    #[target_feature(enable = "avx2")]
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
