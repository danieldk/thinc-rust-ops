use std::arch::x86_64::{
    __m128, __m128d, __m128i, _mm_add_pd, _mm_add_ps, _mm_and_pd, _mm_and_ps, _mm_andnot_pd,
    _mm_andnot_ps, _mm_castsi128_pd, _mm_castsi128_ps, _mm_cmpeq_pd, _mm_cmpeq_ps, _mm_cmpgt_pd,
    _mm_cmpgt_ps, _mm_cmplt_pd, _mm_cmplt_ps, _mm_cvtps_epi32, _mm_cvtsd_f64, _mm_cvtss_f32,
    _mm_div_pd, _mm_div_ps, _mm_floor_pd, _mm_floor_ps, _mm_load_si128, _mm_loadu_pd, _mm_loadu_ps,
    _mm_movehdup_ps, _mm_movehl_ps, _mm_mul_pd, _mm_mul_ps, _mm_or_pd, _mm_or_ps, _mm_set1_pd,
    _mm_set1_ps, _mm_store_pd, _mm_store_ps, _mm_storeu_pd, _mm_storeu_ps, _mm_sub_pd, _mm_sub_ps,
    _mm_unpackhi_pd, _mm_xor_pd, _mm_xor_ps,
};
use std::mem;
use std::ops::Neg;

use aligned::{Aligned, A16};
use num_traits::{Float, Zero};

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
        let sign_mask = Self::splat(Self::FloatScalar::zero().neg());
        _mm_andnot_ps(sign_mask, a)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn add(a: Self::Float, b: Self::Float) -> Self::Float {
        _mm_add_ps(a, b)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn add_lanes(a: Self::Float) -> Self::FloatScalar {
        let sums = _mm_add_ps(a, _mm_movehl_ps(a, a));
        let sums = _mm_add_ps(sums, _mm_movehdup_ps(sums));
        _mm_cvtss_f32(sums)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn add_scalar(a: Self::Float, b: Self::FloatScalar) -> Self::Float {
        _mm_add_ps(a, _mm_set1_ps(b))
    }

    #[target_feature(enable = "sse2")]
    unsafe fn bitwise_select(a: Self::Mask, b: Self::Float, c: Self::Float) -> Self::Float {
        let u = _mm_and_ps(a, b);
        let v = _mm_andnot_ps(a, c);
        _mm_or_ps(u, v)
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
        let sign_bit_mask = Self::splat(Self::FloatScalar::zero().neg());
        Self::bitwise_select(sign_bit_mask, sign_src, dest)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn div(a: Self::Float, b: Self::Float) -> Self::Float {
        _mm_div_ps(a, b)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn fma(a: Self::Float, b: Self::Float, c: Self::Float) -> Self::Float {
        _mm_add_ps(_mm_mul_ps(a, b), c)
    }

    #[target_feature(enable = "sse4.1")]
    unsafe fn floor(a: Self::Float) -> Self::Float {
        _mm_floor_ps(a)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn eq(a: Self::Float, b: Self::Float) -> Self::Mask {
        _mm_cmpeq_ps(a, b)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn gt(a: Self::Float, b: Self::Float) -> Self::Mask {
        _mm_cmpgt_ps(a, b)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn load(a: &[Self::FloatScalar]) -> Self::Float {
        _mm_loadu_ps(a.as_ptr())
    }

    #[target_feature(enable = "sse2")]
    unsafe fn lt(a: Self::Float, b: Self::Float) -> Self::Mask {
        _mm_cmplt_ps(a, b)
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
        _mm_mul_ps(a, b)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn mul_scalar(a: Self::Float, b: Self::FloatScalar) -> Self::Float {
        _mm_mul_ps(a, _mm_set1_ps(b))
    }

    #[target_feature(enable = "sse2")]
    unsafe fn neg(a: Self::Float) -> Self::Float {
        let neg_zero = _mm_set1_ps(Self::FloatScalar::neg_zero());
        _mm_xor_ps(a, neg_zero)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn sub(a: Self::Float, b: Self::Float) -> Self::Float {
        _mm_sub_ps(a, b)
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
        _mm_set1_ps(v)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn sqrt(v: Self::Float) -> Self::Float {
        SSE2Vector32::sqrt(v)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn reinterpret_float_signed(v: Self::Int) -> Self::Float {
        _mm_castsi128_ps(v)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn to_int(v: Self::Float) -> Self::Int {
        _mm_cvtps_epi32(v)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn to_float_scalar_array(v: Self::Float) -> Self::FloatScalarArray {
        let mut a: Aligned<A16, _> = Aligned([0f32; 4]);
        _mm_store_ps(a.as_mut_ptr(), v);
        a
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
        let sign_mask = Self::splat(Self::FloatScalar::zero().neg());
        _mm_andnot_pd(sign_mask, a)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn add(a: Self::Float, b: Self::Float) -> Self::Float {
        _mm_add_pd(a, b)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn add_lanes(a: Self::Float) -> Self::FloatScalar {
        _mm_cvtsd_f64(a) + _mm_cvtsd_f64(_mm_unpackhi_pd(a, a))
    }

    #[target_feature(enable = "sse2")]
    unsafe fn add_scalar(a: Self::Float, b: Self::FloatScalar) -> Self::Float {
        _mm_add_pd(a, _mm_set1_pd(b))
    }

    #[target_feature(enable = "sse2")]
    unsafe fn bitwise_select(a: Self::Mask, b: Self::Float, c: Self::Float) -> Self::Float {
        let u = _mm_and_pd(a, b);
        let v = _mm_andnot_pd(a, c);
        _mm_or_pd(u, v)
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
        let sign_bit_mask = Self::splat(Self::FloatScalar::zero().neg());
        Self::bitwise_select(sign_bit_mask, sign_src, dest)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn div(a: Self::Float, b: Self::Float) -> Self::Float {
        _mm_div_pd(a, b)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn fma(a: Self::Float, b: Self::Float, c: Self::Float) -> Self::Float {
        _mm_add_pd(_mm_mul_pd(a, b), c)
    }

    #[target_feature(enable = "sse4.1")]
    unsafe fn floor(a: Self::Float) -> Self::Float {
        _mm_floor_pd(a)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn eq(a: Self::Float, b: Self::Float) -> Self::Mask {
        _mm_cmpeq_pd(a, b)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn gt(a: Self::Float, b: Self::Float) -> Self::Mask {
        _mm_cmpgt_pd(a, b)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn load(a: &[Self::FloatScalar]) -> Self::Float {
        _mm_loadu_pd(a.as_ptr())
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
    unsafe fn lt(a: Self::Float, b: Self::Float) -> Self::Mask {
        _mm_cmplt_pd(a, b)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn mul(a: Self::Float, b: Self::Float) -> Self::Float {
        _mm_mul_pd(a, b)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn mul_scalar(a: Self::Float, b: Self::FloatScalar) -> Self::Float {
        _mm_mul_pd(a, _mm_set1_pd(b))
    }

    #[target_feature(enable = "sse2")]
    unsafe fn neg(a: Self::Float) -> Self::Float {
        let neg_zero = _mm_set1_pd(Self::FloatScalar::neg_zero());
        _mm_xor_pd(a, neg_zero)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn sub(a: Self::Float, b: Self::Float) -> Self::Float {
        _mm_sub_pd(a, b)
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
        _mm_set1_pd(v)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn sqrt(v: Self::Float) -> Self::Float {
        SSE2Vector64::sqrt(v)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn reinterpret_float_signed(v: Self::Int) -> Self::Float {
        _mm_castsi128_pd(v)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn to_int(v: Self::Float) -> Self::Int {
        // Blegh, no instruction for this before AVX-512.
        let mut data_f64: Aligned<A16, _> = Aligned([0f64; 2]);
        _mm_store_pd(data_f64.as_mut_ptr(), v);
        let data: Aligned<A16, [i64; 2]> = Aligned(data_f64.map(|v| v as i64));
        _mm_load_si128(data.as_ptr().cast())
    }

    #[target_feature(enable = "sse2")]
    unsafe fn to_float_scalar_array(v: Self::Float) -> Self::FloatScalarArray {
        let mut a: Aligned<A16, _> = Aligned([0f64; 2]);
        _mm_store_pd(a.as_mut_ptr(), v);
        a
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
