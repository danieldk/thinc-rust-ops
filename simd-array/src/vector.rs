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
        let v = Self;
        apply_elementwise_generic(&v, f, f_rest, a);
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
        let v = Self;
        apply_elementwise_generic(&v, f, f_rest, a);
    }
}

#[cfg(all(target_arch = "x86_64"))]
pub mod avx {
    use aligned::{Aligned, A32};
    use std::arch::x86_64::{
        __m256, __m256d, __m256i, _mm256_add_pd, _mm256_add_ps, _mm256_and_pd, _mm256_and_ps,
        _mm256_andnot_pd, _mm256_andnot_ps, _mm256_castsi256_pd, _mm256_castsi256_ps,
        _mm256_cmp_pd, _mm256_cmp_ps, _mm256_cvtps_epi32, _mm256_div_pd, _mm256_div_ps,
        _mm256_floor_pd, _mm256_floor_ps, _mm256_loadu_pd, _mm256_loadu_ps, _mm256_loadu_si256,
        _mm256_max_pd, _mm256_max_ps, _mm256_min_pd, _mm256_min_ps, _mm256_mul_pd, _mm256_mul_ps,
        _mm256_or_pd, _mm256_or_ps, _mm256_set1_epi32, _mm256_set1_epi64x, _mm256_set1_pd,
        _mm256_set1_ps, _mm256_store_pd, _mm256_store_ps, _mm256_storeu_pd, _mm256_storeu_ps,
        _mm256_sub_pd, _mm256_sub_ps, _mm256_xor_pd, _mm256_xor_ps, _CMP_EQ_OQ, _CMP_GT_OQ,
        _CMP_LT_OQ,
    };
    use std::mem;
    use std::ops::Neg;

    use super::{ScalarVector32, SimdVector};
    use crate::vector::ScalarVector64;
    use num_traits::{Float, Zero};

    #[derive(Default)]
    pub struct AVXVector32;

    impl SimdVector for AVXVector32 {
        type Lower = ScalarVector32;
        type Float = __m256;
        type FloatScalar = f32;
        type FloatScalarArray = Aligned<
            A32,
            [Self::FloatScalar;
                mem::size_of::<Self::Float>() / mem::size_of::<Self::FloatScalar>()],
        >;
        type Int = __m256i;
        type IntScalar = i32;
        type Mask = __m256;

        unsafe fn abs(a: Self::Float) -> Self::Float {
            let mask = _mm256_set1_epi32(0x7fffffff);
            _mm256_and_ps(a, _mm256_castsi256_ps(mask))
        }

        unsafe fn add(a: Self::Float, b: Self::Float) -> Self::Float {
            _mm256_add_ps(a, b)
        }

        unsafe fn add_scalar(a: Self::Float, b: f32) -> Self::Float {
            let b_simd = _mm256_set1_ps(b);
            _mm256_add_ps(a, b_simd)
        }

        unsafe fn bitwise_select(a: Self::Mask, b: Self::Float, c: Self::Float) -> Self::Float {
            // Self::Float::from_bits((a & b.to_bits()) | ((!a) & c.to_bits()))
            let u = _mm256_and_ps(a, b);
            let v = _mm256_andnot_ps(a, c);
            _mm256_or_ps(u, v)
        }

        unsafe fn copy_sign(sign_src: Self::Float, dest: Self::Float) -> Self::Float {
            // Negative zero has all bits unset, except the sign bit.
            let sign_bit_mask = Self::splat(Self::FloatScalar::zero().neg());
            Self::bitwise_select(sign_bit_mask, sign_src, dest)
        }

        unsafe fn div(a: Self::Float, b: Self::Float) -> Self::Float {
            _mm256_div_ps(a, b)
        }

        unsafe fn floor(a: Self::Float) -> Self::Float {
            _mm256_floor_ps(a)
        }

        unsafe fn fma(a: Self::Float, b: Self::Float, c: Self::Float) -> Self::Float {
            _mm256_add_ps(_mm256_mul_ps(a, b), c)
        }

        unsafe fn eq(a: Self::Float, b: Self::Float) -> Self::Mask {
            _mm256_cmp_ps::<_CMP_EQ_OQ>(a, b)
        }

        unsafe fn gt(a: Self::Float, b: Self::Float) -> Self::Mask {
            _mm256_cmp_ps::<_CMP_GT_OQ>(a, b)
        }

        unsafe fn lt(a: Self::Float, b: Self::Float) -> Self::Mask {
            _mm256_cmp_ps::<_CMP_LT_OQ>(a, b)
        }

        unsafe fn mul(a: Self::Float, b: Self::Float) -> Self::Float {
            _mm256_mul_ps(a, b)
        }

        unsafe fn mul_scalar(a: Self::Float, b: f32) -> Self::Float {
            let b_simd = _mm256_set1_ps(b);
            _mm256_mul_ps(a, b_simd)
        }

        unsafe fn neg(a: Self::Float) -> Self::Float {
            let neg_zero = _mm256_set1_ps(Self::FloatScalar::neg_zero());
            _mm256_xor_ps(a, neg_zero)
        }

        unsafe fn sub(a: Self::Float, b: Self::Float) -> Self::Float {
            _mm256_sub_ps(a, b)
        }

        unsafe fn vmax(a: Self::Float, b: Self::Float) -> Self::Float {
            _mm256_max_ps(a, b)
        }

        unsafe fn vmin(a: Self::Float, b: Self::Float) -> Self::Float {
            _mm256_min_ps(a, b)
        }

        unsafe fn splat(v: f32) -> Self::Float {
            _mm256_set1_ps(v)
        }

        unsafe fn reinterpret_float_signed(v: Self::Int) -> Self::Float {
            _mm256_castsi256_ps(v)
        }

        unsafe fn to_int(v: Self::Float) -> Self::Int {
            _mm256_cvtps_epi32(v)
        }

        unsafe fn to_float_scalar_array(v: Self::Float) -> Self::FloatScalarArray {
            let mut a: Aligned<A32, _> = Aligned([0f32; 8]);
            _mm256_store_ps(a.as_mut_ptr(), v);
            a
        }

        unsafe fn with_load_store(f: &impl Fn(Self::Float) -> Self::Float, a: &mut [f32]) {
            let mut val = _mm256_loadu_ps(a.as_ptr());
            val = f(val);
            _mm256_storeu_ps(a.as_mut_ptr(), val);
        }

        #[target_feature(enable = "avx")]
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
    pub struct AVXVector64;

    impl SimdVector for AVXVector64 {
        type Lower = ScalarVector64;
        type Float = __m256d;
        type FloatScalar = f64;
        type FloatScalarArray = Aligned<
            A32,
            [Self::FloatScalar;
                mem::size_of::<Self::Float>() / mem::size_of::<Self::FloatScalar>()],
        >;
        type Int = __m256i;
        type IntScalar = i64;
        type Mask = __m256d;

        unsafe fn abs(a: Self::Float) -> Self::Float {
            let mask = _mm256_set1_epi64x(0x7fffffffffffffff);
            _mm256_and_pd(a, _mm256_castsi256_pd(mask))
        }

        unsafe fn add(a: Self::Float, b: Self::Float) -> Self::Float {
            _mm256_add_pd(a, b)
        }

        unsafe fn add_scalar(a: Self::Float, b: f64) -> Self::Float {
            let b_simd = _mm256_set1_pd(b);
            _mm256_add_pd(a, b_simd)
        }

        unsafe fn bitwise_select(a: Self::Mask, b: Self::Float, c: Self::Float) -> Self::Float {
            let u = _mm256_and_pd(a, b);
            let v = _mm256_andnot_pd(a, c);
            _mm256_or_pd(u, v)
        }

        unsafe fn copy_sign(sign_src: Self::Float, dest: Self::Float) -> Self::Float {
            // Negative zero has all bits unset, except the sign bit.
            let sign_bit_mask = Self::splat(Self::FloatScalar::zero().neg());
            Self::bitwise_select(sign_bit_mask, sign_src, dest)
        }

        unsafe fn div(a: Self::Float, b: Self::Float) -> Self::Float {
            _mm256_div_pd(a, b)
        }

        unsafe fn floor(a: Self::Float) -> Self::Float {
            _mm256_floor_pd(a)
        }

        unsafe fn fma(a: Self::Float, b: Self::Float, c: Self::Float) -> Self::Float {
            _mm256_add_pd(_mm256_mul_pd(a, b), c)
        }

        unsafe fn eq(a: Self::Float, b: Self::Float) -> Self::Mask {
            _mm256_cmp_pd::<_CMP_EQ_OQ>(a, b)
        }

        unsafe fn gt(a: Self::Float, b: Self::Float) -> Self::Mask {
            _mm256_cmp_pd::<_CMP_GT_OQ>(a, b)
        }

        unsafe fn lt(a: Self::Float, b: Self::Float) -> Self::Mask {
            _mm256_cmp_pd::<_CMP_LT_OQ>(a, b)
        }

        unsafe fn mul(a: Self::Float, b: Self::Float) -> Self::Float {
            _mm256_mul_pd(a, b)
        }

        unsafe fn mul_scalar(a: Self::Float, b: f64) -> Self::Float {
            let b_simd = _mm256_set1_pd(b);
            _mm256_mul_pd(a, b_simd)
        }

        unsafe fn neg(a: Self::Float) -> Self::Float {
            let neg_zero = _mm256_set1_pd(Self::FloatScalar::neg_zero());
            _mm256_xor_pd(a, neg_zero)
        }

        unsafe fn sub(a: Self::Float, b: Self::Float) -> Self::Float {
            _mm256_sub_pd(a, b)
        }

        unsafe fn vmax(a: Self::Float, b: Self::Float) -> Self::Float {
            _mm256_max_pd(a, b)
        }

        unsafe fn vmin(a: Self::Float, b: Self::Float) -> Self::Float {
            _mm256_min_pd(a, b)
        }

        unsafe fn splat(v: f64) -> Self::Float {
            _mm256_set1_pd(v)
        }

        unsafe fn reinterpret_float_signed(v: Self::Int) -> Self::Float {
            _mm256_castsi256_pd(v)
        }

        unsafe fn to_int(v: Self::Float) -> Self::Int {
            // Blegh, no instruction for this before AVX-512.
            let mut data_f64 = [0f64; 4];
            _mm256_storeu_pd(data_f64.as_mut_ptr(), v);
            let data = data_f64.map(|v| v as i64);
            _mm256_loadu_si256(data.as_ptr().cast())
        }

        unsafe fn to_float_scalar_array(v: Self::Float) -> Self::FloatScalarArray {
            let mut a: Aligned<A32, _> = Aligned([0f64; 4]);
            _mm256_store_pd(a.as_mut_ptr(), v);
            a
        }

        unsafe fn with_load_store(f: &impl Fn(Self::Float) -> Self::Float, a: &mut [f64]) {
            let mut val = _mm256_loadu_pd(a.as_ptr());
            val = f(val);
            _mm256_storeu_pd(a.as_mut_ptr(), val);
        }

        #[target_feature(enable = "avx")]
        unsafe fn apply_elementwise(
            f: impl Fn(Self::Float) -> Self::Float,
            f_rest: impl Fn(&mut [f64]),
            a: &mut [f64],
        ) {
            let v = Self;
            super::apply_elementwise_generic(&v, f, f_rest, a);
        }
    }
}

#[cfg(all(target_arch = "aarch64"))]
pub mod neon {
    use num_traits::Zero;
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
    use std::mem;
    use std::ops::Neg;

    use super::{ScalarVector32, SimdVector};
    use crate::vector::ScalarVector64;

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

        unsafe fn abs(a: Self::Float) -> Self::Float {
            vabsq_f32(a)
        }

        unsafe fn add(a: Self::Float, b: Self::Float) -> Self::Float {
            vaddq_f32(a, b)
        }

        unsafe fn add_scalar(a: Self::Float, b: f32) -> Self::Float {
            let b_simd = vdupq_n_f32(b);
            vaddq_f32(a, b_simd)
        }

        unsafe fn bitwise_select(a: Self::Mask, b: Self::Float, c: Self::Float) -> Self::Float {
            // We want to use the bit selection intrinsic, however it is currently broken:
            // https://github.com/rust-lang/stdarch/issues/1191
            // vbslq_f32(a, b, c)

            let b = vreinterpretq_u32_f32(b);
            let c = vreinterpretq_u32_f32(c);
            let r = vorrq_u32(vandq_u32(a, b), vbicq_u32(c, a));
            vreinterpretq_f32_u32(r)
        }

        unsafe fn copy_sign(sign_src: Self::Float, dest: Self::Float) -> Self::Float {
            // Negative zero has all bits unset, except the sign bit.
            let sign_bit_mask = vreinterpretq_u32_f32(Self::splat(Self::FloatScalar::zero().neg()));
            Self::bitwise_select(sign_bit_mask, sign_src, dest)
        }

        unsafe fn div(a: Self::Float, b: Self::Float) -> Self::Float {
            vdivq_f32(a, b)
        }

        unsafe fn floor(a: Self::Float) -> Self::Float {
            vrndmq_f32(a)
        }

        unsafe fn fma(a: Self::Float, b: Self::Float, c: Self::Float) -> Self::Float {
            vfmaq_f32(c, a, b)
        }

        unsafe fn eq(a: Self::Float, b: Self::Float) -> Self::Mask {
            vceqq_f32(a, b)
        }

        unsafe fn gt(a: Self::Float, b: Self::Float) -> Self::Mask {
            vcgtq_f32(a, b)
        }

        unsafe fn lt(a: Self::Float, b: Self::Float) -> Self::Mask {
            vcltq_f32(a, b)
        }

        unsafe fn mul(a: Self::Float, b: Self::Float) -> Self::Float {
            vmulq_f32(a, b)
        }

        unsafe fn mul_scalar(a: Self::Float, b: f32) -> Self::Float {
            let b_simd = vdupq_n_f32(b);
            vmulq_f32(a, b_simd)
        }

        unsafe fn neg(a: Self::Float) -> Self::Float {
            vnegq_f32(a)
        }

        unsafe fn sub(a: Self::Float, b: Self::Float) -> Self::Float {
            vsubq_f32(a, b)
        }

        unsafe fn vmax(a: Self::Float, b: Self::Float) -> Self::Float {
            vmaxq_f32(a, b)
        }

        unsafe fn vmin(a: Self::Float, b: Self::Float) -> Self::Float {
            vminq_f32(a, b)
        }

        unsafe fn splat(v: f32) -> Self::Float {
            vdupq_n_f32(v)
        }

        unsafe fn reinterpret_float_signed(v: Self::Int) -> Self::Float {
            vreinterpretq_f32_s32(v)
        }

        unsafe fn to_int(v: Self::Float) -> Self::Int {
            vcvtq_s32_f32(v)
        }

        unsafe fn to_float_scalar_array(v: Self::Float) -> Self::FloatScalarArray {
            let mut a = [0f32; 4];
            vst1q_f32(a.as_mut_ptr(), v);
            a
        }

        unsafe fn with_load_store(f: &impl Fn(Self::Float) -> Self::Float, a: &mut [f32]) {
            let mut val = vld1q_f32(a.as_ptr());
            val = f(val);
            vst1q_f32(a.as_mut_ptr(), val)
        }

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

        unsafe fn abs(a: Self::Float) -> Self::Float {
            vabsq_f64(a)
        }

        unsafe fn add(a: Self::Float, b: Self::Float) -> Self::Float {
            vaddq_f64(a, b)
        }

        unsafe fn add_scalar(a: Self::Float, b: f64) -> Self::Float {
            let b_simd = vdupq_n_f64(b);
            vaddq_f64(a, b_simd)
        }

        unsafe fn bitwise_select(a: Self::Mask, b: Self::Float, c: Self::Float) -> Self::Float {
            // We want to use the bit selection intrinsic, however it is currently broken:
            // https://github.com/rust-lang/stdarch/issues/1191
            // vbslq_f64(a, b, c)

            let b = vreinterpretq_u64_f64(b);
            let c = vreinterpretq_u64_f64(c);
            let r = vorrq_u64(vandq_u64(a, b), vbicq_u64(c, a));
            vreinterpretq_f64_u64(r)
        }

        unsafe fn copy_sign(sign_src: Self::Float, dest: Self::Float) -> Self::Float {
            // Negative zero has all bits unset, except the sign bit.
            let sign_bit_mask = vreinterpretq_u64_f64(Self::splat(Self::FloatScalar::zero().neg()));
            Self::bitwise_select(sign_bit_mask, sign_src, dest)
        }

        unsafe fn div(a: Self::Float, b: Self::Float) -> Self::Float {
            vdivq_f64(a, b)
        }

        unsafe fn floor(a: Self::Float) -> Self::Float {
            vrndmq_f64(a)
        }

        unsafe fn fma(a: Self::Float, b: Self::Float, c: Self::Float) -> Self::Float {
            vfmaq_f64(c, a, b)
        }

        unsafe fn eq(a: Self::Float, b: Self::Float) -> Self::Mask {
            vceqq_f64(a, b)
        }

        unsafe fn gt(a: Self::Float, b: Self::Float) -> Self::Mask {
            vcgtq_f64(a, b)
        }

        unsafe fn lt(a: Self::Float, b: Self::Float) -> Self::Mask {
            vcltq_f64(a, b)
        }

        unsafe fn mul(a: Self::Float, b: Self::Float) -> Self::Float {
            vmulq_f64(a, b)
        }

        unsafe fn mul_scalar(a: Self::Float, b: f64) -> Self::Float {
            let b_simd = vdupq_n_f64(b);
            vmulq_f64(a, b_simd)
        }

        unsafe fn neg(a: Self::Float) -> Self::Float {
            vnegq_f64(a)
        }

        unsafe fn sub(a: Self::Float, b: Self::Float) -> Self::Float {
            vsubq_f64(a, b)
        }

        unsafe fn vmax(a: Self::Float, b: Self::Float) -> Self::Float {
            vmaxq_f64(a, b)
        }

        unsafe fn vmin(a: Self::Float, b: Self::Float) -> Self::Float {
            vminq_f64(a, b)
        }

        unsafe fn splat(v: f64) -> Self::Float {
            vdupq_n_f64(v)
        }

        unsafe fn reinterpret_float_signed(v: Self::Int) -> Self::Float {
            vreinterpretq_f64_s64(v)
        }

        unsafe fn to_int(v: Self::Float) -> Self::Int {
            vcvtq_s64_f64(v)
        }

        unsafe fn to_float_scalar_array(v: Self::Float) -> Self::FloatScalarArray {
            let mut a = [0f64; 2];
            vst1q_f64(a.as_mut_ptr(), v);
            a
        }

        unsafe fn with_load_store(f: &impl Fn(Self::Float) -> Self::Float, a: &mut [f64]) {
            let mut val = vld1q_f64(a.as_ptr());
            val = f(val);
            vst1q_f64(a.as_mut_ptr(), val)
        }

        unsafe fn apply_elementwise(
            f: impl Fn(Self::Float) -> Self::Float,
            f_rest: impl Fn(&mut [f64]),
            a: &mut [f64],
        ) {
            let v = Self;
            super::apply_elementwise_generic(&v, f, f_rest, a);
        }
    }
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
