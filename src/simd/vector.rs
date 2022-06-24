use std::mem;

pub trait SimdVector<T>: Default + Send + Sync {
    type Lower: SimdVector<T>;
    type Type: Copy;

    /// Add a scalar to every vector element.
    unsafe fn add_scalar(a: Self::Type, b: T) -> Self::Type;

    /// Multiply every vector element by a scalar.
    unsafe fn mul_scalar(a: Self::Type, b: T) -> Self::Type;

    /// Vector element-wise maximum.
    unsafe fn vmax(a: Self::Type, b: Self::Type) -> Self::Type;

    /// Vector element-wise minimum.
    unsafe fn vmin(a: Self::Type, b: Self::Type) -> Self::Type;

    unsafe fn splat(v: T) -> Self::Type;

    unsafe fn with_load_store(f: &impl Fn(Self::Type) -> Self::Type, a: &mut [T]);

    unsafe fn apply_elementwise(
        f: impl Fn(Self::Type) -> Self::Type,
        f_rest: impl Fn(&mut [T]),
        a: &mut [T],
    );
}

#[derive(Default)]
pub struct ScalarVector;

impl SimdVector<f32> for ScalarVector {
    type Lower = ScalarVector;
    type Type = f32;

    unsafe fn add_scalar(a: Self::Type, b: f32) -> Self::Type {
        a + b
    }

    unsafe fn mul_scalar(a: Self::Type, b: f32) -> Self::Type {
        a * b
    }

    unsafe fn vmax(a: Self::Type, b: Self::Type) -> Self::Type {
        if a > b {
            a
        } else {
            b
        }
    }

    unsafe fn vmin(a: Self::Type, b: Self::Type) -> Self::Type {
        if a > b {
            b
        } else {
            a
        }
    }

    unsafe fn splat(v: f32) -> Self::Type {
        v
    }

    unsafe fn with_load_store(f: &impl Fn(Self::Type) -> Self::Type, a: &mut [f32]) {
        a[0] = f(a[0])
    }

    unsafe fn apply_elementwise(
        f: impl Fn(Self::Type) -> Self::Type,
        f_rest: impl Fn(&mut [f32]),
        a: &mut [f32],
    ) {
        let v = Self;
        apply_elementwise_generic(&v, f, f_rest, a);
    }
}

impl SimdVector<f64> for ScalarVector {
    type Lower = ScalarVector;
    type Type = f64;

    unsafe fn add_scalar(a: Self::Type, b: f64) -> Self::Type {
        a + b
    }

    unsafe fn mul_scalar(a: Self::Type, b: f64) -> Self::Type {
        a * b
    }

    unsafe fn vmax(a: Self::Type, b: Self::Type) -> Self::Type {
        if a > b {
            a
        } else {
            b
        }
    }
    unsafe fn vmin(a: Self::Type, b: Self::Type) -> Self::Type {
        if a > b {
            b
        } else {
            a
        }
    }

    unsafe fn splat(v: f64) -> Self::Type {
        v
    }

    unsafe fn with_load_store(f: &impl Fn(Self::Type) -> Self::Type, a: &mut [f64]) {
        a[0] = f(a[0])
    }

    unsafe fn apply_elementwise(
        f: impl Fn(Self::Type) -> Self::Type,
        f_rest: impl Fn(&mut [f64]),
        a: &mut [f64],
    ) {
        let v = Self;
        apply_elementwise_generic(&v, f, f_rest, a);
    }
}

#[cfg(all(target_arch = "x86_64"))]
pub mod avx {
    use std::arch::x86_64::{
        __m256, __m256d, _mm256_add_pd, _mm256_add_ps, _mm256_load_pd, _mm256_loadu_ps,
        _mm256_max_pd, _mm256_max_ps, _mm256_min_pd, _mm256_min_ps, _mm256_mul_pd, _mm256_mul_ps,
        _mm256_set1_pd, _mm256_set1_ps, _mm256_store_pd, _mm256_storeu_ps,
    };

    use super::{ScalarVector, SimdVector};

    #[derive(Default)]
    pub struct AVXVector;

    impl SimdVector<f32> for AVXVector {
        type Lower = ScalarVector;
        type Type = __m256;

        unsafe fn add_scalar(a: Self::Type, b: f32) -> Self::Type {
            let b_simd = _mm256_set1_ps(b);
            _mm256_add_ps(a, b_simd)
        }

        unsafe fn mul_scalar(a: Self::Type, b: f32) -> Self::Type {
            let b_simd = _mm256_set1_ps(b);
            _mm256_mul_ps(a, b_simd)
        }

        unsafe fn vmax(a: Self::Type, b: Self::Type) -> Self::Type {
            _mm256_max_ps(a, b)
        }

        unsafe fn vmin(a: Self::Type, b: Self::Type) -> Self::Type {
            _mm256_min_ps(a, b)
        }

        unsafe fn splat(v: f32) -> Self::Type {
            _mm256_set1_ps(v)
        }

        unsafe fn with_load_store(f: &impl Fn(Self::Type) -> Self::Type, a: &mut [f32]) {
            let mut val = _mm256_loadu_ps(a.as_ptr());
            val = f(val);
            _mm256_storeu_ps(a.as_mut_ptr(), val);
        }

        #[target_feature(enable = "avx")]
        unsafe fn apply_elementwise(
            f: impl Fn(Self::Type) -> Self::Type,
            f_rest: impl Fn(&mut [f32]),
            a: &mut [f32],
        ) {
            let v = Self;
            super::apply_elementwise_generic(&v, f, f_rest, a);
        }
    }

    impl SimdVector<f64> for AVXVector {
        type Lower = ScalarVector;
        type Type = __m256d;

        unsafe fn add_scalar(a: Self::Type, b: f64) -> Self::Type {
            let b_simd = _mm256_set1_pd(b);
            _mm256_add_pd(a, b_simd)
        }

        unsafe fn mul_scalar(a: Self::Type, b: f64) -> Self::Type {
            let b_simd = _mm256_set1_pd(b);
            _mm256_mul_pd(a, b_simd)
        }

        unsafe fn vmax(a: Self::Type, b: Self::Type) -> Self::Type {
            _mm256_max_pd(a, b)
        }

        unsafe fn vmin(a: Self::Type, b: Self::Type) -> Self::Type {
            _mm256_min_pd(a, b)
        }

        unsafe fn splat(v: f64) -> Self::Type {
            _mm256_set1_pd(v)
        }

        unsafe fn with_load_store(f: &impl Fn(Self::Type) -> Self::Type, a: &mut [f64]) {
            let mut val = _mm256_load_pd(a.as_ptr());
            val = f(val);
            _mm256_store_pd(a.as_mut_ptr(), val);
        }

        #[target_feature(enable = "avx")]
        unsafe fn apply_elementwise(
            f: impl Fn(Self::Type) -> Self::Type,
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
    use std::arch::aarch64::{
        float32x4_t, float64x2_t, vaddq_f32, vaddq_f64, vdupq_n_f32, vdupq_n_f64, vld1q_f32,
        vld1q_f64, vmaxq_f32, vmaxq_f64, vminq_f32, vminq_f64, vmulq_f32, vmulq_f64, vst1q_f32,
        vst1q_f64,
    };

    use super::{ScalarVector, SimdVector};

    #[derive(Default)]
    pub struct NeonVector;

    impl SimdVector<f32> for NeonVector {
        type Lower = ScalarVector;
        type Type = float32x4_t;

        unsafe fn add_scalar(a: Self::Type, b: f32) -> Self::Type {
            let b_simd = vdupq_n_f32(b);
            vaddq_f32(a, b_simd)
        }

        unsafe fn mul_scalar(a: Self::Type, b: f32) -> Self::Type {
            let b_simd = vdupq_n_f32(b);
            vmulq_f32(a, b_simd)
        }

        unsafe fn vmax(a: Self::Type, b: Self::Type) -> Self::Type {
            vmaxq_f32(a, b)
        }

        unsafe fn vmin(a: Self::Type, b: Self::Type) -> Self::Type {
            vminq_f32(a, b)
        }

        unsafe fn splat(v: f32) -> Self::Type {
            vdupq_n_f32(v)
        }

        unsafe fn with_load_store(f: &impl Fn(Self::Type) -> Self::Type, a: &mut [f32]) {
            let mut val = vld1q_f32(a.as_ptr());
            val = f(val);
            vst1q_f32(a.as_mut_ptr(), val)
        }

        unsafe fn apply_elementwise(
            f: impl Fn(Self::Type) -> Self::Type,
            f_rest: impl Fn(&mut [f32]),
            a: &mut [f32],
        ) {
            let v = Self;
            super::apply_elementwise_generic(&v, f, f_rest, a);
        }
    }

    impl SimdVector<f64> for NeonVector {
        type Lower = ScalarVector;
        type Type = float64x2_t;

        unsafe fn add_scalar(a: Self::Type, b: f64) -> Self::Type {
            let b_simd = vdupq_n_f64(b);
            vaddq_f64(a, b_simd)
        }

        unsafe fn mul_scalar(a: Self::Type, b: f64) -> Self::Type {
            let b_simd = vdupq_n_f64(b);
            vmulq_f64(a, b_simd)
        }

        unsafe fn vmax(a: Self::Type, b: Self::Type) -> Self::Type {
            vmaxq_f64(a, b)
        }

        unsafe fn vmin(a: Self::Type, b: Self::Type) -> Self::Type {
            vminq_f64(a, b)
        }

        unsafe fn splat(v: f64) -> Self::Type {
            vdupq_n_f64(v)
        }

        unsafe fn with_load_store(f: &impl Fn(Self::Type) -> Self::Type, a: &mut [f64]) {
            let mut val = vld1q_f64(a.as_ptr());
            val = f(val);
            vst1q_f64(a.as_mut_ptr(), val)
        }

        unsafe fn apply_elementwise(
            f: impl Fn(Self::Type) -> Self::Type,
            f_rest: impl Fn(&mut [f64]),
            a: &mut [f64],
        ) {
            let v = Self;
            super::apply_elementwise_generic(&v, f, f_rest, a);
        }
    }
}

// TODO: get rid of the first argument. Needed so far to help type inference.
unsafe fn apply_elementwise_generic<T, V>(
    _v: &V,
    f: impl Fn(V::Type) -> V::Type,
    f_rest: impl Fn(&mut [T]),
    mut a: &mut [T],
) where
    V: SimdVector<T>,
{
    let elem_size = mem::size_of::<V::Type>() / mem::size_of::<T>();

    while a.len() >= elem_size {
        V::with_load_store(&f, a);
        a = &mut a[elem_size..];
    }

    if a.len() > 0 {
        f_rest(a);
    }
}
