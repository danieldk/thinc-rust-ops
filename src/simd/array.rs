#[cfg(target_arch = "aarch64")]
use std::arch::is_aarch64_feature_detected;
use std::ops::Neg;

use num_traits::{NumCast, One, Zero};

#[cfg(target_arch = "x86_64")]
use crate::simd::vector::avx::AVX32;
#[cfg(target_arch = "x86_64")]
use crate::simd::vector::avx::AVX64;
#[cfg(target_arch = "aarch64")]
use crate::simd::vector::neon::NeonVector32;
#[cfg(target_arch = "aarch64")]
use crate::simd::vector::neon::NeonVector64;
use crate::simd::vector::{ScalarVector32, ScalarVector64, SimdVector};

#[cfg(target_arch = "aarch64")]
pub fn platform_arrays() -> (Box<dyn Array<Scalar = f32>>, Box<dyn Array<Scalar = f64>>) {
    if is_aarch64_feature_detected!("neon") {
        (Box::new(NeonVector32), Box::new(NeonVector64))
    } else {
        (Box::new(ScalarVector32), Box::new(ScalarVector64))
    }
}

#[cfg(target_arch = "x86_64")]
pub fn platform_arrays() -> (Box<dyn Array<Scalar = f32>>, Box<dyn Array<Scalar = f64>>) {
    if is_x86_feature_detected!("avx") {
        (Box::new(AVX32), Box::new(AVX64))
    } else {
        (Box::new(ScalarVector32), Box::new(ScalarVector64))
    }
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
pub fn platform_arrays() -> (Box<dyn Array<Scalar = f32>>, Box<dyn Array<Scalar = f64>>) {
    (Box::new(ScalarVector32), Box::new(ScalarVector64))
}

pub trait Array: Send + Sync {
    type Scalar;

    fn clipped_linear(
        &self,
        a: &mut [Self::Scalar],
        slope: Self::Scalar,
        offset: Self::Scalar,
        min_val: Self::Scalar,
        max_val: Self::Scalar,
    );

    fn hard_sigmoid(&self, a: &mut [Self::Scalar]);

    fn hard_tanh(&self, a: &mut [Self::Scalar]);

    fn relu(&self, a: &mut [Self::Scalar]);
}

impl<V> Array for V
where
    V: SimdVector,
{
    type Scalar = V::FloatScalar;

    fn clipped_linear(
        &self,
        a: &mut [Self::Scalar],
        slope: Self::Scalar,
        offset: Self::Scalar,
        min_val: Self::Scalar,
        max_val: Self::Scalar,
    ) {
        let lower = V::Lower::default();
        unsafe {
            let v_min_val = V::splat(min_val);
            let v_max_val = V::splat(max_val);

            V::apply_elementwise(
                |v| {
                    let v = V::add_scalar(V::mul_scalar(v, slope), offset);
                    V::vmin(V::vmax(v, v_min_val), v_max_val)
                },
                |a| lower.clipped_linear(a, slope, offset, min_val, max_val),
                a,
            )
        }
    }

    fn hard_sigmoid(&self, a: &mut [Self::Scalar]) {
        self.clipped_linear(
            a,
            <Self::Scalar as NumCast>::from(0.2).unwrap(),
            <Self::Scalar as NumCast>::from(0.5).unwrap(),
            Self::Scalar::zero(),
            Self::Scalar::one(),
        )
    }

    fn hard_tanh(&self, a: &mut [Self::Scalar]) {
        self.clipped_linear(
            a,
            <Self::Scalar as NumCast>::from(1.).unwrap(),
            <Self::Scalar as NumCast>::from(0.).unwrap(),
            Self::Scalar::one().neg(),
            Self::Scalar::one(),
        )
    }

    fn relu(&self, a: &mut [Self::Scalar]) {
        let smaller = V::Lower::default();
        unsafe {
            let zero = V::splat(Self::Scalar::zero());
            V::apply_elementwise(|v| V::vmax(v, zero), |a| smaller.relu(a), a);
        }
    }
}
