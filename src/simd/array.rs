#[cfg(target_arch = "aarch64")]
use std::arch::is_aarch64_feature_detected;

use crate::simd::activation::Activation;
use num_traits::Float;

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

    fn logistic_function(&self, a: &mut [Self::Scalar]);

    fn relu(&self, a: &mut [Self::Scalar]);

    fn swish(&self, a: &mut [Self::Scalar]);
}

impl<V, T, U> Array for V
where
    T: Copy,
    U: Float,
    V: Activation<Float = T, FloatScalar = U> + SimdVector<Float = T, FloatScalar = U>,
{
    type Scalar = U;

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
            V::apply_elementwise(
                |v| V::clipped_linear(v, slope, offset, min_val, max_val),
                |a| lower.clipped_linear(a, slope, offset, min_val, max_val),
                a,
            )
        }
    }

    fn hard_sigmoid(&self, a: &mut [Self::Scalar]) {
        let lower = V::Lower::default();
        unsafe { V::apply_elementwise(|v| V::hard_sigmoid(v), |a| lower.hard_sigmoid(a), a) }
    }

    fn hard_tanh(&self, a: &mut [Self::Scalar]) {
        let lower = V::Lower::default();
        unsafe { V::apply_elementwise(|v| V::hard_tanh(v), |a| lower.hard_tanh(a), a) }
    }

    fn logistic_function(&self, a: &mut [Self::Scalar]) {
        let smaller = V::Lower::default();
        unsafe {
            V::apply_elementwise(
                |v| V::logistic_function(v),
                |a| smaller.logistic_function(a),
                a,
            );
        }
    }

    fn relu(&self, a: &mut [Self::Scalar]) {
        let smaller = V::Lower::default();
        unsafe {
            V::apply_elementwise(|v| V::relu(v), |a| smaller.relu(a), a);
        }
    }

    fn swish(&self, a: &mut [Self::Scalar]) {
        let smaller = V::Lower::default();
        unsafe {
            V::apply_elementwise(|v| V::swish(v), |a| smaller.swish(a), a);
        }
    }
}
