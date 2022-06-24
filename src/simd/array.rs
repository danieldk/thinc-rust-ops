use num_traits::Float;
#[cfg(target_arch = "aarch64")]
use std::arch::is_aarch64_feature_detected;

#[cfg(target_arch = "x86_64")]
use crate::simd::vector::avx::AVXVector;
#[cfg(target_arch = "aarch64")]
use crate::simd::vector::neon::NeonVector;
use crate::simd::vector::{ScalarVector, SimdVector};

#[cfg(target_arch = "aarch64")]
pub fn platform_arrays() -> (Box<dyn Array<f32>>, Box<dyn Array<f64>>) {
    if is_aarch64_feature_detected!("neon") {
        (Box::new(NeonVector), Box::new(NeonVector))
    } else {
        (Box::new(ScalarVector), Box::new(ScalarVector))
    }
}

#[cfg(target_arch = "x86_64")]
pub fn platform_arrays() -> (Box<dyn Array<f32>>, Box<dyn Array<f64>>) {
    if is_x86_feature_detected!("avx") {
        (Box::new(AVXVector), Box::new(AVXVector))
    } else {
        (Box::new(ScalarVector), Box::new(ScalarVector))
    }
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
pub fn platform_arrays() -> (Box<dyn Array<f32>>, Box<dyn Array<f64>>) {
    (Box::new(ScalarVector), Box::new(ScalarVector))
}

pub trait Array<T>: Send + Sync {
    fn clipped_linear(&self, a: &mut [T], slope: T, offset: T, min_val: T, max_val: T);

    fn hard_sigmoid(&self, a: &mut [T]);

    fn hard_tanh(&self, a: &mut [T]);

    fn relu(&self, a: &mut [T]);
}

impl<T, V> Array<T> for V
where
    T: Float,
    V: SimdVector<T>,
{
    fn clipped_linear(&self, a: &mut [T], slope: T, offset: T, min_val: T, max_val: T) {
        let smaller = V::Lower::default();
        unsafe {
            let v_min_val = V::splat(min_val);
            let v_max_val = V::splat(max_val);

            V::apply_elementwise(
                |v| {
                    let v = V::add_scalar(V::mul_scalar(v, slope), offset);
                    V::vmin(V::vmax(v, v_min_val), v_max_val)
                },
                |a| smaller.clipped_linear(a, slope, offset, min_val, max_val),
                a,
            )
        }
    }

    fn hard_sigmoid(&self, a: &mut [T]) {
        self.clipped_linear(
            a,
            T::from(0.2).unwrap(),
            T::from(0.5).unwrap(),
            T::zero(),
            T::one(),
        )
    }

    fn hard_tanh(&self, a: &mut [T]) {
        self.clipped_linear(
            a,
            T::from(1.).unwrap(),
            T::from(0.).unwrap(),
            T::one().neg(),
            T::one(),
        )
    }

    fn relu(&self, a: &mut [T]) {
        let smaller = V::Lower::default();
        unsafe {
            let zero = V::splat(T::zero());
            V::apply_elementwise(|v| V::vmax(v, zero), |a| smaller.relu(a), a);
        }
    }
}
