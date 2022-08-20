#[cfg(target_arch = "aarch64")]
use std::arch::is_aarch64_feature_detected;
use std::collections::HashMap;

use num_traits::Float;

use crate::activation::Activation;
use crate::distribution::Distribution;
#[cfg(target_arch = "x86_64")]
use crate::vector::avx::AVXVector32;
#[cfg(target_arch = "x86_64")]
use crate::vector::avx::AVXVector64;
#[cfg(target_arch = "x86_64")]
use crate::vector::avx2::{AVX2Vector32, AVX2Vector64};
#[cfg(target_arch = "aarch64")]
use crate::vector::neon::NeonVector32;
#[cfg(target_arch = "aarch64")]
use crate::vector::neon::NeonVector64;
use crate::vector::scalar::{ScalarVector32, ScalarVector64};
#[cfg(target_arch = "x86_64")]
use crate::vector::sse2::{SSE2Vector32, SSE2Vector64};
#[cfg(target_arch = "x86_64")]
use crate::vector::sse41::{SSE41Vector32, SSE41Vector64};
use crate::vector::SimdVector;

#[cfg(target_arch = "aarch64")]
pub fn all_platform_arrays(
) -> HashMap<String, (Box<dyn Array<Scalar = f32>>, Box<dyn Array<Scalar = f64>>)> {
    let mut arrays = HashMap::new();
    arrays.insert(
        "scalar".to_string(),
        (
            Box::new(ScalarVector32) as Box<dyn Array<Scalar = f32>>,
            Box::new(ScalarVector64) as Box<dyn Array<Scalar = f64>>,
        ),
    );

    if is_aarch64_feature_detected!("neon") {
        arrays.insert(
            "neon".to_string(),
            (Box::new(NeonVector32), Box::new(NeonVector64)),
        );
    }

    arrays
}

#[cfg(target_arch = "x86_64")]
pub fn all_platform_arrays(
) -> HashMap<String, (Box<dyn Array<Scalar = f32>>, Box<dyn Array<Scalar = f64>>)> {
    let mut arrays = HashMap::new();
    arrays.insert(
        "scalar".to_string(),
        (
            Box::new(ScalarVector32) as Box<dyn Array<Scalar = f32>>,
            Box::new(ScalarVector64) as Box<dyn Array<Scalar = f64>>,
        ),
    );

    if is_x86_feature_detected!("sse2") {
        arrays.insert(
            "sse2".to_string(),
            (Box::new(SSE2Vector32), Box::new(SSE2Vector64)),
        );
    }
    if is_x86_feature_detected!("sse4.1") {
        arrays.insert(
            "sse4.1".to_string(),
            (Box::new(SSE41Vector32), Box::new(SSE41Vector64)),
        );
    }
    if is_x86_feature_detected!("avx") {
        arrays.insert(
            "avx2".to_string(),
            (Box::new(AVXVector32), Box::new(AVXVector64)),
        );
    }
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        arrays.insert(
            "avx2".to_string(),
            (Box::new(AVX2Vector32), Box::new(AVX2Vector64)),
        );
    }

    arrays
}

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
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        (Box::new(AVX2Vector32), Box::new(AVX2Vector64))
    } else if is_x86_feature_detected!("avx") {
        (Box::new(AVXVector32), Box::new(AVXVector64))
    } else if is_x86_feature_detected!("sse4.1") {
        (Box::new(SSE41Vector32), Box::new(SSE41Vector64))
    } else if is_x86_feature_detected!("sse2") {
        (Box::new(SSE2Vector32), Box::new(SSE2Vector64))
    } else {
        (Box::new(ScalarVector32), Box::new(ScalarVector64))
    }
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
pub fn platform_arrays() -> (Box<dyn Array<Scalar = f32>>, Box<dyn Array<Scalar = f64>>) {
    (Box::new(ScalarVector32), Box::new(ScalarVector64))
}

macro_rules! unary_activation {
    ($j:ident) => {
        fn $j(&self, a: &mut [Self::Scalar]) {
            let lower = V::Lower::default();
            unsafe { V::apply_elementwise(|v| V::$j(v), |a| lower.$j(a), a) }
        }
    };
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

    fn gelu(&self, a: &mut [Self::Scalar]);

    fn hard_sigmoid(&self, a: &mut [Self::Scalar]);

    fn hard_tanh(&self, a: &mut [Self::Scalar]);

    fn logistic_cdf(&self, a: &mut [Self::Scalar]);

    fn relu(&self, a: &mut [Self::Scalar]);

    fn sum(&self, a: &[Self::Scalar]) -> Self::Scalar;

    fn swish(&self, a: &mut [Self::Scalar]);
}

impl<V, T, U> Array for V
where
    T: Copy,
    U: Float,
    V: Activation<Float = T, FloatScalar = U>
        + Distribution<Float = T>
        + SimdVector<Float = T, FloatScalar = U>,
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

    fn sum(&self, a: &[Self::Scalar]) -> Self::Scalar {
        let lower = V::Lower::default();
        unsafe {
            V::reduce(
                |acc, v| V::add(acc, v),
                |v| V::add_lanes(v),
                |init, a| init + lower.sum(a),
                Self::Scalar::from(0.0).unwrap(),
                a,
            )
        }
    }

    unary_activation!(gelu);
    unary_activation!(hard_sigmoid);
    unary_activation!(hard_tanh);
    unary_activation!(logistic_cdf);
    unary_activation!(relu);
    unary_activation!(swish);
}

#[cfg(test)]
mod tests {
    use std::fmt;

    use num_traits::Float;

    use super::{all_platform_arrays, Array};

    fn test_sum_special_values<S>(array: &dyn Array<Scalar = S>)
    where
        S: Float,
    {
        for i in 0..17 {
            let mut special = [S::zero(); 17];
            special[i] = S::nan();
            assert!(array.sum(&special).is_nan());
            special[i] = S::infinity();
            assert!(array.sum(&special).is_infinite());
            assert!(array.sum(&special).is_sign_positive());
            special[i] = -S::infinity();
            assert!(array.sum(&special).is_infinite());
            assert!(array.sum(&special).is_sign_negative());
        }
    }

    fn test_sum_triangular<S>(array: &dyn Array<Scalar = S>)
    where
        S: fmt::Debug + Float,
    {
        for i in 1..=128 {
            let check = S::from((i * (i + 1)) / 2).unwrap();
            let a = (1..=i).map(|v| S::from(v).unwrap()).collect::<Vec<_>>();
            let r = array.sum(&a);
            assert_eq!(r, check);
        }
    }

    #[test]
    fn test_sum_f32() {
        for (array_f32, _) in all_platform_arrays().values() {
            test_sum_triangular(array_f32.as_ref());
            test_sum_special_values(array_f32.as_ref());
        }
    }

    #[test]
    fn test_sum_f64() {
        for (_, array_f64) in all_platform_arrays().values() {
            test_sum_triangular(array_f64.as_ref());
            test_sum_special_values(array_f64.as_ref());
        }
    }
}
