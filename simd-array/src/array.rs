#[cfg(target_arch = "aarch64")]
use std::arch::is_aarch64_feature_detected;
use std::collections::HashMap;

use crate::activation::Activation;
use crate::distribution::Distribution;
use crate::elementary::Elementary;
use crate::util::maximum;
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
    ($t:ty, $j:ident) => {
        unsafe fn $j(&self, a: &mut [Self::Scalar]) {
            let lower = <Self as SimdVector>::Lower::default();
            unsafe { <Self as SimdVector>::apply_elementwise(|v| <Self as $t>::$j(v), |a| lower.$j(a), a) }
        }
    };
}

pub trait Array: Send + Sync {
    type Scalar;

    unsafe fn clipped_linear(
        &self,
        a: &mut [Self::Scalar],
        slope: Self::Scalar,
        offset: Self::Scalar,
        min_val: Self::Scalar,
        max_val: Self::Scalar,
    );

    unsafe fn div(&self, a: &mut [Self::Scalar], b: Self::Scalar);

    unsafe fn exp(&self, a: &mut [Self::Scalar]);

    unsafe fn gelu(&self, a: &mut [Self::Scalar]);

    unsafe fn hard_sigmoid(&self, a: &mut [Self::Scalar]);

    unsafe fn hard_tanh(&self, a: &mut [Self::Scalar]);

    unsafe fn logistic_cdf(&self, a: &mut [Self::Scalar]);

    unsafe fn max(&self, a: &[Self::Scalar]) -> Option<Self::Scalar>;

    unsafe fn relu(&self, a: &mut [Self::Scalar]);

    unsafe fn softmax(&self, a: &mut [Self::Scalar], n_class: usize, temperature: Option<Self::Scalar>);

    unsafe fn sub(&self, a: &mut [Self::Scalar], b: Self::Scalar);

    unsafe fn sum(&self, a: &[Self::Scalar]) -> Self::Scalar;

    unsafe fn swish(&self, a: &mut [Self::Scalar]);
}

macro_rules! array_impl {
    ($vector:ty, $target_feat:literal) => {
        impl Array for $vector {
            type Scalar = <$vector as SimdVector>::FloatScalar;

            #[target_feature(enable = $target_feat)]
            unsafe fn clipped_linear(
                &self,
                a: &mut [Self::Scalar],
                slope: Self::Scalar,
                offset: Self::Scalar,
                min_val: Self::Scalar,
                max_val: Self::Scalar,
            ) {
                let lower = <Self as SimdVector>::Lower::default();
                unsafe {
                    <Self as SimdVector>::apply_elementwise(
                        |v| <Self as Activation>::clipped_linear(v, slope, offset, min_val, max_val),
                        |a| lower.clipped_linear(a, slope, offset, min_val, max_val),
                        a,
                    )
                }
            }

            #[target_feature(enable = $target_feat)]
            unsafe fn div(&self, a: &mut [Self::Scalar], b: Self::Scalar) {
                let lower = <Self as SimdVector>::Lower::default();
                unsafe { <Self as SimdVector>::apply_elementwise(|v| <Self as SimdVector>::div_scalar(v, b), |a| lower.div(a, b), a) };
            }

            #[target_feature(enable = $target_feat)]
            unsafe fn max(&self, a: &[Self::Scalar]) -> Option<Self::Scalar> {
                if a.is_empty() {
                    return None;
                }

                let lower = <Self as SimdVector>::Lower::default();
                Some(unsafe {
                    <Self as SimdVector>::reduce(
                        |acc, v| <Self as SimdVector>::max(acc, v),
                        |v| <Self as SimdVector>::max_lanes(v),
                        |init, a| maximum(init, lower.max(a).unwrap()),
                        a[0],
                        a,
                    )
                })
            }

            #[target_feature(enable = $target_feat)]
            unsafe fn softmax(&self, a: &mut [Self::Scalar], n_class: usize, temperature: Option<Self::Scalar>) {
                assert!(n_class > 0);

                if let Some(temperature) = temperature {
                    self.div(a, temperature);
                }

                // Subtract maximum from each class to improve numeric stability.
                let mut tmp = &mut *a;
                while !tmp.is_empty() {
                    let max = self.max(&tmp[..n_class]).expect("Cannot get maximum, zero classes?");
                    self.sub(&mut tmp[..n_class], max);
                    tmp = &mut tmp[n_class..];
                }

                // Exponentiate.
                tmp = a;
                self.exp(tmp);

                // Normalize
                while !tmp.is_empty() {
                    let sum = self.sum(&tmp[..n_class]);
                    self.div(&mut tmp[..n_class], sum);
                    tmp = &mut tmp[n_class..];
                }
            }

            #[target_feature(enable = $target_feat)]
            unsafe fn sub(&self, a: &mut [Self::Scalar], b: Self::Scalar) {
                let lower = <Self as SimdVector>::Lower::default();
                unsafe { <Self as SimdVector>::apply_elementwise(|v| <Self as SimdVector>::sub_scalar(v, b), |a| lower.sub(a, b), a) };
            }

            #[target_feature(enable = $target_feat)]
            unsafe fn sum(&self, a: &[Self::Scalar]) -> Self::Scalar {
                let lower = <Self as SimdVector>::Lower::default();
                unsafe {
                    <Self as SimdVector>::reduce(
                        |acc, v| <Self as SimdVector>::add(acc, v),
                        |v| <Self as SimdVector>::add_lanes(v),
                        |init, a| init + lower.sum(a),
                        Self::Scalar::from(0.0),
                        a,
                    )
                }
            }

            unary_activation!(Elementary, exp);
            unary_activation!(Activation, gelu);
            unary_activation!(Activation, hard_sigmoid);
            unary_activation!(Activation, hard_tanh);
            unary_activation!(Distribution, logistic_cdf);
            unary_activation!(Activation, relu);
            unary_activation!(Activation, swish);
        }
    };
}


#[cfg(target_arch = "aarch64")]
array_impl!(NeonVector32, "neon");
#[cfg(target_arch = "aarch64")]
array_impl!(NeonVector64, "neon");
#[cfg(target_arch = "aarch64")]
array_impl!(ScalarVector32, "neon");
#[cfg(target_arch = "aarch64")]
array_impl!(ScalarVector64, "neon");

#[cfg(target_arch = "x86_64")]
array_impl!(ScalarVector32, "sse2");
#[cfg(target_arch = "x86_64")]
array_impl!(ScalarVector64, "sse2");
#[cfg(target_arch = "x86_64")]
array_impl!(SSE2Vector32, "sse2");
#[cfg(target_arch = "x86_64")]
array_impl!(SSE2Vector64, "sse2");
#[cfg(target_arch = "x86_64")]
array_impl!(SSE41Vector32, "sse4.1");
#[cfg(target_arch = "x86_64")]
array_impl!(SSE41Vector64, "sse4.1");
#[cfg(target_arch = "x86_64")]
array_impl!(AVXVector32, "avx");
#[cfg(target_arch = "x86_64")]
array_impl!(AVXVector64, "avx");
#[cfg(target_arch = "x86_64")]
array_impl!(AVX2Vector32, "avx2");
#[cfg(target_arch = "x86_64")]
array_impl!(AVX2Vector64, "avx2");



#[cfg(test)]
mod tests {
    use std::fmt;

    use num_traits::Float;
    use ordered_float::OrderedFloat;
    use quickcheck_macros::quickcheck;

    use super::{all_platform_arrays, Array};

    fn test_max<S: Float>(arrays: &[Box<dyn Array<Scalar = S>>], a: &[S]) -> bool {
        for array in arrays {
            let check = a.iter().max_by_key(|&&v| OrderedFloat(v)).cloned();
            let r = array.max(&a);

            if r.map(OrderedFloat) != check.map(OrderedFloat) {
                return false;
            }
        }

        true
    }

    #[quickcheck]
    fn test_max_f32(a: Vec<f32>) -> bool {
        let arrays_f32 = all_platform_arrays()
            .into_values()
            .map(|a| a.0)
            .collect::<Vec<_>>();
        test_max(&arrays_f32, &a)
    }

    #[quickcheck]
    fn test_max_f64(a: Vec<f64>) -> bool {
        let arrays_f64 = all_platform_arrays()
            .into_values()
            .map(|a| a.1)
            .collect::<Vec<_>>();
        test_max(&arrays_f64, &a)
    }

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
