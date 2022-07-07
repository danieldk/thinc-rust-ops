use num_traits::{Float, FloatConst, NumCast, One, Zero};

use crate::vector::{FloatingPointProps, SimdVector};

mod fasterf_poly_coeff {
    // Constants and approximation from Abramowitz and Stegun, 1964.
    pub const COEFF_A1: f64 = 0.254829592;
    pub const COEFF_A2: f64 = -0.284496736;
    pub const COEFF_A3: f64 = 1.421413741;
    pub const COEFF_A4: f64 = -1.453152027;
    pub const COEFF_A5: f64 = 1.061405429;
    pub const COEFF_P: f64 = 0.3275911;
}

mod fastexp_poly_coeff {
    // Constants from Malossi et al., 2015
    pub const POLY_COEFF_5_0: f64 = 1.068_237_537_102_394_8e-7;
    pub const POLY_COEFF_5_1: f64 = 3.068_452_496_566_328_5e-1;
    pub const POLY_COEFF_5_2: f64 = -2.401_397_219_822_308e-1;
    pub const POLY_COEFF_5_3: f64 = -5.586_622_824_128_225e-2;
    pub const POLY_COEFF_5_4: f64 = -8.942_838_909_312_74e-3;
    pub const POLY_COEFF_5_5: f64 = -1.896_460_523_807_077_3e-3;
}

pub trait Elementary {
    type Float;

    unsafe fn erf(x: Self::Float) -> Self::Float;

    unsafe fn exp(x: Self::Float) -> Self::Float;
}

impl<V> Elementary for V
where
    V: SimdVector,
{
    type Float = V::Float;

    unsafe fn erf(x: Self::Float) -> Self::Float {
        let one = V::splat(V::FloatScalar::one());
        let coeff_p = V::from_f64(fasterf_poly_coeff::COEFF_P);
        let coeff_a1 = V::from_f64(fasterf_poly_coeff::COEFF_A1);
        let coeff_a2 = V::from_f64(fasterf_poly_coeff::COEFF_A2);
        let coeff_a3 = V::from_f64(fasterf_poly_coeff::COEFF_A3);
        let coeff_a4 = V::from_f64(fasterf_poly_coeff::COEFF_A4);
        let coeff_a5 = V::from_f64(fasterf_poly_coeff::COEFF_A5);

        let x_abs = V::abs(x);
        let neg_x_sq = V::neg(V::mul(x, x));
        let t = V::div(one, V::add(V::mul(x_abs, coeff_p), one));

        let mut tp = V::mul(t, coeff_a5);
        tp = V::mul(t, V::add(tp, coeff_a4));
        tp = V::mul(t, V::add(tp, coeff_a3));
        tp = V::mul(t, V::add(tp, coeff_a2));
        tp = V::mul(t, V::add(tp, coeff_a1));

        let erf_abs = V::sub(one, V::mul(tp, Self::exp(neg_x_sq)));

        V::copy_sign(x, erf_abs)
    }

    unsafe fn exp(mut x: Self::Float) -> Self::Float {
        let inf = V::splat(V::FloatScalar::infinity());
        let zero = V::splat(V::FloatScalar::zero());

        let coeff_a = <V::FloatScalar as NumCast>::from(
            V::IntScalar::one() << V::FloatScalar::mantissa_bits(),
        )
        .unwrap();
        let coeff_b = <V::FloatScalar as NumCast>::from(
            (V::IntScalar::one() << V::FloatScalar::mantissa_bits())
                * <V::IntScalar as NumCast>::from(V::FloatScalar::bias()).unwrap(),
        )
        .unwrap();
        let poly_coeff_5_0 = V::from_f64(fastexp_poly_coeff::POLY_COEFF_5_0);
        let poly_coeff_5_1 = V::from_f64(fastexp_poly_coeff::POLY_COEFF_5_1);
        let poly_coeff_5_2 = V::from_f64(fastexp_poly_coeff::POLY_COEFF_5_2);
        let poly_coeff_5_3 = V::from_f64(fastexp_poly_coeff::POLY_COEFF_5_3);
        let poly_coeff_5_4 = V::from_f64(fastexp_poly_coeff::POLY_COEFF_5_4);
        let poly_coeff_5_5 = V::from_f64(fastexp_poly_coeff::POLY_COEFF_5_5);

        // Maximum positive value.
        let max_mask = V::gt(x, V::splat(V::FloatScalar::max_value().ln()));

        // Smallest positive normalized value.
        let smallest_positive_mask = V::lt(x, V::splat(V::FloatScalar::min_positive_value().ln()));

        // Elements that are not NaN.
        let not_nan = V::eq(x, x);

        x = V::mul_scalar(x, V::FloatScalar::LOG2_E());
        let xf = V::sub(x, V::floor(x));

        let mut factor = poly_coeff_5_5;
        factor = V::add(V::mul(factor, xf), poly_coeff_5_4);
        factor = V::add(V::mul(factor, xf), poly_coeff_5_3);
        factor = V::add(V::mul(factor, xf), poly_coeff_5_2);
        factor = V::add(V::mul(factor, xf), poly_coeff_5_1);
        factor = V::add(V::mul(factor, xf), poly_coeff_5_0);

        x = V::sub(x, factor);

        let cast = V::to_int(V::add_scalar(V::mul_scalar(x, coeff_a), coeff_b));

        x = V::reinterpret_float_signed(cast);

        x = V::bitwise_select(max_mask, inf, x);
        x = V::bitwise_select(smallest_positive_mask, zero, x);
        V::bitwise_select(not_nan, x, V::splat(V::FloatScalar::nan()))
    }
}

#[cfg(test)]
mod tests {
    use std::mem;

    use approx::relative_eq;
    use as_slice::AsSlice;
    use libm::erf;
    use num_traits::{Float, NumCast, ToPrimitive};
    use quickcheck::quickcheck;

    use super::Elementary;
    #[cfg(feature = "test_avx")]
    use crate::vector::avx::{AVXVector32, AVXVector64};
    #[cfg(target_arch = "aarch64")]
    use crate::vector::neon::{NeonVector32, NeonVector64};
    use crate::vector::{ScalarVector32, ScalarVector64, SimdVector};

    fn erf_close_to_libm_erf<S>(v: S::FloatScalar) -> bool
    where
        S: SimdVector,
    {
        let check_erf =
            <S::FloatScalar as NumCast>::from(erf(<f64 as NumCast>::from(v).unwrap())).unwrap();
        let r = {
            unsafe {
                S::to_float_scalar_array(S::erf(S::splat(v))).as_slice()[0]
                    .to_f64()
                    .unwrap()
            }
        };
        assert_eq!(r.is_nan(), check_erf.is_nan());
        if v.is_nan() {
            return true;
        }

        let max_relative = if mem::size_of::<S::FloatScalar>() == 4 {
            // 1e-5 fails sometimes in many repetitions of the test, e.g.: 51.304375
            1e-6
        } else {
            1e-7
        };

        relative_eq!(
            r,
            check_erf.to_f64().unwrap(),
            max_relative = max_relative,
            epsilon = 1e-7
        )
    }

    fn exp_close_to_std_exp<S>(v: S::FloatScalar) -> bool
    where
        S: SimdVector,
    {
        let check_exp = v.exp();
        let r = {
            unsafe {
                S::to_float_scalar_array(S::exp(S::splat(v))).as_slice()[0]
                    .to_f64()
                    .unwrap()
            }
        };
        assert_eq!(r.is_nan(), check_exp.is_nan());
        if v.is_nan() {
            return true;
        }

        let max_relative = if mem::size_of::<S::FloatScalar>() == 4 {
            // 1e-5 fails sometimes in many repetitions of the test, e.g.: 51.304375
            1e-4
        } else {
            1e-6
        };

        relative_eq!(r, check_exp.to_f64().unwrap(), max_relative = max_relative)
    }

    quickcheck! {
        #[cfg(feature = "test_avx")]
        fn avx_erf_close_to_libm_erf_f32(v: f32) -> bool {
            erf_close_to_libm_erf::<AVXVector32>(v)
        }

        #[cfg(feature = "test_avx")]
        fn avx_erf_close_to_libm_erf_f64(v: f64) -> bool {
            erf_close_to_libm_erf::<AVXVector64>(v)
        }

        #[cfg(target_arch = "aarch64")]
        fn neon_erf_close_to_libm_erf_f32(v: f32) -> bool {
            erf_close_to_libm_erf::<NeonVector32>(v)
        }

        #[cfg(target_arch = "aarch64")]
        fn neon_erf_close_to_libm_erf_f64(v: f64) -> bool {
            erf_close_to_libm_erf::<NeonVector64>(v)
        }

        fn scalar_erf_close_to_libm_erf_f32(v: f32) -> bool {
            erf_close_to_libm_erf::<ScalarVector32>(v)
        }

        fn scalar_erf_close_to_libm_erf_f64(v: f64) -> bool {
            erf_close_to_libm_erf::<ScalarVector64>(v)
        }

        fn scalar_exp_close_to_std_exp_f32(v: f32) -> bool {
            exp_close_to_std_exp::<ScalarVector32>(v)
        }

        fn scalar_exp_close_to_std_exp_f64(v: f64) -> bool {
            exp_close_to_std_exp::<ScalarVector64>(v)
        }

        #[cfg(feature = "test_avx")]
        fn avx_exp_close_to_std_exp_f32(v: f32) -> bool {
            exp_close_to_std_exp::<AVXVector32>(v)
        }

        #[cfg(feature = "test_avx")]
        fn avx_exp_close_to_std_exp_f64(v: f64) -> bool {
            exp_close_to_std_exp::<AVXVector64>(v)
        }

        #[cfg(target_arch = "aarch64")]
        fn neon_exp_close_to_std_exp_f32(v: f32) -> bool {
            exp_close_to_std_exp::<NeonVector32>(v)
        }

        #[cfg(target_arch = "aarch64")]
        fn neon_exp_close_to_std_exp_f64(v: f64) -> bool {
            exp_close_to_std_exp::<NeonVector64>(v)
        }
    }
}
