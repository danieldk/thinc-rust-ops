use num_traits::{Float, FloatConst, NumCast, One, Zero};

use crate::simd::vector::{FloatingPointProps, SimdVector};

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
    unsafe fn exp(x: Self::Float) -> Self::Float;
}

impl<V> Elementary for V
where
    V: SimdVector,
{
    type Float = V::Float;

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
        let poly_coeff_5_0 =
            <V::FloatScalar as NumCast>::from(fastexp_poly_coeff::POLY_COEFF_5_0).unwrap();
        let poly_coeff_5_1 =
            <V::FloatScalar as NumCast>::from(fastexp_poly_coeff::POLY_COEFF_5_1).unwrap();
        let poly_coeff_5_2 =
            <V::FloatScalar as NumCast>::from(fastexp_poly_coeff::POLY_COEFF_5_2).unwrap();
        let poly_coeff_5_3 =
            <V::FloatScalar as NumCast>::from(fastexp_poly_coeff::POLY_COEFF_5_3).unwrap();
        let poly_coeff_5_4 =
            <V::FloatScalar as NumCast>::from(fastexp_poly_coeff::POLY_COEFF_5_4).unwrap();
        let poly_coeff_5_5 =
            <V::FloatScalar as NumCast>::from(fastexp_poly_coeff::POLY_COEFF_5_5).unwrap();

        // Maximum positive value.
        let max_mask = V::gt(x, V::splat(V::FloatScalar::max_value().ln()));

        // Smallest positive normalized value.
        let smallest_positive_mask = V::lt(x, V::splat(V::FloatScalar::min_positive_value().ln()));

        // Elements that are not NaN.
        let not_nan = V::eq(x, x);

        x = V::mul_scalar(x, V::FloatScalar::LOG2_E());
        let xf = V::sub(x, V::floor(x));

        let mut factor = V::splat(poly_coeff_5_5);
        factor = V::add_scalar(V::mul(factor, xf), poly_coeff_5_4);
        factor = V::add_scalar(V::mul(factor, xf), poly_coeff_5_3);
        factor = V::add_scalar(V::mul(factor, xf), poly_coeff_5_2);
        factor = V::add_scalar(V::mul(factor, xf), poly_coeff_5_1);
        factor = V::add_scalar(V::mul(factor, xf), poly_coeff_5_0);

        x = V::sub(x, factor);

        //let cast = V::to_int(V::add_scalar(V::mul_scalar(x, coeff_a), coeff_b));
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
    use num_traits::{Float, ToPrimitive};
    use quickcheck::quickcheck;

    use super::Elementary;
    #[cfg(feature = "test_avx")]
    use crate::simd::vector::avx::{AVXVector32, AVXVector64};
    #[cfg(target_arch = "aarch64")]
    use crate::simd::vector::neon::{NeonVector32, NeonVector64};
    use crate::simd::vector::{ScalarVector32, ScalarVector64, SimdVector};

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
