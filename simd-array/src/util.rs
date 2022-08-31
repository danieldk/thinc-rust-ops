use num_traits::Float;

// NaN-propagating maximum like {f32,f64}::maximum. Replace once
// these are stabilized.
pub fn maximum<F: Float>(a: F, b: F) -> F {
    if a > b {
        a
    } else if b > a {
        b
    } else if a == b {
        if a.is_sign_positive() && b.is_sign_negative() {
            a
        } else {
            b
        }
    } else {
        a + b
    }
}

// NaN-propagating minimum like {f32,f64}::minimum. Replace once
// these are stabilized.
pub fn minimum<F: Float>(a: F, b: F) -> F {
    if a < b {
        a
    } else if b < a {
        b
    } else if a == b {
        if a.is_sign_negative() && b.is_sign_positive() {
            a
        } else {
            b
        }
    } else {
        a + b
    }
}
