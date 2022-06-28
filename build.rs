fn main() {
    check_avx();
}

#[cfg(target_arch = "x86_64")]
fn check_avx() {
    if is_x86_feature_detected!("avx") {
        println!("cargo:rustc-cfg=feature=\"test_avx\"");
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn check_avx() {}
