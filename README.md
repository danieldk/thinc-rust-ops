# 🧪 thinc-rust-ops

This is an experimental, non-official, Python package that implements some Thinc
[ops](https://thinc.ai/docs/api-backends) using vectorized SIMD functions in Rust 🦀.
The following SIMD instruction sets are currently supported:

* Scalar
* ARM64:
  - NEON
* x86_64:
  - SSE2
  - SSE4.1
  - AVX
  - AVX2 + FMA

## ⏳ Install

Make sure that you have a [Rust toolchain](https://rustup.rs) installed and then
install with `pip`:

```
python -m pip install git+https://github.com/danieldk/thinc-rust-ops#subdirectory=thinc-rust-ops
```

## 🚀 Quickstart

After installation, you can import `RustOps` and use it like any other `Ops` implementation:

```
>>> import numpy as np
>>> from rust_ops import RustOps
>>> ops = RustOps()
>>> ops.gelu(np.arange(-5., 5.))
array([-1.43552511e-06, -1.26744153e-04, -4.04990215e-03, -4.55001283e-02,
       -1.58655251e-01,  0.00000000e+00,  8.41344749e-01,  1.95449987e+00,
        2.99595010e+00,  3.99987326e+00])
```
