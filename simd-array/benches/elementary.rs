use std::mem;

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use ndarray::{Array, Array1};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use simd_array::PlatformSimdSlice;

fn exp_benchmark(c: &mut Criterion) {
    let source: Array1<f32> = Array::random((2949120,), Uniform::new(0., 10.));
    let mut group = c.benchmark_group("exp");
    group.throughput(Throughput::Bytes(
        (source.len() * mem::size_of::<f32>()) as u64,
    ));
    for (name, array) in f32::all_simd_slice() {
        let mut test_array = source.clone();
        group.bench_function(&format!("exp {}", name), |b| {
            b.iter(|| array.exp(test_array.as_slice_mut().unwrap()))
        });
    }
    let mut t2 = source.clone();
    group.bench_function("std exp", |b| {
        b.iter(|| {
            black_box(
                t2.as_slice_mut()
                    .unwrap()
                    .iter_mut()
                    .for_each(|v| *v = v.exp()),
            )
        })
    });
}

criterion_group!(elementary, exp_benchmark);
criterion_main!(elementary);
