use std::mem;

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use ndarray::{Array, Array1};
use ndarray_rand::rand_distr::uniform::{SampleBorrow, SampleUniform};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use num_traits::Float;
use simd_array::{PlatformSimdSlice, SimdSlice};

fn reduction_benchmark<T, F>(c: &mut Criterion, name: &str, reduce: F)
where
    T: Float + PlatformSimdSlice + SampleBorrow<T> + SampleUniform,
    F: Fn(&dyn SimdSlice<Scalar = T>, &[T]) -> T,
{
    let test_array: Array1<T> =
        Array::random((2949120,), Uniform::new(T::zero(), T::from(10.0).unwrap()));

    let mut group = c.benchmark_group(name);
    group.throughput(Throughput::Bytes(
        (test_array.len() * mem::size_of::<T>()) as u64,
    ));

    for (name, simd_slice) in T::all_simd_slice() {
        group.bench_function(name, |b| {
            b.iter(|| black_box(reduce(&*simd_slice, test_array.as_slice().unwrap())))
        });
    }
}

fn max_benchmark(c: &mut Criterion) {
    reduction_benchmark::<f32, _>(c, "max f32", |simd_slice, slice| {
        simd_slice.max(slice).unwrap()
    });
    reduction_benchmark::<f64, _>(c, "max f64", |simd_slice, slice| {
        simd_slice.max(slice).unwrap()
    });
}

fn min_benchmark(c: &mut Criterion) {
    reduction_benchmark::<f32, _>(c, "min f32", |simd_slice, slice| {
        simd_slice.min(slice).unwrap()
    });
    reduction_benchmark::<f64, _>(c, "min f64", |simd_slice, slice| {
        simd_slice.min(slice).unwrap()
    });
}

fn sum_benchmark(c: &mut Criterion) {
    reduction_benchmark::<f32, _>(c, "sum f32", |simd_slice, slice| simd_slice.sum(slice));
    reduction_benchmark::<f64, _>(c, "sum f64", |simd_slice, slice| simd_slice.sum(slice));
}

criterion_group!(slice, max_benchmark, min_benchmark, sum_benchmark);
criterion_main!(slice);
