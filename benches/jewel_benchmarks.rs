/*use criterion::{criterion_group, criterion_main, Criterion, black_box};
use rand::prelude::*;
use triple_accel::jewel::*;

fn bench_jewel_fn(c: &mut Criterion) {
    let mut group = c.benchmark_group("bench_jewel_fn");
    let length = 30;

    group.bench_function("create_adds_1", |b| b.iter(|| {
        unsafe {
            let mut a = AvxNx32x8::repeating_max(black_box(length));
            let b = AvxNx32x8::repeating_max(black_box(length));
            a.adds(&b);
            a
        }
    }));

    group.bench_function("create_adds_10", |b| b.iter(|| {
        unsafe {
            let mut a = AvxNx32x8::repeating_max(black_box(length));
            let b = AvxNx32x8::repeating_max(black_box(length));
            a.adds(&b);
            a.adds(&b);
            a.adds(&b);
            a.adds(&b);
            a.adds(&b);
            a.adds(&b);
            a.adds(&b);
            a.adds(&b);
            a.adds(&b);
            a.adds(&b);
            a
        }
    }));

    group.bench_function("create_shift_insert", |b| b.iter(|| {
        unsafe {
            let mut a = AvxNx32x8::repeating_max(black_box(length));
            a.shift_left_1();
            a.insert_last_max();
            a.clone()
        }
    }));


    group.bench_function("create_vector", |b| b.iter(|| {
        let mut a = vec![0u8; length];
        a[0] = 255u8;
        a.clone()
    }));

    group.finish();
}

criterion_group!(bench_jewel, bench_jewel_fn);
criterion_main!(bench_jewel);*/
