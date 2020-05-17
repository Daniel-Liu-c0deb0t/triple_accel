use criterion::{criterion_group, criterion_main, Criterion, black_box};
use triple_accel::jewel::*;

#[cfg(target_arch = "x86")]
use std::arch::x86::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

fn bench_jewel_fn(c: &mut Criterion) {
    let mut group = c.benchmark_group("bench_jewel_fn");
    let length = 100;

    let mut a = unsafe {Avx1x32x8::repeating_max(black_box(length))};
    let b = unsafe {Avx1x32x8::repeating_max(black_box(length))};

    group.bench_function("avx_jewel_create_adds_1", |bencher| bencher.iter(|| {
        unsafe {
            a.adds_mut(&b);
        }
    }));

    black_box(a);
    black_box(b);

    let mut a = unsafe {Avx1x32x8::repeating_max(black_box(length))};
    let b = unsafe {Avx1x32x8::repeating_max(black_box(length))};

    group.bench_function("avx_jewel_create_adds_10", |bencher| bencher.iter(|| {
        unsafe {
            a.adds_mut(&b);
            a.adds_mut(&b);
            a.adds_mut(&b);
            a.adds_mut(&b);
            a.adds_mut(&b);
            a.adds_mut(&b);
            a.adds_mut(&b);
            a.adds_mut(&b);
            a.adds_mut(&b);
            a.adds_mut(&b);
        }
    }));

    black_box(a);
    black_box(b);

    let mut a = black_box(127i8);
    let b = black_box(127i8);

    group.bench_function("regular_create_adds_1", |bencher| bencher.iter(|| {
        a = a + b
    }));

    black_box(a);
    black_box(b);

    let mut a = black_box(127i8);
    let b = black_box(127i8);

    group.bench_function("regular_create_adds_10", |bencher| bencher.iter(|| {
        a = a + b;
        a = a + b;
        a = a + b;
        a = a + b;
        a = a + b;
        a = a + b;
        a = a + b;
        a = a + b;
        a = a + b;
        a = a + b
    }));

    black_box(a);
    black_box(b);
	
	let mut a = black_box(unsafe {_mm256_set1_epi8(127i8)});
    let b = black_box(unsafe {_mm256_set1_epi8(127i8)});

    group.bench_function("avx_create_adds_1", |bencher| bencher.iter(|| {
        unsafe {
            a = _mm256_adds_epi8(a, b);
        }
    }));

    black_box(a);
    black_box(b);

    let mut a = black_box(unsafe {_mm256_set1_epi8(127i8)});
    let b = black_box(unsafe {_mm256_set1_epi8(127i8)});

    group.bench_function("avx_create_adds_10", |bencher| bencher.iter(|| {
        unsafe {
            a = _mm256_adds_epi8(a, b);
            a = _mm256_adds_epi8(a, b);
            a = _mm256_adds_epi8(a, b);
            a = _mm256_adds_epi8(a, b);
            a = _mm256_adds_epi8(a, b);
            a = _mm256_adds_epi8(a, b);
            a = _mm256_adds_epi8(a, b);
            a = _mm256_adds_epi8(a, b);
            a = _mm256_adds_epi8(a, b);
            a = _mm256_adds_epi8(a, b);
        }
    }));

    black_box(a);
    black_box(b);

    let mut a = unsafe {Sse1x16x8::repeating_max(black_box(length))};
    let b = unsafe {Sse1x16x8::repeating_max(black_box(length))};

    group.bench_function("jewel_sse_create_adds_1", |bencher| bencher.iter(|| {
        unsafe {
            a.adds_mut(&b);
        }
    }));

    black_box(a);
    black_box(b);

    let mut a = unsafe {Sse1x16x8::repeating_max(black_box(length))};
    let b = unsafe {Sse1x16x8::repeating_max(black_box(length))};

    group.bench_function("jewel_sse_create_adds_10", |bencher| bencher.iter(|| {
        unsafe {
            a.adds_mut(&b);
            a.adds_mut(&b);
            a.adds_mut(&b);
            a.adds_mut(&b);
            a.adds_mut(&b);
            a.adds_mut(&b);
            a.adds_mut(&b);
            a.adds_mut(&b);
            a.adds_mut(&b);
            a.adds_mut(&b);
        }
    }));

    black_box(a);
    black_box(b);

	let mut a = black_box(unsafe {_mm_set1_epi8(127i8)});
    let b = black_box(unsafe {_mm_set1_epi8(127i8)});

    group.bench_function("sse_create_adds_1", |bencher| bencher.iter(|| {
        unsafe {
            a = _mm_adds_epi8(a, b);
        }
    }));

    black_box(a);
    black_box(b);

    let mut a = black_box(unsafe {_mm_set1_epi8(127i8)});
    let b = black_box(unsafe {_mm_set1_epi8(127i8)});

    group.bench_function("sse_create_adds_10", |bencher| bencher.iter(|| {
        unsafe {
            a = _mm_adds_epi8(a, b);
            a = _mm_adds_epi8(a, b);
            a = _mm_adds_epi8(a, b);
            a = _mm_adds_epi8(a, b);
            a = _mm_adds_epi8(a, b);
            a = _mm_adds_epi8(a, b);
            a = _mm_adds_epi8(a, b);
            a = _mm_adds_epi8(a, b);
            a = _mm_adds_epi8(a, b);
            a = _mm_adds_epi8(a, b);
        }
    }));

    black_box(a);
    black_box(b);

    group.finish();
}

criterion_group!(bench_jewel, bench_jewel_fn);
criterion_main!(bench_jewel);
