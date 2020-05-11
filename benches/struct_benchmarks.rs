use criterion::{criterion_group, criterion_main, Criterion, black_box};
use core::arch::x86_64::*;

mod other {
    use core::arch::x86_64::*;

    pub struct A {
        pub v: __m256i
    }

    impl A {
        #[target_feature(enable = "avx2")]
        #[inline]
        pub unsafe fn adds(&mut self, b: &Self) {
            self.v = _mm256_adds_epi8(self.v, b.v);
        }
    }
}

fn bench_fn(c: &mut Criterion) {
    let mut group = c.benchmark_group("bench_fn");

    let mut a = black_box(unsafe {_mm256_set1_epi8(127i8)});
    let b = black_box(unsafe {_mm256_set1_epi8(127i8)});

    group.bench_function("avx_raw", |bencher| bencher.iter(|| {
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

    let mut a = black_box(unsafe {other::A{v: _mm256_set1_epi8(127i8)}});
    let b = black_box(unsafe {other::A{v: _mm256_set1_epi8(127i8)}});

    group.bench_function("avx_with_struct_fn", |bencher| bencher.iter(|| {
        unsafe {
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
        }
    }));

    black_box(a);
    black_box(b);

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn adds_inline(a: __m256i, b: __m256i) -> __m256i {
        _mm256_adds_epi8(a, b)
    }

    let mut a = black_box(unsafe {_mm256_set1_epi8(127i8)});
    let b = black_box(unsafe {_mm256_set1_epi8(127i8)});

    group.bench_function("avx_with_fn", |bencher| bencher.iter(|| {
        unsafe {
            a = adds_inline(a, b);
            a = adds_inline(a, b);
            a = adds_inline(a, b);
            a = adds_inline(a, b);
            a = adds_inline(a, b);
            a = adds_inline(a, b);
            a = adds_inline(a, b);
            a = adds_inline(a, b);
            a = adds_inline(a, b);
            a = adds_inline(a, b);
        }
    }));

    black_box(a);
    black_box(b);

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn adds_10(mut a: __m256i, b: __m256i) -> __m256i {
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
        a
    }

    let mut a = black_box(unsafe {_mm256_set1_epi8(127i8)});
    let b = black_box(unsafe {_mm256_set1_epi8(127i8)});

    group.bench_function("avx_with_fn_10", |bencher| bencher.iter(|| {
        unsafe {
            a = adds_10(a, b);
        }
    }));

    black_box(a);
    black_box(b);

    group.finish();
}

criterion_group!(bench, bench_fn);
criterion_main!(bench);
