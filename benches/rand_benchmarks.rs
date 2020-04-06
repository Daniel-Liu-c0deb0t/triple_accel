use criterion::{criterion_group, criterion_main, Criterion};
use rand::prelude::*;
use triple_accel::*;

fn bench_rand_hamming(c: &mut Criterion) {
    let mut rng = thread_rng();
    let (a_str, b_str) = rand_hamming_pair(1000, 30, &mut rng);

    let mut group = c.benchmark_group("bench_rand_hamming");

    group.bench_function("hamming_naive", |b| b.iter(|| hamming_naive(&a_str, &b_str)));
    group.bench_function("hamming_words", |b| b.iter(|| hamming_words(&a_str, &b_str)));
    group.bench_function("hamming_simd", |b| b.iter(|| hamming_simd(&a_str, &b_str)));

    group.finish();
}

fn bench_rand_hamming_search(c: &mut Criterion) {
    let mut rng = thread_rng();
    let needle_len = 32;
    let haystack_len = 1000;
    let k = 16;
    let (needle, haystack) = rand_hamming_needle_haystack(needle_len, haystack_len, 50, k, &mut rng);

    let mut group = c.benchmark_group("bench_rand_hamming_search");

    group.bench_function("hamming_search_naive", |b| b.iter(|| hamming_search_naive(&needle, needle_len, &haystack, haystack_len, k)));
    group.bench_function("hamming_search_simd", |b| b.iter(|| hamming_search_simd(&needle, needle_len, &haystack, haystack_len, k)));

    group.finish();
}

fn bench_rand_levenshtein(c: &mut Criterion) {
    let mut rng = thread_rng();
    let (a_str, b_str) = rand_levenshtein_pair(1000, 30, &mut rng);

    let mut group = c.benchmark_group("bench_rand_levenshtein");

    group.bench_function("levenshtein_naive", |b| b.iter(|| levenshtein_naive(&a_str, a_str.len(), &b_str, b_str.len(), false)));
    group.bench_function("levenshtein_naive_k", |b| b.iter(|| levenshtein_naive_k(&a_str, a_str.len(), &b_str, b_str.len(), 30, false)));
    group.bench_function("levenshtein_simd", |b| b.iter(|| levenshtein_simd(&a_str, a_str.len(), &b_str, b_str.len(), false)));

    group.finish();
}

criterion_group!(bench_rand, bench_rand_hamming, bench_rand_hamming_search, bench_rand_levenshtein);
criterion_main!(bench_rand);

fn rand_hamming_needle_haystack<R: Rng>(needle_len: usize, haystack_len: usize, num_match: usize, k: u32, rng: &mut R) -> (Vec<u8>, Vec<u8>) {
    let mut idx: Vec<usize> = (0usize..haystack_len).collect();
    idx.shuffle(rng);
    let mut insert = vec![false; haystack_len];

    for i in 0..num_match {
        insert[idx[i]] = true;
    }

    let bytes: Vec<u8> = (33u8..127u8).collect();
    let needle = rand_alloc_str(needle_len, rng);
    let mut haystack: Vec<u8> = vec![];

    for i in 0..haystack_len {
        if insert[i] {
            let s = rand_hamming_mutate(&needle, needle_len, k, rng);
            haystack.extend(&s[..needle_len]);
        }else{
            haystack.push(*bytes.choose(rng).unwrap());
        }
    }

    (needle, haystack)
}

fn rand_hamming_pair<R: Rng>(length: usize, k: u32, rng: &mut R) -> (Vec<u8>, Vec<u8>) {
    let a = rand_alloc_str(length, rng);
    let b = rand_hamming_mutate(&a, length, k, rng);

    (a, b)
}

fn rand_hamming_mutate<R: Rng>(a: &[u8], length: usize, k: u32, rng: &mut R) -> Vec<u8> {
    let mut b = alloc_str(length);
    fill_str(&mut b, a);
    let curr_k: usize = rng.gen_range(0, k as usize + 1);
    let mut idx: Vec<usize> = (0usize..length).collect();
    idx.shuffle(rng);

    for i in 0..curr_k {
        b[idx[i]] = 32u8;
    }

    b
}

fn rand_levenshtein_pair<R: Rng>(length: usize, k: u32, rng: &mut R) -> (Vec<u8>, Vec<u8>) {
    let a = rand_str(length, rng);
    let mut edits = vec![0u8; length];
    let curr_k: usize = rng.gen_range(0usize, k as usize + 1);
    let mut idx: Vec<usize> = (0usize..length).collect();
    idx.shuffle(rng);

    for i in 0..curr_k {
        edits[idx[i]] = rng.gen_range(1u8, 4u8);
    }

    let bytes: Vec<u8> = (33u8..127u8).collect();
    let mut b = vec![];

    for i in 0..length {
        match edits[i] {
            0u8 => { // same
                b.push(a[i]);
            },
            1u8 => { // diff
                b.push(32u8);
            },
            2u8 => { // insert
                b.push(*bytes.choose(rng).unwrap());
                b.push(a[i]);
            },
            3u8 => (), // delete
            _ => panic!("This should not have been reached!")
        }
    }

    (a, b)
}

fn rand_str<R: Rng>(length: usize, rng: &mut R) -> Vec<u8> {
    let bytes: Vec<u8> = (33u8..127u8).collect();
    let mut res = vec![0u8; length];

    for i in 0..length {
        res[i] = *bytes.choose(rng).unwrap();
    }

    res
}

fn rand_alloc_str<R: Rng>(length: usize, rng: &mut R) -> Vec<u8> {
    let bytes: Vec<u8> = (33u8..127u8).collect();
    let mut res = alloc_str(length);

    for i in 0..length {
        res[i] = *bytes.choose(rng).unwrap();
    }

    res
}

