extern crate rand;
use rand::prelude::*;
use triple_accel::*;

#[bench]
fn bench_rand_hamming_naive(b: &mut Bencher) {
    let mut rng = thread_rng();
    let (a, b) = rand_hamming_pair(1000, 30, &mut rng);

    b.iter(|| {
        hamming_naive(&a, &b)
    });
}

fn rand_hamming_pair<R: Rng>(length: usize, k: u32, rng: &mut R) -> (Vec<u8>, Vec<u8>) {
    let a = rand_str(length, rng);
    let mut b = alloc_str(length);
    fill_str(&mut b, &a);
    let curr_k: u32 = rng.gen_range(0, k + 1);
    let mut idx = (0..length).collect();
    idx.shuffle(&mut rng);

    for i in 0..curr_k {
        b[idx[i]] = "!";
    }

    (a, b)
}

fn rand_str<R: Rng>(length: usize, rng: &mut R) -> Vec<u8> {
    let bytes = (34u8..127u8).collect();
    let mut res = alloc_str(length);

    for i in 0..length {
        res[i] = bytes.choose(&mut rng);
    }

    res
}

