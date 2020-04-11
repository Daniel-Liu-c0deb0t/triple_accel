use std;

mod hamming;
mod levenshtein;

// re-export
pub use hamming::*;
pub use levenshtein::*;

#[cfg(target_arch = "x86")]
use core::arch::x86::*;

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

// some shared utility stuff below

#[derive(Debug, PartialEq)]
pub struct Match {
    pub start: usize, // inclusive
    pub end: usize, // exclusive
    pub k: u32
}

#[derive(Debug, PartialEq)]
pub enum Edit {
    Match,
    Mismatch,
    AGap,
    BGap
}

// create a vector with alignment for u128, then convert it to u8
// (u8 has looser alignment requirements than u128)
// then, this vector can be easily converted back to u8 later
// mutable
#[inline]
pub fn alloc_str(len: usize) -> Vec<u8> {
    let words_len = (len >> 4) + (if (len & 15) > 0 {1} else {0});
    let words = vec![0u128; words_len];
    let mut words = std::mem::ManuallyDrop::new(words);

    unsafe {
        Vec::from_raw_parts(words.as_mut_ptr() as *mut u8, len, words_len << 4)
    }
}

// directly copy from one string to another
#[inline]
pub fn fill_str(dest: &mut [u8], src: &[u8]) {
    assert!(dest.len() >= src.len());

    unsafe {
        std::ptr::copy_nonoverlapping(src.as_ptr(), dest.as_mut_ptr(), src.len());
    }
}

#[allow(dead_code)]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn print_x86_avx2(a: &str, b: __m256i){
    let mut arr = [0u8; 32];
    _mm256_storeu_si256(arr.as_mut_ptr() as *mut __m256i, b);
    print!("{}\t", a);

    for i in 0..32 {
        print!(" {:>3}", arr[i]);
    }

    println!();
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn shift_left_x86_avx2(a: __m256i) -> __m256i {
    _mm256_alignr_epi8(_mm256_permute2x128_si256(a, a, 0b10000001i32), a, 1i32)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn shift_right_x86_avx2(a: __m256i) -> __m256i {
    _mm256_alignr_epi8(a, _mm256_permute2x128_si256(a, a, 0b00001000i32), 15i32)
}

