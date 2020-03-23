use std;

#[cfg(target_arch = "x86")]
use core::arch::x86::*;

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

pub fn hamming_naive(a: &[u8], b: &[u8]) -> u32 {
    let len = a.len();
    assert!(len == b.len());

    let mut res = 0u32;

    for i in 0..len {
        if a[i] != b[i] {
            res += 1;
        }
    }

    res
}

// create a vector with alignment for u128, then convert it to u8
// (u8 has looser alignment requirements than u128)
// then, this vector can be easily converted back to u8 later
// mutable
#[inline(always)]
pub fn alloc_str(len: usize) -> Vec<u8> {
    let words_len = (len >> 4) + (if (len & 15) > 0 {1} else {0});
    let words = vec![0u128; words_len];

    let mut words = std::mem::ManuallyDrop::new(words);
    let vec_ptr = words.as_mut_ptr();
    let vec_len = words.len();
    let vec_cap = words.capacity();

    unsafe {
        Vec::from_raw_parts(vec_ptr as *mut u8, vec_len << 4, vec_cap)
    }
}

// directly copy from one string to another
#[inline(always)]
pub fn fill_str(dest: &mut [u8], src: &[u8]) {
    assert!(dest.len() >= src.len());

    unsafe {
        std::ptr::copy_nonoverlapping(src.as_ptr(), dest.as_mut_ptr(), src.len());
    }
}

pub fn hamming_words(a: &[u8], b: &[u8]) -> u32 {
    assert!(a.len() == b.len());

    unsafe {
        let mut res = 0u32;
        // the address better be aligned for u128
        // may not be in little endian
        let a_ptr = a.as_ptr() as *const u128;
        let b_ptr = b.as_ptr() as *const u128;
        let words_len = (a.len() >> 4) as isize;

        for i in 0..words_len {
            // change to little endian omitted because it is not necessary in this case
            let mut r = (*a_ptr.offset(i)) ^ (*b_ptr.offset(i));
            // reduce or by "folding" one half of each byte onto the other multiple times
            r |= r >> 4;
            // ...00001111
            r &= 0x0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0fu128;
            r |= r >> 2;
            // ...00110011
            r &= 0x33333333333333333333333333333333u128;
            r |= r >> 1;
            // ...01010101
            r &= 0x55555555555555555555555555555555u128;
            res += r.count_ones();
        }

        res
    }
}

pub fn hamming_simd(a: &[u8], b: &[u8]) -> u32 {
    assert!(a.len() == b.len());

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe {hamming_simd_x86_avx2(a, b)};
        }
    }

    // todo: sse support and fallback to hamming_words

    unimplemented!();
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn hamming_simd_x86_avx2(a: &[u8], b: &[u8]) -> u32 {
    let mut res = 0u32;
    let len = a.len();
    let word_len = (len >> 5) as isize;
    let word_rem = len & 31;
    let a_ptr = a.as_ptr() as *const __m256i;
    let b_ptr = b.as_ptr() as *const __m256i;

    for i in 0..word_len {
        // unaligned, so must use loadu
        let a = _mm256_loadu_si256(a_ptr.offset(i));
        let b = _mm256_loadu_si256(b_ptr.offset(i));
        // directly check if equal
        let r = _mm256_cmpeq_epi8(a, b);
        // pack the 32 bytes into a 32-bit int and count not equal
        res += _mm256_movemask_epi8(r).count_zeros();
    }

    if word_rem == 16 {
        // same algorithm as hamming_words
        let a_ptr = a.as_ptr() as *const u128;
        let b_ptr = b.as_ptr() as *const u128;
        let offset = word_len << 1;

        let mut r = (*a_ptr.offset(offset)) ^ (*b_ptr.offset(offset));
        r |= r >> 4;
        r &= 0x0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0fu128;
        r |= r >> 2;
        r &= 0x33333333333333333333333333333333u128;
        r |= r >> 1;
        r &= 0x55555555555555555555555555555555u128;
        res += r.count_ones();
    }else{
        // this should not happen since a and b should be zero padded to a multiple of 128
        unimplemented!();
    }

    res
}

// naive
// bounded fast
// bounded fast, equal length
// exp search for k
// search
// simd-based bounded fast
// simd-based search

pub fn levenshtein_simd(a: &[u8], a_len: usize, b: &[u8], b_len: usize) -> u32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe {levenshtein_simd_x86_avx2(a, a_len, b, b_len)};
        }
    }

    // todo: sse support and fallback

    unimplemented!();
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn levenshtein_simd_x86_avx2(a_old: &[u8], a_old_len: usize, b_old: &[u8], b_old_len: usize) -> u32 {
    if a_old_len == 0 && b_old_len == 0 {
        return 0u32;
    }

    // swap a and b so that a is shorter than b, if applicable
    // makes operations later on slightly easier, since length of a <= length of b
    let swap = a_old_len > b_old_len;
    let a = if swap {b_old} else {a_old};
    let a_len = if swap {b_old_len} else {a_old_len};
    let b = if swap {a_old} else {b_old};
    let b_len = if swap {a_old_len} else {b_old_len};

    // lengths of the (anti) diagonals
    // constant length for simplicity because similar speeds even if this is shorter
    let k1 = 31usize;
    let k1_div2 = k1 >> 1;
    let k2 = 30usize;
    let k2_div2 = k2 >> 1;

    // initialized with max value of i8
    // must use saturated additions afterwards to not overflow
    let mut dp1 = _mm256_set1_epi8(127i8);
    let mut dp2 = _mm256_set1_epi8(127i8);
    // set dp[0][0] = 0
    dp1 = _mm256_insert_epi8(dp1, 0i8, 15i32); // k1 / 2
    // set dp[0][1] = 1 and dp[1][0] = 1
    dp2 = _mm256_insert_epi8(dp2, 1i8, 14i32); // k2 / 2 - 1
    dp2 = _mm256_insert_epi8(dp2, 1i8, 15i32); // k2 / 2

    // a_k1_window and a_k2_window represent reversed portions of the string a
    // copy in half of k1/k2 number of characters
    // these characters are placed in the second half of b windows
    // since a windows are reversed, the characters are placed in reverse in the first half of b windows
    let mut a_k1_window = {
        let mut a_k1_window_arr = [0u8; 32];

        for i in 0..std::cmp::min(k1_div2, a_len) {
            a_k1_window_arr[k1_div2 - 1 - i] = a[i];
        }

        // unaligned, so must use loadu
        _mm256_loadu_si256(a_k1_window_arr.as_ptr() as *const __m256i)
    };

    let mut b_k1_window = {
        let mut b_k1_window_arr = [0u8; 32];

        for i in 0..std::cmp::min(k1_div2, b_len) {
            b_k1_window_arr[k1_div2 + 1 + i] = b[i];
        }

        _mm256_loadu_si256(b_k1_window_arr.as_ptr() as *const __m256i)
    };

    let mut a_k2_window = {
        let mut a_k2_window_arr = [0u8; 32];

        for i in 0..std::cmp::min(k2_div2, a_len) {
            a_k2_window_arr[k2_div2 - 1 - i] = a[i];
        }

        _mm256_loadu_si256(a_k2_window_arr.as_ptr() as *const __m256i)
    };

    let mut b_k2_window = {
        let mut b_k2_window_arr = [0u8; 32];

        for i in 0..std::cmp::min(k2_div2, b_len) {
            b_k2_window_arr[k2_div2 + i] = b[i];
        }

        _mm256_loadu_si256(b_k2_window_arr.as_ptr() as *const __m256i)
    };

    // used to keep track of the next characters to place in the windows
    let mut k1_idx = k1_div2 - 1;
    let mut k2_idx = k2_div2 - 1;

    let ones = _mm256_set1_epi8(1i8);

    let len_diff = b_len - a_len;
    let len = a_len + b_len + 1;
    // every diff between the length of a and b results in a shift from the main diagonal
    let final_idx = {
        if len & 1 == 0 { // divisible by 2, ends with state = true, k2
            k2_div2 + ((len_diff - 1) >> 1)
        }else{ // not divisible by 2, ends with state = false, k1
            k1_div2 + (len_diff >> 1)
        }
    };

    // state alternates between k1 and k2 diagonals
    let mut state = false;

    // example: allow k = 2 edits for two strings of length 3
    //
    // xx */*
    //  x /*/*
    //    */*/ x
    //     */* xx
    //
    // each (anti) diagonal is represented with '*' or '/'
    // state = true: '/', use k2 = 2
    // state = false: '*', use k1 = 3
    // 'x' represents cells not in the "traditional" dp array
    //
    // dp2 is one diagonal before current
    // dp1 is two diagonals before current
    // we are trying to calculate the current diagonal
    // note that a state = false '*' dp diagonal has its center cell on the main diagonal
    //
    // the a windows and b windows are queues of a fixed length
    // a is reversed, so that elementwise comparison can be done between a and b
    // this operation obtains the comparison of characters along the (anti) diagonal
    //
    // example of moving the windows:
    // a windows: [5 4 3 2 1] -> [6 5 4 3 2] (right shift + insert)
    // b windows: [1 2 3 4 5] -> [2 3 4 5 6] (left shift + insert)
    //
    // initially:
    // a windows: [2 1 0 0 0]
    // b windows: [0 0 0 1 2]
    //
    // note that there will be left over cells not filled in the SIMD vector
    // this is because k1 and k2 are not long enough
    // all of these empty cells should be at the end of the SIMD vectors

    for _i in 2..len {
        // move windows for the strings a and b
        if state {
            k2_idx += 1;

            // right shift
            a_k2_window = shift_right_x86_avx2(a_k2_window);

            if k2_idx < a_len {
                a_k2_window = _mm256_insert_epi8(a_k2_window, a[k2_idx] as i8, 0i32);
            }

            // left shift
            b_k2_window = shift_left_x86_avx2(b_k2_window);

            if k2_idx < b_len {
                b_k2_window = _mm256_insert_epi8(b_k2_window, b[k2_idx] as i8, 29i32); // k2 - 1
            }
        }else{
            k1_idx += 1;

            // right shift
            a_k1_window = shift_right_x86_avx2(a_k1_window);

            if k1_idx < a_len {
                a_k1_window = _mm256_insert_epi8(a_k1_window, a[k1_idx] as i8, 0i32);
            }

            // left shift
            b_k1_window = shift_left_x86_avx2(b_k1_window);

            if k1_idx < b_len {
                b_k1_window = _mm256_insert_epi8(b_k1_window, b[k1_idx] as i8, 30i32); // k1 - 1
            }
        }

        let match_mask = {
            if state { // shorter, k2
                _mm256_cmpeq_epi8(a_k2_window, b_k2_window)
            }else{ // longer, k1
                _mm256_cmpeq_epi8(a_k1_window, b_k1_window)
            }
        };

        // add ones to cells that have mismatching characters from a and b
        let sub = _mm256_adds_epi8(dp1, _mm256_andnot_si256(match_mask, ones));

        let a_gap = {
            if state { // shorter, k2
                // left shift
                let mut a_gap_prev = shift_left_x86_avx2(dp2);
                a_gap_prev = _mm256_insert_epi8(a_gap_prev, 127i8, 31i32); // k1
                _mm256_adds_epi8(a_gap_prev, ones)
            }else{ // longer, k1
                _mm256_adds_epi8(dp2, ones)
            }
        };

        let b_gap = {
            if state { // shorter, k2
                _mm256_adds_epi8(dp2, ones)
            }else{ // longer, k1
                // right shift
                let mut b_gap_prev = shift_right_x86_avx2(dp2);
                b_gap_prev = _mm256_insert_epi8(b_gap_prev, 127i8, 0i32);
                _mm256_adds_epi8(b_gap_prev, ones)
            }
        };

        dp1 = dp2;
        // min of the cost of all three edit operations
        dp2 = _mm256_min_epi8(sub, _mm256_min_epi8(a_gap, b_gap));

        state = !state;
    }

    let mut final_arr = [0u8; 32];
    _mm256_storeu_si256(final_arr.as_mut_ptr() as *mut __m256i, dp2);
    final_arr[final_idx] as u32
}

#[allow(dead_code)]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn print_x86_avx2(a: &str, b: __m256i){
    let mut arr = [0u8; 32];
    _mm256_storeu_si256(arr.as_mut_ptr() as *mut __m256i, b);
    println!("{}:\t{:?}", a, arr);
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

