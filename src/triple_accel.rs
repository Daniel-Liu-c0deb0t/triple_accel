use std;

#[cfg(target_arch = "x86")]
use core::arch::x86::*;

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

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

pub fn hamming_search_naive(needle: &[u8], needle_len: usize, haystack: &[u8], haystack_len: usize, k: u32) -> Vec<Match> {
    assert!(needle_len <= haystack_len);

    let mut res = vec![];
    let len = haystack_len + 1 - needle_len;

    'outer: for i in 0..len {
        let mut final_res = 0u32;

        for j in 0..needle_len {
            if needle[j] != haystack[i + j] {
                final_res += 1;
            }

            if final_res > k {
                continue 'outer;
            }
        }

        res.push(Match{start: i, end: i + needle_len, k: final_res});
    }

    res
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
    let vec_ptr = words.as_mut_ptr();
    let vec_len = words.len();
    let vec_cap = words.capacity();

    unsafe {
        Vec::from_raw_parts(vec_ptr as *mut u8, vec_len << 4, vec_cap)
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
        panic!("This should not be happening!");
    }

    res
}

unsafe fn hamming_search_simd(needle: &[u8], needle_len: usize, haystack: &[u8], haystack_len: usize, k: u32) -> Vec<Match> {
    assert!(needle_len <= haystack_len);

    let mut res = vec![];
    let len = haystack.len() + 1 - needle.len();
    let needle_ptr = needle.as_ptr() as *const __m256i;
    let haystack_ptr = haystack.as_ptr();

    for i in 0..len {
        let a = _mm256_loadu_si256(needle_ptr);
        let b = _mm256_loadu_si256(haystack_ptr.offset(i) as *const __m256i);

        res.push(Match{start: i, end: i + needle_len, k: final_res});
    }

    res
}

pub fn levenshtein_naive(a: &[u8], a_len: usize, b: &[u8], b_len: usize, trace_on: bool) -> (u32, Option<Vec<Edit>>) {

}

pub fn levenshtein_naive_k(a: &[u8], a_len: usize, b: &[u8], b_len: usize, k: u32, trace_on: bool) -> (u32, Option<Vec<Edit>>) {

}

pub fn levenshtein_simd(a: &[u8], a_len: usize, b: &[u8], b_len: usize, trace_on: bool) -> (u32, Option<Vec<Edit>>) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe {levenshtein_simd_x86_avx2(a, a_len, b, b_len, trace_on)};
        }
    }

    // todo: sse support and fallback

    unimplemented!();
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn levenshtein_simd_x86_avx2(a_old: &[u8], a_old_len: usize, b_old: &[u8], b_old_len: usize, trace_on: bool) -> (u32, Option<Vec<Edit>>) {
    if a_old_len == 0 && b_old_len == 0 {
        return if trace_on {(0u32, Some(vec![]))} else {(0u32, None)};
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

    // reusable constants
    let ones = _mm256_set1_epi8(1i8);
    let twos = _mm256_set1_epi8(2i8);

    let len_diff = b_len - a_len;
    let len = a_len + b_len + 1;
    let len_div2 = (len >> 1) + (len & 1);
    let ends_with_k2 = len & 1 == 0;
    // every diff between the length of a and b results in a shift from the main diagonal
    let final_idx = {
        if ends_with_k2 { // divisible by 2, ends with k2
            k2_div2 + ((len_diff - 1) >> 1)
        }else{ // not divisible by 2, ends with k1
            k1_div2 + (len_diff >> 1)
        }
    };

    // 0 = match/mismatch, 1 = a gap, 2 = b gap
    let mut traceback_arr = if trace_on {vec![[0u8; 32]; len + (len & 1)]} else {vec![]};

    if trace_on {
        traceback_arr[1][k2_div2 - 1] = 2u8;
        traceback_arr[1][k2_div2] = 1u8;
    }

    // example: allow k = 2 edits for two strings of length 3
    //      -b--   
    // | xx */*
    // a  x /*/*
    // |    */*/ x
    // |     */* xx
    //
    // each (anti) diagonal is represented with '*' or '/'
    // '/', use k2 = 2
    // '*', use k1 = 3
    // 'x' represents cells not in the "traditional" dp array
    // these out of bounds dp cells are shown because they represent
    // a horizontal sliding window of length 5 (2 * k + 1)
    //
    // dp2 is one diagonal before current
    // dp1 is two diagonals before current
    // we are trying to calculate the "current" diagonal
    // note that a k1 '*' dp diagonal has its center cell on the main diagonal
    // in general, the diagonals are centered on the main diagonal
    // each diagonal is represented using a 256-bit vector
    // each vector goes from bottom-left to top-right
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
    //
    // each iteration of the loop below results in processing both a k1 diagonal and a k2 diagonal
    // this could be done with an alternating state flag but it is unrolled for less branching
    //
    // note: in traditional dp array
    // dp[i][j] -> dp[i + 1][j] is a gap in string b
    // dp[i][j] -> dp[i][j + 1] is a gap in string a

    for i in 1..len_div2 {
        // move indexes in strings forward
        k1_idx += 1;
        k2_idx += 1;

        // move windows for the strings a and b
        a_k1_window = shift_right_x86_avx2(a_k1_window);

        if k1_idx < a_len {
            a_k1_window = _mm256_insert_epi8(a_k1_window, a[k1_idx] as i8, 0i32);
        }

        b_k1_window = shift_left_x86_avx2(b_k1_window);

        if k1_idx < b_len {
            b_k1_window = _mm256_insert_epi8(b_k1_window, b[k1_idx] as i8, 30i32); // k1 - 1
        }

        a_k2_window = shift_right_x86_avx2(a_k2_window);

        if k2_idx < a_len {
            a_k2_window = _mm256_insert_epi8(a_k2_window, a[k2_idx] as i8, 0i32);
        }

        b_k2_window = shift_left_x86_avx2(b_k2_window);

        if k2_idx < b_len {
            b_k2_window = _mm256_insert_epi8(b_k2_window, b[k2_idx] as i8, 29i32); // k2 - 1
        }

        // (anti) diagonal that matches in the a and b windows
        let match_mask_k1 = _mm256_cmpeq_epi8(a_k1_window, b_k1_window);
        // add ones to cells that have mismatching characters from a and b
        let sub_k1 = _mm256_adds_epi8(dp1, _mm256_andnot_si256(match_mask_k1, ones));
        // cost of gaps in a
        let a_gap_k1 = {
            let mut a_gap_prev = shift_right_x86_avx2(dp2);
            a_gap_prev = _mm256_insert_epi8(a_gap_prev, 127i8, 0i32);
            _mm256_adds_epi8(a_gap_prev, ones)
        };
        // cost of gaps in b
        let b_gap_k1 = _mm256_adds_epi8(dp2, ones);

        dp1 = dp2;

        // min of the cost of all three edit operations
        if trace_on {
            let min = triple_argmin_x86_avx2(sub_k1, a_gap_k1, b_gap_k1, ones, twos);
            dp2 = min.0;
            _mm256_storeu_si256(traceback_arr[i << 1].as_mut_ptr() as *mut __m256i, min.1);
        }else{
            dp2 = _mm256_min_epi8(sub_k1, _mm256_min_epi8(a_gap_k1, b_gap_k1));
        }

        // (anti) diagonal that matches in the a and b windows
        let match_mask_k2 = _mm256_cmpeq_epi8(a_k2_window, b_k2_window);
        // add ones to cells that have mismatching characters from a and b
        let sub_k2 = _mm256_adds_epi8(dp1, _mm256_andnot_si256(match_mask_k2, ones));
        // cost of gaps in b
        let b_gap_k2 = {
            let mut b_gap_prev = shift_left_x86_avx2(dp2);
            b_gap_prev = _mm256_insert_epi8(b_gap_prev, 127i8, 31i32); // k1
            _mm256_adds_epi8(b_gap_prev, ones)
        };
        // cost of gaps in a
        let a_gap_k2 = _mm256_adds_epi8(dp2, ones);

        dp1 = dp2;

        // min of the cost of all three edit operations
        if trace_on {
            let min = triple_argmin_x86_avx2(sub_k2, a_gap_k2, b_gap_k2, ones, twos);
            dp2 = min.0;
            _mm256_storeu_si256(traceback_arr[(i << 1) + 1].as_mut_ptr() as *mut __m256i, min.1);
        }else{
            dp2 = _mm256_min_epi8(sub_k2, _mm256_min_epi8(a_gap_k2, b_gap_k2));
        }
    }

    let mut final_arr = [0u8; 32];

    if ends_with_k2 {
        _mm256_storeu_si256(final_arr.as_mut_ptr() as *mut __m256i, dp2);
    }else{
        _mm256_storeu_si256(final_arr.as_mut_ptr() as *mut __m256i, dp1);
    }

    if trace_on {
        (final_arr[final_idx] as u32, Some(traceback(&traceback_arr, final_idx, a, a_len, b, b_len, swap, ends_with_k2)))
    }else{
        (final_arr[final_idx] as u32, None)
    }
}

fn traceback(arr: &[[u8; 32]], mut idx: usize, a: &[u8], a_len: usize, b: &[u8], b_len: usize, swap: bool, mut is_k2: bool) -> Vec<Edit> {
    // keep track of position in traditional dp array and strings
    let mut i = a_len; // index in a
    let mut j = b_len; // index in b

    // last diagonal may overshoot, so ignore it
    let mut arr_idx = arr.len() - 1 - (if is_k2 {0} else {1});
    let mut res = vec![];

    while arr_idx > 0 {
        let edit = arr[arr_idx][idx];

        match edit {
            0u8 => { // match/mismatch
                res.push(if a[i - 1] == b[j - 1] {Edit::Match} else {Edit::Mismatch});
                arr_idx -= 2;
                i -= 1;
                j -= 1;
            },
            1u8 => { // a gap
                res.push(if swap {Edit::BGap} else {Edit::AGap}); // account for the swap in the beginning
                arr_idx -= 1;

                if !is_k2 {
                    idx -= 1;
                }

                j -= 1;
                is_k2 = !is_k2; // must account for alternating k1/k2 diagonals
            },
            2u8 => { // b gap
                res.push(if swap {Edit::AGap} else {Edit::BGap});
                arr_idx -= 1;

                if is_k2 {
                    idx += 1;
                }

                i -= 1;
                is_k2 = !is_k2;
            },
            _ => panic!("This should not be happening!")
        }
    }

    res.reverse();
    res
}

pub fn levenshtein_search_naive(needle: &[u8], needle_len: usize, haystack: &[u8], haystack_len: usize, k: u32) -> Vec<Match> {
    let len = needle_len + 1;
    let mut dp1 = vec![0u32; len];
    let mut dp2 = vec![0u32; len];
    let mut length1 = vec![0u32; len];
    let mut length2 = vec![0u32; len];
    let mut res = vec![];

    for i in 0..len {
        dp1[i] = i;
    }

    if dp1[len - 1] <= k {
        res.push(Match{start: 0, end: 0, k: dp1[len - 1]});
    }

    for i in 0..haystack_len {
        dp2[0] = 0;
        length2[0] = 0;

        for j in 1..len {
            let sub = dp1[j - 1] + (if needle[j - 1] == haystack[i] {0} else {1});
            let a_gap = dp1[j] + 1;
            let b_gap = dp2[j - 1] + 1;

            dp2[j] = a_gap;
            length2[j] = length1[j] + 1;

            if b_gap < dp2[j] {
                dp2[j] = b_gap;
                length2[j] = length2[j - 1];
            }

            if sub <= dp2[j] {
                dp2[j] = sub;
                length2[j] = length1[j - 1] + 1;
            }
        }

        let final_res = dp2[len - 1];

        if final_res <= k {
            res.push(Match{start: i + 1 - length2[len - 1], end: i + 1, k: final_res});
        }

        let temp = dp1;
        dp1 = dp2;
        dp2 = temp;
    }

    res
}

pub fn levenshtein_search_simd(needle: &[u8], needle_len: usize, haystack: &[u8], haystack_len: usize, k: u32) -> Vec<Match> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe {levenshtein_search_simd_x86_avx2(needle, needle_len, haystack, haystack_len, k)};
        }
    }

    // todo: sse support and fallback

    unimplemented!();
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn levenshtein_search_simd_x86_avx2(needle: &[u8], needle_len: usize, haystack: &[u8], haystack_len: usize, k: u32) -> Vec<Match> {
    assert!(needle_len <= 32);

    if needle_len == 0 {
        return vec![];
    }

    let mut dp1 = _mm256_set1_epi8(127i8);
    let mut dp2 = _mm256_set1_epi8(127i8);
    dp2 = _mm256_insert_epi8(dp2, 1i8, 31i32); // last cell

    // save length instead of start idx due to int size constraints
    let mut length1 = _mm256_setzero_si256();
    let mut length2 = _mm256_setzero_si256();

    let ones = _mm256_set1_epi8(1i8);

    let len = haystack_len + needle_len;
    let final_idx = 32 - needle_len;

    let mut dp_arr = [0u8; 32];
    let dp_arr_ptr = dp_arr.as_mut_ptr() as *mut __m256i;
    let mut length_arr = [0u8; 32];
    let length_arr_ptr = length_arr.as_mut_ptr() as *mut __m256i;

    let mut res = vec![];

    let needle_window = {
        let mut needle_window_arr = [0u8; 32];

        // needle window is in reverse order
        for i in 0..std::cmp::min(needle_len, 32) {
            needle_window_arr[31 - i] = needle[i];
        }

        _mm256_loadu_si256(needle_window_arr.as_ptr() as *const __m256i)
    };

    let mut haystack_window = _mm256_setzero_si256();
    let mut haystack_idx = 0usize;

    //       ..i...
    //       --h---
    // |     //////xxxx
    // |    x//////xxx
    // n   xx//////xx
    // |  xxx//////x
    // | xxxx//////
    //
    // 'n' = needle, 'h' = haystack
    // each (anti) diagonal is denoted using '/' and 'x'
    // 'x' marks cells that are not in the traditional dp array
    // every (anti) diagonal is calculated simultaneously using a 256-bit vector
    // note: each vector goes from bottom-left to top-right

    for i in 1..len {
        // shift the haystack window
        haystack_window = shift_left_x86_avx2(haystack_window);

        if haystack_idx < haystack_len {
            haystack_window = _mm256_insert_epi8(haystack_window, haystack[haystack_idx] as i8, 31i32);
            haystack_idx += 1;
        }

        let match_mask = _mm256_cmpeq_epi8(needle_window, haystack_window);

        // match/mismatch
        let mut sub = shift_left_x86_avx2(dp1); // zeros are shifted in
        sub = _mm256_adds_epi8(sub, _mm256_andnot_si256(match_mask, ones));

        let sub_length = _mm256_adds_epi8(shift_left_x86_avx2(length1), ones);

        // gap in needle
        let needle_gap = _mm256_adds_epi8(dp2, ones);

        let needle_gap_length = _mm256_adds_epi8(length2, ones);

        // gap in haystack
        let mut haystack_gap = shift_left_x86_avx2(dp2); // zeros are shifted in
        haystack_gap = _mm256_adds_epi8(haystack_gap, ones);

        let haystack_gap_length = shift_left_x86_avx2(length2);

        dp1 = dp2;
        length1 = length2;

        let min = triple_min_length_x86_avx2(sub, needle_gap, haystack_gap, sub_length, needle_gap_length, haystack_gap_length);
        dp2 = min.0;
        length2 = min.1;

        if i >= needle_len - 1 {
            // manually storing the dp and length arrays is necessary
            // because the accessed index is not a compile time constant
            _mm256_storeu_si256(dp_arr_ptr, dp2);
            let final_res = dp_arr[final_idx] as u32;

            _mm256_storeu_si256(length_arr_ptr, length2);

            if final_res <= k {
                let end_idx = i + 1 - needle_len;
                res.push(Match{start: end_idx - (length_arr[final_idx] as usize), end: end_idx, k: final_res});
            }
        }
    }

    res
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
unsafe fn triple_argmin_x86_avx2(sub: __m256i, a_gap: __m256i, b_gap: __m256i, a_gap_arg: __m256i, b_gap_arg: __m256i) -> (__m256i, __m256i) {
    // return the edit used in addition to doing a min operation
    let mut res_min = _mm256_min_epi8(a_gap, b_gap);
    let a_gap_mask = _mm256_cmpeq_epi8(res_min, a_gap);
    let mut res_arg = _mm256_blendv_epi8(b_gap_arg, a_gap_arg, a_gap_mask);

    res_min = _mm256_min_epi8(sub, res_min);
    let sub_mask = _mm256_cmpeq_epi8(res_min, sub);
    res_arg = _mm256_andnot_si256(sub_mask, res_arg); // substitution arguments will automatically be zero

    return (res_min, res_arg);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn triple_min_length_x86_avx2(sub: __m256i, a_gap: __m256i, b_gap: __m256i, sub_length: __m256i, a_gap_length: __m256i, b_gap_length: __m256i) -> (__m256i, __m256i) {
    // choose the length based on which edit is chosen during the min operation
    let mut res_min = _mm256_min_epi8(a_gap, b_gap);
    let a_gap_mask = _mm256_cmpeq_epi8(res_min, a_gap);
    let mut res_length = _mm256_blendv_epi8(b_gap_length, a_gap_length, a_gap_mask);

    res_min = _mm256_min_epi8(sub, res_min);
    let sub_mask = _mm256_cmpeq_epi8(res_min, sub);
    res_length = _mm256_blendv_epi8(res_length, sub_length, sub_mask);

    return (res_min, res_length);
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

