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
        res += (a[i] != b[i]) as u32;
    }

    res
}


pub fn hamming_search_naive(needle: &[u8], haystack: &[u8]) -> Vec<Match> {
    hamming_search_naive_k(needle, haystack, needle.len() as u32, true)
}

pub fn hamming_search_naive_k(needle: &[u8], haystack: &[u8], k: u32, best: bool) -> Vec<Match> {
    let needle_len = needle.len();
    let haystack_len = haystack.len();

    assert!(needle_len <= haystack_len);

    let len = haystack_len + 1 - needle_len;
    let mut res = Vec::with_capacity(len >> 2);
    let mut curr_k = k;

    'outer: for i in 0..len {
        let mut final_res = 0u32;

        for j in 0..needle_len {
            final_res += (needle[j] != haystack[i + j]) as u32;

            if final_res > curr_k {
                continue 'outer;
            }
        }

        res.push(Match{start: i, end: i + needle_len, k: final_res});

        if best {
            curr_k = final_res;
        }
    }

    if best {
        res.retain(|m| m.k == curr_k);
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

pub fn hamming_words_64(a: &[u8], b: &[u8]) -> u32 {
    assert!(a.len() == b.len());

    unsafe {
        let mut res = 0u32;
        // the pointer address better be aligned for u64
        // may not be in little endian
        let a_ptr = a.as_ptr() as *const u64;
        let b_ptr = b.as_ptr() as *const u64;
        let words_len = (a.len() >> 3) as isize;

        for i in 0..words_len {
            // change to little endian omitted because it is not necessary in this case
            let mut r = (*a_ptr.offset(i)) ^ (*b_ptr.offset(i));
            // reduce or by "folding" one half of each byte onto the other multiple times
            r |= r >> 4;
            // ...00001111
            r &= 0x0f0f0f0f0f0f0f0fu64;
            r |= r >> 2;
            // ...00110011
            r &= 0x3333333333333333u64;
            r |= r >> 1;
            // ...01010101
            r &= 0x5555555555555555u64;
            res += r.count_ones();
        }

        let words_rem = a.len() & 7;

        if words_rem > 0 {
            let mut r = (*a_ptr.offset(words_len)) ^ (*b_ptr.offset(words_len));
            r |= r >> 4;
            r &= 0x0f0f0f0f0f0f0f0fu64;
            r |= r >> 2;
            r &= 0x3333333333333333u64;
            r |= r >> 1;
            r &= 0x5555555555555555u64;
            // make sure to mask out bits outside the string lengths
            res += (r & ((1u64 << ((words_rem as u64) << 3u64)) - 1u64)).count_ones();
        }

        res
    }
}

pub fn hamming_words_128(a: &[u8], b: &[u8]) -> u32 {
    assert!(a.len() == b.len());

    unsafe {
        let mut res = 0u32;
        // the pointer address better be aligned for u128
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

        let words_rem = a.len() & 15;

        if words_rem > 0 {
            let mut r = (*a_ptr.offset(words_len)) ^ (*b_ptr.offset(words_len));
            r |= r >> 4;
            r &= 0x0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0fu128;
            r |= r >> 2;
            r &= 0x33333333333333333333333333333333u128;
            r |= r >> 1;
            r &= 0x55555555555555555555555555555555u128;
            // make sure to mask out bits outside the string lengths
            res += (r & ((1u128 << ((words_rem as u128) << 3u128)) - 1u128)).count_ones();
        }

        res
    }
}

pub fn hamming_simd_parallel(a: &[u8], b: &[u8]) -> u32 {
    assert!(a.len() == b.len());

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe {hamming_simd_parallel_x86_avx2(a, b)};
        }
    }

    hamming_naive(a, b)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn hamming_simd_parallel_x86_avx2(a: &[u8], b: &[u8]) -> u32 {
    let len = a.len();
    let refresh_len = (len / (255 * 32)) as isize;
    let a_ptr = a.as_ptr() as *const __m256i;
    let b_ptr = b.as_ptr() as *const __m256i;
    let zeros = _mm256_setzero_si256();
    let mut sad = zeros;

    for i in 0..refresh_len {
        let mut curr = zeros;

        for j in (i * 255)..((i + 1) * 255) {
            let a = _mm256_loadu_si256(a_ptr.offset(j));
            let b = _mm256_loadu_si256(b_ptr.offset(j));
            let r = _mm256_cmpeq_epi8(a, b);
            curr = _mm256_subs_epu8(curr, r); // subtract -1 = add 1 when matching
            // counting matches instead of mismatches for speed
        }

        // subtract 0 and sum up 8 bytes at once horizontally into four 64 bit ints
        // accumulate those 64 bit ints
        sad = _mm256_add_epi64(sad, _mm256_sad_epu8(curr, zeros));
    }

    let word_len = len >> 5;
    let mut curr = zeros;

    // leftover blocks of 32 bytes
    for i in (refresh_len * 255)..word_len as isize {
        let a = _mm256_loadu_si256(a_ptr.offset(i));
        let b = _mm256_loadu_si256(b_ptr.offset(i));
        let r = _mm256_cmpeq_epi8(a, b);
        curr = _mm256_subs_epu8(curr, r); // subtract -1 = add 1 when matching
    }

    sad = _mm256_add_epi64(sad, _mm256_sad_epu8(curr, zeros));
    let mut sad_arr = [0u32; 8];
    _mm256_storeu_si256(sad_arr.as_mut_ptr() as *mut __m256i, sad);
    let mut res = sad_arr[0] + sad_arr[2] + sad_arr[4] + sad_arr[6];

    // leftover characters
    for i in (word_len << 5)..len {
        res += (*a.get_unchecked(i) == *b.get_unchecked(i)) as u32;
    }

    len as u32 - res
}

pub fn hamming_simd_movemask(a: &[u8], b: &[u8]) -> u32 {
    assert!(a.len() == b.len());

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe {hamming_simd_movemask_x86_avx2(a, b)};
        }
    }

    hamming_naive(a, b)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn hamming_simd_movemask_x86_avx2(a: &[u8], b: &[u8]) -> u32 {
    let mut res = 0u32;
    let len = a.len();
    let word_len = len >> 5;
    let a_ptr = a.as_ptr() as *const __m256i;
    let b_ptr = b.as_ptr() as *const __m256i;

    for i in 0..word_len as isize {
        // unaligned, so must use loadu
        let a = _mm256_loadu_si256(a_ptr.offset(i));
        let b = _mm256_loadu_si256(b_ptr.offset(i));
        // directly check if equal
        let r = _mm256_cmpeq_epi8(a, b);
        // pack the 32 bytes into a 32-bit int and count not equal
        res += _mm256_movemask_epi8(r).count_zeros();
    }

    // leftover characters
    for i in (word_len << 5)..len {
        res += (*a.get_unchecked(i) != *b.get_unchecked(i)) as u32;
    }

    res
}

pub fn hamming_search_simd(needle: &[u8], haystack: &[u8]) -> Vec<Match> {
    assert!(needle.len() <= 32);
    assert!(needle.len() <= haystack.len());

    if needle.len() == 0 {
        return vec![];
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe {hamming_search_simd_x86_avx2(needle, haystack, needle.len() as u32, true)};
        }
    }

    hamming_search_naive_k(needle, haystack, needle.len() as u32, true)
}

pub fn hamming_search_simd_k(needle: &[u8], haystack: &[u8], k: u32, best: bool) -> Vec<Match> {
    assert!(needle.len() <= 32);
    assert!(needle.len() <= haystack.len());

    if needle.len() == 0 {
        return vec![];
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe {hamming_search_simd_x86_avx2(needle, haystack, k, best)};
        }
    }

    hamming_search_naive_k(needle, haystack, k, best)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn hamming_search_simd_x86_avx2(needle: &[u8], haystack: &[u8], k: u32, best: bool) -> Vec<Match> {
    let needle_len = needle.len();
    let haystack_len = haystack.len();
    let real_len = haystack_len + 1 - needle_len;
    let mut res = Vec::with_capacity(real_len >> 2);
    // length if needle is a block of 32 bytes
    let len = if haystack_len < 31 {0} else {haystack_len - 31};
    let haystack_ptr = haystack.as_ptr();
    let mask = (1 << needle_len) - 1;
    let mut curr_k = k;
    let a = {
        if needle_len == 32 {
            _mm256_loadu_si256(needle.as_ptr() as *const __m256i)
        }else{ // padding
            let mut arr = [0u8; 32];
            fill_str(&mut arr, needle);
            _mm256_loadu_si256(arr.as_ptr() as *const __m256i)
        }
    };

    // do blocks of 32 bytes at once
    for i in 0..len {
        let b = _mm256_loadu_si256(haystack_ptr.offset(i as isize) as *const __m256i);
        let r = _mm256_movemask_epi8(_mm256_cmpeq_epi8(a, b));
        let final_res = ((!r) & mask).count_ones(); // mask out characters not in needle

        if final_res <= curr_k {
            res.push(Match{start: i, end: i + needle_len, k: final_res});

            if best {
                curr_k = final_res;
            }
        }
    }

    // do leftover characters
    'outer: for i in len..real_len {
        let mut final_res = 0u32;

        for j in 0..needle_len {
            final_res += (*needle.get_unchecked(j) != *haystack.get_unchecked(i + j)) as u32;

            if final_res > curr_k {
                continue 'outer;
            }
        }

        res.push(Match{start: i, end: i + needle_len, k: final_res});

        if best {
            curr_k = final_res;
        }
    }

    if best {
        res.retain(|m| m.k == curr_k);
    }

    res
}

pub fn levenshtein_naive(a: &[u8], b: &[u8], trace_on: bool) -> (u32, Option<Vec<Edit>>) {
    let swap = a.len() > b.len(); // swap so that a len <= b len
    let a_new = if swap {b} else {a};
    let a_new_len = a_new.len();
    let b_new = if swap {a} else {b};
    let b_new_len = b_new.len();

    let len = a_new_len + 1;
    let mut dp1 = vec![0u32; len]; // in each iteration, dp1 is already calculated
    let mut dp2 = vec![0u32; len]; // dp2 the currently calculated column
    let mut traceback = if trace_on {vec![0u8; (b_new_len + 1) * len]} else {vec![]};

    for i in 0..len {
        dp1[i] = i as u32;

        if trace_on {
            traceback[0 * len + i] = 2u8;
        }
    }

    for i in 1..(b_new_len + 1) {
        dp2[0] = i as u32;

        if trace_on {
            traceback[i * len + 0] = 1u8;
        }

        for j in 1..len {
            let sub = dp1[j - 1] + (a_new[j - 1] != b_new[i - 1]) as u32;
            let a_gap = dp1[j] + 1;
            let b_gap = dp2[j - 1] + 1;
            let traceback_idx = i * len + j;

            dp2[j] = a_gap;

            if trace_on {
                traceback[traceback_idx] = 1u8;
            }

            if b_gap < dp2[j] {
                dp2[j] = b_gap;

                if trace_on {
                    traceback[traceback_idx] = 2u8;
                }
            }

            if sub <= dp2[j] {
                dp2[j] = sub;

                if trace_on {
                    traceback[traceback_idx] = 0u8;
                }
            }
        }

        let temp = dp1;
        dp1 = dp2;
        dp2 = temp;
    }

    if trace_on {
        let mut res = Vec::with_capacity(a_new_len + b_new_len);
        let mut i = b_new_len;
        let mut j = a_new_len;

        while i > 0 || j > 0 {
            let edit = traceback[i * len + j];

            match edit {
                0 => {
                    res.push(if a_new[j - 1] == b_new[i - 1] {Edit::Match} else {Edit::Mismatch});
                    i -= 1;
                    j -= 1;
                },
                1 => {
                    res.push(if swap {Edit::BGap} else {Edit::AGap});
                    i -= 1;
                },
                2 => {
                    res.push(if swap {Edit::AGap} else {Edit::BGap});
                    j -= 1;
                },
                _ => {
                    panic!("This should not be reached!");
                }
            }
        }

        res.reverse();
        (dp1[a_new_len], Some(res))
    }else{
        (dp1[a_new_len], None)
    }
}

pub fn levenshtein_naive_k(a: &[u8], b: &[u8], k: u32, trace_on: bool) -> (u32, Option<Vec<Edit>>) {
    let swap = a.len() > b.len(); // swap so that a len <= b len
    let a_new = if swap {b} else {a};
    let a_new_len = a_new.len();
    let b_new = if swap {a} else {b};
    let b_new_len = b_new.len();
    let k_usize = k as usize;

    if b_new_len - a_new_len > k_usize {
        return ((b_new_len - a_new_len) as u32, None);
    }

    let len = a_new_len + 1;
    let mut lo = 0usize;
    let mut hi = std::cmp::min(k_usize + 1, b_new_len + 1);
    let mut prev_lo;
    let mut prev_hi;
    let k_len = std::cmp::min((k_usize << 1) + 1, b_new_len + 1);
    let mut dp1 = vec![0u32; k_len]; // in each iteration, dp1 is already calculated
    let mut dp2 = vec![0u32; k_len]; // dp2 the currently calculated row
    let mut traceback = if trace_on {vec![0u8; len * k_len]} else {vec![]};

    for i in 0..(hi - lo) {
        dp1[i] = i as u32;

        if trace_on {
            traceback[0 * k_len + i] = 1u8;
        }
    }

    for i in 1..len {
        prev_lo = lo;
        prev_hi = hi;
        hi = std::cmp::min(hi + 1, b_new_len + 1);

        if i > k_usize {
            lo += 1;
        }

        for j in 0..(hi - lo) {
            let idx = lo + j;
            let sub = {
                if idx == 0 {
                    u32::max_value()
                }else{
                    dp1[idx - 1 - prev_lo] + (a_new[i - 1] != b_new[idx - 1]) as u32
                }
            };
            let a_gap = if j == 0 {u32::max_value()} else {dp2[j - 1] + 1};
            let b_gap = if idx >= prev_hi {u32::max_value()} else {dp1[idx - prev_lo] + 1};

            dp2[j] = sub;

            let traceback_idx = i * k_len + j;

            if trace_on {
                traceback[traceback_idx] = 0u8;
            }

            if a_gap < dp2[j] {
                dp2[j] = a_gap;

                if trace_on {
                    traceback[traceback_idx] = 1u8;
                }
            }

            if b_gap < dp2[j] {
                dp2[j] = b_gap;

                if trace_on {
                    traceback[traceback_idx] = 2u8;
                }
            }
        }

        let temp = dp1;
        dp1 = dp2;
        dp2 = temp;
    }

    if !trace_on || dp1[hi - lo - 1] > k {
        return (dp1[hi - lo - 1], None);
    }

    let mut res = Vec::with_capacity(a_new_len + b_new_len);
    let mut i = a_new_len;
    let mut j = b_new_len;

    while i > 0 || j > 0 {
        let edit = traceback[i * k_len + (j - (if i > k_usize {i - k_usize} else {0}))];

        match edit {
            0 => {
                res.push(if a_new[i - 1] == b_new[j - 1] {Edit::Match} else {Edit::Mismatch});
                i -= 1;
                j -= 1;
            },
            1 => {
                res.push(if swap {Edit::BGap} else {Edit::AGap});
                j -= 1;
            },
            2 => {
                res.push(if swap {Edit::AGap} else {Edit::BGap});
                i -= 1;
            },
            _ => {
                panic!("This should not be reached!");
            }
        }
    }

    res.reverse();
    (dp1[hi - lo - 1], Some(res))
}

pub fn levenshtein_simd_k(a: &[u8], b: &[u8], k: u32, trace_on: bool) -> (u32, Option<Vec<Edit>>) {
    assert!(k <= 30);

    if a.len() == 0 && b.len() == 0 {
        return if trace_on {(0u32, Some(vec![]))} else {(0u32, None)};
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe {levenshtein_simd_x86_avx2(a, b, k, trace_on)};
        }
    }

    levenshtein_naive_k(a, b, k, trace_on)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn levenshtein_simd_x86_avx2(a_old: &[u8], b_old: &[u8], k: u32, trace_on: bool) -> (u32, Option<Vec<Edit>>) {
    // swap a and b so that a is shorter than b, if applicable
    // makes operations later on slightly easier, since length of a <= length of b
    let swap = a_old.len() > b_old.len();
    let a = if swap {b_old} else {a_old};
    let a_len = a.len();
    let b = if swap {a_old} else {b_old};
    let b_len = b.len();

    // lengths of the (anti) diagonals
    // constant length for simplicity because similar speeds even if this is shorter
    let k1 = 31usize;
    let k1_div2 = k1 >> 1;
    let k2 = 30usize;
    let k2_div2 = k2 >> 1;

    if b_len - a_len > k as usize {
        return ((b_len - a_len) as u32, None);
    }

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
            *a_k1_window_arr.get_unchecked_mut(k1_div2 - 1 - i) = *a.get_unchecked(i);
        }

        // unaligned, so must use loadu
        _mm256_loadu_si256(a_k1_window_arr.as_ptr() as *const __m256i)
    };

    let mut b_k1_window = {
        let mut b_k1_window_arr = [0u8; 32];

        for i in 0..std::cmp::min(k1_div2, b_len) {
            *b_k1_window_arr.get_unchecked_mut(k1_div2 + 1 + i) = *b.get_unchecked(i);
        }

        _mm256_loadu_si256(b_k1_window_arr.as_ptr() as *const __m256i)
    };

    let mut a_k2_window = {
        let mut a_k2_window_arr = [0u8; 32];

        for i in 0..std::cmp::min(k2_div2, a_len) {
            *a_k2_window_arr.get_unchecked_mut(k2_div2 - 1 - i) = *a.get_unchecked(i);
        }

        _mm256_loadu_si256(a_k2_window_arr.as_ptr() as *const __m256i)
    };

    let mut b_k2_window = {
        let mut b_k2_window_arr = [0u8; 32];

        for i in 0..std::cmp::min(k2_div2, b_len) {
            *b_k2_window_arr.get_unchecked_mut(k2_div2 + i) = *b.get_unchecked(i);
        }

        _mm256_loadu_si256(b_k2_window_arr.as_ptr() as *const __m256i)
    };

    // used to keep track of the next characters to place in the windows
    let mut k1_idx = k1_div2 - 1;
    let mut k2_idx = k2_div2 - 1;

    // reusable constants
    let ones = _mm256_set1_epi8(1i8);
    // reversed bytes for setting highest/lowest byte to max value
    let end_max = _mm256_set_epi8(127i8, 0i8, 0i8, 0i8, 0i8, 0i8, 0i8, 0i8, 0i8, 0i8, 0i8, 0i8, 0i8, 0i8, 0i8, 0i8, 0i8, 0i8, 0i8, 0i8, 0i8, 0i8, 0i8, 0i8, 0i8, 0i8, 0i8, 0i8, 0i8, 0i8, 0i8, 0i8);
    let start_max = _mm256_set_epi8(0i8, 0i8, 0i8, 0i8, 0i8, 0i8, 0i8, 0i8, 0i8, 0i8, 0i8, 0i8, 0i8, 0i8, 0i8, 0i8, 0i8, 0i8, 0i8, 0i8, 0i8, 0i8, 0i8, 0i8, 0i8, 0i8, 0i8, 0i8, 0i8, 0i8, 0i8, 127i8);

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
    let mut traceback_arr = if trace_on {vec![0u8; (len + (len & 1)) * 32]} else {vec![]};

    if trace_on {
        *traceback_arr.get_unchecked_mut(1 * 32 + (k2_div2 - 1)) = 2u8;
        *traceback_arr.get_unchecked_mut(1 * 32 + k2_div2) = 1u8;
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
            a_k1_window = _mm256_insert_epi8(a_k1_window, *a.get_unchecked(k1_idx) as i8, 0i32);
        }

        b_k1_window = shift_left_x86_avx2(b_k1_window);

        if k1_idx < b_len {
            b_k1_window = _mm256_insert_epi8(b_k1_window, *b.get_unchecked(k1_idx) as i8, 30i32); // k1 - 1
        }

        a_k2_window = shift_right_x86_avx2(a_k2_window);

        if k2_idx < a_len {
            a_k2_window = _mm256_insert_epi8(a_k2_window, *a.get_unchecked(k2_idx) as i8, 0i32);
        }

        b_k2_window = shift_left_x86_avx2(b_k2_window);

        if k2_idx < b_len {
            b_k2_window = _mm256_insert_epi8(b_k2_window, *b.get_unchecked(k2_idx) as i8, 29i32); // k2 - 1
        }

        // (anti) diagonal that matches in the a and b windows
        let match_mask_k1 = _mm256_cmpeq_epi8(a_k1_window, b_k1_window);
        // add negative ones to cells that have matching characters from a and b
        let sub_k1 = _mm256_adds_epi8(dp1, match_mask_k1);
        // cost of gaps in a
        let a_gap_k1 = {
            let a_gap_prev = shift_right_x86_avx2(dp2);
            _mm256_or_si256(a_gap_prev, start_max) // shift in max value
        };
        // cost of gaps in b: dp2

        dp1 = dp2;

        // min of the cost of all three edit operations
        if trace_on {
            let min = triple_argmin_x86_avx2(sub_k1, a_gap_k1, dp2, ones);
            dp2 = _mm256_adds_epi8(min.0, ones);
            _mm256_storeu_si256(traceback_arr.as_mut_ptr().offset(((i << 1) * 32) as isize) as *mut __m256i, min.1);
        }else{
            dp2 = _mm256_adds_epi8(_mm256_min_epi8(sub_k1, _mm256_min_epi8(a_gap_k1, dp2)), ones);
        }

        // (anti) diagonal that matches in the a and b windows
        let match_mask_k2 = _mm256_cmpeq_epi8(a_k2_window, b_k2_window);
        // add negative ones to cells that have matching characters from a and b
        let sub_k2 = _mm256_adds_epi8(dp1, match_mask_k2);
        // cost of gaps in b
        let b_gap_k2 = {
            let b_gap_prev = shift_left_x86_avx2(dp2);
            _mm256_or_si256(b_gap_prev, end_max) // k1, shift in max value
        };
        // cost of gaps in a: dp2

        dp1 = dp2;

        // min of the cost of all three edit operations
        if trace_on {
            let min = triple_argmin_x86_avx2(sub_k2, dp2, b_gap_k2, ones);
            dp2 = _mm256_adds_epi8(min.0, ones);
            _mm256_storeu_si256(traceback_arr.as_mut_ptr().offset((((i << 1) + 1) * 32) as isize) as *mut __m256i, min.1);
        }else{
            dp2 = _mm256_adds_epi8(_mm256_min_epi8(sub_k2, _mm256_min_epi8(dp2, b_gap_k2)), ones);
        }
    }

    let mut final_arr = [0u8; 32];

    if ends_with_k2 {
        _mm256_storeu_si256(final_arr.as_mut_ptr() as *mut __m256i, dp2);
    }else{
        _mm256_storeu_si256(final_arr.as_mut_ptr() as *mut __m256i, dp1);
    }

    let final_res = final_arr[final_idx] as u32;

    if !trace_on || final_res > k {
        return (final_res, None);
    }

    (final_res, Some(traceback(&traceback_arr, final_idx, a, b, swap, ends_with_k2)))
}

unsafe fn traceback(arr: &[u8], mut idx: usize, a: &[u8], b: &[u8], swap: bool, mut is_k2: bool) -> Vec<Edit> {
    // keep track of position in traditional dp array and strings
    let mut i = a.len(); // index in a
    let mut j = b.len(); // index in b

    // last diagonal may overshoot, so ignore it
    let mut arr_idx = (arr.len() >> 5) - 1 - (if is_k2 {0} else {1});
    let mut res = Vec::with_capacity(a.len() + b.len());

    while arr_idx > 0 {
        let edit = *arr.get_unchecked(arr_idx * 32 + idx);

        match edit {
            0u8 => { // match/mismatch
                res.push(if *a.get_unchecked(i - 1) == *b.get_unchecked(j - 1) {Edit::Match} else {Edit::Mismatch});
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

pub fn levenshtein_exp(a: &[u8], b: &[u8], trace_on: bool) -> (u32, Option<Vec<Edit>>) {
    let mut k = 30;
    let mut res = levenshtein_simd_k(a, b, k, false);

    while res.0 > k {
        k <<= 2; // multiply by 4 every time (instead of the usual multiple by 2) for speed
        res = levenshtein_naive_k(a, b, k, false);
    }

    if trace_on { // save memory by only calculating the traceback at the end, with an extra step
        if res.0 <= 30 {levenshtein_simd_k(a, b, res.0, true)} else {levenshtein_naive_k(a, b, res.0, true)}
    }else{
        res
    }
}

pub fn levenshtein_search_naive(needle: &[u8], haystack: &[u8]) -> Vec<Match> {
    levenshtein_search_naive_k(needle, haystack, needle.len() as u32, true)
}

pub fn levenshtein_search_naive_k(needle: &[u8], haystack: &[u8], k: u32, best: bool) -> Vec<Match> {
    let needle_len = needle.len();
    let haystack_len = haystack.len();

    if needle_len == 0 {
        return vec![];
    }

    let len = needle_len + 1;
    let mut dp1 = vec![0u32; len];
    let mut dp2 = vec![0u32; len];
    let mut length1 = vec![0usize; len];
    let mut length2 = vec![0usize; len];
    let mut res = Vec::with_capacity(haystack_len >> 2);
    let mut curr_k = k;

    for i in 0..len {
        dp1[i] = i as u32;
    }

    if dp1[len - 1] <= curr_k {
        res.push(Match{start: 0, end: 0, k: dp1[len - 1]});

        if best {
            curr_k = dp1[len - 1];
        }
    }

    for i in 0..haystack_len {
        dp2[0] = 0;
        length2[0] = 0;

        for j in 1..len {
            let sub = dp1[j - 1] + (needle[j - 1] != haystack[i]) as u32;
            let a_gap = dp1[j] + 1;
            let b_gap = dp2[j - 1] + 1;

            dp2[j] = a_gap;
            length2[j] = length1[j] + 1;

            if (b_gap < dp2[j]) || (b_gap == dp2[j] && length2[j - 1] > length2[j]) {
                dp2[j] = b_gap;
                length2[j] = length2[j - 1];
            }

            if (sub < dp2[j]) || (sub == dp2[j] && (length1[j - 1] + 1) > length2[j]) {
                dp2[j] = sub;
                length2[j] = length1[j - 1] + 1;
            }
        }

        let final_res = dp2[len - 1];

        if final_res <= curr_k {
            res.push(Match{start: i + 1 - length2[len - 1], end: i + 1, k: final_res});

            if best {
                curr_k = final_res;
            }
        }

        let temp_dp = dp1;
        dp1 = dp2;
        dp2 = temp_dp;

        let temp_length = length1;
        length1 = length2;
        length2 = temp_length;
    }

    if best {
        res.retain(|m| m.k == curr_k);
    }

    res
}

pub fn levenshtein_search_simd(needle: &[u8], haystack: &[u8]) -> Vec<Match> {
    assert!(needle.len() <= 32);

    if needle.len() == 0 {
        return vec![];
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe {levenshtein_search_simd_x86_avx2(needle, haystack, needle.len() as u32, true)};
        }
    }

    levenshtein_search_naive_k(needle, haystack, needle.len() as u32, true)
}

pub fn levenshtein_search_simd_k(needle: &[u8], haystack: &[u8], k: u32, best: bool) -> Vec<Match> {
    assert!(needle.len() <= 32);

    if needle.len() == 0 {
        return vec![];
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe {levenshtein_search_simd_x86_avx2(needle, haystack, k, best)};
        }
    }

    levenshtein_search_naive_k(needle, haystack, k, best)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn levenshtein_search_simd_x86_avx2(needle: &[u8], haystack: &[u8], k: u32, best: bool) -> Vec<Match> {
    let needle_len = needle.len();
    let haystack_len = haystack.len();
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

    let mut res = Vec::with_capacity(haystack_len >> 2);

    let needle_window = {
        let mut needle_window_arr = [0u8; 32];

        // needle window is in reverse order
        for i in 0..std::cmp::min(needle_len, 32) {
            *needle_window_arr.get_unchecked_mut(31 - i) = *needle.get_unchecked(i);
        }

        _mm256_loadu_si256(needle_window_arr.as_ptr() as *const __m256i)
    };

    let mut haystack_window = _mm256_setzero_si256();
    let mut haystack_idx = 0usize;
    let mut curr_k = k;

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
            haystack_window = _mm256_insert_epi8(haystack_window, *haystack.get_unchecked(haystack_idx) as i8, 31i32);
            haystack_idx += 1;
        }

        let match_mask = _mm256_cmpeq_epi8(needle_window, haystack_window);

        // match/mismatch
        let mut sub = shift_left_x86_avx2(dp1); // zeros are shifted in
        sub = _mm256_adds_epi8(sub, match_mask); // add -1 if match

        let sub_length = _mm256_add_epi8(shift_left_x86_avx2(length1), ones);

        // gap in needle: dp2
        let needle_gap_length = _mm256_add_epi8(length2, ones);

        // gap in haystack
        let haystack_gap = shift_left_x86_avx2(dp2); // zeros are shifted in
        let haystack_gap_length = shift_left_x86_avx2(length2);

        dp1 = dp2;
        length1 = length2;

        let min = triple_min_length_x86_avx2(sub, dp2, haystack_gap, sub_length, needle_gap_length, haystack_gap_length);
        dp2 = _mm256_adds_epi8(min.0, ones);
        length2 = min.1;

        if i >= needle_len - 1 {
            // manually storing the dp and length arrays is necessary
            // because the accessed index is not a compile time constant
            _mm256_storeu_si256(dp_arr_ptr, dp2);
            let final_res = *dp_arr.get_unchecked(final_idx) as u32;

            _mm256_storeu_si256(length_arr_ptr, length2);

            if final_res <= curr_k {
                let end_idx = i + 1 - needle_len;
                res.push(Match{start: end_idx - (*length_arr.get_unchecked(final_idx) as usize), end: end_idx, k: final_res});

                if best { // if we want the best, then we can shrink the k threshold
                    curr_k = final_res;
                }
            }
        }
    }

    if best {
        res.retain(|m| m.k == curr_k); // only retain matches with the lowest k
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
unsafe fn triple_argmin_x86_avx2(sub: __m256i, a_gap: __m256i, b_gap: __m256i, ones: __m256i) -> (__m256i, __m256i) {
    // return the edit used in addition to doing a min operation
    // hide latency by minimizing dependencies
    let res_min = _mm256_min_epi8(a_gap, b_gap);
    let a_gap_mask = _mm256_cmpgt_epi8(a_gap, b_gap);
    let res_arg = _mm256_subs_epi8(ones, a_gap_mask); // a gap: 1 - 0 = 1, b gap: 1 - -1 = 2

    let res_min2 = _mm256_min_epi8(sub, res_min);
    let sub_mask = _mm256_cmpgt_epi8(sub, res_min);
    let res_arg2 = _mm256_and_si256(sub_mask, res_arg); // sub: 0

    return (res_min2, res_arg2);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn triple_min_length_x86_avx2(sub: __m256i, a_gap: __m256i, b_gap: __m256i, sub_length: __m256i, a_gap_length: __m256i, b_gap_length: __m256i) -> (__m256i, __m256i) {
    // choose the length based on which edit is chosen during the min operation
    // hide latency by minimizing dependencies
    // secondary objective of maximizing length if edit costs equal
    let res_min = _mm256_min_epi8(a_gap, b_gap);
    let a_gap_mask = _mm256_cmpgt_epi8(a_gap, b_gap); // a gap: 0, b gap: -1
    let mut res_length = _mm256_blendv_epi8(a_gap_length, b_gap_length, a_gap_mask); // lengths based on edits
    let a_b_eq_mask = _mm256_cmpeq_epi8(a_gap, b_gap); // equal: -1
    let a_b_max_len = _mm256_max_epi8(a_gap_length, b_gap_length);
    res_length = _mm256_blendv_epi8(res_length, a_b_max_len, a_b_eq_mask); // maximize length if edits equal

    let res_min2 = _mm256_min_epi8(sub, res_min);
    let sub_mask = _mm256_cmpgt_epi8(sub, res_min); // sub: 0, prev a or b gap: -1
    let mut res_length2 = _mm256_blendv_epi8(sub_length, res_length, sub_mask); // length based on edits
    let sub_eq_mask = _mm256_cmpeq_epi8(sub, res_min);
    let sub_max_len = _mm256_max_epi8(sub_length, res_length);
    res_length2 = _mm256_blendv_epi8(res_length2, sub_max_len, sub_eq_mask);

    return (res_min2, res_length2);
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

