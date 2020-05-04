use super::*;
use super::jewel::*;

/// Returns the hamming distance between two strings by naively counting mismatches.
///
/// The length of `a` and `b` must be the same.
///
/// # Arguments
/// * `a` - first string (slice)
/// * `b` - second string (slice)
///
/// # Panics
/// * If the length of `a` does not equal the length of `b`.
///
/// # Example
/// ```
/// # use triple_accel::*;
///
/// let dist = hamming_naive(b"abc", b"abd");
///
/// assert!(dist == 1);
/// ```
pub fn hamming_naive(a: &[u8], b: &[u8]) -> u32 {
    let len = a.len();
    assert!(len == b.len());

    let mut res = 0u32;

    for i in 0..len {
        res += (a[i] != b[i]) as u32;
    }

    res
}

/// Returns a vector of best `Match`s by naively searching through the text `haystack` for the pattern `needle`.
///
/// This is done by naively counting mismatches at every position in `haystack`.
/// Only the matches with the lowest Hamming distance are returned.
/// The length of `needle` must be less than or equal to the length of `haystack`.
///
/// # Arguments
/// * `needle` - pattern string (slice)
/// * `haystack` - text string (slice)
///
/// # Example
/// ```
/// # use triple_accel::*;
///
/// let matches = hamming_search_naive(b"abc", b"  abd");
///
/// assert!(matches == vec![Match{start: 2, end: 5, k: 1}]);
/// ```
pub fn hamming_search_naive(needle: &[u8], haystack: &[u8]) -> Vec<Match> {
    hamming_search_naive_with_opts(needle, haystack, needle.len() as u32, SearchType::Best)
}

/// Returns a vector of `Match`s by naively searching through the text `haystack` for the pattern `needle`.
///
/// Only matches with less than `k` mismatches are returned.
/// This is done by naively counting mismatches at every position in `haystack`.
/// The length of `needle` must be less than or equal to the length of `haystack`.
///
/// # Arguments
/// * `needle` - pattern string (slice)
/// * `haystack` - text string (slice)
/// * `k` - number of mismatches allowed
/// * `search_type` - whether to only return the "best" matches with the lowest Hamming distance, or
/// the first match that is encountered
///
/// # Example
/// ```
/// # use triple_accel::*;
///
/// let matches = hamming_search_naive_with_opts(b"abc", b"  abd", 1, SearchType::All);
///
/// assert!(matches == vec![Match{start: 2, end: 5, k: 1}]);
/// ```
pub fn hamming_search_naive_with_opts(needle: &[u8], haystack: &[u8], k: u32, search_type: SearchType) -> Vec<Match> {
    let needle_len = needle.len();
    let haystack_len = haystack.len();

    if needle_len > haystack_len {
        return vec![];
    }

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

        match search_type {
            SearchType::First => break,
            SearchType::Best => curr_k = final_res,
            _ => ()
        }
    }

    if search_type == SearchType::Best {
        res.retain(|m| m.k == curr_k);
    }

    res
}

/// Returns the hamming distance between two strings by efficiently counting mismatches in chunks of 64 bits.
///
/// The length of `a` and `b` must be the same.
/// Both `a` and `b` must be aligned and padded so they can be directly casted to chunks of `u64`.
/// Use `alloc_str` to create aligned and padded strings.
/// This should be faster than `hamming_naive` and maybe even `hamming_words_128`. This should be slower
/// than `hamming_simd_parallel/movemask`.
///
/// # Arguments
/// * `a` - first string (slice)
/// * `b` - second string (slice)
///
/// # Panics
/// * If the length of `a` does not equal the length of `b`.
///
/// # Example
/// ```
/// # use triple_accel::*;
///
/// let mut a = alloc_str(3);
/// let mut b = alloc_str(3);
/// fill_str(&mut a, b"abc");
/// fill_str(&mut b, b"abd");
///
/// let dist = hamming_words_64(&a, &b);
///
/// assert!(dist == 1);
/// ```
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

/// Returns the hamming distance between two strings by counting mismatches in chunks of 128 bits.
///
/// The length of `a` and `b` must be the same.
/// Both `a` and `b` must be aligned and padded so they can be directly casted to chunks of `u128`.
/// Use `alloc_str` to create aligned and padded strings.
/// This may be slower than `hamming_words_64` in practice, probably since Rust `u128` is not as
/// optimized. This should be slower than `hamming_simd_parallel/movemask`.
///
/// # Arguments
/// * `a` - first string (slice)
/// * `b` - second string (slice)
///
/// # Panics
/// * If the length of `a` does not equal the length of `b`.
///
/// # Example
/// ```
/// # use triple_accel::*;
///
/// let mut a = alloc_str(3);
/// let mut b = alloc_str(3);
/// fill_str(&mut a, b"abc");
/// fill_str(&mut b, b"abd");
///
/// let dist = hamming_words_128(&a, &b);
///
/// assert!(dist == 1);
/// ```
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

/// Returns the hamming distance between two strings by counting mismatches in chunks of 256 bits, by using AVX2 to increment multiple counters in parallel.
///
/// The length of `a` and `b` must be the same.
/// There are no constraints on how `a` and `b` are aligned and padded.
/// This should be faster than both `hamming_word_64/128` and `hamming_simd_movemask`.
///
/// # Arguments
/// * `a` - first string (slice)
/// * `b` - second string (slice)
///
/// # Panics
/// * If the length of `a` does not equal the length of `b`.
///
/// # Example
/// ```
/// # use triple_accel::*;
///
/// let dist = hamming_simd_parallel(b"abc", b"abd");
///
/// assert!(dist == 1);
/// ```
pub fn hamming_simd_parallel(a: &[u8], b: &[u8]) -> u32 {
    assert!(a.len() == b.len());

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe {hamming_simd_parallel_core::<AvxNx32x8>(a, b)};
        }
    }

    hamming_naive(a, b)
}

unsafe fn hamming_simd_parallel_core<T: Jewel>(a: &[u8], b: &[u8]) -> u32 {
    T::count_mismatches(a.as_ptr(), b.as_ptr(), a.len())
}

/// Returns the hamming distance between two strings by counting mismatches in chunks of 256 bits, by using AVX2's movemask instruction.
///
/// The length of `a` and `b` must be the same.
/// There are no constraints on how `a` and `b` are aligned and padded.
/// This should be faster than `hamming_word_64/128`, but slower than `hamming_simd_parallel`.
///
/// # Arguments
/// * `a` - first string (slice)
/// * `b` - second string (slice)
///
/// # Panics
/// * If the length of `a` does not equal the length of `b`.
///
/// # Example
/// ```
/// # use triple_accel::*;
///
/// let dist = hamming_simd_movemask(b"abc", b"abd");
///
/// assert!(dist == 1);
/// ```
pub fn hamming_simd_movemask(a: &[u8], b: &[u8]) -> u32 {
    assert!(a.len() == b.len());

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe {hamming_simd_movemask_core::<AvxNx32x8>(a, b)};
        }
    }

    hamming_naive(a, b)
}

unsafe fn hamming_simd_movemask_core<T: Jewel>(a: &[u8], b: &[u8]) -> u32 {
    T::mm_count_mismatches(a.as_ptr(), b.as_ptr(), a.len())
}

/// Returns a vector of best `Match`s by searching through the text `haystack` for the pattern `needle` using AVX2.
///
/// This is done by using AVX2 to count mismatches at every position in `haystack`.
/// The length of `needle` must be less than or equal to the length of `haystack`. Additionally,
/// the length of needle must be less than or equal to 32.
/// Only the matches with the lowest Hamming distance are returned.
/// This should be faster than `hamming_search_naive`.
///
/// # Arguments
/// * `needle` - pattern string (slice)
/// * `haystack` - text string (slice)
///
/// # Panics
/// * If needle length is greater than 32.
///
/// # Example
/// ```
/// # use triple_accel::*;
///
/// let matches = hamming_search_simd(b"abc", b"  abd");
///
/// assert!(matches == vec![Match{start: 2, end: 5, k: 1}]);
/// ```
pub fn hamming_search_simd(needle: &[u8], haystack: &[u8]) -> Vec<Match> {
    hamming_search_simd_with_opts(needle, haystack, needle.len() as u32, SearchType::Best)
}

/// Returns a vector of `Match`s by searching through the text `haystack` for the pattern `needle` using AVX2.
///
/// This is done by using AVX2 to count mismatches at every position in `haystack`.
/// The length of `needle` must be less than or equal to the length of `haystack`. Additionally,
/// the length of needle must be less than or equal to 32.
/// This should be faster than `hamming_search_naive_k`.
///
/// # Arguments
/// * `needle` - pattern string (slice)
/// * `haystack` - text string (slice)
/// * `k` - number of mismatches allowed
/// * `search_type` - whether to only return the "best" matches with the lowest Hamming distance or
/// the first match that is encountered
///
/// # Panics
/// * If needle length is greater than 32.
///
/// # Example
/// ```
/// # use triple_accel::*;
///
/// let matches = hamming_search_simd_with_opts(b"abc", b"  abd", 1, SearchType::All);
///
/// assert!(matches == vec![Match{start: 2, end: 5, k: 1}]);
/// ```
pub fn hamming_search_simd_with_opts(needle: &[u8], haystack: &[u8], k: u32, search_type: SearchType) -> Vec<Match> {
    if needle.len() > haystack.len() {
        return vec![];
    }

    if needle.len() == 0 {
        return vec![];
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe {hamming_search_simd_core::<AvxNx32x8>(needle, haystack, k, search_type)};
        }
    }

    hamming_search_naive_with_opts(needle, haystack, k, search_type)
}

unsafe fn hamming_search_simd_core<T: Jewel>(needle: &[u8], haystack: &[u8], k: u32, search_type: SearchType) -> Vec<Match> {
    let needle_len = needle.len();
    let haystack_len = haystack.len();
    let needle_vector = T::loadu(needle.as_ptr(), needle_len);
    let len = if needle_vector.upper_bound() > haystack_len {0} else {haystack_len + 1 - needle_vector.upper_bound()};
    let real_len = haystack_len + 1 - needle_len;
    let mut res = Vec::with_capacity(len >> 2);
    let haystack_ptr = haystack.as_ptr();
    let mut curr_k = k;

    for i in 0..len {
        let final_res = T::vector_count_mismatches(&needle_vector, haystack_ptr.offset(i as isize));

        if final_res <= curr_k {
            res.push(Match{start: i, end: i + needle_len, k: final_res});

            match search_type {
                SearchType::First => break,
                SearchType::Best => curr_k = final_res,
                _ => ()
            }
        }
    }

    'outer: for i in len..real_len {
        let mut final_res = 0u32;

        for j in 0..needle_len {
            final_res += (*needle.get_unchecked(j) != *haystack.get_unchecked(i + j)) as u32;

            if final_res > curr_k {
                continue 'outer;
            }
        }

        res.push(Match{start: i, end: i + needle_len, k: final_res});

        match search_type {
            SearchType::First => break,
            SearchType::Best => curr_k = final_res,
            _ => ()
        }
    }

    if search_type == SearchType::Best {
        res.retain(|m| m.k == curr_k);
    }

    res
}

