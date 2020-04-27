use std;
use super::*;
use super::jewel::*;

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
            return unsafe {levenshtein_simd_core::<AvxNx32x8>(a, b, k, trace_on)};
        }
    }

    levenshtein_naive_k(a, b, k, trace_on)
}

unsafe fn levenshtein_simd_core<T: Jewel>(a_old: &[u8], b_old: &[u8], k: u32, trace_on: bool) -> (u32, Option<Vec<Edit>>) {
    // swap a and b so that a is shorter than b, if applicable
    // makes operations later on slightly easier, since length of a <= length of b
    let swap = a_old.len() > b_old.len();
    let a = if swap {b_old} else {a_old};
    let a_len = a.len();
    let b = if swap {a_old} else {b_old};
    let b_len = b.len();

    if b_len - a_len > k as usize {
        return ((b_len - a_len) as u32, None);
    }

    // initialized with max value of i8
    // must use saturated additions afterwards to not overflow
    let mut dp1 = T::repeating_max((k + 2) as usize);
    let mut dp2 = T::repeating_max(dp1.upper_bound());

    // lengths of the (anti) diagonals
    // assumes upper_bound is even
    let k1 = dp1.upper_bound() - 1;
    let k1_div2 = k1 >> 1;
    let k2 = dp1.upper_bound() - 2;
    let k2_div2 = k2 >> 1;

    // set dp[0][0] = 0
    dp1.slow_insert(k1_div2, 0);
    // set dp[0][1] = 1 and dp[1][0] = 1
    dp2.slow_insert(k2_div2 - 1, 1);
    dp2.slow_insert(k2_div2, 1);

    // a_k1_window and a_k2_window represent reversed portions of the string a
    // copy in half of k1/k2 number of characters
    // these characters are placed in the second half of b windows
    // since a windows are reversed, the characters are placed in reverse in the first half of b windows
    let mut a_k1_window = T::repeating(0, dp1.upper_bound());
    a_k1_window.slow_loadu(k1_div2 - 1, a.as_ptr(), std::cmp::min(k1_div2, a_len), true);

    let mut b_k1_window = T::repeating(0, dp1.upper_bound());
    b_k1_window.slow_loadu(k1_div2 + 1, b.as_ptr(), std::cmp::min(k1_div2, b_len), false);

    let mut a_k2_window = T::repeating(0, dp1.upper_bound());
    a_k2_window.slow_loadu(k2_div2 - 1, a.as_ptr(), std::cmp::min(k2_div2, a_len), true);

    let mut b_k2_window = T::repeating(0, dp1.upper_bound());
    b_k2_window.slow_loadu(k2_div2, b.as_ptr(), std::cmp::min(k2_div2, b_len), false);

    // used to keep track of the next characters to place in the windows
    let mut k1_idx = k1_div2 - 1;
    let mut k2_idx = k2_div2 - 1;

    // reusable constants
    let ones = T::repeating(1, dp1.upper_bound());

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
    let mut traceback_arr = if trace_on {Vec::with_capacity(len + (len & 1))} else {vec![]};

    if trace_on {
        traceback_arr.push(T::repeating(0, dp1.upper_bound()));
        traceback_arr.push(T::repeating(0, dp2.upper_bound()));
        traceback_arr.get_unchecked_mut(1).slow_insert(k2_div2 - 1, 2);
        traceback_arr.get_unchecked_mut(1).slow_insert(k2_div2, 1);
    }

    let mut sub = T::repeating(0, dp1.upper_bound());
    let mut gap = T::repeating(0, dp1.upper_bound());

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

    for _ in 1..len_div2 {
        // move indexes in strings forward
        k1_idx += 1;
        k2_idx += 1;

        // move windows for the strings a and b
        a_k1_window.shift_right_1_mut();

        if k1_idx < a_len {
            a_k1_window.insert_first(*a.get_unchecked(k1_idx) as u32);
        }

        b_k1_window.shift_left_1_mut();

        if k1_idx < b_len {
            b_k1_window.insert_last_1(*b.get_unchecked(k1_idx) as u32); // k1 - 1
        }

        a_k2_window.shift_right_1_mut();

        if k2_idx < a_len {
            a_k2_window.insert_first(*a.get_unchecked(k2_idx) as u32);
        }

        b_k2_window.shift_left_1_mut();

        if k2_idx < b_len {
            b_k2_window.insert_last_2(*b.get_unchecked(k2_idx) as u32);
        }

        // (anti) diagonal that matches in the a and b windows
        T::cmpeq(&a_k1_window, &b_k1_window, &mut sub);
        // add negative ones to cells that have matching characters from a and b
        sub.adds_mut(&dp1);
        // cost of gaps in a
        T::shift_right_1(&dp2, &mut gap);
        gap.insert_first_max();
        // cost of gaps in b: dp2

        // min of the cost of all three edit operations
        if trace_on {
            let args = T::triple_argmin(&sub, &gap, &dp2, &mut dp1);
            let temp = dp1;
            dp1 = dp2;
            dp2 = temp;
            dp2.adds_mut(&ones);
            traceback_arr.push(args);
        }else{
            T::min(&gap, &dp2, &mut dp1);
            dp1.min_mut(&sub);
            let temp = dp1;
            dp1 = dp2;
            dp2 = temp;
            dp2.adds_mut(&ones);
        }

        // (anti) diagonal that matches in the a and b windows
        T::cmpeq(&a_k2_window, &b_k2_window, &mut sub);
        // add negative ones to cells that have matching characters from a and b
        sub.adds_mut(&dp1);
        // cost of gaps in b
        T::shift_left_1(&dp2, &mut gap);
        gap.insert_last_max(); // k1, shift in max value
        // cost of gaps in a: dp2

        // min of the cost of all three edit operations
        if trace_on {
            let args = T::triple_argmin(&sub, &dp2, &gap, &mut dp1);
            let temp = dp1;
            dp1 = dp2;
            dp2 = temp;
            dp2.adds_mut(&ones);
            traceback_arr.push(args);
        }else{
            T::min(&dp2, &gap, &mut dp1);
            dp1.min_mut(&sub);
            let temp = dp1;
            dp1 = dp2;
            dp2 = temp;
            dp2.adds_mut(&ones);
        }
    }

    let final_res = if ends_with_k2 {dp2.slow_extract(final_idx)} else {dp1.slow_extract(final_idx)};

    if !trace_on || final_res > k {
        return (final_res, None);
    }

    (final_res, Some(traceback(&traceback_arr, final_idx, a, b, swap, ends_with_k2)))
}

unsafe fn traceback<T: Jewel>(arr: &[T], mut idx: usize, a: &[u8], b: &[u8], swap: bool, mut is_k2: bool) -> Vec<Edit> {
    // keep track of position in traditional dp array and strings
    let mut i = a.len(); // index in a
    let mut j = b.len(); // index in b

    // last diagonal may overshoot, so ignore it
    let mut arr_idx = arr.len() - 1 - (if is_k2 {0} else {1});
    let mut res = Vec::with_capacity(a.len() + b.len());

    while arr_idx > 0 {
        // each Jewel vector in arr is only visited once, so extract (which is costly) is fine
        let edit = arr.get_unchecked(arr_idx).slow_extract(idx);

        match edit {
            0 => { // match/mismatch
                res.push(if *a.get_unchecked(i - 1) == *b.get_unchecked(j - 1) {Edit::Match} else {Edit::Mismatch});
                arr_idx -= 2;
                i -= 1;
                j -= 1;
            },
            1 => { // a gap
                res.push(if swap {Edit::BGap} else {Edit::AGap}); // account for the swap in the beginning
                arr_idx -= 1;

                if !is_k2 {
                    idx -= 1;
                }

                j -= 1;
                is_k2 = !is_k2; // must account for alternating k1/k2 diagonals
            },
            2 => { // b gap
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
            return unsafe {levenshtein_search_simd_core::<AvxNx32x8>(needle, haystack, needle.len() as u32, true)};
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
            return unsafe {levenshtein_search_simd_core::<AvxNx32x8>(needle, haystack, k, best)};
        }
    }

    levenshtein_search_naive_k(needle, haystack, k, best)
}

unsafe fn levenshtein_search_simd_core<T: Jewel>(needle: &[u8], haystack: &[u8], k: u32, best: bool) -> Vec<Match> {
    let needle_len = needle.len();
    let haystack_len = haystack.len();
    let mut dp1 = T::repeating_max(needle_len);
    let mut dp2 = T::repeating_max(needle_len);
    dp2.insert_last_0(1); // last cell

    // save length instead of start idx due to int size constraints
    let mut length1 = T::repeating(0, needle_len);
    let mut length2 = T::repeating(0, needle_len);

    let ones = T::repeating(1, needle_len);

    let len = haystack_len + needle_len;
    let final_idx = dp1.upper_bound() - needle_len;

    let mut res = Vec::with_capacity(haystack_len >> 2);

    // load needle characters into needle_window in reversed order
    let mut needle_window = T::repeating(0, needle_len);
    needle_window.slow_loadu(needle_window.upper_bound() - 1, needle.as_ptr(), needle_len, true);

    let mut haystack_window = T::repeating(0, needle_len);
    let mut haystack_idx = 0usize;
    let mut curr_k = k;

    let mut match_mask = T::repeating(0, needle_len);
    let mut sub = T::repeating(0, needle_len);
    let mut sub_length = T::repeating(0, needle_len);
    let mut needle_gap_length = T::repeating(0, needle_len);
    let mut haystack_gap = T::repeating(0, needle_len);
    let mut haystack_gap_length = T::repeating(0, needle_len);

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
    // every (anti) diagonal is calculated simultaneously using vectors
    // note: each vector goes from bottom-left to top-right

    for i in 1..len {
        // shift the haystack window
        haystack_window.shift_left_1_mut();

        if haystack_idx < haystack_len {
            haystack_window.insert_last_0(*haystack.get_unchecked(haystack_idx) as u32);
            haystack_idx += 1;
        }

        T::cmpeq(&needle_window, &haystack_window, &mut match_mask);

        // match/mismatch
        T::shift_left_1(&dp1, &mut sub); // zeros are shifted in
        sub.add_mut(&match_mask); // add -1 if match

        T::shift_left_1(&length1, &mut sub_length);
        sub_length.add_mut(&ones);

        // gap in needle: dp2
        T::add(&length2, &ones, &mut needle_gap_length);

        // gap in haystack
        T::shift_left_1(&dp2, &mut haystack_gap); // zeros are shifted in

        T::shift_left_1(&length2, &mut haystack_gap_length);

        T::triple_min_length(&sub, &dp2, &haystack_gap, &sub_length,
                             &needle_gap_length, &haystack_gap_length, &mut dp1, &mut length1);
        let mut temp = dp1;
        dp1 = dp2;
        dp2 = temp;
        temp = length1;
        length1 = length2;
        length2 = temp;
        dp2.adds_mut(&ones);

        if i >= needle_len - 1 {
            let final_res = dp2.slow_extract(final_idx);
            let final_length = length2.slow_extract(final_idx) as usize;

            if final_res <= curr_k {
                let end_idx = i + 1 - needle_len;
                res.push(Match{start: end_idx - final_length, end: end_idx, k: final_res});

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
/*
unsafe fn triple_argmin<T: Jewel>(sub: &T, a_gap: &T, b_gap: &T, ones: &T) -> (T, T) {
    // return the edit used in addition to doing a min operation
    // hide latency by minimizing dependencies
    let res_min = T::min(a_gap, b_gap);
    let mut res_arg = T::cmpgt(a_gap, b_gap);
    res_arg.neg_add(&ones); // a gap: 1 - 0 = 1, b gap: 1 - -1 = 2

    let res_min2 = T::min(sub, &res_min);
    let mut res_arg2 = T::cmpgt(sub, &res_min);
    res_arg2.and(&res_arg); // sub: 0

    (res_min2, res_arg2)
}

unsafe fn triple_min_length<T: Jewel>(sub: &T, a_gap: &T, b_gap: &T, sub_length: &T, a_gap_length: &T, b_gap_length: &T) -> (T, T) {
    // choose the length based on which edit is chosen during the min operation
    // hide latency by minimizing dependencies
    // secondary objective of maximizing length if edit costs equal
    let res_min = T::min(a_gap, b_gap);
    let mut res_length = T::cmpgt(a_gap, b_gap); // a gap: 0, b gap: -1
    res_length.blendv(a_gap_length, b_gap_length); // lengths based on edits
    let mut a_b_eq_mask = T::cmpeq(a_gap, b_gap); // equal: -1
    let a_b_max_len = T::max(a_gap_length, b_gap_length);
    a_b_eq_mask.blendv(&res_length, &a_b_max_len); // maximize length if edits equal
    res_length = a_b_eq_mask;

    let res_min2 = T::min(sub, &res_min);
    let mut res_length2 = T::cmpgt(sub, &res_min); // sub: 0, prev a or b gap: -1
    res_length2.blendv(sub_length, &res_length); // length based on edits
    let mut sub_eq_mask = T::cmpeq(sub, &res_min);
    let sub_max_len = T::max(sub_length, &res_length);
    sub_eq_mask.blendv(&res_length2, &sub_max_len);
    res_length2 = sub_eq_mask;

    (res_min2, res_length2)
}
*/
