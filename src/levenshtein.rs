use std;
use super::*;
use super::jewel::*;

pub struct EditCosts {
    pub match_cost: i8,
    pub mismatch_cost: i8,
    pub gap_cost: i8,
    pub signed: bool
}

pub const LEVENSHTEIN_COSTS: EditCosts = EditCosts{match_cost: 0, mismatch_cost: 1, gap_cost: 1, signed: false};

pub fn levenshtein_naive(a: &[u8], b: &[u8], trace_on: bool) -> (i32, Option<Vec<Edit>>) {
    let swap = a.len() > b.len(); // swap so that a len <= b len
    let a_new = if swap {b} else {a};
    let a_new_len = a_new.len();
    let b_new = if swap {a} else {b};
    let b_new_len = b_new.len();

    let len = a_new_len + 1;
    let mut dp1 = vec![0i32; len]; // in each iteration, dp1 is already calculated
    let mut dp2 = vec![0i32; len]; // dp2 the currently calculated column
    let mut traceback = if trace_on {vec![0u8; (b_new_len + 1) * len]} else {vec![]};

    for i in 0..len {
        dp1[i] = i as i32;

        if trace_on {
            traceback[0 * len + i] = 2u8;
        }
    }

    for i in 1..(b_new_len + 1) {
        dp2[0] = i as i32;

        if trace_on {
            traceback[i * len + 0] = 1u8;
        }

        for j in 1..len {
            let sub = dp1[j - 1] + (a_new[j - 1] != b_new[i - 1]) as i32;
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
        let mut res: Vec<Edit> = Vec::with_capacity(a_new_len);
        let mut i = b_new_len;
        let mut j = a_new_len;

        while i > 0 || j > 0 {
            let edit = traceback[i * len + j];

            let e = match edit {
                0 => {
                    i -= 1;
                    j -= 1;
                    if a_new[j] == b_new[i] {EditType::Match} else {EditType::Mismatch}
                },
                1 => {
                    i -= 1;
                    if swap {EditType::BGap} else {EditType::AGap}
                },
                2 => {
                    j -= 1;
                    if swap {EditType::AGap} else {EditType::BGap}
                },
                _ => unreachable!()
            };

            if res.len() > 0 && res.last().unwrap().edit == e {
                res.last_mut().unwrap().count += 1;
            }else{
                res.push(Edit{edit: e, count: 1});
            }
        }

        res.reverse();
        (dp1[a_new_len], Some(res))
    }else{
        (dp1[a_new_len], None)
    }
}

pub fn levenshtein_naive_k(a: &[u8], b: &[u8], k: i32, trace_on: bool) -> Option<(i32, Option<Vec<Edit>>)> {
    let swap = a.len() > b.len(); // swap so that a len <= b len
    let a_new = if swap {b} else {a};
    let a_new_len = a_new.len();
    let b_new = if swap {a} else {b};
    let b_new_len = b_new.len();
    let k_usize = k as usize;

    if b_new_len - a_new_len > k_usize {
        return None;
    }

    let len = a_new_len + 1;
    let mut lo = 0usize;
    let mut hi = std::cmp::min(k_usize + 1, b_new_len + 1);
    let mut prev_lo;
    let mut prev_hi;
    let k_len = std::cmp::min((k_usize << 1) + 1, b_new_len + 1);
    let mut dp1 = vec![0i32; k_len]; // in each iteration, dp1 is already calculated
    let mut dp2 = vec![0i32; k_len]; // dp2 the currently calculated row
    let mut traceback = if trace_on {vec![0u8; len * k_len]} else {vec![]};

    for i in 0..(hi - lo) {
        dp1[i] = i as i32;

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
                    i32::max_value()
                }else{
                    dp1[idx - 1 - prev_lo] + (a_new[i - 1] != b_new[idx - 1]) as i32
                }
            };
            let a_gap = if j == 0 {i32::max_value()} else {dp2[j - 1] + 1};
            let b_gap = if idx >= prev_hi {i32::max_value()} else {dp1[idx - prev_lo] + 1};

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

    if dp1[hi - lo - 1] > k {
        return None;
    }

    if !trace_on {
        return Some((dp1[hi - lo - 1], None));
    }

    let mut res: Vec<Edit> = Vec::with_capacity(a_new_len);
    let mut i = a_new_len;
    let mut j = b_new_len;

    while i > 0 || j > 0 {
        let edit = traceback[i * k_len + (j - (if i > k_usize {i - k_usize} else {0}))];

        let e = match edit {
            0 => {
                i -= 1;
                j -= 1;
                if a_new[i] == b_new[j] {EditType::Match} else {EditType::Mismatch}
            },
            1 => {
                j -= 1;
                if swap {EditType::BGap} else {EditType::AGap}
            },
            2 => {
                i -= 1;
                if swap {EditType::AGap} else {EditType::BGap}
            },
            _ => unreachable!()
        };

        if res.len() > 0 && res.last().unwrap().edit == e {
            res.last_mut().unwrap().count += 1;
        }else{
            res.push(Edit{edit: e, count: 1});
        }
    }

    res.reverse();
    Some((dp1[hi - lo - 1], Some(res)))
}

pub fn levenshtein_simd_k(a: &[u8], b: &[u8], k: i32, trace_on: bool) -> Option<(i32, Option<Vec<Edit>>)> {
    if a.len() == 0 && b.len() == 0 {
        return if trace_on {Some((0i32, Some(vec![])))} else {Some((0i32, None))};
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            if k <= 30 {
                return unsafe {levenshtein_simd_core::<Avx1x32x8>(a, b, k, trace_on, LEVENSHTEIN_COSTS)};
            }else{
                return unsafe {levenshtein_simd_core::<AvxNx32x8>(a, b, k, trace_on, LEVENSHTEIN_COSTS)};
            }
        }
    }

    levenshtein_naive_k(a, b, k, trace_on)
}

unsafe fn levenshtein_simd_core<T: Jewel>(a_old: &[u8], b_old: &[u8], k: i32, trace_on: bool, costs: EditCosts) -> Option<(i32, Option<Vec<Edit>>)> {
    // swap a and b so that a is shorter than b, if applicable
    // makes operations later on slightly easier, since length of a <= length of b
    let swap = a_old.len() > b_old.len();
    let a = if swap {b_old} else {a_old};
    let a_len = a.len();
    let b = if swap {a_old} else {b_old};
    let b_len = b.len();

    if b_len - a_len > k as usize {
        return None;
    }

    // initialized with max values
    // must use saturated additions afterwards to not overflow
    let mut dp1 = T::repeating_max((k + 2) as usize);
    let max_len = dp1.upper_bound();
    let mut dp2 = T::repeating_max(max_len);

    // lengths of the (anti) diagonals
    // assumes max_len is even
    let k1 = max_len - 1;
    let k1_div2 = k1 >> 1;
    let k2 = max_len - 2;
    let k2_div2 = k2 >> 1;

    // set dp[0][0] = 0
    dp1.slow_insert(k1_div2, 0, costs.signed);
    // set dp[0][1] = gap_cost and dp[1][0] = gap_cost
    dp2.slow_insert(k2_div2 - 1, costs.gap_cost as u32, costs.signed);
    dp2.slow_insert(k2_div2, costs.gap_cost as u32, costs.signed);

    // a_k1_window and a_k2_window represent reversed portions of the string a
    // copy in half of k1/k2 number of characters
    // these characters are placed in the second half of b windows
    // since a windows are reversed, the characters are placed in reverse in the first half of b windows
    let mut a_k1_window = T::repeating(0, max_len, false);
    a_k1_window.slow_loadu(k1_div2 - 1, a.as_ptr(), std::cmp::min(k1_div2, a_len), true);

    let mut b_k1_window = T::repeating(0, max_len, false);
    b_k1_window.slow_loadu(k1_div2 + 1, b.as_ptr(), std::cmp::min(k1_div2, b_len), false);

    let mut a_k2_window = T::repeating(0, max_len, false);
    a_k2_window.slow_loadu(k2_div2 - 1, a.as_ptr(), std::cmp::min(k2_div2, a_len), true);

    let mut b_k2_window = T::repeating(0, max_len, false);
    b_k2_window.slow_loadu(k2_div2, b.as_ptr(), std::cmp::min(k2_div2, b_len), false);

    // used to keep track of the next characters to place in the windows
    let mut k1_idx = k1_div2 - 1;
    let mut k2_idx = k2_div2 - 1;

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
        traceback_arr.push(T::repeating(0, max_len, false));
        traceback_arr.push(T::repeating(0, max_len, false));
        traceback_arr.get_unchecked_mut(1).slow_insert(k2_div2 - 1, 2, false);
        traceback_arr.get_unchecked_mut(1).slow_insert(k2_div2, 1, false);
    }

    let mut sub = T::repeating(0, max_len, false);
    let mut a_gap = T::repeating(0, max_len, false);
    let mut b_gap = T::repeating(0, max_len, false);

    let match_cost = T::repeating(costs.match_cost as u32, max_len, false);
    let mismatch_cost = T::repeating(costs.mismatch_cost as u32, max_len, false);
    let gap_cost = T::repeating(costs.gap_cost as u32, max_len, false);

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
            a_k1_window.insert_first(*a.get_unchecked(k1_idx) as u32, false);
        }

        b_k1_window.shift_left_1_mut();

        if k1_idx < b_len {
            b_k1_window.insert_last_1(*b.get_unchecked(k1_idx) as u32); // k1 - 1
        }

        a_k2_window.shift_right_1_mut();

        if k2_idx < a_len {
            a_k2_window.insert_first(*a.get_unchecked(k2_idx) as u32, false);
        }

        b_k2_window.shift_left_1_mut();

        if k2_idx < b_len {
            b_k2_window.insert_last_2(*b.get_unchecked(k2_idx) as u32);
        }

        // (anti) diagonal that matches in the a and b windows
        T::cmpeq(&a_k1_window, &b_k1_window, &mut sub);
        sub.blendv_mut(&mismatch_cost, &match_cost);
        sub.adds_mut(&dp1);
        // cost of gaps in a
        T::shift_right_1(&dp2, &mut a_gap);
        a_gap.insert_first_max();
        a_gap.adds_mut(&gap_cost);
        // cost of gaps in b
        T::adds(&dp2, &gap_cost, &mut b_gap);

        // min of the cost of all three edit operations
        if trace_on {
            let args = T::triple_argmin(&sub, &a_gap, &b_gap, &mut dp1);
            std::mem::swap(&mut dp1, &mut dp2);
            traceback_arr.push(args);
        }else{
            T::min(&a_gap, &b_gap, &mut dp1);
            dp1.min_mut(&sub);
            std::mem::swap(&mut dp1, &mut dp2);
        }

        // (anti) diagonal that matches in the a and b windows
        T::cmpeq(&a_k2_window, &b_k2_window, &mut sub);
        sub.blendv_mut(&mismatch_cost, &match_cost);
        sub.adds_mut(&dp1);
        // cost of gaps in b
        T::shift_left_1(&dp2, &mut b_gap);
        b_gap.insert_last_max(); // k1, shift in max value
        b_gap.adds_mut(&gap_cost);
        // cost of gaps in a
        T::adds(&dp2, &gap_cost, &mut a_gap);

        // min of the cost of all three edit operations
        if trace_on {
            let args = T::triple_argmin(&sub, &a_gap, &b_gap, &mut dp1);
            std::mem::swap(&mut dp1, &mut dp2);
            traceback_arr.push(args);
        }else{
            T::min(&a_gap, &b_gap, &mut dp1);
            dp1.min_mut(&sub);
            std::mem::swap(&mut dp1, &mut dp2);
        }
    }

    let final_res = if ends_with_k2 {
        dp2.slow_extract(final_idx, costs.signed)
    }else{
        dp1.slow_extract(final_idx, costs.signed)
    };

    if final_res > k {
        return None;
    }

    if !trace_on {
        return Some((final_res, None));
    }

    Some((final_res, Some(traceback(&traceback_arr, final_idx, a, b, swap, ends_with_k2))))
}

unsafe fn traceback<T: Jewel>(arr: &[T], mut idx: usize, a: &[u8], b: &[u8], swap: bool, mut is_k2: bool) -> Vec<Edit> {
    // keep track of position in traditional dp array and strings
    let mut i = a.len(); // index in a
    let mut j = b.len(); // index in b

    // last diagonal may overshoot, so ignore it
    let mut arr_idx = arr.len() - 1 - (if is_k2 {0} else {1});
    let mut res: Vec<Edit> = Vec::with_capacity(a.len());

    while arr_idx > 0 {
        // each Jewel vector in arr is only visited once, so extract (which is costly) is fine
        let edit = arr.get_unchecked(arr_idx).slow_extract(idx, false);

        let e = match edit {
            0 => { // match/mismatch
                arr_idx -= 2;
                i -= 1;
                j -= 1;
                if *a.get_unchecked(i) == *b.get_unchecked(j) {EditType::Match} else {EditType::Mismatch}
            },
            1 => { // a gap
                arr_idx -= 1;

                if !is_k2 {
                    idx -= 1;
                }

                j -= 1;
                is_k2 = !is_k2; // must account for alternating k1/k2 diagonals
                if swap {EditType::BGap} else {EditType::AGap} // account for the swap in the beginning
            },
            2 => { // b gap
                arr_idx -= 1;

                if is_k2 {
                    idx += 1;
                }

                i -= 1;
                is_k2 = !is_k2;
                if swap {EditType::AGap} else {EditType::BGap}
            },
            _ => unreachable!()
        };

        if res.len() > 0 && res.last().unwrap().edit == e {
            res.last_mut().unwrap().count += 1;
        }else{
            res.push(Edit{edit: e, count: 1});
        }
    }

    res.reverse();
    res
}

pub fn levenshtein_exp(a: &[u8], b: &[u8], trace_on: bool) -> (i32, Option<Vec<Edit>>) {
    let mut k = 30;
    let mut res = levenshtein_simd_k(a, b, k, false);

    while res.is_none() {
        k <<= 2; // multiply by 4 every time (instead of the usual multiple by 2) for speed
        res = levenshtein_simd_k(a, b, k, false);
    }

    if trace_on { // save memory by only calculating the traceback at the end, with an extra step
        levenshtein_simd_k(a, b, res.unwrap().0, true).unwrap()
    }else{
        res.unwrap()
    }
}

pub fn levenshtein_search_naive(needle: &[u8], haystack: &[u8]) -> Vec<Match> {
    levenshtein_search_naive_k(needle, haystack, needle.len() as i32, true)
}

pub fn levenshtein_search_naive_k(needle: &[u8], haystack: &[u8], k: i32, best: bool) -> Vec<Match> {
    let needle_len = needle.len();
    let haystack_len = haystack.len();

    if needle_len == 0 {
        return vec![];
    }

    let len = needle_len + 1;
    let mut dp1 = vec![0i32; len];
    let mut dp2 = vec![0i32; len];
    let mut length1 = vec![0usize; len];
    let mut length2 = vec![0usize; len];
    let mut res = Vec::with_capacity(haystack_len >> 2);
    let mut curr_k = k;

    for i in 0..len {
        dp1[i] = i as i32;
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
            let sub = dp1[j - 1] + (needle[j - 1] != haystack[i]) as i32;
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
    if needle.len() == 0 {
        return vec![];
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            if needle.len() <= 32 {
                return unsafe {levenshtein_search_simd_core::<Avx1x32x8>(needle, haystack, needle.len() as i32, true, LEVENSHTEIN_COSTS)};
            }else{
                return unsafe {levenshtein_search_simd_core::<AvxNx32x8>(needle, haystack, needle.len() as i32, true, LEVENSHTEIN_COSTS)};
            }
        }
    }

    levenshtein_search_naive_k(needle, haystack, needle.len() as i32, true)
}

pub fn levenshtein_search_simd_k(needle: &[u8], haystack: &[u8], k: i32, best: bool) -> Vec<Match> {
    if needle.len() == 0 {
        return vec![];
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            if needle.len() <= 32 {
                return unsafe {levenshtein_search_simd_core::<Avx1x32x8>(needle, haystack, k, best, LEVENSHTEIN_COSTS)};
            }else{
                return unsafe {levenshtein_search_simd_core::<AvxNx32x8>(needle, haystack, k, best, LEVENSHTEIN_COSTS)};
            }
        }
    }

    levenshtein_search_naive_k(needle, haystack, k, best)
}

unsafe fn levenshtein_search_simd_core<T: Jewel>(needle: &[u8], haystack: &[u8], k: i32, best: bool, costs: EditCosts) -> Vec<Match> {
    let needle_len = needle.len();
    let haystack_len = haystack.len();
    let mut dp1 = T::repeating_max(needle_len);
    let mut dp2 = T::repeating_max(needle_len);
    dp2.slow_insert(dp2.upper_bound() - 1, costs.gap_cost as u32, costs.signed); // last cell

    // save length instead of start idx due to int size constraints
    let mut length1 = T::repeating(0, needle_len, false);
    let mut length2 = T::repeating(0, needle_len, false);

    let ones = T::repeating(1, needle_len, false);

    let len = haystack_len + needle_len;
    let final_idx = dp1.upper_bound() - needle_len;

    let mut res = Vec::with_capacity(haystack_len >> 2);

    // load needle characters into needle_window in reversed order
    let mut needle_window = T::repeating(0, needle_len, false);
    needle_window.slow_loadu(needle_window.upper_bound() - 1, needle.as_ptr(), needle_len, true);

    let mut haystack_window = T::repeating(0, needle_len, false);
    let mut haystack_idx = 0usize;
    let mut curr_k = k;

    let mut match_mask = T::repeating(0, needle_len, false);
    let mut sub = T::repeating(0, needle_len, false);
    let mut sub_length = T::repeating(0, needle_len, false);
    let mut needle_gap = T::repeating(0, needle_len, false);
    let mut needle_gap_length = T::repeating(0, needle_len, false);
    let mut haystack_gap = T::repeating(0, needle_len, false);
    let mut haystack_gap_length = T::repeating(0, needle_len, false);

    let match_cost = T::repeating(costs.match_cost as u32, needle_len, false);
    let mismatch_cost = T::repeating(costs.mismatch_cost as u32, needle_len, false);
    let gap_cost = T::repeating(costs.gap_cost as u32, needle_len, false);

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
            haystack_window.insert_last_0(*haystack.get_unchecked(haystack_idx) as u32, false);
            haystack_idx += 1;
        }

        T::cmpeq(&needle_window, &haystack_window, &mut match_mask);
        match_mask.blendv_mut(&mismatch_cost, &match_cost);

        // match/mismatch
        T::shift_left_1(&dp1, &mut sub);

        if costs.signed {
            sub.insert_last_0(0, true);
        } // otherwise, zeros are shifted in

        sub.adds_mut(&match_mask);

        T::shift_left_1(&length1, &mut sub_length);
        sub_length.add_mut(&ones);

        // gap in needle
        T::adds(&dp2, &gap_cost, &mut needle_gap);
        T::add(&length2, &ones, &mut needle_gap_length);

        // gap in haystack
        T::shift_left_1(&dp2, &mut haystack_gap);

        if costs.signed {
            haystack_gap.insert_last_0(0, true);
        } // otherwise, zeros are shifted in

        haystack_gap.adds_mut(&gap_cost);

        T::shift_left_1(&length2, &mut haystack_gap_length);

        T::triple_min_length(&sub, &needle_gap, &haystack_gap, &sub_length,
                             &needle_gap_length, &haystack_gap_length, &mut dp1, &mut length1);
        std::mem::swap(&mut dp1, &mut dp2);
        std::mem::swap(&mut length1, &mut length2);

        if i >= needle_len - 1 {
            let final_res = dp2.slow_extract(final_idx, costs.signed);
            let final_length = length2.slow_extract(final_idx, false) as usize;

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

