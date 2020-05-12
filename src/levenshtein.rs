use std;
use super::*;
use super::jewel::*;

#[derive(Copy, Clone, Debug)]
pub struct EditCosts {
    mismatch_cost: u8,
    gap_cost: u8,
    transpose_cost: Option<u8>
}

impl EditCosts {
    pub fn new(mismatch_cost: u8, gap_cost: u8, transpose_cost: Option<u8>) -> Self {
        assert!(mismatch_cost > 0);
        assert!(gap_cost > 0);

        if let Some(cost) = transpose_cost {
            assert!(cost > 0);
            // transpose cost must be cheaper than doing the equivalent with other edits
            assert!((cost >> 1) < mismatch_cost);
            assert!((cost >> 1) < gap_cost);
        }

        Self{
            mismatch_cost: mismatch_cost,
            gap_cost: gap_cost,
            transpose_cost: transpose_cost
        }
    }

    fn check_search(&self) {
        if let Some(cost) = self.transpose_cost {
            assert!(cost <= self.gap_cost);
        }
    }
}

pub const LEVENSHTEIN_COSTS: EditCosts = EditCosts{mismatch_cost: 1, gap_cost: 1, transpose_cost: None};
pub const RDAMERAU_COSTS: EditCosts = EditCosts{mismatch_cost: 1, gap_cost: 1, transpose_cost: Some(1)};

pub fn levenshtein_naive(a: &[u8], b: &[u8]) -> u32 {
    levenshtein_naive_with_opts(a, b, false, LEVENSHTEIN_COSTS).0
}

pub fn levenshtein_naive_with_opts(a: &[u8], b: &[u8], trace_on: bool, costs: EditCosts) -> (u32, Option<Vec<Edit>>) {
    let swap = a.len() > b.len(); // swap so that a len <= b len
    let a_new = if swap {b} else {a};
    let a_new_len = a_new.len();
    let b_new = if swap {a} else {b};
    let b_new_len = b_new.len();
    let mismatch_cost = costs.mismatch_cost as u32;
    let gap_cost = costs.gap_cost as u32;
    let transpose_cost = match costs.transpose_cost {
        Some(cost) => cost as u32,
        None => 0
    };
    let allow_transpose = costs.transpose_cost.is_some();

    let len = a_new_len + 1;
    let mut dp0 = vec![0u32; len];
    let mut dp1 = vec![0u32; len]; // in each iteration, dp1 is already calculated
    let mut dp2 = vec![0u32; len]; // dp2 the currently calculated column
    let mut traceback = if trace_on {vec![0u8; (b_new_len + 1) * len]} else {vec![]};

    for i in 0..len {
        dp1[i] = (i as u32) * gap_cost;

        if trace_on {
            traceback[0 * len + i] = 2u8;
        }
    }

    for i in 1..(b_new_len + 1) {
        dp2[0] = (i as u32) * gap_cost;

        if trace_on {
            traceback[i * len + 0] = 1u8;
        }

        for j in 1..len {
            let sub = dp1[j - 1] + ((a_new[j - 1] != b_new[i - 1]) as u32) * mismatch_cost;
            let a_gap = dp1[j] + gap_cost;
            let b_gap = dp2[j - 1] + gap_cost;
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

            if allow_transpose && i > 1 && j > 1
                && a_new[j - 1] == b_new[i - 2] && a_new[j - 2] == b_new[i - 1] {
                let transpose = dp0[j - 2] + transpose_cost;

                if transpose <= dp2[j] {
                    dp2[j] = transpose;

                    if trace_on {
                        traceback[traceback_idx] = 3u8;
                    }
                }
            }
        }

        std::mem::swap(&mut dp0, &mut dp1);
        std::mem::swap(&mut dp1, &mut dp2);
    }

    if trace_on {
        let mut upper_bound_edits = dp1[a_new_len] / std::cmp::min(mismatch_cost, gap_cost);

        if allow_transpose {
            upper_bound_edits = std::cmp::max(upper_bound_edits, (dp1[a_new_len] >> 1) / transpose_cost + 1);
        }

        let mut res: Vec<Edit> = Vec::with_capacity(((upper_bound_edits << 1) + 1) as usize);
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
                3 => {
                    i -= 2;
                    j -= 2;
                    EditType::Transpose
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

pub fn levenshtein_naive_k(a: &[u8], b: &[u8], k: u32) -> Option<u32> {
    let res = levenshtein_naive_k_with_opts(a, b, k, false, LEVENSHTEIN_COSTS);

    match res {
        Some((edits, _)) => Some(edits),
        None => None
    }
}

pub fn levenshtein_naive_k_with_opts(a: &[u8], b: &[u8], k: u32, trace_on: bool, costs: EditCosts) -> Option<(u32, Option<Vec<Edit>>)> {
    let swap = a.len() > b.len(); // swap so that a len <= b len
    let a_new = if swap {b} else {a};
    let a_new_len = a_new.len();
    let b_new = if swap {a} else {b};
    let b_new_len = b_new.len();
    let mismatch_cost = costs.mismatch_cost as u32;
    let gap_cost = costs.gap_cost as u32;
    let transpose_cost = match costs.transpose_cost {
        Some(cost) => cost as u32,
        None => 0
    };
    let allow_transpose = costs.transpose_cost.is_some();
    let unit_k = (k / gap_cost) as usize;

    if b_new_len - a_new_len > unit_k {
        return None;
    }

    let len = a_new_len + 1;
    let mut lo = 0usize;
    let mut hi = std::cmp::min(unit_k + 1, b_new_len + 1);
    let mut prev_lo0;
    let mut prev_lo1 = 0; // unused value
    let mut prev_hi;
    let k_len = std::cmp::min((unit_k << 1) + 1, b_new_len + 1);
    let mut dp0 = vec![0u32; k_len];
    let mut dp1 = vec![0u32; k_len]; // in each iteration, dp1 is already calculated
    let mut dp2 = vec![0u32; k_len]; // dp2 the currently calculated row
    let mut traceback = if trace_on {vec![0u8; len * k_len]} else {vec![]};

    for i in 0..(hi - lo) {
        dp1[i] = (i as u32) * gap_cost;

        if trace_on {
            traceback[0 * k_len + i] = 1u8;
        }
    }

    for i in 1..len {
        prev_lo0 = prev_lo1;
        prev_lo1 = lo;
        prev_hi = hi;
        hi = std::cmp::min(hi + 1, b_new_len + 1);

        if i > unit_k {
            lo += 1;
        }

        for j in 0..(hi - lo) {
            let idx = lo + j;
            let sub = {
                if idx == 0 {
                    u32::max_value()
                }else{
                    dp1[idx - 1 - prev_lo1] + ((a_new[i - 1] != b_new[idx - 1]) as u32) * mismatch_cost
                }
            };
            let a_gap = if j == 0 {u32::max_value()} else {dp2[j - 1] + gap_cost};
            let b_gap = if idx >= prev_hi {u32::max_value()} else {dp1[idx - prev_lo1] + gap_cost};

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

            if allow_transpose && i > 1 && idx > 1
                && a_new[i - 1] == b_new[idx - 2] && a_new[i - 2] == b_new[idx - 1] {
                let transpose = dp0[idx - prev_lo0 - 2] + transpose_cost;

                if transpose <= dp2[j] {
                    dp2[j] = transpose;

                    if trace_on {
                        traceback[traceback_idx] = 3u8;
                    }
                }
            }
        }

        std::mem::swap(&mut dp0, &mut dp1);
        std::mem::swap(&mut dp1, &mut dp2);
    }

    if dp1[hi - lo - 1] > k {
        return None;
    }

    if !trace_on {
        return Some((dp1[hi - lo - 1], None));
    }

    let mut upper_bound_edits = dp1[hi - lo - 1] / std::cmp::min(mismatch_cost, gap_cost);

    if allow_transpose {
        upper_bound_edits = std::cmp::max(upper_bound_edits, (dp1[hi - lo - 1] >> 1) / transpose_cost + 1);
    }

    let mut res: Vec<Edit> = Vec::with_capacity(((upper_bound_edits << 1) + 1) as usize);
    let mut i = a_new_len;
    let mut j = b_new_len;

    while i > 0 || j > 0 {
        let edit = traceback[i * k_len + (j - (if i > unit_k {i - unit_k} else {0}))];

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
            3 => {
                i -= 2;
                j -= 2;
                EditType::Transpose
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

pub fn levenshtein_simd_k(a: &[u8], b: &[u8], k: u32) -> Option<u32> {
    let res = levenshtein_simd_k_with_opts(a, b, k, false, LEVENSHTEIN_COSTS);

    match res {
        Some((edits, _)) => Some(edits),
        None => None
    }
}

pub fn levenshtein_simd_k_with_opts(a: &[u8], b: &[u8], k: u32, trace_on: bool, costs: EditCosts) -> Option<(u32, Option<Vec<Edit>>)> {
    if a.len() == 0 && b.len() == 0 {
        return if trace_on {Some((0u32, Some(vec![])))} else {Some((0u32, None))};
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        let unit_k = std::cmp::min(k / (costs.gap_cost as u32), std::cmp::max(a.len(), b.len()) as u32);

        // note: do not use the max_value, because it indicates overflow/inaccuracy
        if is_x86_feature_detected!("avx2") {
            if unit_k <= (1 * 32 - 2) && k <= ((u8::max_value() - 1) as u32) {
                return unsafe {levenshtein_simd_core::<Avx1x32x8>(a, b, k, trace_on, costs)};
            }else if unit_k <= (2 * 32 - 2) && k <= ((u8::max_value() - 1) as u32) {
                return unsafe {levenshtein_simd_core::<Avx2x32x8>(a, b, k, trace_on, costs)};
            }else if unit_k <= (4 * 32 - 2) && k <= ((u8::max_value() - 1) as u32) {
                return unsafe {levenshtein_simd_core::<Avx4x32x8>(a, b, k, trace_on, costs)};
            }else if unit_k <= (8 * 32 - 2) && k <= ((u8::max_value() - 1) as u32) {
                return unsafe {levenshtein_simd_core::<Avx8x32x8>(a, b, k, trace_on, costs)};
            }else if k <= ((u16::max_value() - 1) as u32) {

            }
        }else if is_x86_feature_detected!("sse4.1") {
            if unit_k <= (1 * 16 - 2) && k <= ((u8::max_value() - 1) as u32) {
                return unsafe {levenshtein_simd_core::<Avx1x32x8>(a, b, k, trace_on, costs)};
            }else if unit_k <= (2 * 16 - 2) && k <= ((u8::max_value() - 1) as u32) {
                return unsafe {levenshtein_simd_core::<Avx2x32x8>(a, b, k, trace_on, costs)};
            }else if unit_k <= (4 * 16 - 2) && k <= ((u8::max_value() - 1) as u32) {
                return unsafe {levenshtein_simd_core::<Avx4x32x8>(a, b, k, trace_on, costs)};
            }else if unit_k <= (8 * 16 - 2) && k <= ((u8::max_value() - 1) as u32) {
                return unsafe {levenshtein_simd_core::<Avx8x32x8>(a, b, k, trace_on, costs)};
            }else if k <= ((u16::max_value() - 1) as u32) {

            }
        }
    }

    levenshtein_naive_k_with_opts(a, b, k, trace_on, costs)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,sse4.1")]
unsafe fn levenshtein_simd_core<T: Jewel>(a_old: &[u8], b_old: &[u8], k: u32, trace_on: bool, costs: EditCosts) -> Option<(u32, Option<Vec<Edit>>)> {
    // swap a and b so that a is shorter than b, if applicable
    // makes operations later on slightly easier, since length of a <= length of b
    let swap = a_old.len() > b_old.len();
    let a = if swap {b_old} else {a_old};
    let a_len = a.len();
    let b = if swap {a_old} else {b_old};
    let b_len = b.len();
    let unit_k = std::cmp::min((k / (costs.gap_cost as u32)) as usize, b_len);

    if b_len - a_len > unit_k {
        return None;
    }

    // initialized with max values
    // must use saturated additions afterwards to not overflow
    let mut dp1 = T::repeating_max((unit_k + 2) as usize);
    let max_len = dp1.upper_bound();
    let mut dp2 = T::repeating_max(max_len);
    let mut dp0 = T::repeating_max(max_len);
    let mut dp_temp = T::repeating_max(max_len);
    // dp0 -> dp_temp -> dp1 -> dp2 -> current diagonal

    // lengths of the (anti) diagonals
    // assumes max_len is even
    let k1 = max_len - 1;
    let k1_div2 = k1 >> 1;
    let k2 = max_len - 2;
    let k2_div2 = k2 >> 1;

    // set dp[0][0] = 0
    dp1.slow_insert(k1_div2, 0);
    // set dp[0][1] = gap_cost and dp[1][0] = gap_cost
    dp2.slow_insert(k2_div2 - 1, costs.gap_cost as u32);
    dp2.slow_insert(k2_div2, costs.gap_cost as u32);

    // a_k1_window and a_k2_window represent reversed portions of the string a
    // copy in half of k1/k2 number of characters
    // these characters are placed in the second half of b windows
    // since a windows are reversed, the characters are placed in reverse in the first half of b windows
    let mut a_k1_window = T::repeating(0, max_len);
    a_k1_window.slow_loadu(k1_div2 - 1, a.as_ptr(), std::cmp::min(k1_div2, a_len), true);

    let mut b_k1_window = T::repeating(0, max_len);
    b_k1_window.slow_loadu(k1_div2 + 1, b.as_ptr(), std::cmp::min(k1_div2, b_len), false);

    let mut a_k2_window = T::repeating(0, max_len);
    a_k2_window.slow_loadu(k2_div2 - 1, a.as_ptr(), std::cmp::min(k2_div2, a_len), true);

    let mut b_k2_window = T::repeating(0, max_len);
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
        traceback_arr.push(T::repeating(0, max_len));
        traceback_arr.push(T::repeating(0, max_len));
        traceback_arr.get_unchecked_mut(1).slow_insert(k2_div2 - 1, 2);
        traceback_arr.get_unchecked_mut(1).slow_insert(k2_div2, 1);
    }

    // reusable constant
    let threes = T::repeating(3, max_len);

    let mut sub = T::repeating(0, max_len);
    let mut match_mask0 = T::repeating(0, max_len);
    let mut match_mask1 = T::repeating(0, max_len);
    let mut a_gap = T::repeating(0, max_len);
    let mut b_gap = T::repeating(0, max_len);
    let mut transpose = T::repeating(0, max_len);

    let mismatch_cost = T::repeating(costs.mismatch_cost as u32, max_len);
    let gap_cost = T::repeating(costs.gap_cost as u32, max_len);
    let transpose_cost = match costs.transpose_cost {
        Some(cost) => T::repeating(cost as u32, max_len),
        None => T::repeating(0, max_len) // value does not matter
    };
    let allow_transpose = costs.transpose_cost.is_some();

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
        T::cmpeq(&a_k1_window, &b_k1_window, &mut match_mask1);
        T::andnot(&match_mask1, &mismatch_cost, &mut sub);
        sub.adds_mut(&dp1);
        // cost of gaps in a
        T::shift_right_1(&dp2, &mut a_gap);
        a_gap.insert_first_max();
        a_gap.adds_mut(&gap_cost);
        // cost of gaps in b
        T::adds(&dp2, &gap_cost, &mut b_gap);

        if allow_transpose {
            T::shift_right_1(&match_mask0, &mut transpose); // reuse transpose
            transpose.and_mut(&match_mask0);
            T::andnot(&match_mask1, &transpose, &mut match_mask0); // reuse match_mask0 to represent transpose mask
            T::adds(&dp0, &transpose_cost, &mut transpose);
        }

        // min of the cost of all three edit operations
        if trace_on {
            let mut args = T::triple_argmin(&sub, &a_gap, &b_gap, &mut dp0);

            if allow_transpose {
                // blend using transpose mask
                dp0.blendv_mut(&transpose, &match_mask0);
                args.blendv_mut(&threes, &match_mask0);
                std::mem::swap(&mut match_mask0, &mut match_mask1);
            }

            traceback_arr.push(args);
        }else{
            T::min(&a_gap, &b_gap, &mut dp0);
            dp0.min_mut(&sub);

            if allow_transpose {
                // blend using transpose mask
                dp0.blendv_mut(&transpose, &match_mask0);
                std::mem::swap(&mut match_mask0, &mut match_mask1);
            }
        }

        std::mem::swap(&mut dp0, &mut dp_temp);
        std::mem::swap(&mut dp_temp, &mut dp1);
        std::mem::swap(&mut dp1, &mut dp2);

        // (anti) diagonal that matches in the a and b windows
        T::cmpeq(&a_k2_window, &b_k2_window, &mut match_mask1);
        T::andnot(&match_mask1, &mismatch_cost, &mut sub);
        sub.adds_mut(&dp1);
        // cost of gaps in b
        T::shift_left_1(&dp2, &mut b_gap);
        b_gap.insert_last_max(); // k1, shift in max value
        b_gap.adds_mut(&gap_cost);
        // cost of gaps in a
        T::adds(&dp2, &gap_cost, &mut a_gap);

        if allow_transpose {
            T::shift_left_1(&match_mask0, &mut transpose); // reuse transpose
            transpose.and_mut(&match_mask0);
            T::andnot(&match_mask1, &transpose, &mut match_mask0); // reuse match_mask0 to represent transpose mask
            T::adds(&dp0, &transpose_cost, &mut transpose);
        }

        // min of the cost of all three edit operations
        if trace_on {
            let mut args = T::triple_argmin(&sub, &a_gap, &b_gap, &mut dp0);

            if allow_transpose {
                // blend using transpose mask
                dp0.blendv_mut(&transpose, &match_mask0);
                args.blendv_mut(&threes, &match_mask0);
                std::mem::swap(&mut match_mask0, &mut match_mask1);
            }

            traceback_arr.push(args);
        }else{
            T::min(&a_gap, &b_gap, &mut dp0);
            dp0.min_mut(&sub);

            if allow_transpose {
                // blend using transpose mask
                dp0.blendv_mut(&transpose, &match_mask0);
                std::mem::swap(&mut match_mask0, &mut match_mask1);
            }
        }

        std::mem::swap(&mut dp0, &mut dp_temp);
        std::mem::swap(&mut dp_temp, &mut dp1);
        std::mem::swap(&mut dp1, &mut dp2);
    }

    let final_res = if ends_with_k2 {
        dp2.slow_extract(final_idx)
    }else{
        dp1.slow_extract(final_idx)
    };

    if final_res > k {
        return None;
    }

    if !trace_on {
        return Some((final_res, None));
    }

    // upper bound the number of edit operations, to reduce memory allocations for saving the traceback
    let mut upper_bound_edits = final_res / (std::cmp::min(costs.mismatch_cost, costs.gap_cost) as u32);

    if let Some(cost) = costs.transpose_cost {
        upper_bound_edits = std::cmp::max(upper_bound_edits, (final_res >> 1) / (cost as u32) + 1);
    }

    Some((final_res, Some(traceback(&traceback_arr, upper_bound_edits as usize, final_idx, a, b, swap, ends_with_k2))))
}

unsafe fn traceback<T: Jewel>(arr: &[T], k: usize, mut idx: usize, a: &[u8], b: &[u8], swap: bool, mut is_k2: bool) -> Vec<Edit> {
    // keep track of position in traditional dp array and strings
    let mut i = a.len(); // index in a
    let mut j = b.len(); // index in b

    // last diagonal may overshoot, so ignore it
    let mut arr_idx = arr.len() - 1 - (if is_k2 {0} else {1});
    let mut res: Vec<Edit> = Vec::with_capacity((k << 1) + 1);

    while arr_idx > 0 {
        // each Jewel vector in arr is only visited once, so extract (which is costly) is fine
        let edit = arr.get_unchecked(arr_idx).slow_extract(idx);

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
            3 => { // transpose
                arr_idx -= 4;
                i -= 2;
                j -= 2;
                EditType::Transpose
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

pub fn levenshtein_exp(a: &[u8], b: &[u8], trace_on: bool) -> (u32, Option<Vec<Edit>>) {
    let mut k = 30;
    let mut res = levenshtein_simd_k_with_opts(a, b, k, false, LEVENSHTEIN_COSTS);

    while res.is_none() {
        k <<= 1;
        res = levenshtein_simd_k_with_opts(a, b, k, false, LEVENSHTEIN_COSTS);
    }

    if trace_on { // save memory by only calculating the traceback at the end, with an extra step
        levenshtein_simd_k_with_opts(a, b, res.unwrap().0, true, LEVENSHTEIN_COSTS).unwrap()
    }else{
        res.unwrap()
    }
}

pub fn levenshtein_search_naive(needle: &[u8], haystack: &[u8]) -> Vec<Match> {
    let max_k = std::cmp::min((needle.len() as u32) * (LEVENSHTEIN_COSTS.mismatch_cost as u32),
                              (needle.len() as u32) * (LEVENSHTEIN_COSTS.gap_cost as u32) * 2);
    levenshtein_search_naive_with_opts(needle, haystack, max_k, SearchType::Best, LEVENSHTEIN_COSTS, false)
}

pub fn levenshtein_search_naive_with_opts(needle: &[u8], haystack: &[u8], k: u32, search_type: SearchType, costs: EditCosts, anchored: bool) -> Vec<Match> {
    let needle_len = needle.len();
    let haystack_len = haystack.len();

    if needle_len == 0 {
        return vec![];
    }

    costs.check_search();

    let len = needle_len + 1;
    let iter_len = if anchored {
        std::cmp::min(haystack_len, needle_len + (k as usize) / (costs.gap_cost as usize))
    }else{
        haystack_len
    };

    let mut dp0 = vec![0u32; len];
    let mut dp1 = vec![0u32; len];
    let mut dp2 = vec![0u32; len];
    let mut length0 = vec![0usize; len];
    let mut length1 = vec![0usize; len];
    let mut length2 = vec![0usize; len];
    let mut res = Vec::with_capacity(iter_len >> 2);
    let mut curr_k = k;
    let mismatch_cost = costs.mismatch_cost as u32;
    let gap_cost = costs.gap_cost as u32;
    let transpose_cost = match costs.transpose_cost {
        Some(cost) => cost as u32,
        None => 0
    };
    let allow_transpose = costs.transpose_cost.is_some();

    for i in 0..len {
        dp1[i] = (i as u32) * gap_cost;
    }

    if dp1[len - 1] <= curr_k {
        res.push(Match{start: 0, end: 0, k: dp1[len - 1]});

        if search_type == SearchType::Best {
            curr_k = dp1[len - 1];
        }
    }

    for i in 0..iter_len {
        dp2[0] = if anchored {(i as u32 + 1) * (costs.gap_cost as u32)} else {0};
        length2[0] = 0;

        for j in 1..len {
            let sub = dp1[j - 1] + ((needle[j - 1] != haystack[i]) as u32) * mismatch_cost;
            let a_gap = dp1[j] + gap_cost;
            let b_gap = dp2[j - 1] + gap_cost;

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

            if allow_transpose && i > 0 && j > 1
                && needle[j - 1] == haystack[i - 1] && needle[j - 2] == haystack[i] {
                let transpose = dp0[j - 2] + transpose_cost;

                if transpose <= dp2[j] {
                    dp2[j] = transpose;
                    length2[j] = length0[j - 2] + 2;
                }
            }
        }

        let final_res = dp2[len - 1];

        if final_res <= curr_k {
            res.push(Match{start: i + 1 - length2[len - 1], end: i + 1, k: final_res});

            match search_type {
                SearchType::First => break,
                SearchType::Best => curr_k = final_res,
                _ => ()
            }
        }

        std::mem::swap(&mut dp0, &mut dp1);
        std::mem::swap(&mut dp1, &mut dp2);
        std::mem::swap(&mut length0, &mut length1);
        std::mem::swap(&mut length1, &mut length2);
    }

    if search_type == SearchType::Best {
        res.retain(|m| m.k == curr_k);
    }

    res
}

pub fn levenshtein_search_simd(needle: &[u8], haystack: &[u8]) -> Vec<Match> {
    let max_k = std::cmp::min((needle.len() as u32) * (LEVENSHTEIN_COSTS.mismatch_cost as u32),
                              (needle.len() as u32) * (LEVENSHTEIN_COSTS.gap_cost as u32) * 2);
    levenshtein_search_simd_with_opts(needle, haystack, max_k, SearchType::Best, LEVENSHTEIN_COSTS, false)
}

pub fn levenshtein_search_simd_with_opts(needle: &[u8], haystack: &[u8], k: u32, search_type: SearchType, costs: EditCosts, anchored: bool) -> Vec<Match> {
    if needle.len() == 0 {
        return vec![];
    }

    costs.check_search();

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        let unit_k = k / (costs.gap_cost as u32);
        // either the length of the match or the number of edits may exceed the maximum
        // available int size; additionally, max_value is used to indicate overflow
        let max_k = std::cmp::max(needle.len() as u32 + unit_k, k + 1);

        if is_x86_feature_detected!("avx2") {
            if needle.len() <= (1 * 32) && max_k <= u8::max_value() as u32 {
                return unsafe {levenshtein_search_simd_core::<Avx1x32x8>(needle, haystack, k, search_type, costs, anchored)};
            }else if needle.len() <= (2 * 32) && max_k <= u8::max_value() as u32 {
                return unsafe {levenshtein_search_simd_core::<Avx2x32x8>(needle, haystack, k, search_type, costs, anchored)};
            }else if needle.len() <= (4 * 32) && max_k <= u8::max_value() as u32 {
                return unsafe {levenshtein_search_simd_core::<Avx4x32x8>(needle, haystack, k, search_type, costs, anchored)};
            }else if needle.len() <= (8 * 32) && max_k <= u8::max_value() as u32 {
                return unsafe {levenshtein_search_simd_core::<Avx8x32x8>(needle, haystack, k, search_type, costs, anchored)};
            }else if max_k <= u16::max_value() as u32 {

            }
        }else if is_x86_feature_detected!("sse4.1") {
            if needle.len() <= (1 * 16) && max_k <= u8::max_value() as u32 {
                return unsafe {levenshtein_search_simd_core::<Avx1x32x8>(needle, haystack, k, search_type, costs, anchored)};
            }else if needle.len() <= (2 * 16) && max_k <= u8::max_value() as u32 {
                return unsafe {levenshtein_search_simd_core::<Avx2x32x8>(needle, haystack, k, search_type, costs, anchored)};
            }else if needle.len() <= (4 * 16) && max_k <= u8::max_value() as u32 {
                return unsafe {levenshtein_search_simd_core::<Avx4x32x8>(needle, haystack, k, search_type, costs, anchored)};
            }else if needle.len() <= (8 * 16) && max_k <= u8::max_value() as u32 {
                return unsafe {levenshtein_search_simd_core::<Avx8x32x8>(needle, haystack, k, search_type, costs, anchored)};
            }else if max_k <= u16::max_value() as u32 {

            }
        }
    }

    levenshtein_search_naive_with_opts(needle, haystack, k, search_type, costs, anchored)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,sse4.1")]
unsafe fn levenshtein_search_simd_core<T: Jewel>(needle: &[u8], haystack: &[u8], k: u32, search_type: SearchType, costs: EditCosts, anchored: bool) -> Vec<Match> {
    let needle_len = needle.len();
    let haystack_len = haystack.len();
    let mut dp0 = T::repeating_max(needle_len);
    let mut dp_temp = T::repeating_max(needle_len);
    let mut dp1 = T::repeating_max(needle_len);
    let mut dp2 = T::repeating_max(needle_len);
    dp2.slow_insert(dp2.upper_bound() - 1, costs.gap_cost as u32); // last cell

    // save length instead of start idx due to int size constraints
    let mut length0 = T::repeating(0, needle_len);
    let mut length_temp = T::repeating(0, needle_len);
    let mut length1 = T::repeating(0, needle_len);
    let mut length2 = T::repeating(0, needle_len);

    let ones = T::repeating(1, needle_len);
    let twos = T::repeating(2, needle_len);

    // the suffix of haystack can be ignored if needle must be anchored
    let len = if anchored {
        needle_len + std::cmp::min(haystack_len, needle_len + (k as usize) / (costs.gap_cost as usize))
    }else{
        needle_len + haystack_len
    };

    let final_idx = dp1.upper_bound() - needle_len;

    let mut res = Vec::with_capacity((len - needle_len) >> 2);

    // load needle characters into needle_window in reversed order
    let mut needle_window = T::repeating(0, needle_len);
    needle_window.slow_loadu(needle_window.upper_bound() - 1, needle.as_ptr(), needle_len, true);

    let mut haystack_window = T::repeating(0, needle_len);
    let mut haystack_idx = 0usize;
    let mut curr_k = k;

    let mut match_mask0 = T::repeating(0, needle_len);
    let mut match_mask1 = T::repeating(0, needle_len);
    let mut match_mask_cost = T::repeating(0, needle_len);
    let mut sub = T::repeating(0, needle_len);
    let mut sub_length = T::repeating(0, needle_len);
    let mut needle_gap = T::repeating(0, needle_len);
    let mut needle_gap_length = T::repeating(0, needle_len);
    let mut haystack_gap = T::repeating(0, needle_len);
    let mut haystack_gap_length = T::repeating(0, needle_len);
    let mut transpose = T::repeating(0, needle_len);
    let mut transpose_length = T::repeating(0, needle_len);

    let mismatch_cost = T::repeating(costs.mismatch_cost as u32, needle_len);
    let gap_cost = T::repeating(costs.gap_cost as u32, needle_len);
    let transpose_cost = match costs.transpose_cost {
        Some(cost) => T::repeating(cost as u32, needle_len),
        None => T::repeating(0, needle_len)
    };
    let allow_transpose = costs.transpose_cost.is_some();

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

        T::cmpeq(&needle_window, &haystack_window, &mut match_mask1);
        T::andnot(&match_mask1, &mismatch_cost, &mut match_mask_cost);

        // match/mismatch
        T::shift_left_1(&dp1, &mut sub);

        if anchored {
            // dp1 is 2 diagonals behind the current i
            // must be capped at k to prevent overflow when inserting
            sub.insert_last_0(std::cmp::min((i as u32 - 1) * (costs.gap_cost as u32), k + 1));
        }

        sub.adds_mut(&match_mask_cost);

        T::shift_left_1(&length1, &mut sub_length); // zeros are shifted in
        sub_length.add_mut(&ones);

        // gap in needle
        T::adds(&dp2, &gap_cost, &mut needle_gap);
        T::add(&length2, &ones, &mut needle_gap_length);

        // gap in haystack
        T::shift_left_1(&dp2, &mut haystack_gap); // zeros are shifted in

        if anchored {
            // dp2 is one diagonal behind the current i
            haystack_gap.insert_last_0(std::cmp::min((i as u32) * (costs.gap_cost as u32), k + 1));
        }

        haystack_gap.adds_mut(&gap_cost);

        T::shift_left_1(&length2, &mut haystack_gap_length); // zeros are shifted in

        if allow_transpose {
            T::shift_left_1(&match_mask0, &mut transpose); // reuse transpose
            transpose.and_mut(&match_mask0);
            T::andnot(&match_mask1, &transpose, &mut match_mask0); // reuse match_mask0 to represent transpose mask
            dp0.shift_left_2_mut();

            if anchored && i > 3 {
                dp0.insert_last_1(std::cmp::min((i as u32 - 2) * (costs.gap_cost as u32), k + 1));
            }

            length0.shift_left_2_mut();
            T::adds(&dp0, &transpose_cost, &mut transpose);
            T::add(&length0, &twos, &mut transpose_length);
        }

        T::triple_min_length(&sub, &needle_gap, &haystack_gap, &sub_length,
                             &needle_gap_length, &haystack_gap_length, &mut dp0, &mut length0);

        if allow_transpose {
            // blend using transpose mask
            dp0.blendv_mut(&transpose, &match_mask0);
            length0.blendv_mut(&transpose_length, &match_mask0);
            std::mem::swap(&mut match_mask0, &mut match_mask1);
        }

        std::mem::swap(&mut dp0, &mut dp_temp);
        std::mem::swap(&mut dp_temp, &mut dp1);
        std::mem::swap(&mut dp1, &mut dp2);
        std::mem::swap(&mut length0, &mut length_temp);
        std::mem::swap(&mut length_temp, &mut length1);
        std::mem::swap(&mut length1, &mut length2);

        if i >= needle_len - 1 {
            let final_res = dp2.slow_extract(final_idx);
            let final_length = length2.slow_extract(final_idx) as usize;

            if final_res <= curr_k {
                let end_idx = i + 1 - needle_len;
                res.push(Match{start: end_idx - final_length, end: end_idx, k: final_res});

                match search_type {
                    SearchType::First => break,
                    // if we want the best, then we can shrink the k threshold
                    SearchType::Best => curr_k = final_res,
                    _ => ()
                }
            }
        }
    }

    if search_type == SearchType::Best {
        res.retain(|m| m.k == curr_k); // only retain matches with the lowest k
    }

    res
}

