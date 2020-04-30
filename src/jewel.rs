use std::fmt;

#[cfg(target_arch = "x86")]
use core::arch::x86::*;

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Jewel provides a uniform interface for SIMD operations.
///
/// To save space, most operations are modify in place.
pub trait Jewel: fmt::Display {
    /// Functions for creating a Jewel vector.
    unsafe fn repeating(val: u32, len: usize) -> Self;
    unsafe fn repeating_max(len: usize) -> Self;
    unsafe fn loadu(ptr: *const u8, len: usize) -> Self;

    /// Figure out the length of the created vector, which may
    /// be higher than the length given by the caller.
    fn upper_bound(&self) -> usize;

    /// These operations are modify in place, so less memory allocations are needed
    /// on long sequences of operations.
    unsafe fn slow_loadu(&mut self, idx: usize, ptr: *const u8, len: usize, reverse: bool);

    unsafe fn slow_extract(&self, i: usize) -> u32;
    unsafe fn slow_insert(&mut self, i: usize, val: u32);
    /// last_0 is the last element, last_1 is the second to last, etc.
    unsafe fn insert_last_0(&mut self, val: u32);
    unsafe fn insert_last_1(&mut self, val: u32);
    unsafe fn insert_last_2(&mut self, val: u32);
    unsafe fn insert_last_max(&mut self);
    unsafe fn insert_first(&mut self, val: u32);
    unsafe fn insert_first_max(&mut self);

    /// For speed, the `count_mismatches` functions do not require creating a Jewel vector.
    unsafe fn mm_count_mismatches(a_ptr: *const u8, b_ptr: *const u8, len: usize) -> u32;
    unsafe fn count_mismatches(a_ptr: *const u8, b_ptr: *const u8, len: usize) -> u32;
    unsafe fn vector_count_mismatches(a: &Self, b_ptr: *const u8) -> u32;

    unsafe fn add_mut(&mut self, b: &Self);
    unsafe fn adds_mut(&mut self, b: &Self);
    unsafe fn and_mut(&mut self, b: &Self);
    unsafe fn cmpeq_mut(&mut self, b: &Self);
    unsafe fn min_mut(&mut self, b: &Self);
    unsafe fn max_mut(&mut self, b: &Self);
    unsafe fn shift_left_1_mut(&mut self);
    unsafe fn shift_right_1_mut(&mut self);

    /// Overwrite a res vector to reduce memory allocations
    unsafe fn add(a: &Self, b: &Self, res: &mut Self);
    unsafe fn adds(a: &Self, b: &Self, res: &mut Self);
    unsafe fn and(a: &Self, b: &Self, res: &mut Self);
    unsafe fn cmpeq(a: &Self, b: &Self, res: &mut Self);
    unsafe fn min(a: &Self, b: &Self, res: &mut Self);
    unsafe fn max(a: &Self, b: &Self, res: &mut Self);
    unsafe fn shift_left_1(a: &Self, res: &mut Self);
    unsafe fn shift_right_1(a: &Self, res: &mut Self);
    unsafe fn blendv(a: &Self, b: &Self, mask: &Self, res: &mut Self);

    unsafe fn triple_min_length(sub: &Self, a_gap: &Self, b_gap: &Self, sub_length: &Self,
                                a_gap_length: &Self, b_gap_length: &Self, res_min: &mut Self, res_length: &mut Self);
    unsafe fn triple_argmin(sub: &Self, a_gap: &Self, b_gap: &Self, res_min: &mut Self) -> Self;
}

macro_rules! operation_param2 {
    ($target:literal, $struct_name:ident, $fn_name:ident, $intrinsic:ident) => {
        #[target_feature(enable = $target)]
        #[inline]
        unsafe fn $fn_name(a: &$struct_name, b: &$struct_name, res: &mut $struct_name) {
            for i in 0..a.v.len() {
                *res.v.get_unchecked_mut(i) = $intrinsic(*a.v.get_unchecked(i), *b.v.get_unchecked(i));
            }
        }
    };
}

macro_rules! single_operation_param2 {
    ($target:literal, $struct_name:ident, $fn_name:ident, $intrinsic:ident) => {
        #[target_feature(enable = $target)]
        #[inline]
        unsafe fn $fn_name(a: &$struct_name, b: &$struct_name, res: &mut $struct_name) {
            res.v = $intrinsic(a.v, b.v);
        }
    };
}

macro_rules! operation_mut_param2 {
    ($target:literal, $struct_name:ident, $fn_name:ident, $intrinsic:ident) => {
        #[target_feature(enable = $target)]
        #[inline]
        unsafe fn $fn_name(&mut self, b: &$struct_name) {
            for i in 0..self.v.len() {
                *self.v.get_unchecked_mut(i) = $intrinsic(*self.v.get_unchecked(i), *b.v.get_unchecked(i));
            }
        }
    };
}

macro_rules! single_operation_mut_param2 {
    ($target:literal, $struct_name:ident, $fn_name:ident, $intrinsic:ident) => {
        #[target_feature(enable = $target)]
        #[inline]
        unsafe fn $fn_name(&mut self, b: &$struct_name) {
            self.v = $intrinsic(self.v, b.v);
        }
    };
}

/// N x 32 x 8 vector backed with 256-bit AVX2 vectors
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub struct AvxNx32x8 {
    v: Vec<__m256i>
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl Jewel for AvxNx32x8 {
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn repeating(val: u32, len: usize) -> AvxNx32x8 {
        let v = vec![_mm256_set1_epi8(val as i8); (len >> 5) + if (len & 31) > 0 {1} else {0}];

        AvxNx32x8{
            v: v
        }
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn repeating_max(len: usize) -> AvxNx32x8 {
        let v = vec![_mm256_set1_epi8(-1i8); (len >> 5) + if (len & 31) > 0 {1} else {0}];

        AvxNx32x8{
            v: v
        }
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn loadu(ptr: *const u8, len: usize) -> AvxNx32x8 {
        let word_len = len >> 5;
        let word_rem = len & 31;
        let mut v = Vec::with_capacity(word_len + if word_rem > 0 {1} else {0});
        let avx2_ptr = ptr as *const __m256i;

        for i in 0..word_len {
            v.push(_mm256_loadu_si256(avx2_ptr.offset(i as isize)));
        }

        if word_rem > 0 {
            let mut arr = [0u8; 32];
            let end_ptr = ptr.offset((word_len << 5) as isize);

            for i in 0..word_rem {
                *arr.get_unchecked_mut(i) = *end_ptr.offset(i as isize);
            }

            v.push(_mm256_loadu_si256(arr.as_ptr() as *const __m256i));
        }

        AvxNx32x8{
            v: v
        }
    }

    #[inline]
    fn upper_bound(&self) -> usize {
        self.v.len() << 5
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn slow_loadu(&mut self, idx: usize, ptr: *const u8, len: usize, reverse: bool) {
        if len == 0 {
            return;
        }

        let mut arr = [0u8; 32];
        let arr_ptr = arr.as_mut_ptr() as *mut __m256i;

        for i in 0..len {
            let curr_idx = if reverse {idx - i} else {idx + i};
            let arr_idx = curr_idx & 31;

            if arr_idx == 0 || i == 0 {
                _mm256_storeu_si256(arr_ptr, *self.v.get_unchecked(curr_idx >> 5));
            }

            *arr.get_unchecked_mut(arr_idx) = *ptr.offset(i as isize);

            if arr_idx == 31 || i == len - 1 {
                *self.v.get_unchecked_mut(curr_idx >> 5) = _mm256_loadu_si256(arr_ptr);
            }
        }
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn slow_extract(&self, i: usize) -> u32 {
        let idx = i >> 5;
        let j = i & 31;
        let mut arr = [0u8; 32];
        _mm256_storeu_si256(arr.as_mut_ptr() as *mut __m256i, *self.v.get_unchecked(idx));
        *arr.get_unchecked(j) as u32
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn slow_insert(&mut self, i: usize, val: u32) {
        let idx = i >> 5;
        let j = i & 31;
        let mut arr = [0u8; 32];
        let arr_ptr = arr.as_mut_ptr() as *mut __m256i;
        _mm256_storeu_si256(arr_ptr, *self.v.get_unchecked(idx));
        *arr.get_unchecked_mut(j) = val as u8;
        *self.v.get_unchecked_mut(idx) = _mm256_loadu_si256(arr_ptr);
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn insert_last_0(&mut self, val: u32) {
        let last = self.v.len() - 1;
        *self.v.get_unchecked_mut(last) = _mm256_insert_epi8(*self.v.get_unchecked(last), val as i8, 31i32);
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn insert_last_1(&mut self, val: u32) {
        let last = self.v.len() - 1;
        *self.v.get_unchecked_mut(last) = _mm256_insert_epi8(*self.v.get_unchecked(last), val as i8, 30i32);
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn insert_last_2(&mut self, val: u32) {
        let last = self.v.len() - 1;
        *self.v.get_unchecked_mut(last) = _mm256_insert_epi8(*self.v.get_unchecked(last), val as i8, 29i32);
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn insert_last_max(&mut self) {
        let last = self.v.len() - 1;
        *self.v.get_unchecked_mut(last) = _mm256_insert_epi8(*self.v.get_unchecked(last), -1i8, 31i32);
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn insert_first(&mut self, val: u32) {
        *self.v.get_unchecked_mut(0) = _mm256_insert_epi8(*self.v.get_unchecked(0), val as i8, 0i32);
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn insert_first_max(&mut self) {
        *self.v.get_unchecked_mut(0) = _mm256_insert_epi8(*self.v.get_unchecked(0), -1i8, 0i32);
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn mm_count_mismatches(a_ptr: *const u8, b_ptr: *const u8, len: usize) -> u32 {
        let mut res = 0u32;
        let div_len = (len >> 5) as isize;
        let avx2_a_ptr = a_ptr as *const __m256i;
        let avx2_b_ptr = b_ptr as *const __m256i;

        for i in 0..div_len {
            let a = _mm256_loadu_si256(avx2_a_ptr.offset(i));
            let b = _mm256_loadu_si256(avx2_b_ptr.offset(i));
            let eq = _mm256_cmpeq_epi8(a, b);
            res += _mm256_movemask_epi8(eq).count_ones();
        }

        for i in (div_len << 5)..len as isize {
            res += (*a_ptr.offset(i) == *b_ptr.offset(i)) as u32;
        }

        len as u32 - res
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn count_mismatches(a_ptr: *const u8, b_ptr: *const u8, len: usize) -> u32 {
        let refresh_len = (len / (255 * 32)) as isize;
        let zeros = _mm256_setzero_si256();
        let mut sad = zeros;
        let avx2_a_ptr = a_ptr as *const __m256i;
        let avx2_b_ptr = b_ptr as *const __m256i;

        for i in 0..refresh_len {
            let mut curr = zeros;

            for j in (i * 255)..((i + 1) * 255) {
                let a = _mm256_loadu_si256(avx2_a_ptr.offset(j));
                let b = _mm256_loadu_si256(avx2_b_ptr.offset(j));
                let eq = _mm256_cmpeq_epi8(a, b);
                curr = _mm256_sub_epi8(curr, eq); // subtract -1 = add 1 when matching
                // counting matches instead of mismatches for speed
            }

            // subtract 0 and sum up 8 bytes at once horizontally into four 64 bit ints
            // accumulate those 64 bit ints
            sad = _mm256_add_epi64(sad, _mm256_sad_epu8(curr, zeros));
        }

        let word_len = (len >> 5) as isize;
        let mut curr = zeros;

        // leftover blocks of 32 bytes
        for i in (refresh_len * 255)..word_len {
            let a = _mm256_loadu_si256(avx2_a_ptr.offset(i));
            let b = _mm256_loadu_si256(avx2_b_ptr.offset(i));
            let eq = _mm256_cmpeq_epi8(a, b);
            curr = _mm256_sub_epi8(curr, eq); // subtract -1 = add 1 when matching
        }

        sad = _mm256_add_epi64(sad, _mm256_sad_epu8(curr, zeros));
        let mut sad_arr = [0u32; 8];
        _mm256_storeu_si256(sad_arr.as_mut_ptr() as *mut __m256i, sad);
        let mut res = *sad_arr.get_unchecked(0) + *sad_arr.get_unchecked(2)
            + *sad_arr.get_unchecked(4) + *sad_arr.get_unchecked(6);

        for i in (word_len << 5)..len as isize {
            res += (*a_ptr.offset(i) == *b_ptr.offset(i)) as u32;
        }

        len as u32 - res
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn vector_count_mismatches(a: &AvxNx32x8, b_ptr: *const u8) -> u32 {
        let refresh_len = (a.v.len() / 255) as isize;
        let zeros = _mm256_setzero_si256();
        let mut sad = zeros;
        let avx2_b_ptr = b_ptr as *const __m256i;

        for i in 0..refresh_len {
            let mut curr = zeros;

            for j in (i * 255)..((i + 1) * 255) {
                let a = *a.v.get_unchecked(j as usize);
                let b = _mm256_loadu_si256(avx2_b_ptr.offset(j));
                let eq = _mm256_cmpeq_epi8(a, b);
                curr = _mm256_sub_epi8(curr, eq); // subtract -1 = add 1 when matching
                // counting matches instead of mismatches for speed
            }

            // subtract 0 and sum up 8 bytes at once horizontally into four 64 bit ints
            // accumulate those 64 bit ints
            sad = _mm256_add_epi64(sad, _mm256_sad_epu8(curr, zeros));
        }

        let mut curr = zeros;

        // leftover blocks of 32 bytes
        for i in (refresh_len * 255)..a.v.len() as isize {
            let a = *a.v.get_unchecked(i as usize);
            let b = _mm256_loadu_si256(avx2_b_ptr.offset(i));
            let eq = _mm256_cmpeq_epi8(a, b);
            curr = _mm256_sub_epi8(curr, eq); // subtract -1 = add 1 when matching
        }

        sad = _mm256_add_epi64(sad, _mm256_sad_epu8(curr, zeros));
        let mut sad_arr = [0u32; 8];
        _mm256_storeu_si256(sad_arr.as_mut_ptr() as *mut __m256i, sad);
        let res = *sad_arr.get_unchecked(0) + *sad_arr.get_unchecked(2)
            + *sad_arr.get_unchecked(4) + *sad_arr.get_unchecked(6);

        (a.v.len() << 5) as u32 - res
    }

    operation_mut_param2!("avx2", AvxNx32x8, add_mut, _mm256_add_epi8);
    operation_mut_param2!("avx2", AvxNx32x8, adds_mut, _mm256_adds_epu8);
    operation_mut_param2!("avx2", AvxNx32x8, and_mut, _mm256_and_si256);
    operation_mut_param2!("avx2", AvxNx32x8, cmpeq_mut, _mm256_cmpeq_epi8);
    operation_mut_param2!("avx2", AvxNx32x8, min_mut, _mm256_min_epu8);
    operation_mut_param2!("avx2", AvxNx32x8, max_mut, _mm256_max_epu8);

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn shift_left_1_mut(&mut self) {
        for i in 0..(self.v.len() - 1) {
            let curr = *self.v.get_unchecked(i);
            // permute concatenates the second half of the current vector and the first half of the next vector
            *self.v.get_unchecked_mut(i) = _mm256_alignr_epi8(
                _mm256_permute2x128_si256(curr, *self.v.get_unchecked(i + 1), 0b00100001i32), curr, 1i32);
        }

        // last one gets to shift in zeros
        let last = self.v.len() - 1;
        let curr = *self.v.get_unchecked(last);
        // permute concatenates the second half of the last vector and a vector of zeros
        *self.v.get_unchecked_mut(last) = _mm256_alignr_epi8(_mm256_permute2x128_si256(curr, curr, 0b10000001i32), curr, 1i32);
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn shift_right_1_mut(&mut self) {
        for i in (1..self.v.len()).rev() {
            let curr = *self.v.get_unchecked(i);
            // permute concatenates the second half of the previous vector and the first half of the current vector
            *self.v.get_unchecked_mut(i) = _mm256_alignr_epi8(
                curr, _mm256_permute2x128_si256(curr, *self.v.get_unchecked(i - 1), 0b00000011i32), 15i32);
        }

        // first one gets to shift in zeros
        let curr = *self.v.get_unchecked(0);
        // permute concatenates a vector of zeros and the first half of the first vector
        *self.v.get_unchecked_mut(0) = _mm256_alignr_epi8(curr, _mm256_permute2x128_si256(curr, curr, 0b00001000i32), 15i32);
    }

    operation_param2!("avx2", AvxNx32x8, add, _mm256_add_epi8);
    operation_param2!("avx2", AvxNx32x8, adds, _mm256_adds_epu8);
    operation_param2!("avx2", AvxNx32x8, and, _mm256_and_si256);
    operation_param2!("avx2", AvxNx32x8, cmpeq, _mm256_cmpeq_epi8);
    operation_param2!("avx2", AvxNx32x8, min, _mm256_min_epu8);
    operation_param2!("avx2", AvxNx32x8, max, _mm256_max_epu8);

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn shift_left_1(a: &AvxNx32x8, res: &mut AvxNx32x8) {
        for i in 0..(a.v.len() - 1) {
            let curr = *a.v.get_unchecked(i);
            // permute concatenates the second half of the current vector and the first half of the next vector
            *res.v.get_unchecked_mut(i) = _mm256_alignr_epi8(
                _mm256_permute2x128_si256(curr, *a.v.get_unchecked(i + 1), 0b00100001i32), curr, 1i32);
        }

        // last one gets to shift in zeros
        let last = a.v.len() - 1;
        let curr = *a.v.get_unchecked(last);
        // permute concatenates the second half of the last vector and a vector of zeros
        *res.v.get_unchecked_mut(last) = _mm256_alignr_epi8(_mm256_permute2x128_si256(curr, curr, 0b10000001i32), curr, 1i32);
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn shift_right_1(a: &AvxNx32x8, res: &mut AvxNx32x8) {
        for i in (1..a.v.len()).rev() {
            let curr = *a.v.get_unchecked(i);
            // permute concatenates the second half of the previous vector and the first half of the current vector
            *res.v.get_unchecked_mut(i) = _mm256_alignr_epi8(
                curr, _mm256_permute2x128_si256(curr, *a.v.get_unchecked(i - 1), 0b00000011i32), 15i32);
        }

        // first one gets to shift in zeros
        let curr = *a.v.get_unchecked(0);
        // permute concatenates a vector of zeros and the first half of the first vector
        *res.v.get_unchecked_mut(0) = _mm256_alignr_epi8(curr, _mm256_permute2x128_si256(curr, curr, 0b00001000i32), 15i32);
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn blendv(a: &AvxNx32x8, b: &AvxNx32x8, mask: &AvxNx32x8, res: &mut AvxNx32x8) {
        for i in 0..a.v.len() {
            *res.v.get_unchecked_mut(i) = _mm256_blendv_epi8(
                *a.v.get_unchecked(i), *b.v.get_unchecked(i), *mask.v.get_unchecked(i));
        }
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn triple_argmin(sub: &AvxNx32x8, a_gap: &AvxNx32x8, b_gap: &AvxNx32x8, res_min: &mut AvxNx32x8) -> AvxNx32x8 {
        // return the edit used in addition to doing a min operation
        // hide latency by minimizing dependencies
        let mut v = Vec::with_capacity(sub.v.len());
        let twos = _mm256_set1_epi8(2);

        for i in 0..sub.v.len() {
            let sub = *sub.v.get_unchecked(i);
            let a_gap = *a_gap.v.get_unchecked(i);
            let b_gap = *b_gap.v.get_unchecked(i);

            let res_min1 = _mm256_min_epu8(a_gap, b_gap);
            // a gap: 2 + -1 = 1, b gap: 2 + 0 = 2
            let res_arg1 = _mm256_add_epi8(twos, _mm256_cmpeq_epi8(a_gap, res_min1));

            let res_min2 = _mm256_min_epu8(sub, res_min1);
            // sub: 0
            let res_arg2 = _mm256_andnot_si256(res_arg1, _mm256_cmpeq_epi8(sub, res_min2));

            *res_min.v.get_unchecked_mut(i) = res_min2;
            v.push(res_arg2);
        }

        AvxNx32x8{
            v: v
        }
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn triple_min_length(sub: &AvxNx32x8, a_gap: &AvxNx32x8,
                                b_gap: &AvxNx32x8, sub_length: &AvxNx32x8, a_gap_length: &AvxNx32x8,
                                b_gap_length: &AvxNx32x8, res_min: &mut AvxNx32x8, res_length: &mut AvxNx32x8) {
        // choose the length based on which edit is chosen during the min operation
        // hide latency by minimizing dependencies
        // secondary objective of maximizing length if edit costs equal
        for i in 0..sub.v.len() {
            let sub = *sub.v.get_unchecked(i);
            let a_gap = *a_gap.v.get_unchecked(i);
            let b_gap = *b_gap.v.get_unchecked(i);
            let sub_length = *sub_length.v.get_unchecked(i);
            let a_gap_length = *a_gap_length.v.get_unchecked(i);
            let b_gap_length = *b_gap_length.v.get_unchecked(i);

            let res_min1 = _mm256_min_epu8(a_gap, b_gap);
            let a_b_gt_mask = _mm256_cmpeq_epi8(a_gap, res_min1); // a gap: -1, b gap: 0
            let mut res_length1 = _mm256_blendv_epi8(b_gap_length, a_gap_length, a_b_gt_mask); // lengths based on edits
            let a_b_eq_mask = _mm256_cmpeq_epi8(a_gap, b_gap); // equal: -1
            let a_b_max_len = _mm256_max_epu8(a_gap_length, b_gap_length);
            res_length1 = _mm256_blendv_epi8(res_length1, a_b_max_len, a_b_eq_mask); // maximize length if edits equal

            let res_min2 = _mm256_min_epu8(sub, res_min1);
            let sub_gt_mask = _mm256_cmpeq_epi8(sub, res_min2); // sub: -1, prev a or b gap: 0
            let mut res_length2 = _mm256_blendv_epi8(res_length1, sub_length, sub_gt_mask); // length based on edits
            let sub_eq_mask = _mm256_cmpeq_epi8(sub, res_min1);
            let sub_max_len = _mm256_max_epu8(sub_length, res_length1);
            res_length2 = _mm256_blendv_epi8(res_length2, sub_max_len, sub_eq_mask); // maximize length if edits equal

            *res_min.v.get_unchecked_mut(i) = res_min2;
            *res_length.v.get_unchecked_mut(i) = res_length2;
        }
    }
}

// this implementation will probably only be used for debugging
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl fmt::Display for AvxNx32x8 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        unsafe {
            #![target_feature(enable = "avx2")]
            write!(f, "[")?;

            let mut arr = [0u8; 32];
            let arr_ptr = arr.as_mut_ptr() as *mut __m256i;

            for i in 0..(self.v.len() - 1) {
                _mm256_storeu_si256(arr_ptr, *self.v.get_unchecked(i));

                for j in 0..32 {
                    write!(f, "{:>3}, ", *arr.get_unchecked(j))?;
                }
            }

            // leftover elements

            _mm256_storeu_si256(arr_ptr, *self.v.get_unchecked(self.v.len() - 1));

            let start = (self.v.len() - 1) << 5;

            for i in 0..(self.upper_bound() - start) {
                if i == self.upper_bound() - start - 1 {
                    write!(f, "{:>3}", *arr.get_unchecked(i))?;
                }else{
                    write!(f, "{:>3}, ", *arr.get_unchecked(i))?;
                }
            }

            write!(f, "]")
        }
    }
}

/// 1 x 32 x 8 vector backed with one AVX2 vector
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub struct Avx1x32x8 {
    v: __m256i
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl Jewel for Avx1x32x8 {
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn repeating(val: u32, _len: usize) -> Avx1x32x8 {
        Avx1x32x8{
            v: _mm256_set1_epi8(val as i8)
        }
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn repeating_max(_len: usize) -> Avx1x32x8 {
        Avx1x32x8{
            v: _mm256_set1_epi8(i8::max_value())
        }
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn loadu(ptr: *const u8, len: usize) -> Avx1x32x8 {
        let mut arr = [0u8; 32];

        for i in 0..len {
            *arr.get_unchecked_mut(i) = *ptr.offset(i as isize);
        }

        let v = _mm256_loadu_si256(arr.as_ptr() as *const __m256i);

        Avx1x32x8{
            v: v
        }
    }

    #[inline]
    fn upper_bound(&self) -> usize {
        32
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn slow_loadu(&mut self, idx: usize, ptr: *const u8, len: usize, reverse: bool) {
        if len == 0 {
            return;
        }

        let mut arr = [0u8; 32];
        let arr_ptr = arr.as_mut_ptr() as *mut __m256i;

        for i in 0..len {
            let curr_idx = if reverse {idx - i} else {idx + i};
            *arr.get_unchecked_mut(curr_idx) = *ptr.offset(i as isize);
        }

        self.v = _mm256_loadu_si256(arr_ptr);
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn slow_extract(&self, i: usize) -> u32 {
        let mut arr = [0u8; 32];
        _mm256_storeu_si256(arr.as_mut_ptr() as *mut __m256i, self.v);
        *arr.get_unchecked(i) as u32
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn slow_insert(&mut self, i: usize, val: u32) {
        let mut arr = [0u8; 32];
        let arr_ptr = arr.as_mut_ptr() as *mut __m256i;
        _mm256_storeu_si256(arr_ptr, self.v);
        *arr.get_unchecked_mut(i) = val as u8;
        self.v = _mm256_loadu_si256(arr_ptr);
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn insert_last_0(&mut self, val: u32) {
        self.v = _mm256_insert_epi8(self.v, val as i8, 31i32);
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn insert_last_1(&mut self, val: u32) {
        self.v = _mm256_insert_epi8(self.v, val as i8, 30i32);
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn insert_last_2(&mut self, val: u32) {
        self.v = _mm256_insert_epi8(self.v, val as i8, 29i32);
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn insert_last_max(&mut self) {
        self.v = _mm256_insert_epi8(self.v, i8::max_value(), 31i32);
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn insert_first(&mut self, val: u32) {
        self.v = _mm256_insert_epi8(self.v, val as i8, 0i32);
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn insert_first_max(&mut self) {
        self.v = _mm256_insert_epi8(self.v, i8::max_value(), 0i32);
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn mm_count_mismatches(_a_ptr: *const u8, _b_ptr: *const u8, _len: usize) -> u32 {
        unimplemented!();
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn count_mismatches(_a_ptr: *const u8, _b_ptr: *const u8, _len: usize) -> u32 {
        unimplemented!();
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn vector_count_mismatches(_a: &Avx1x32x8, _b_ptr: *const u8) -> u32 {
        unimplemented!();
    }

    single_operation_mut_param2!("avx2", Avx1x32x8, add_mut, _mm256_add_epi8);
    single_operation_mut_param2!("avx2", Avx1x32x8, adds_mut, _mm256_adds_epi8);
    single_operation_mut_param2!("avx2", Avx1x32x8, and_mut, _mm256_and_si256);
    single_operation_mut_param2!("avx2", Avx1x32x8, cmpeq_mut, _mm256_cmpeq_epi8);
    single_operation_mut_param2!("avx2", Avx1x32x8, min_mut, _mm256_min_epi8);
    single_operation_mut_param2!("avx2", Avx1x32x8, max_mut, _mm256_max_epi8);

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn shift_left_1_mut(&mut self) {
        // permute concatenates the second half of the last vector and a vector of zeros
        self.v = _mm256_alignr_epi8(_mm256_permute2x128_si256(self.v, self.v, 0b10000001i32), self.v, 1i32);
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn shift_right_1_mut(&mut self) {
        // permute concatenates a vector of zeros and the first half of the first vector
        self.v = _mm256_alignr_epi8(self.v, _mm256_permute2x128_si256(self.v, self.v, 0b00001000i32), 15i32);
    }

    single_operation_param2!("avx2", Avx1x32x8, add, _mm256_add_epi8);
    single_operation_param2!("avx2", Avx1x32x8, adds, _mm256_adds_epi8);
    single_operation_param2!("avx2", Avx1x32x8, and, _mm256_and_si256);
    single_operation_param2!("avx2", Avx1x32x8, cmpeq, _mm256_cmpeq_epi8);
    single_operation_param2!("avx2", Avx1x32x8, min, _mm256_min_epi8);
    single_operation_param2!("avx2", Avx1x32x8, max, _mm256_max_epi8);

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn shift_left_1(a: &Avx1x32x8, res: &mut Avx1x32x8) {
        // permute concatenates the second half of the last vector and a vector of zeros
        res.v = _mm256_alignr_epi8(_mm256_permute2x128_si256(a.v, a.v, 0b10000001i32), a.v, 1i32);
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn shift_right_1(a: &Avx1x32x8, res: &mut Avx1x32x8) {
        // permute concatenates a vector of zeros and the first half of the first vector
        res.v = _mm256_alignr_epi8(a.v, _mm256_permute2x128_si256(a.v, a.v, 0b00001000i32), 15i32);
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn blendv(a: &AvxNx32x8, b: &AvxNx32x8, mask: &AvxNx32x8, res: &mut AvxNx32x8) {
        res.v = _mm256_blendv_epi8(a.v, b.v, mask.v);
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn triple_argmin(sub: &Avx1x32x8, a_gap: &Avx1x32x8, b_gap: &Avx1x32x8, res_min: &mut Avx1x32x8) -> Avx1x32x8 {
        // return the edit used in addition to doing a min operation
        // hide latency by minimizing dependencies

        let res_min1 = _mm256_min_epi8(a_gap.v, b_gap.v);
        // a gap: 1 - 0 = 1, b gap: 1 - -1 = 2
        let res_arg1 = _mm256_sub_epi8(_mm256_set1_epi8(1), _mm256_cmpgt_epi8(a_gap.v, b_gap.v));

        res_min.v = _mm256_min_epi8(sub.v, res_min1);
        // sub: 0
        let res_arg2 = _mm256_and_si256(_mm256_cmpgt_epi8(sub.v, res_min1), res_arg1);

        Avx1x32x8{
            v: res_arg2
        }
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn triple_min_length(sub: &Avx1x32x8, a_gap: &Avx1x32x8,
                                b_gap: &Avx1x32x8, sub_length: &Avx1x32x8, a_gap_length: &Avx1x32x8,
                                b_gap_length: &Avx1x32x8, res_min: &mut Avx1x32x8, res_length: &mut Avx1x32x8) {
        // choose the length based on which edit is chosen during the min operation
        // hide latency by minimizing dependencies
        // secondary objective of maximizing length if edit costs equal
        let res_min1 = _mm256_min_epi8(a_gap.v, b_gap.v);
        let a_b_gt_mask = _mm256_cmpgt_epi8(a_gap.v, b_gap.v); // a gap: 0, b gap: -1
        let mut res_length1 = _mm256_blendv_epi8(a_gap_length.v, b_gap_length.v, a_b_gt_mask); // lengths based on edits
        let a_b_eq_mask = _mm256_cmpeq_epi8(a_gap.v, b_gap.v); // equal: -1
        let a_b_max_len = _mm256_max_epi8(a_gap_length.v, b_gap_length.v);
        res_length1 = _mm256_blendv_epi8(res_length1, a_b_max_len, a_b_eq_mask); // maximize length if edits equal

        res_min.v = _mm256_min_epi8(sub.v, res_min1);
        let sub_gt_mask = _mm256_cmpgt_epi8(sub.v, res_min1); // sub: 0, prev a or b gap: -1
        let mut res_length2 = _mm256_blendv_epi8(sub_length.v, res_length1, sub_gt_mask); // length based on edits
        let sub_eq_mask = _mm256_cmpeq_epi8(sub.v, res_min1);
        let sub_max_len = _mm256_max_epi8(sub_length.v, res_length1);
        res_length2 = _mm256_blendv_epi8(res_length2, sub_max_len, sub_eq_mask); // maximize length if edits equal
        res_length.v = res_length2;
    }
}

// this implementation will probably only be used for debugging
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl fmt::Display for Avx1x32x8 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        unsafe {
            #![target_feature(enable = "avx2")]
            write!(f, "[")?;

            let mut arr = [0u8; 32];
            let arr_ptr = arr.as_mut_ptr() as *mut __m256i;

            _mm256_storeu_si256(arr_ptr, self.v);

            for i in 0..self.upper_bound() {
                if i == self.upper_bound() - 1 {
                    write!(f, "{:>3}", *arr.get_unchecked(i))?;
                }else{
                    write!(f, "{:>3}, ", *arr.get_unchecked(i))?;
                }
            }

            write!(f, "]")
        }
    }
}
