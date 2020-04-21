use std::fmt;

#[cfg(target_arch = "x86")]
use core::arch::x86::*;

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Jewel provides a uniform interface for SIMD operations.
///
/// To save space, most operations are modify in place.
trait Jewel {
    unsafe fn new(val: u32, len: usize);
    unsafe fn slow_loadu(&mut self, idx: usize, ptr: *const u8, len: usize, reverse: bool);
    unsafe fn fast_loadu(&mut self, ptr: *const u8);
    unsafe fn add(&mut self, o: Self);
    unsafe fn adds(&mut self, o: Self);
    unsafe fn subs(&mut self, o: Self);
    unsafe fn min(&mut self, o: Self);
    unsafe fn and(&mut self, o: Self);
    unsafe fn cmpeq(&mut self, o: Self);
    unsafe fn cmpgt(&mut self, o: Self);
    unsafe fn blendv(&mut self, o: Self, mask: Self);
    unsafe fn mm_count_zeros(&mut self, o: Self) -> u32;
    unsafe fn mm_count_ones(&mut self, o: Self) -> u32;
    unsafe fn count_ones(&mut self, o: Self) -> u32;
    unsafe fn shift_left(&mut self);
    unsafe fn shift_right(&mut self);
    unsafe fn extract(&mut self, i: usize) -> u32;
    unsafe fn insert_last_0(&mut self, val: u32);
    unsafe fn insert_last_1(&mut self, val: u32);
    unsafe fn insert_last_2(&mut self, val: u32);
    unsafe fn insert_last_max(&mut self);
    unsafe fn insert_first(&mut self, val: u32);
    unsafe fn insert_first_max(&mut self);
}

/// N x 32 x 8 vector backed with 256-bit AVX2 vectors
#[derive(Clone)]
struct AvxNx32x8 {
    len: usize,
    v: Vec<__m256i>
}

impl fmt::Display for AvxNx32x8 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[")?;

        let mut arr = [0u8; 32];
        let arr_ptr = arr.as_mut_ptr() as *mut __m256i;

        for i in 0..(self.v.len() - 1) {
            _mm256_storeu_si256(arr_ptr, *self.v.get_unchecked(i));

            for j in 0..32 {
                write!(f, "{:>3}, ", arr[i])?;
            }
        }

        // leftover elements

        _mm256_storeu_si256(arr_ptr, *self.v.get_unchecked(self.v.len() - 1));

        let start = (self.v.len() - 1) << 5;

        for i in 0..(self.len - start) {
            if i == self.len - start - 1 {
                write!(f, "{:>3}", arr[i])?;
            }else{
                write!(f, "{:>3}, ", arr[i])?;
            }
        }

        write!(f, "]")
    }
}

impl Jewel for AvxNx32x8 {
    unsafe fn new(val: u32, len: usize) {
        let mut v = vec![_mm256_set1_epi8(val as i8; (len >> 5) + if (len & 31) > 0 {1} else {0}];

        AvxNx32x8{
            len: len,
            v: v
        }
    }

    unsafe fn slow_loadu(&mut self, idx: usize, ptr: *const u8, len: usize, reverse: bool) {
        if len == 0 {
            return;
        }

        let mut arr = [0u8; 32];
        let arr_ptr = arr.as_mut_ptr() as *mut __m256i;
        let sign = if reverse {-1} else {1};

        for i in 0..len {
            let curr_idx = idx + sign * i;
            let arr_idx = curr_idx & 31;

            if arr_idx == 0 || i == 0 {
                _mm256_storeu_si256(arr_ptr, *v.get_unchecked(curr_idx >> 5));
            }

            *arr.get_unchecked_mut(arr_idx) = *ptr.offset(i as isize);

            if arr_idx == 31 || i == len - 1 {
                *v.get_unchecked_mut(curr_idx >> 5) = _mm256_loadu_si256(arr_ptr);
            }
        }
    }

    unsafe fn fast_loadu(&mut self, ptr: *const u8) {
        let avx2_ptr = ptr as *const __m256i;

        for i in 0..self.v.len() as isize {
            *v.get_unchecked_mut(i) = _mm256_loadu_si256(*avx2_ptr.offset(i));
        }
    }

    unsafe fn extract(&mut self, i: usize) -> u32 {
        let idx = i >> 5;
        let j = i & 31;
        let mut arr = [0u8; 32];
        _mm256_storeu_si256(arr.as_mut_ptr() as *mut __m256i, *v.get_unchecked(idx));
        *arr.get_unchecked(j) as u32
    }

    unsafe fn add(&mut self, o: AvxNx32x8) {
        for i in 0..self.v.len() {
            *self.v.get_unchecked_mut(i) = _mm256_add_epi8(*self.v.get_unchecked(i), *o.v.get_unchecked(i));
        }
    }

    unsafe fn adds(&mut self, o: AvxNx32x8) {
        for i in 0..self.v.len() {
            *self.v.get_unchecked_mut(i) = _mm256_adds_epi8(*self.v.get_unchecked(i), *o.v.get_unchecked(i));
        }
    }

    unsafe fn subs(&mut self, o: AvxNx32x8) {
        for i in 0..self.v.len() {
            *self.v.get_unchecked_mut(i) = _mm256_subs_epi8(*self.v.get_unchecked(i), *o.v.get_unchecked(i));
        }
    }

    unsafe fn min(&mut self, o: AvxNx32x8) {
        for i in 0..self.v.len() {
            *self.v.get_unchecked_mut(i) = _mm256_min_epi8(*self.v.get_unchecked(i), *o.v.get_unchecked(i));
        }
    }

    unsafe fn and(&mut self, o: AvxNx32x8) {
        for i in 0..self.v.len() {
            *self.v.get_unchecked_mut(i) = _mm256_and_si256(*self.v.get_unchecked(i), *o.v.get_unchecked(i));
        }
    }

    unsafe fn cmpeq(&mut self, o: AvxNx32x8) {
        for i in 0..self.v.len() {
            *self.v.get_unchecked_mut(i) = _mm256_cmpeq_epi8(*self.v.get_unchecked(i), *o.v.get_unchecked(i));
        }
    }

    unsafe fn cmpgt(&mut self, o: AvxNx32x8) {
        for i in 0..self.v.len() {
            *self.v.get_unchecked_mut(i) = _mm256_cmpgt_epi8(*self.v.get_unchecked(i), *o.v.get_unchecked(i));
        }
    }

    unsafe fn blendv(&mut self, o: AvxNx32x8, mask: AvxNx32x8) {
        for i in 0..self.v.len() {
            *self.v.get_unchecked_mut(i) = _mm256_blendv_epi8(*self.v.get_unchecked(i), *o.v.get_unchecked(i), *mask.v.get_unchecked(i));
        }
    }

    unsafe fn mm_count_ones(&mut self, len: usize) -> u32 {
        let mut res = 0u32;
        let div_len = len >> 5;

        for i in 0..div_len {
            res += _mm256_movemask_epi8(*self.v.get_unchecked(i)).count_ones();
        }

        let rem_len = len & 31;
        res += (_mm256_movemask_epi8(*self.v.get_unchecked(i)) & ((1 << rem_len) - 1)).count_ones();

        res
    }

    unsafe fn mm_count_zeros(&mut self, len: usize) -> u32 {
        let mut res = 0u32;
        let div_len = len >> 5;

        for i in 0..div_len {
            res += _mm256_movemask_epi8(*self.v.get_unchecked(i)).count_zeros();
        }

        let rem_len = len & 31;
        res += ((!_mm256_movemask_epi8(*self.v.get_unchecked(i))) & ((1 << rem_len) - 1)).count_ones();

        res
    }

    unsafe fn count_ones(&mut self, len: usize) -> u32 {

    }

    unsafe fn shift_left(&mut self) {
        for i in 0..(self.v.len() - 1) {
            let curr = self.v.get_unchecked(i);
            // permute concatenates the second half of the current vector and the first half of the next vector
            *self.v.get_unchecked_mut(i) = _mm256_alignr_epi8(
                _mm256_permute2x128_si256(*curr, *self.v.get_unchecked(i + 1), 0b00100001i32), *curr, 1i32);
        }

        // last one gets to shift in zeros
        let last = self.v.len() - 1;
        let curr = self.v.get_unchecked(last);
        // permute concatenates the second half of the last vector and a vector of zeros
        *self.v.get_unchecked_mut(last) = _mm256_alignr_epi8(_mm256_permute2x128_si256(*curr, *curr, 0b10000001i32), *curr, 1i32);
    }

    unsafe fn shift_right(&mut self) {
        for i in (1..self.len()).rev() {
            let curr = self.v.get_unchecked(i);
            // permute concatenates the second half of the previous vector and the first half of the current vector
            *self.v.get_unchecked_mut(i) = _mm256_alignr_epi8(
                *curr, _mm256_permute2x128_si256(*curr, *self.v.get_unchecked(i - 1), 0b00000011i32), 15i32);
        }

        // first one gets to shift in zeros
        let curr = self.v.get_unchecked(0);
        // permute concatenates a vector of zeros and the first half of the first vector
        *self.v.get_uncheched_mut(0) = _mm256_alignr_epi8(*curr, _mm256_permute2x128_si256(*curr, *curr, 0b00001000i32), 15i32);
    }

    unsafe fn extract(&mut self, i: usize) -> u32 {
        let idx = i >> 5;
        let j = i & 31;
        let mut arr = [0u8; 32];
        _mm256_storeu_si256(arr.as_mut_ptr() as *mut __m256i, *v.get_unchecked(idx));
        *arr.get_unchecked(j) as u32
    }

    unsafe fn insert_last_0(&mut self, val: u32) {
        let last = self.v.len() - 1;
        *self.v.get_unchecked_mut(last) = _mm256_insert_epi8(*self.v.get_unchecked(last), val as i8, 31i32);
    }

    unsafe fn insert_last_1(&mut self, val: u32) {
        let last = self.v.len() - 1;
        *self.v.get_unchecked_mut(last) = _mm256_insert_epi8(*self.v.get_unchecked(last), val as i8, 30i32);
    }

    unsafe fn insert_last_2(&mut self, val: u32) {
        let last = self.v.len() - 1;
        *self.v.get_unchecked_mut(last) = _mm256_insert_epi8(*self.v.get_unchecked(last), val as i8, 29i32);
    }

    unsafe fn insert_last_max(&mut self) {
        let last = self.v.len() - 1;
        *self.v.get_unchecked_mut(last) = _mm256_insert_epi8(*self.v.get_unchecked(last), i8::max_value(), 31i32);
    }

    unsafe fn insert_first(&mut self, val: u32) {
        *self.v.get_unchecked_mut(0) = _mm256_insert_epi8(*self.v.get_unchecked(0), val as i8, 0i32);
    }

    unsafe fn insert_first_max(&mut self) {
        *self.v.get_unchecked_mut(0) = _mm256_insert_epi8(*self.v.get_unchecked(0), i8::max_value(), 0i32);
    }
}
