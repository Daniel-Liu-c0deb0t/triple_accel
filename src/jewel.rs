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
    unsafe fn extract(&mut self, i: usize) -> u32;
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
    unsafe fn insert_last(&mut self, c: u32);
    unsafe fn insert_last_1(&mut self, c: u32);
    unsafe fn insert_last_2(&mut self, c: u32);
    unsafe fn insert_last_max(&mut self);
    unsafe fn insert_first(&mut self, c: u32);
    unsafe fn insert_first_max(&mut self);
}

/// N x 32 x 8 vector backed with 256-bit AVX2 vectors
#[derive(Clone)]
struct AvxNx32x8 {
    len: usize,
    v: Vec<__m256i>
}

impl Jewel<AvxNx32x8> for AvxNx32x8 {
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

        for i in 0..vec.len() as isize {
            *v.get_unchecked_mut(i) = _mm256_loadu_si256(*avx2_ptr.offset(i));
        }
    }

    unsafe fn extract(&mut self, i: usize) -> u32{
        let idx = i >> 5;
        let j = i & 31;
        let mut arr = [0u8; 32];
        _mm256_storeu_si256(arr.as_mut_ptr() as *mut __m256i, *v.get_unchecked(idx));
        arr[j] as u32
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

    unsafe fn cmpeq(&mut self, o: AvxNx32x8) {
        for i in 0..self.v.len() {
            *self.v.get_unchecked_mut(i) = _mm256_cmpeq_epi8(*self.v.get_unchecked(i), *o.v.get_unchecked(i));
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
}
