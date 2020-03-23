pub mod triple_accel;

#[cfg(test)]
mod tests {
    use crate::triple_accel::*;

    #[test]
    fn test_basic_hamming_naive() {
        let a_str = b"abc";
        let b_str = b"abd";

        let mut a = alloc_str(a_str.len());
        fill_str(&mut a, a_str);

        let mut b = alloc_str(b_str.len());
        fill_str(&mut b, b_str);

        let h_dist_naive = hamming_naive(&a, &b);
        assert!(h_dist_naive == 1);
    }

    #[test]
    fn test_basic_hamming_words() {
        let a_str = b"abc";
        let b_str = b"abd";

        let mut a = alloc_str(a_str.len());
        fill_str(&mut a, a_str);

        let mut b = alloc_str(b_str.len());
        fill_str(&mut b, b_str);

        let h_dist_words = hamming_words(&a, &b);
        assert!(h_dist_words == 1);
    }

    #[test]
    fn test_basic_hamming_simd() {
        let a_str = b"abc";
        let b_str = b"abd";

        let mut a = alloc_str(a_str.len());
        fill_str(&mut a, a_str);

        let mut b = alloc_str(b_str.len());
        fill_str(&mut b, b_str);

        let h_dist_simd = hamming_simd(&a, &b);
        assert!(h_dist_simd == 1);
    }

    #[test]
    fn test_basic_levenshtein_simd() {
        let a = b"abcde";
        let b = b" ab cde";

        let l_dist_simd = levenshtein_simd(a, a.len(), b, b.len());
        assert!(l_dist_simd == 2);
    }
}

