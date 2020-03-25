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

        let dist = hamming_naive(&a, &b);
        assert!(dist == 1);
    }

    #[test]
    fn test_basic_hamming_words() {
        let a_str = b"abc";
        let b_str = b"abd";

        let mut a = alloc_str(a_str.len());
        fill_str(&mut a, a_str);

        let mut b = alloc_str(b_str.len());
        fill_str(&mut b, b_str);

        let dist = hamming_words(&a, &b);
        assert!(dist == 1);
    }

    #[test]
    fn test_basic_hamming_simd() {
        let a_str = b"abc";
        let b_str = b"abd";

        let mut a = alloc_str(a_str.len());
        fill_str(&mut a, a_str);

        let mut b = alloc_str(b_str.len());
        fill_str(&mut b, b_str);

        let dist = hamming_simd(&a, &b);
        assert!(dist == 1);
    }

    #[test]
    fn test_basic_levenshtein_simd() {
        let a1 = b"abcde";
        let b1 = b" ab cde";
        let mut dist = levenshtein_simd(a1, a1.len(), b1, b1.len());
        assert!(dist == 2);

        let a2 = b"abcde";
        let b2 = b"";
        dist = levenshtein_simd(a2, a2.len(), b2, b2.len());
        assert!(dist == 5);

        let a3 = b"abcde";
        let b3 = b"abcdee";
        dist = levenshtein_simd(a3, a3.len(), b3, b3.len());
        assert!(dist == 1);

        let a4 = b"abcde";
        let b4 = b"acde";
        dist = levenshtein_simd(a4, a4.len(), b4, b4.len());
        assert!(dist == 1);

        let a5 = b"abcde";
        let b5 = b"abbde";
        dist = levenshtein_simd(a5, a5.len(), b5, b5.len());
        assert!(dist == 1);
    }

    #[test]
    fn test_basic_levenshtein_search_simd() {
        let a1 = b"bcc";
        let b1 = b"abcde";
        let k1 = 1u32;
        let mut res = levenshtein_search_simd(a1, a1.len(), b1, b1.len(), k1);
        assert!(res == vec![Match{idx: 2, k: 1}, Match{idx: 3, k: 1}]);

        let a2 = b"";
        let b2 = b"";
        let k2 = 1u32;
        res = levenshtein_search_simd(a2, a2.len(), b2, b2.len(), k2);
        assert!(res == vec![]);

        let a3 = b"tast";
        let b3 = b"testing 123 tating!";
        let k3 = 1u32;
        res = levenshtein_search_simd(a3, a3.len(), b3, b3.len(), k3);
        assert!(res == vec![Match{idx: 3, k: 1}, Match{idx: 14, k: 1}]);

        let a4 = b"tst";
        let b4 = b"testing 123 tasting!";
        let k4 = 1u32;
        res = levenshtein_search_simd(a4, a4.len(), b4, b4.len(), k4);
        assert!(res == vec![Match{idx: 3, k: 1}, Match{idx: 15, k: 1}]);
    }
}

