use triple_accel::*;

#[test]
fn test_basic_hamming_naive() {
    let a = b"abc";
    let b = b"abd";
    let dist = hamming_naive(a, b);
    assert!(dist == 1);
}

#[test]
fn test_basic_hamming_search_naive() {
    let a = b"abc";
    let b = b"  abc  abb";
    let res = hamming_search_naive(a, b, 1);
    assert!(res == vec![Match{start: 2, end: 5, k: 0}, Match{start: 7, end: 10, k: 1}]);
}

#[test]
fn test_basic_hamming_search_simd() {
    let a = b"abc";
    let b = b"  abc  abb";
    let res = hamming_search_simd(a, b, 1);
    assert!(res == vec![Match{start: 2, end: 5, k: 0}, Match{start: 7, end: 10, k: 1}]);
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
    let a = b"abc";
    let b = b"abd";
    let dist = hamming_simd(a, b);
    assert!(dist == 1);
}

#[test]
fn test_basic_levenshtein_naive() {
    let a1 = b"abcde";
    let b1 = b" ab cde";
    let mut res = levenshtein_naive(a1, b1, false);
    assert!(res.0 == 2);
    assert!(res.1.is_none());

    let a2 = b"abcde";
    let b2 = b"";
    res = levenshtein_naive(a2, b2, false);
    assert!(res.0 == 5);
    assert!(res.1.is_none());

    let a3 = b"abcde";
    let b3 = b"abcdee";
    res = levenshtein_naive(a3, b3, false);
    assert!(res.0 == 1);
    assert!(res.1.is_none());

    let a4 = b"abcde";
    let b4 = b"acde";
    res = levenshtein_naive(a4, b4, false);
    assert!(res.0 == 1);
    assert!(res.1.is_none());

    let a5 = b"abcde";
    let b5 = b"abbde";
    res = levenshtein_naive(a5, b5, false);
    assert!(res.0 == 1);
    assert!(res.1.is_none());
}

#[test]
fn test_trace_on_levenshtein_naive() {
    let a1 = b"abcde";
    let b1 = b" ab cde";
    let mut res = levenshtein_naive(a1, b1, true);
    assert!(res.0 == 2);
    assert!(res.1.unwrap() == vec![Edit::AGap, Edit::Match, Edit::Match, Edit::AGap, Edit::Match, Edit::Match, Edit::Match]);

    let a2 = b"abcde";
    let b2 = b"";
    res = levenshtein_naive(a2, b2, true);
    assert!(res.0 == 5);
    assert!(res.1.unwrap() == vec![Edit::BGap, Edit::BGap, Edit::BGap, Edit::BGap, Edit::BGap]);

    let a3 = b"abcde";
    let b3 = b"abcce";
    res = levenshtein_naive(a3, b3, true);
    assert!(res.0 == 1);
    assert!(res.1.unwrap() == vec![Edit::Match, Edit::Match, Edit::Match, Edit::Mismatch, Edit::Match]);
}

#[test]
fn test_basic_levenshtein_naive_k() {
    let a1 = b"abcde";
    let b1 = b" ab cde";
    let mut res = levenshtein_naive_k(a1, b1, 2, false);
    assert!(res.0 == 2);
    assert!(res.1.is_none());

    let a2 = b"abcde";
    let b2 = b"";
    res = levenshtein_naive_k(a2, b2, 10, false);
    assert!(res.0 == 5);
    assert!(res.1.is_none());

    let a3 = b"abcde";
    let b3 = b"abcdee";
    res = levenshtein_naive_k(a3, b3, 2, false);
    assert!(res.0 == 1);
    assert!(res.1.is_none());

    let a4 = b"abcde";
    let b4 = b"acde";
    res = levenshtein_naive_k(a4, b4, 2, false);
    assert!(res.0 == 1);
    assert!(res.1.is_none());

    let a5 = b"abcde";
    let b5 = b"abbde";
    res = levenshtein_naive_k(a5, b5, 2, false);
    assert!(res.0 == 1);
    assert!(res.1.is_none());

    let a6 = b"abcde";
    let b6 = b"abbde";
    res = levenshtein_naive_k(a6, b6, 1, false);
    assert!(res.0 == 1);
    assert!(res.1.is_none());
}

#[test]
fn test_trace_on_levenshtein_naive_k() {
    let a1 = b"abcde";
    let b1 = b" ab cde";
    let mut res = levenshtein_naive_k(a1, b1, 2, true);
    assert!(res.0 == 2);
    assert!(res.1.unwrap() == vec![Edit::AGap, Edit::Match, Edit::Match, Edit::AGap, Edit::Match, Edit::Match, Edit::Match]);

    let a2 = b"abcde";
    let b2 = b"";
    res = levenshtein_naive_k(a2, b2, 10, true);
    assert!(res.0 == 5);
    assert!(res.1.unwrap() == vec![Edit::BGap, Edit::BGap, Edit::BGap, Edit::BGap, Edit::BGap]);

    let a3 = b"abcde";
    let b3 = b"abcce";
    res = levenshtein_naive_k(a3, b3, 2, true);
    assert!(res.0 == 1);
    assert!(res.1.unwrap() == vec![Edit::Match, Edit::Match, Edit::Match, Edit::Mismatch, Edit::Match]);
}

#[test]
fn test_basic_levenshtein_simd() {
    let a1 = b"abcde";
    let b1 = b" ab cde";
    let mut res = levenshtein_simd(a1, b1, false);
    assert!(res.0 == 2);
    assert!(res.1.is_none());

    let a2 = b"abcde";
    let b2 = b"";
    res = levenshtein_simd(a2, b2, false);
    assert!(res.0 == 5);
    assert!(res.1.is_none());

    let a3 = b"abcde";
    let b3 = b"abcdee";
    res = levenshtein_simd(a3, b3, false);
    assert!(res.0 == 1);
    assert!(res.1.is_none());

    let a4 = b"abcde";
    let b4 = b"acde";
    res = levenshtein_simd(a4, b4, false);
    assert!(res.0 == 1);
    assert!(res.1.is_none());

    let a5 = b"abcde";
    let b5 = b"abbde";
    res = levenshtein_simd(a5, b5, false);
    assert!(res.0 == 1);
    assert!(res.1.is_none());
}

#[test]
fn test_trace_on_levenshtein_simd() {
    let a1 = b"abcde";
    let b1 = b" ab cde";
    let mut res = levenshtein_simd(a1, b1, true);
    assert!(res.0 == 2);
    assert!(res.1.unwrap() == vec![Edit::AGap, Edit::Match, Edit::Match, Edit::AGap, Edit::Match, Edit::Match, Edit::Match]);

    let a2 = b"abcde";
    let b2 = b"";
    res = levenshtein_simd(a2, b2, true);
    assert!(res.0 == 5);
    assert!(res.1.unwrap() == vec![Edit::BGap, Edit::BGap, Edit::BGap, Edit::BGap, Edit::BGap]);

    let a3 = b"abcde";
    let b3 = b"abcce";
    res = levenshtein_simd(a3, b3, true);
    assert!(res.0 == 1);
    assert!(res.1.unwrap() == vec![Edit::Match, Edit::Match, Edit::Match, Edit::Mismatch, Edit::Match]);
}

#[test]
fn test_basic_levenshtein_search_naive() {
    let a1 = b"bcc";
    let b1 = b"abcde";
    let k1 = 1u32;
    let mut res = levenshtein_search_naive(a1, b1, k1);
    assert!(res == vec![Match{start: 1, end: 3, k: 1}, Match{start: 1, end: 4, k: 1}]);

    let a2 = b"";
    let b2 = b"";
    let k2 = 1u32;
    res = levenshtein_search_naive(a2, b2, k2);
    assert!(res == vec![]);

    let a3 = b"tast";
    let b3 = b"testing 123 tating!";
    let k3 = 1u32;
    res = levenshtein_search_naive(a3, b3, k3);
    assert!(res == vec![Match{start: 0, end: 4, k: 1}, Match{start: 12, end: 15, k: 1}]);

    let a4 = b"tst";
    let b4 = b"testing 123 tasting!";
    let k4 = 1u32;
    res = levenshtein_search_naive(a4, b4, k4);
    assert!(res == vec![Match{start: 1, end: 4, k: 1}, Match{start: 13, end: 16, k: 1}]);
}

#[test]
fn test_basic_levenshtein_search_simd() {
    let a1 = b"bcc";
    let b1 = b"abcde";
    let k1 = 1u32;
    let mut res = levenshtein_search_simd(a1, b1, k1);
    assert!(res == vec![Match{start: 1, end: 3, k: 1}, Match{start: 1, end: 4, k: 1}]);

    let a2 = b"";
    let b2 = b"";
    let k2 = 1u32;
    res = levenshtein_search_simd(a2, b2, k2);
    assert!(res == vec![]);

    let a3 = b"tast";
    let b3 = b"testing 123 tating!";
    let k3 = 1u32;
    res = levenshtein_search_simd(a3, b3, k3);
    assert!(res == vec![Match{start: 0, end: 4, k: 1}, Match{start: 12, end: 15, k: 1}]);

    let a4 = b"tst";
    let b4 = b"testing 123 tasting!";
    let k4 = 1u32;
    res = levenshtein_search_simd(a4, b4, k4);
    assert!(res == vec![Match{start: 1, end: 4, k: 1}, Match{start: 13, end: 16, k: 1}]);
}

