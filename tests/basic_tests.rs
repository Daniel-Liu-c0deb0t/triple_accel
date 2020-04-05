use triple_accel::*;

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
fn test_basic_hamming_search_naive() {
    let a_str = b"abc";
    let b_str = b"  abc  abb";

    let mut a = alloc_str(a_str.len());
    fill_str(&mut a, a_str);

    let mut b = alloc_str(b_str.len());
    fill_str(&mut b, b_str);

    let res = hamming_search_naive(&a, a_str.len(), &b, b_str.len(), 1);
    assert!(res == vec![Match{start: 2, end: 5, k: 0}, Match{start: 7, end: 10, k: 1}]);
}

#[test]
fn test_basic_hamming_search_simd() {
    let a_str = b"abc";
    let b_str = b"  abc  abb";

    let mut a = alloc_str(a_str.len());
    fill_str(&mut a, a_str);

    let mut b = alloc_str(b_str.len());
    fill_str(&mut b, b_str);

    let res = hamming_search_simd(&a, a_str.len(), &b, b_str.len(), 1);
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
fn test_basic_levenshtein_naive() {
    let a1 = b"abcde";
    let b1 = b" ab cde";
    let mut res = levenshtein_naive(a1, a1.len(), b1, b1.len(), false);
    assert!(res.0 == 2);
    assert!(res.1.is_none());

    let a2 = b"abcde";
    let b2 = b"";
    res = levenshtein_naive(a2, a2.len(), b2, b2.len(), false);
    assert!(res.0 == 5);
    assert!(res.1.is_none());

    let a3 = b"abcde";
    let b3 = b"abcdee";
    res = levenshtein_naive(a3, a3.len(), b3, b3.len(), false);
    assert!(res.0 == 1);
    assert!(res.1.is_none());

    let a4 = b"abcde";
    let b4 = b"acde";
    res = levenshtein_naive(a4, a4.len(), b4, b4.len(), false);
    assert!(res.0 == 1);
    assert!(res.1.is_none());

    let a5 = b"abcde";
    let b5 = b"abbde";
    res = levenshtein_naive(a5, a5.len(), b5, b5.len(), false);
    assert!(res.0 == 1);
    assert!(res.1.is_none());
}

#[test]
fn test_trace_on_levenshtein_naive() {
    let a1 = b"abcde";
    let b1 = b" ab cde";
    let mut res = levenshtein_naive(a1, a1.len(), b1, b1.len(), true);
    assert!(res.0 == 2);
    assert!(res.1.unwrap() == vec![Edit::AGap, Edit::Match, Edit::Match, Edit::AGap, Edit::Match, Edit::Match, Edit::Match]);

    let a2 = b"abcde";
    let b2 = b"";
    res = levenshtein_naive(a2, a2.len(), b2, b2.len(), true);
    assert!(res.0 == 5);
    assert!(res.1.unwrap() == vec![Edit::BGap, Edit::BGap, Edit::BGap, Edit::BGap, Edit::BGap]);

    let a3 = b"abcde";
    let b3 = b"abcce";
    res = levenshtein_naive(a3, a3.len(), b3, b3.len(), true);
    assert!(res.0 == 1);
    assert!(res.1.unwrap() == vec![Edit::Match, Edit::Match, Edit::Match, Edit::Mismatch, Edit::Match]);
}

#[test]
fn test_basic_levenshtein_naive_k() {
    let a1 = b"abcde";
    let b1 = b" ab cde";
    let mut res = levenshtein_naive_k(a1, a1.len(), b1, b1.len(), 2, false);
    assert!(res.0 == 2);
    assert!(res.1.is_none());

    let a2 = b"abcde";
    let b2 = b"";
    res = levenshtein_naive_k(a2, a2.len(), b2, b2.len(), 10, false);
    assert!(res.0 == 5);
    assert!(res.1.is_none());

    let a3 = b"abcde";
    let b3 = b"abcdee";
    res = levenshtein_naive_k(a3, a3.len(), b3, b3.len(), 2, false);
    assert!(res.0 == 1);
    assert!(res.1.is_none());

    let a4 = b"abcde";
    let b4 = b"acde";
    res = levenshtein_naive_k(a4, a4.len(), b4, b4.len(), 2, false);
    assert!(res.0 == 1);
    assert!(res.1.is_none());

    let a5 = b"abcde";
    let b5 = b"abbde";
    res = levenshtein_naive_k(a5, a5.len(), b5, b5.len(), 2, false);
    assert!(res.0 == 1);
    assert!(res.1.is_none());

    let a6 = b"abcde";
    let b6 = b"abbde";
    res = levenshtein_naive_k(a6, a6.len(), b6, b6.len(), 1, false);
    assert!(res.0 == 1);
    assert!(res.1.is_none());
}

#[test]
fn test_trace_on_levenshtein_naive_k() {
    let a1 = b"abcde";
    let b1 = b" ab cde";
    let mut res = levenshtein_naive_k(a1, a1.len(), b1, b1.len(), 2, true);
    assert!(res.0 == 2);
    assert!(res.1.unwrap() == vec![Edit::AGap, Edit::Match, Edit::Match, Edit::AGap, Edit::Match, Edit::Match, Edit::Match]);

    let a2 = b"abcde";
    let b2 = b"";
    res = levenshtein_naive_k(a2, a2.len(), b2, b2.len(), 10, true);
    assert!(res.0 == 5);
    assert!(res.1.unwrap() == vec![Edit::BGap, Edit::BGap, Edit::BGap, Edit::BGap, Edit::BGap]);

    let a3 = b"abcde";
    let b3 = b"abcce";
    res = levenshtein_naive_k(a3, a3.len(), b3, b3.len(), 2, true);
    assert!(res.0 == 1);
    assert!(res.1.unwrap() == vec![Edit::Match, Edit::Match, Edit::Match, Edit::Mismatch, Edit::Match]);
}

#[test]
fn test_basic_levenshtein_simd() {
    let a1 = b"abcde";
    let b1 = b" ab cde";
    let mut res = levenshtein_simd(a1, a1.len(), b1, b1.len(), false);
    assert!(res.0 == 2);
    assert!(res.1.is_none());

    let a2 = b"abcde";
    let b2 = b"";
    res = levenshtein_simd(a2, a2.len(), b2, b2.len(), false);
    assert!(res.0 == 5);
    assert!(res.1.is_none());

    let a3 = b"abcde";
    let b3 = b"abcdee";
    res = levenshtein_simd(a3, a3.len(), b3, b3.len(), false);
    assert!(res.0 == 1);
    assert!(res.1.is_none());

    let a4 = b"abcde";
    let b4 = b"acde";
    res = levenshtein_simd(a4, a4.len(), b4, b4.len(), false);
    assert!(res.0 == 1);
    assert!(res.1.is_none());

    let a5 = b"abcde";
    let b5 = b"abbde";
    res = levenshtein_simd(a5, a5.len(), b5, b5.len(), false);
    assert!(res.0 == 1);
    assert!(res.1.is_none());
}

#[test]
fn test_trace_on_levenshtein_simd() {
    let a1 = b"abcde";
    let b1 = b" ab cde";
    let mut res = levenshtein_simd(a1, a1.len(), b1, b1.len(), true);
    assert!(res.0 == 2);
    assert!(res.1.unwrap() == vec![Edit::AGap, Edit::Match, Edit::Match, Edit::AGap, Edit::Match, Edit::Match, Edit::Match]);

    let a2 = b"abcde";
    let b2 = b"";
    res = levenshtein_simd(a2, a2.len(), b2, b2.len(), true);
    assert!(res.0 == 5);
    assert!(res.1.unwrap() == vec![Edit::BGap, Edit::BGap, Edit::BGap, Edit::BGap, Edit::BGap]);

    let a3 = b"abcde";
    let b3 = b"abcce";
    res = levenshtein_simd(a3, a3.len(), b3, b3.len(), true);
    assert!(res.0 == 1);
    assert!(res.1.unwrap() == vec![Edit::Match, Edit::Match, Edit::Match, Edit::Mismatch, Edit::Match]);
}

#[test]
fn test_basic_levenshtein_search_naive() {
    let a1 = b"bcc";
    let b1 = b"abcde";
    let k1 = 1u32;
    let mut res = levenshtein_search_naive(a1, a1.len(), b1, b1.len(), k1);
    assert!(res == vec![Match{start: 1, end: 3, k: 1}, Match{start: 1, end: 4, k: 1}]);

    let a2 = b"";
    let b2 = b"";
    let k2 = 1u32;
    res = levenshtein_search_naive(a2, a2.len(), b2, b2.len(), k2);
    assert!(res == vec![]);

    let a3 = b"tast";
    let b3 = b"testing 123 tating!";
    let k3 = 1u32;
    res = levenshtein_search_naive(a3, a3.len(), b3, b3.len(), k3);
    assert!(res == vec![Match{start: 0, end: 4, k: 1}, Match{start: 12, end: 15, k: 1}]);

    let a4 = b"tst";
    let b4 = b"testing 123 tasting!";
    let k4 = 1u32;
    res = levenshtein_search_naive(a4, a4.len(), b4, b4.len(), k4);
    assert!(res == vec![Match{start: 1, end: 4, k: 1}, Match{start: 13, end: 16, k: 1}]);
}

#[test]
fn test_basic_levenshtein_search_simd() {
    let a1 = b"bcc";
    let b1 = b"abcde";
    let k1 = 1u32;
    let mut res = levenshtein_search_simd(a1, a1.len(), b1, b1.len(), k1);
    assert!(res == vec![Match{start: 1, end: 3, k: 1}, Match{start: 1, end: 4, k: 1}]);

    let a2 = b"";
    let b2 = b"";
    let k2 = 1u32;
    res = levenshtein_search_simd(a2, a2.len(), b2, b2.len(), k2);
    assert!(res == vec![]);

    let a3 = b"tast";
    let b3 = b"testing 123 tating!";
    let k3 = 1u32;
    res = levenshtein_search_simd(a3, a3.len(), b3, b3.len(), k3);
    assert!(res == vec![Match{start: 0, end: 4, k: 1}, Match{start: 12, end: 15, k: 1}]);

    let a4 = b"tst";
    let b4 = b"testing 123 tasting!";
    let k4 = 1u32;
    res = levenshtein_search_simd(a4, a4.len(), b4, b4.len(), k4);
    assert!(res == vec![Match{start: 1, end: 4, k: 1}, Match{start: 13, end: 16, k: 1}]);
}

