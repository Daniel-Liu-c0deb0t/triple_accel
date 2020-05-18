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
    let a1 = b"abc";
    let b1 = b"  abc  abb";
    let mut res = hamming_search_naive_with_opts(a1, b1, 1, SearchType::All);
    assert!(res == vec![Match{start: 2, end: 5, k: 0}, Match{start: 7, end: 10, k: 1}]);

    let a2 = b"abc";
    let b2 = b"  abc  abb";
    res = hamming_search_naive(a2, b2);
    assert!(res == vec![Match{start: 2, end: 5, k: 0}]);
}

#[test]
fn test_basic_hamming_search_simd() {
    let a1 = b"abc";
    let b1 = b"  abc  abb aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    let mut res = hamming_search_simd_with_opts(a1, b1, 1, SearchType::All);
    assert!(res == vec![Match{start: 2, end: 5, k: 0}, Match{start: 7, end: 10, k: 1}]);

    let a2 = b"abc";
    let b2 = b"  abc  abb aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    res = hamming_search_simd(a2, b2);
    assert!(res == vec![Match{start: 2, end: 5, k: 0}]);
}

#[test]
fn test_basic_hamming_words_64() {
    let a_str = b"abc";
    let b_str = b"abd";

    let mut a = alloc_str(a_str.len());
    fill_str(&mut a, a_str);

    let mut b = alloc_str(b_str.len());
    fill_str(&mut b, b_str);

    let dist = hamming_words_64(&a, &b);
    assert!(dist == 1);
}

#[test]
fn test_basic_hamming_words_128() {
    let a_str = b"abc";
    let b_str = b"abd";

    let mut a = alloc_str(a_str.len());
    fill_str(&mut a, a_str);

    let mut b = alloc_str(b_str.len());
    fill_str(&mut b, b_str);

    let dist = hamming_words_128(&a, &b);
    assert!(dist == 1);
}

#[test]
fn test_basic_hamming_simd_movemask() {
    let a = b"abcaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    let b = b"abdaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    let dist = hamming_simd_movemask(a, b);
    assert!(dist == 1);
}

#[test]
fn test_basic_hamming_simd_parallel() {
    let a = b"abcaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    let b = b"abdaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    let dist = hamming_simd_parallel(a, b);
    assert!(dist == 1);
}

#[test]
fn test_basic_levenshtein_naive() {
    let a1 = b"abcde";
    let b1 = b" ab cde";
    let mut res = levenshtein_naive(a1, b1);
    assert!(res == 2);

    let a2 = b"abcde";
    let b2 = b"";
    res = levenshtein_naive(a2, b2);
    assert!(res == 5);

    let a3 = b"abcde";
    let b3 = b"abcdee";
    res = levenshtein_naive(a3, b3);
    assert!(res == 1);

    let a4 = b"abcde";
    let b4 = b"acde";
    res = levenshtein_naive(a4, b4);
    assert!(res == 1);

    let a5 = b"abcde";
    let b5 = b"abbde";
    res = levenshtein_naive(a5, b5);
    assert!(res == 1);

    let a6 = b"abcde";
    let b6 = b"acbde";
    res = levenshtein_naive_with_opts(a6, b6, false, EditCosts::new(1, 1, Some(1))).0;
    assert!(res == 1);

    let a7 = b"ab";
    let b7 = b"ba";
    res = levenshtein_naive_with_opts(a7, b7, false, EditCosts::new(1, 1, Some(1))).0;
    assert!(res == 1);
}

#[test]
fn test_trace_on_levenshtein_naive() {
    let a1 = b"abcde";
    let b1 = b" ab cde";
    let mut res = levenshtein_naive_with_opts(a1, b1, true, LEVENSHTEIN_COSTS);
    assert!(res.0 == 2);
    assert!(res.1.unwrap() == vec![Edit{edit: EditType::AGap, count: 1},
                                   Edit{edit: EditType::Match, count: 2},
                                   Edit{edit: EditType::AGap, count: 1},
                                   Edit{edit: EditType::Match, count: 3}]);

    let a2 = b"abcde";
    let b2 = b"";
    res = levenshtein_naive_with_opts(a2, b2, true, LEVENSHTEIN_COSTS);
    assert!(res.0 == 5);
    assert!(res.1.unwrap() == vec![Edit{edit: EditType::BGap, count: 5}]);

    let a3 = b"abcde";
    let b3 = b"abcce";
    res = levenshtein_naive_with_opts(a3, b3, true, LEVENSHTEIN_COSTS);
    assert!(res.0 == 1);
    assert!(res.1.unwrap() == vec![Edit{edit: EditType::Match, count: 3},
                                   Edit{edit: EditType::Mismatch, count: 1},
                                   Edit{edit: EditType::Match, count: 1}]);

    let a4 = b"abcde";
    let b4 = b"acbde";
    res = levenshtein_naive_with_opts(a4, b4, true, EditCosts::new(1, 1, Some(1)));
    assert!(res.0 == 1);
    assert!(res.1.unwrap() == vec![Edit{edit: EditType::Match, count: 1},
                                   Edit{edit: EditType::Transpose, count: 1},
                                   Edit{edit: EditType::Match, count: 2}]);
}

#[test]
fn test_basic_levenshtein_exp() {
    let a1 = b"abcde";
    let b1 = b" ab cde";
    let mut res = levenshtein_exp(a1, b1, false);
    assert!(res.0 == 2);
    assert!(res.1.is_none());

    let a2 = b"abcde";
    let b2 = b"";
    res = levenshtein_exp(a2, b2, false);
    assert!(res.0 == 5);
    assert!(res.1.is_none());

    let a3 = b"abcde";
    let b3 = b"abcdee";
    res = levenshtein_exp(a3, b3, false);
    assert!(res.0 == 1);
    assert!(res.1.is_none());

    let a4 = b"abcde";
    let b4 = b"acde";
    res = levenshtein_exp(a4, b4, false);
    assert!(res.0 == 1);
    assert!(res.1.is_none());

    let a5 = b"abcde";
    let b5 = b"abbde";
    res = levenshtein_exp(a5, b5, false);
    assert!(res.0 == 1);
    assert!(res.1.is_none());
}

#[test]
fn test_trace_on_levenshtein_exp() {
    let a1 = b"abcde";
    let b1 = b" ab cde";
    let mut res = levenshtein_exp(a1, b1, true);
    assert!(res.0 == 2);
    assert!(res.1.unwrap() == vec![Edit{edit: EditType::AGap, count: 1},
                                   Edit{edit: EditType::Match, count: 2},
                                   Edit{edit: EditType::AGap, count: 1},
                                   Edit{edit: EditType::Match, count: 3}]);

    let a2 = b"abcde";
    let b2 = b"";
    res = levenshtein_exp(a2, b2, true);
    assert!(res.0 == 5);
    assert!(res.1.unwrap() == vec![Edit{edit: EditType::BGap, count: 5}]);

    let a3 = b"abcde";
    let b3 = b"abcce";
    res = levenshtein_exp(a3, b3, true);
    assert!(res.0 == 1);
    assert!(res.1.unwrap() == vec![Edit{edit: EditType::Match, count: 3},
                                   Edit{edit: EditType::Mismatch, count: 1},
                                   Edit{edit: EditType::Match, count: 1}]);
}

#[test]
fn test_basic_levenshtein_naive_k_with_opts() {
    let a1 = b"abcde";
    let b1 = b" ab cde";
    let mut res = levenshtein_naive_k_with_opts(a1, b1, 2, false, LEVENSHTEIN_COSTS).unwrap();
    assert!(res.0 == 2);
    assert!(res.1.is_none());

    let a2 = b"abcde";
    let b2 = b"";
    res = levenshtein_naive_k_with_opts(a2, b2, 10, false, LEVENSHTEIN_COSTS).unwrap();
    assert!(res.0 == 5);
    assert!(res.1.is_none());

    let a3 = b"abcde";
    let b3 = b"abcdee";
    res = levenshtein_naive_k_with_opts(a3, b3, 2, false, LEVENSHTEIN_COSTS).unwrap();
    assert!(res.0 == 1);
    assert!(res.1.is_none());

    let a4 = b"abcde";
    let b4 = b"acde";
    res = levenshtein_naive_k_with_opts(a4, b4, 2, false, LEVENSHTEIN_COSTS).unwrap();
    assert!(res.0 == 1);
    assert!(res.1.is_none());

    let a5 = b"abcde";
    let b5 = b"abbde";
    res = levenshtein_naive_k_with_opts(a5, b5, 2, false, LEVENSHTEIN_COSTS).unwrap();
    assert!(res.0 == 1);
    assert!(res.1.is_none());

    let a6 = b"abcde";
    let b6 = b"abbde";
    res = levenshtein_naive_k_with_opts(a6, b6, 1, false, LEVENSHTEIN_COSTS).unwrap();
    assert!(res.0 == 1);
    assert!(res.1.is_none());

    let a7 = b"abcde";
    let b7 = b"acbde";
    res = levenshtein_naive_k_with_opts(a7, b7, 1, false, EditCosts::new(1, 1, Some(1))).unwrap();
    assert!(res.0 == 1);
    assert!(res.1.is_none());

    let a8 = b"ab";
    let b8 = b"ba";
    res = levenshtein_naive_k_with_opts(a8, b8, 1, false, EditCosts::new(1, 1, Some(1))).unwrap();
    assert!(res.0 == 1);
    assert!(res.1.is_none());
}

#[test]
fn test_trace_on_levenshtein_naive_k_with_opts() {
    let a1 = b"abcde";
    let b1 = b" ab cde";
    let mut res = levenshtein_naive_k_with_opts(a1, b1, 2, true, LEVENSHTEIN_COSTS).unwrap();
    assert!(res.0 == 2);
    assert!(res.1.unwrap() == vec![Edit{edit: EditType::AGap, count: 1},
                                   Edit{edit: EditType::Match, count: 2},
                                   Edit{edit: EditType::AGap, count: 1},
                                   Edit{edit: EditType::Match, count: 3}]);

    let a2 = b"abcde";
    let b2 = b"";
    res = levenshtein_naive_k_with_opts(a2, b2, 10, true, LEVENSHTEIN_COSTS).unwrap();
    assert!(res.0 == 5);
    assert!(res.1.unwrap() == vec![Edit{edit: EditType::BGap, count: 5}]);

    let a3 = b"abcde";
    let b3 = b"abcce";
    res = levenshtein_naive_k_with_opts(a3, b3, 2, true, LEVENSHTEIN_COSTS).unwrap();
    assert!(res.0 == 1);
    assert!(res.1.unwrap() == vec![Edit{edit: EditType::Match, count: 3},
                                   Edit{edit: EditType::Mismatch, count: 1},
                                   Edit{edit: EditType::Match, count: 1}]);

    let a4 = b"abcde";
    let b4 = b"acbde";
    res = levenshtein_naive_k_with_opts(a4, b4, 2, true, EditCosts::new(1, 1, Some(1))).unwrap();
    assert!(res.0 == 1);
    assert!(res.1.unwrap() == vec![Edit{edit: EditType::Match, count: 1},
                                   Edit{edit: EditType::Transpose, count: 1},
                                   Edit{edit: EditType::Match, count: 2}]);
}

#[test]
fn test_basic_levenshtein_simd_k_with_opts() {
    let a1 = b"abcde";
    let b1 = b" ab cde";
    let mut res = levenshtein_simd_k_with_opts(a1, b1, 2, false, LEVENSHTEIN_COSTS).unwrap();
    assert!(res.0 == 2);
    assert!(res.1.is_none());

    let a2 = b"abcde";
    let b2 = b"";
    res = levenshtein_simd_k_with_opts(a2, b2, 30, false, LEVENSHTEIN_COSTS).unwrap();
    assert!(res.0 == 5);
    assert!(res.1.is_none());

    let a3 = b"abcde";
    let b3 = b"abcdee";
    res = levenshtein_simd_k_with_opts(a3, b3, 20, false, LEVENSHTEIN_COSTS).unwrap();
    assert!(res.0 == 1);
    assert!(res.1.is_none());

    let a4 = b"abcde";
    let b4 = b"acde";
    res = levenshtein_simd_k_with_opts(a4, b4, 1, false, LEVENSHTEIN_COSTS).unwrap();
    assert!(res.0 == 1);
    assert!(res.1.is_none());

    let a5 = b"abcde";
    let b5 = b"abbde";
    res = levenshtein_simd_k_with_opts(a5, b5, 2, false, LEVENSHTEIN_COSTS).unwrap();
    assert!(res.0 == 1);
    assert!(res.1.is_none());

    let a6 = b"abcde";
    let b6 = b"acbde";
    res = levenshtein_simd_k_with_opts(a6, b6, 2, false, EditCosts::new(1, 1, Some(1))).unwrap();
    assert!(res.0 == 1);
    assert!(res.1.is_none());

    let a7 = b"ab";
    let b7 = b"ba";
    res = levenshtein_simd_k_with_opts(a7, b7, 2, false, EditCosts::new(1, 1, Some(1))).unwrap();
    assert!(res.0 == 1);
    assert!(res.1.is_none());
}

#[test]
fn test_trace_on_levenshtein_simd_k_with_opts() {
    let a1 = b"abcde";
    let b1 = b" ab cde";
    let mut res = levenshtein_simd_k_with_opts(a1, b1, 30, true, LEVENSHTEIN_COSTS).unwrap();
    assert!(res.0 == 2);
    assert!(res.1.unwrap() == vec![Edit{edit: EditType::AGap, count: 1},
                                   Edit{edit: EditType::Match, count: 2},
                                   Edit{edit: EditType::AGap, count: 1},
                                   Edit{edit: EditType::Match, count: 3}]);

    let a2 = b"abcde";
    let b2 = b"";
    res = levenshtein_simd_k_with_opts(a2, b2, 5, true, LEVENSHTEIN_COSTS).unwrap();
    assert!(res.0 == 5);
    assert!(res.1.unwrap() == vec![Edit{edit: EditType::BGap, count: 5}]);

    let a3 = b"abcde";
    let b3 = b"abcce";
    res = levenshtein_simd_k_with_opts(a3, b3, 1, true, LEVENSHTEIN_COSTS).unwrap();
    assert!(res.0 == 1);
    assert!(res.1.unwrap() == vec![Edit{edit: EditType::Match, count: 3},
                                   Edit{edit: EditType::Mismatch, count: 1},
                                   Edit{edit: EditType::Match, count: 1}]);

    let a4 = b"abcde";
    let b4 = b"acbde";
    res = levenshtein_simd_k_with_opts(a4, b4, 2, true, EditCosts::new(1, 1, Some(1))).unwrap();
    assert!(res.0 == 1);
    assert!(res.1.unwrap() == vec![Edit{edit: EditType::Match, count: 1},
                                   Edit{edit: EditType::Transpose, count: 1},
                                   Edit{edit: EditType::Match, count: 2}]);
}

#[test]
fn test_basic_levenshtein_search_naive() {
    let a1 = b"bcc";
    let b1 = b"abcde";
    let k1 = 1;
    let mut res = levenshtein_search_naive_with_opts(a1, b1, k1, SearchType::All, LEVENSHTEIN_COSTS, false);
    assert!(res == vec![Match{start: 1, end: 3, k: 1}, Match{start: 1, end: 4, k: 1}]);

    let a2 = b"";
    let b2 = b"";
    let k2 = 1;
    res = levenshtein_search_naive_with_opts(a2, b2, k2, SearchType::All, LEVENSHTEIN_COSTS, false);
    assert!(res == vec![]);

    let a3 = b"tast";
    let b3 = b"testing 123 tating!";
    let k3 = 1;
    res = levenshtein_search_naive_with_opts(a3, b3, k3, SearchType::All, LEVENSHTEIN_COSTS, false);
    assert!(res == vec![Match{start: 0, end: 4, k: 1}, Match{start: 12, end: 15, k: 1}]);

    let a4 = b"tst";
    let b4 = b"testing 123 tasting!";
    let k4 = 1;
    res = levenshtein_search_naive_with_opts(a4, b4, k4, SearchType::All, LEVENSHTEIN_COSTS, false);
    assert!(res == vec![Match{start: 0, end: 4, k: 1}, Match{start: 12, end: 16, k: 1}]);

    let a5 = b"tst";
    let b5 = b"testing 123 tasting!";
    res = levenshtein_search_naive(a5, b5);
    assert!(res == vec![Match{start: 0, end: 4, k: 1}, Match{start: 12, end: 16, k: 1}]);

    let a6 = b"ab";
    let b6 = b"ba";
    let k6 = 1;
    res = levenshtein_search_naive_with_opts(a6, b6, k6, SearchType::All, EditCosts::new(1, 1, Some(1)), false);
    assert!(res == vec![Match{start: 0, end: 1, k: 1}, Match{start: 0, end: 2, k: 1}]);

    let a7 = b"test";
    let b7 = b"...tseting!";
    let k7 = 1;
    res = levenshtein_search_naive_with_opts(a7, b7, k7, SearchType::All, EditCosts::new(1, 1, Some(1)), false);
    assert!(res == vec![Match{start: 3, end: 7, k: 1}]);
}

#[test]
fn test_basic_levenshtein_search_simd() {
    let a1 = b"bcc";
    let b1 = b"abcde";
    let k1 = 1;
    let mut res = levenshtein_search_simd_with_opts(a1, b1, k1, SearchType::All, LEVENSHTEIN_COSTS, false);
    assert!(res == vec![Match{start: 1, end: 3, k: 1}, Match{start: 1, end: 4, k: 1}]);

    let a2 = b"";
    let b2 = b"";
    let k2 = 1;
    res = levenshtein_search_simd_with_opts(a2, b2, k2, SearchType::All, LEVENSHTEIN_COSTS, false);
    assert!(res == vec![]);

    let a3 = b"tast";
    let b3 = b"testing 123 tating!";
    let k3 = 1;
    res = levenshtein_search_simd_with_opts(a3, b3, k3, SearchType::All, LEVENSHTEIN_COSTS, false);
    assert!(res == vec![Match{start: 0, end: 4, k: 1}, Match{start: 12, end: 15, k: 1}]);

    let a4 = b"tst";
    let b4 = b"testing 123 tasting!";
    let k4 = 1;
    res = levenshtein_search_simd_with_opts(a4, b4, k4, SearchType::All, LEVENSHTEIN_COSTS, false);
    assert!(res == vec![Match{start: 0, end: 4, k: 1}, Match{start: 12, end: 16, k: 1}]);

    let a5 = b"tst";
    let b5 = b"testing 123 tasting!";
    res = levenshtein_search_simd(a5, b5);
    assert!(res == vec![Match{start: 0, end: 4, k: 1}, Match{start: 12, end: 16, k: 1}]);

    let a6 = b"ab";
    let b6 = b"ba";
    let k6 = 1;
    res = levenshtein_search_simd_with_opts(a6, b6, k6, SearchType::All, EditCosts::new(1, 1, Some(1)), false);
    assert!(res == vec![Match{start: 0, end: 1, k: 1}, Match{start: 0, end: 2, k: 1}]);

    let a7 = b"test";
    let b7 = b"...tseting!";
    let k7 = 1;
    res = levenshtein_search_simd_with_opts(a7, b7, k7, SearchType::All, EditCosts::new(1, 1, Some(1)), false);
    assert!(res == vec![Match{start: 3, end: 7, k: 1}]);
}

