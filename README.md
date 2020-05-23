# triple_accel
![Build and Test](https://github.com/Daniel-Liu-c0deb0t/triple_accel/workflows/Build%20and%20Test/badge.svg)

Rust edit distance routines accelerated using SIMD, with support for Hamming, Levenshtein, etc. distances.

## Example
`triple_accel` provides a very simple and easy to use framework for common edit distance operations. Calculating the Hamming distance (number of mismatches) between two strings is extremely simple:
```Rust
use triple_accel::*;

let a = b"abcd";
let b = b"abcc";

let dist = hamming(&a, &b);
assert!(dist == 1);
```
By default, `triple_accel` will choose to use the fastest implementation at runtime, based on CPU features, going down the list:

1. Vectorized implementation with 256-bit AVX vectors, if AVX2 support is available.
2. Vectorized implementation with 128-bit SSE vectors, if SSE4.1 support is available.
3. Scalar implementation without SIMD.

This means that the routines provided by `triple_accel` are safe to run on CPUs without AVX2 or SSE4.1 support, after being compiled with a CPU that support those features.

Similarly, we can easily calculate the Levenshtein distance (character mismatches and gaps all have a cost of 1) between two strings with the following code:
```Rust
let a = b"abc";
let b = b"abcd";

let dist = levenshtein_exp(&a, &b);
assert!(dist == 1);
```
In addition to edit distance routines, `triple_accel` also provides search routines. These routines return a vector of matches that indicate where the `needle` string matches the `haystack` string. `triple_accel` will attempt to maximize the length of matches that end at the same position.
```Rust
let needle = b"helllo";
let haystack = b"hello world";

let matches = levenshtein_search(&needle, &haystack);
// note: start index is inclusive, end index is exclusive!
assert!(matches == vec![Match{start: 0, end: 5, k: 1}]);
```
Sometimes, it is necessary to use the slightly lower level, but also more powerful routines that `triple_accel` provides. For example, it is possible to allow transpositions (character swaps) that have a cost of 1, in addition to mismatches and gaps:
```Rust
use triple_accel::levenshtein::*;

let a = b"abcd";
let b = b"abdc";
let k = 2; // upper bound on allowed cost
let trace_on = false; // return edit traceback?

let dist = levenshtein_simd_k_with_opts(&a, &b, k, trace_on, RDAMERAU_COSTS);
// note: dist may be None if a and b do not match within a cost of k
assert!(dist.unwrap().0 == 1);
```
Don't let the name of the function fool you! `levenshtein_simd_k_with_opts` will still fall back to the scalar implementation if AVX2 or SSE4.1 support is not available. It just prefers to use SIMD where possible.

## Features


## Why the name "triple_accel"?
Because "Time Altar - Triple Accel" is a magical ability used by Kiritsugu Emiya to boost his speed and reaction time in Fate/Zero.
