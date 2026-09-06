# SparseSoftmax problem 298 — correctness baseline

This branch implements the first executable milestone for the Sparse Softmax operator.

## Implemented semantics

- index mode and CSR `ptr` mode
- stable softmax: `exp(x - group_max) / (group_sum + eps)`
- negative `dim`
- arbitrary-rank ND tensors using `(outer, dim, inner)` logical addressing
- common 1-D PyG-style `index` broadcasting along `dim`
- full-shape `index` when its element count equals `src`
- empty CSR groups
- float16 / float32 / bfloat16 registration
- int64 `index` and `ptr`
- output shape and dtype inference

## Why the first kernel is single-core

The first goal is to validate correctness and the complete build/submission path without introducing cross-core races for arbitrary unsorted index groups. The current index implementation discovers each unique group within an independent `(outer, inner)` slice, performs stable max/sum normalization, and writes the final output. The CSR path processes contiguous segments directly.

This is intentionally a correctness baseline, not the final performance submission.

## Next optimization milestones

1. Split independent `(outer, inner)` slices across cores.
2. Replace scalar GM access with tiled GM→UB copies.
3. Batch group exponentiation and reduction in UB.
4. CSR: specialize small/medium/large groups and avoid repeated pointer reads.
5. index: add scatter/reduction strategy and a sorted-index fast path.
6. Tune FP16/BF16 paths while retaining FP32 accumulation where needed.

## Local build

The template requires an Ascend C/CANN development environment because `code/CMakeLists.txt` uses `find_package(ASC REQUIRED)`. A generic GitHub-hosted runner cannot perform the real device build unless the Ascend SDK/toolchain is installed.

A lightweight GitHub Actions check is therefore kept separate and only validates the reference Sparse Softmax semantics and repository source invariants. The real acceptance signal remains compilation/execution in the competition's Ascend environment.
