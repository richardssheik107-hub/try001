#pragma once

#include <cstdint>

// Correctness-first tiling data.  The first optimization pass deliberately runs
// on one AI Vector core, but keeps enough logical shape information for generic
// dim handling and for both index and CSR(ptr) grouping modes.
struct SparseSoftmaxTilingData {
    uint64_t totalLength;
    uint64_t outerSize;
    uint64_t dimSize;
    uint64_t innerSize;
    uint64_t indexLength;
    uint64_t ptrLength;
    uint32_t mode;  // 0: index, 1: ptr
    float eps;
};
