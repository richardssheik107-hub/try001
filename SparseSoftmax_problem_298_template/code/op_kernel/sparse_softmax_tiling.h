#pragma once

#include <cstdint>

// Resource budgets, not shape special-cases.  Fast paths tile dynamically and
// the scalar fallback preserves arbitrary supported shapes.
constexpr uint32_t SPARSE_SOFTMAX_RAW_BUFFER_BYTES = 8192;
constexpr uint32_t SPARSE_SOFTMAX_INDEX_BUFFER_BYTES = 4096;
constexpr uint32_t SPARSE_SOFTMAX_WORK_BUFFER_ELEMS = 1024;
constexpr uint32_t SPARSE_SOFTMAX_AXIS_OUT_QUEUE_BYTES = 4096;
constexpr uint32_t SPARSE_SOFTMAX_MAX_DMA_BLOCKS = 4095;
constexpr uint32_t SPARSE_SOFTMAX_MAX_AIV = 8;

struct SparseSoftmaxTilingData {
    uint64_t totalLength;
    uint64_t outerSize;
    uint64_t dimSize;
    uint64_t innerSize;
    uint64_t indexLength;
    uint64_t ptrLength;
    uint32_t mode;            // 0: index, 1: ptr
    uint32_t fastPath;        // 0 scalar, 1 tiled DMA, 2 ptr-axis, 3 index-axis MTE3
    uint32_t blockDim;
    uint32_t dtypeSize;
    uint64_t innerTileWidth;
    uint64_t tileCount;
    float eps;
};
