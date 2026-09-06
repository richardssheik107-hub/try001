#pragma once

#include <cstdint>

// Resource budgets are layout-independent. 910B has ample UB, but microsecond
// kernels benefit from keeping per-AIV state compact while making DMA tiles
// large enough to amortize setup cost.
constexpr uint32_t SPARSE_SOFTMAX_RAW_BUFFER_BYTES = 16384;
constexpr uint32_t SPARSE_SOFTMAX_INDEX_BUFFER_BYTES = 4096;
constexpr uint32_t SPARSE_SOFTMAX_WORK_BUFFER_ELEMS = 2048;
constexpr uint32_t SPARSE_SOFTMAX_AXIS_OUT_QUEUE_BYTES = 4096;
constexpr uint32_t SPARSE_SOFTMAX_MAX_DMA_BLOCKS = 4095;
constexpr uint32_t SPARSE_SOFTMAX_MAX_AIV = 40;

struct SparseSoftmaxTilingData {
    uint64_t totalLength;
    uint64_t outerSize;
    uint64_t dimSize;
    uint64_t innerSize;
    uint64_t indexLength;
    uint64_t ptrLength;
    uint32_t mode;            // 0: index, 1: ptr
    uint32_t fastPath;        // 0 scalar, 1 tiled DMA, 2 ptr-axis, 3 index-axis
    uint32_t blockDim;
    uint32_t dtypeSize;
    uint64_t innerTileWidth;
    uint64_t tileCount;
    float eps;
};
