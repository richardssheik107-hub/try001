#pragma once

#include <cstdint>

// UB budget shared by host tiling and kernel runtime.  These are resource
// limits, not shape special-cases: the kernel dynamically tiles any supported
// tensor that exceeds them and falls back to the fully generic scalar path only
// when a DMA tile cannot be represented safely.
constexpr uint32_t SPARSE_SOFTMAX_RAW_BUFFER_BYTES = 32768;
constexpr uint32_t SPARSE_SOFTMAX_INDEX_BUFFER_BYTES = 32768;
constexpr uint32_t SPARSE_SOFTMAX_GROUP_BUFFER_ELEMS = 4096;
constexpr uint32_t SPARSE_SOFTMAX_MAX_DMA_BLOCKS = 4095;

struct SparseSoftmaxTilingData {
    uint64_t totalLength;
    uint64_t outerSize;
    uint64_t dimSize;
    uint64_t innerSize;
    uint64_t indexLength;
    uint64_t ptrLength;
    uint32_t mode;       // 0: index, 1: ptr
    uint32_t fastPath;   // 0: scalar fallback, 1: outer DMA tiles, 2: ptr-group DMA tasks
    uint32_t blockDim;
    uint32_t dtypeSize;
    float eps;
};
