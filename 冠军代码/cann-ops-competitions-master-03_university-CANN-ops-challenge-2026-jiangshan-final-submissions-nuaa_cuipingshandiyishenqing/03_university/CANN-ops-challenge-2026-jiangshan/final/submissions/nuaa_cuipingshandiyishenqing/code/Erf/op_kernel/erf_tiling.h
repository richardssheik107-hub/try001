// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct ErfTilingData {
    uint64_t length;
    uint32_t usedCoreNum;
    uint32_t tileLength;
    uint64_t smallCoreDataNum;  // per-core data count for small cores
    uint64_t bigCoreDataNum;    // per-core data count for big cores
    uint32_t tailBlockNum;      // number of big cores
    uint32_t tailNum;           // global trailing elements (< ALIGN_NUM) handled by last core
};