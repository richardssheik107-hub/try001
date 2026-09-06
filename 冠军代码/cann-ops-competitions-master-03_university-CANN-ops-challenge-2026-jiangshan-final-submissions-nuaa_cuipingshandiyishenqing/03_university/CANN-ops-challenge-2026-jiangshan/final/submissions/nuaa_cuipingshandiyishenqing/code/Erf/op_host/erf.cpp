#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/erf_tiling.h"
#include "../op_kernel/tiling_key_erf.h"

namespace {
constexpr uint32_t ALIGN_NUM = 8;
constexpr uint32_t MIN_TILE_LENGTH = 8;
constexpr uint32_t RESERVED_UB_BYTES = 8192;



static uint32_t AlignDown(uint64_t value, uint32_t align) {
    return static_cast<uint32_t>((value / align) * align);
}

static uint32_t AlignUp(uint64_t value, uint32_t align) {
    uint64_t rem = value % align;
    if (rem == 0) {
        return static_cast<uint32_t>(value);
    }
    return static_cast<uint32_t>(value + (align - rem));
}

static bool TilingTraceEnabled() {
    const char *value = std::getenv("OP_AGENT_TILING_TRACE");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

static const char *SafeEnv(const char *name) {
    const char *value = std::getenv(name);
    return value != nullptr ? value : "";
}

static uint32_t CalcTileLength(uint64_t ubSizeBytes, uint64_t totalLength) {
    // Kernel uses 5 float32 tile buffers: inQueueX_(×2) + outQueueY_(×2) + tmp1_(×1)
    const uint64_t usable = ubSizeBytes > RESERVED_UB_BYTES ? (ubSizeBytes - RESERVED_UB_BYTES) : 0;
    const uint64_t maxTileDataNumRaw = usable / (5U * sizeof(float));
    const uint32_t maxTileDataNum = AlignDown(maxTileDataNumRaw, ALIGN_NUM);

    uint32_t tileDataNum;
    if (totalLength <= maxTileDataNum) {
        // Small shape: use a tile that covers the whole input (aligned up)
        tileDataNum = AlignUp(totalLength, ALIGN_NUM);
    } else {
        // Large shape: use the maximum tile the UB can hold
        tileDataNum = maxTileDataNum;
    }

    if (tileDataNum < MIN_TILE_LENGTH) {
        tileDataNum = MIN_TILE_LENGTH;
    }
    return tileDataNum;
}

static uint32_t CalcBlockDim(uint64_t length, uint32_t tileLength, uint32_t maxCoreNum) {
    (void)tileLength;
    if (length <= 32768ULL) {
        return std::min<uint32_t>(8, maxCoreNum);
    }
    if (length <= 131072ULL) {
        return std::min<uint32_t>(16, maxCoreNum);
    }
    if (length <= 262144ULL) {
        return std::min<uint32_t>(20, maxCoreNum);
    }
    if (length <= 524288ULL) {
        return std::min<uint32_t>(32, maxCoreNum);
    }
    return std::min<uint32_t>(40, maxCoreNum);
}

static void FormatShape(
    char *shapeText,
    size_t shapeTextSize,
    int32_t dimNum,
    int64_t dim0,
    int64_t dim1,
    int64_t dim2,
    int64_t dim3) {
    if (dimNum <= 1) {
        std::snprintf(shapeText, shapeTextSize, "%lld", static_cast<long long>(dim0));
    } else if (dimNum == 2) {
        std::snprintf(
            shapeText,
            shapeTextSize,
            "%lldx%lld",
            static_cast<long long>(dim0),
            static_cast<long long>(dim1));
    } else if (dimNum == 3) {
        std::snprintf(
            shapeText,
            shapeTextSize,
            "%lldx%lldx%lld",
            static_cast<long long>(dim0),
            static_cast<long long>(dim1),
            static_cast<long long>(dim2));
    } else {
        std::snprintf(
            shapeText,
            shapeTextSize,
            "%lldx%lldx%lldx%lld",
            static_cast<long long>(dim0),
            static_cast<long long>(dim1),
            static_cast<long long>(dim2),
            static_cast<long long>(dim3));
    }
}

static void EmitTilingTrace(
    uint64_t length,
    int32_t dimNum,
    int64_t dim0,
    int64_t dim1,
    int64_t dim2,
    int64_t dim3,
    uint32_t tileLength,
    uint64_t tileCount,
    uint32_t usedCoreNum,
    uint32_t vectorCoreNum,
    uint32_t aicNum,
    uint32_t aivNum,
    uint64_t ubSizeBytes,
    uint64_t perCoreLength,
    uint64_t lastCoreLength,
    uint32_t isSingleTile) {
    if (!TilingTraceEnabled()) {
        return;
    }

    char shapeText[128] = {0};
    FormatShape(shapeText, sizeof(shapeText), dimNum, dim0, dim1, dim2, dim3);
    const uint64_t estimatedUbBytes = static_cast<uint64_t>(tileLength) * sizeof(float) * 2U;

    std::fprintf(
        stderr,
        "[TILING_TRACE]{"
        "\"case_id\":\"%s\","
        "\"shape\":\"%s\","
        "\"length\":%llu,"
        "\"tiling_key\":\"DT_X=float32,IS_SINGLE_TILE=%u\","
        "\"path\":\"NaiveSingleCoreOfficialErfApi\","
        "\"is_single_tile\":%s,"
        "\"tile_length\":%u,"
        "\"tile_count\":%llu,"
        "\"used_core_num\":%u,"
        "\"available_core_num\":%u,"
        "\"vector_core_num\":%u,"
        "\"aic_num\":%u,"
        "\"aiv_num\":%u,"
        "\"ub_size_bytes\":%llu,"
        "\"ub_size_kb\":%llu,"
        "\"per_core_length\":%llu,"
        "\"last_core_length\":%llu,"
        "\"estimated_ub_bytes\":%llu,"
        "\"buffer_num\":1"
        "}\n",
        SafeEnv("CASE_ID"),
        shapeText,
        static_cast<unsigned long long>(length),
        isSingleTile,
        isSingleTile ? "true" : "false",
        tileLength,
        static_cast<unsigned long long>(tileCount),
        usedCoreNum,
        vectorCoreNum,
        vectorCoreNum,
        aicNum,
        aivNum,
        static_cast<unsigned long long>(ubSizeBytes),
        static_cast<unsigned long long>(ubSizeBytes / 1024ULL),
        static_cast<unsigned long long>(perCoreLength),
        static_cast<unsigned long long>(lastCoreLength),
        static_cast<unsigned long long>(estimatedUbBytes));
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    const auto rawAicNum = ascendcPlatform.GetCoreNumAic();
    const auto rawAivNum = ascendcPlatform.GetCoreNumAiv();
    const uint32_t aicNum = rawAicNum > 0 ? static_cast<uint32_t>(rawAicNum) : 0U;
    const uint32_t aivNum = rawAivNum > 0 ? static_cast<uint32_t>(rawAivNum) : 0U;
    uint32_t vectorCoreNum = aivNum > 0U ? aivNum : aicNum;
    if (vectorCoreNum == 0U) {
        vectorCoreNum = 1U;
    }

    uint64_t ubSizeBytes = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizeBytes);

    const gert::Tensor *tensorX = context->GetRequiredInputTensor(0);
    if (tensorX == nullptr || tensorX->GetDataType() != ge::DT_FLOAT) {
        return ge::GRAPH_FAILED;
    }

    const int64_t shapeSize = tensorX->GetShapeSize();
    const uint64_t length = shapeSize > 0 ? static_cast<uint64_t>(shapeSize) : 0ULL;
    const gert::StorageShape *xShape = context->GetInputShape(0);
    if (xShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto storageShape = xShape->GetStorageShape();
    const int32_t dimNum = storageShape.GetDimNum();
    const int64_t firstDim = dimNum > 0 ? storageShape.GetDim(0) : 0;
    const int64_t secondDim = dimNum > 1 ? storageShape.GetDim(1) : 0;
    const int64_t thirdDim = dimNum > 2 ? storageShape.GetDim(2) : 0;
    const int64_t fourthDim = dimNum > 3 ? storageShape.GetDim(3) : 0;

    const uint32_t tileLength = CalcTileLength(ubSizeBytes, length);
    const uint64_t tileCount = tileLength > 0 ? (length + tileLength - 1) / tileLength : 0ULL;
    uint32_t usedCoreNum = CalcBlockDim(length, tileLength, vectorCoreNum);

    // Allow environment override for blockDim benchmarking
    const char *envBlockDim = std::getenv("ERF_BLOCK_DIM");
    if (envBlockDim != nullptr) {
        int parsed = std::atoi(envBlockDim);
        if (parsed > 0) {
            usedCoreNum = static_cast<uint32_t>(parsed);
        }
    }

    // Align total length down to 32B (8 float32 elements) granularity.
    // Distribute aligned blocks across cores; global tail (<8 elements)
    // is appended to the last active core.
    const uint64_t alignedLength = (length / ALIGN_NUM) * ALIGN_NUM;
    uint32_t tailNum = static_cast<uint32_t>(length - alignedLength);

    const uint64_t alignedBlockNum = alignedLength / ALIGN_NUM;
    uint64_t smallCoreDataNum;
    uint64_t bigCoreDataNum;
    uint32_t tailBlockNum;
    uint64_t remBlocks = 0;

    if (alignedBlockNum == 0) {
        // No aligned body: the entire payload is the tail.
        // Absorb tail into the first big core so the kernel does not
        // double-count it via the last-core append path.
        smallCoreDataNum = 0;
        bigCoreDataNum = tailNum;
        tailBlockNum = 1;
        tailNum = 0;
    } else {
        const uint64_t blocksPerCore = alignedBlockNum / usedCoreNum;
        remBlocks = alignedBlockNum % usedCoreNum;
        smallCoreDataNum = blocksPerCore * ALIGN_NUM;
        bigCoreDataNum = (blocksPerCore + 1) * ALIGN_NUM;
        tailBlockNum = static_cast<uint32_t>(remBlocks);
    }

    // SingleTile check: max per-core length must fit in one tile.
    // When there are big cores, they hold the max. Otherwise the last
    // small core may get the tailNum and become the longest.
    const uint64_t maxCoreLength = (remBlocks > 0) ? bigCoreDataNum : (smallCoreDataNum + tailNum);
    const uint32_t isSingleTile = (maxCoreLength <= tileLength) ? 1U : 0U;

    std::fprintf(
        stderr,
        "[HW_INFO][Erf] opType=Erf aicNum=%u aivNum=%u vectorCoreNum=%u "
        "ubSizeBytes=%llu ubSizeKB=%llu\n",
        aicNum,
        aivNum,
        vectorCoreNum,
        static_cast<unsigned long long>(ubSizeBytes),
        static_cast<unsigned long long>(ubSizeBytes / 1024ULL));
    std::fprintf(
        stderr,
        "[TILING_INFO][Erf] totalLength=%llu blockDim=%u coreNum=%u alignedLength=%llu tailNum=%u "
        "smallCore=%llu bigCore=%llu tileLength=%u tileNum=%llu tilingKey=DT_X=float32,IS_SINGLE_TILE=%u\n",
        static_cast<unsigned long long>(length),
        usedCoreNum,
        vectorCoreNum,
        static_cast<unsigned long long>(alignedLength),
        tailNum,
        static_cast<unsigned long long>(smallCoreDataNum),
        static_cast<unsigned long long>(bigCoreDataNum),
        tileLength,
        static_cast<unsigned long long>(tileCount),
        isSingleTile);

    uint32_t DT_X = static_cast<uint32_t>(tensorX->GetDataType());
    ASCENDC_TPL_SEL_PARAM(context, DT_X, isSingleTile);

    ErfTilingData *tiling = context->GetTilingData<ErfTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->length = length;
    tiling->usedCoreNum = usedCoreNum;
    tiling->tileLength = tileLength;
    tiling->smallCoreDataNum = smallCoreDataNum;
    tiling->bigCoreDataNum = bigCoreDataNum;
    tiling->tailBlockNum = tailBlockNum;
    tiling->tailNum = tailNum;

    EmitTilingTrace(
        length,
        dimNum,
        firstDim,
        secondDim,
        thirdDim,
        fourthDim,
        tileLength,
        tileCount,
        usedCoreNum,
        vectorCoreNum,
        aicNum,
        aivNum,
        ubSizeBytes,
        smallCoreDataNum,
        maxCoreLength,
        isSingleTile);

    context->SetBlockDim(usedCoreNum);

    size_t *workspace = context->GetWorkspaceSizes(1);
    if (workspace != nullptr) {
        workspace[0] = 0;
    }

    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    const gert::Shape *xShape = context->GetInputShape(0);
    gert::Shape *yShape = context->GetOutputShape(0);
    if (xShape == nullptr || yShape == nullptr) {
        return GRAPH_FAILED;
    }
    *yShape = *xShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class Erf : public OpDef {
public:
    explicit Erf(const char *name) : OpDef(name) {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});

        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(Erf);
}  // namespace ops
