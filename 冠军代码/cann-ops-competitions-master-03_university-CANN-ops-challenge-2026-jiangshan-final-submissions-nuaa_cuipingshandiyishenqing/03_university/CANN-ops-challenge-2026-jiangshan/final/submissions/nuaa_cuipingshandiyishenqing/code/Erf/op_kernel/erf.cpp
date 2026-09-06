// Kernel侧核函数实现：Double Buffer + SingleTile快速路径 + 标量常量多项式逼近 Erf。
// IS_SINGLE_TILE 编译期特化：tileNum==1 时编译为 SingleTile 类，否则编译为 DoubleBuffer 类。
#include "kernel_operator.h"

#include "erf_tiling.h"
#include "tiling_key_erf.h"

using namespace AscendC;

constexpr uint32_t ALIGN_NUM = 8;
constexpr uint32_t BUFFER_NUM = 2;

constexpr float C0 = 1.12419727f;
constexpr float C1 = -0.357146274f;
constexpr float C2 = 0.0879197606f;
constexpr float C3 = -0.0123576f;
constexpr float C4 = 0.00072783462f;
constexpr float CLAMP_MIN = -2.25f;
constexpr float CLAMP_MAX = 2.25f;

// ========== Shared compute body (free function) ==========
template <class DT_X>
__aicore__ inline void ComputeErfBody(
    LocalTensor<DT_X> xLocal,
    LocalTensor<DT_X> yLocal,
    LocalTensor<DT_X> tLocal,
    uint32_t calCount) {
    const int32_t count = static_cast<int32_t>(calCount);

    Maxs(xLocal, xLocal, CLAMP_MIN, count);
    Mins(xLocal, xLocal, CLAMP_MAX, count);

    Mul(tLocal, xLocal, xLocal, count);

    Muls(yLocal, tLocal, static_cast<DT_X>(C4), count);
    Adds(yLocal, yLocal, static_cast<DT_X>(C3), count);
    Mul(yLocal, tLocal, yLocal, count);
    Adds(yLocal, yLocal, static_cast<DT_X>(C2), count);
    Mul(yLocal, tLocal, yLocal, count);
    Adds(yLocal, yLocal, static_cast<DT_X>(C1), count);
    Mul(yLocal, tLocal, yLocal, count);
    Adds(yLocal, yLocal, static_cast<DT_X>(C0), count);

    Mul(yLocal, xLocal, yLocal, count);
}

// ========== SingleTile class: TQue with BUFFER_NUM=1, implicit sync ==========
// Replaces TBuf + explicit WaitFlag with TQue EnQue/DeQue for lighter scalar
// overhead.  For a single tile there is no inter-tile overlap, but avoiding
// FetchEventID/SetFlag/WaitFlag saves scalar instructions.
template <class DT_X>
class KernelErfSingleTile {
public:
    __aicore__ inline KernelErfSingleTile() {}

    // Init + Process merged into one function to eliminate one call/return overhead.
    __aicore__ inline void InitAndProcess(GM_ADDR x, GM_ADDR y, uint64_t totalLength, uint32_t tileLength) {
        // ---- Init ----
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y));

        pipe_.InitBuffer(inQueueX_, 1, tileLength * sizeof(DT_X));
        pipe_.InitBuffer(outQueueY_, 1, tileLength * sizeof(DT_X));
        pipe_.InitBuffer(tmp1_, tileLength * sizeof(DT_X));

        // ---- Process (inlined) ----
        const uint64_t tl = totalLength;
        const uint64_t offset = 0;
        const uint32_t calCount = static_cast<uint32_t>(tl);

        // CopyIn (MTE2) — implicit sync via EnQue
        {
            LocalTensor<DT_X> xLocal = inQueueX_.AllocTensor<DT_X>();
            if (((offset | static_cast<uint64_t>(calCount)) & (ALIGN_NUM - 1)) == 0) {
                DataCopy(xLocal, xGm_[offset], calCount);
            } else {
                DataCopyExtParams copyParams;
                copyParams.blockCount = 1;
                copyParams.blockLen = calCount * sizeof(DT_X);
                copyParams.srcStride = 0;
                copyParams.dstStride = 0;
                copyParams.rsv = 0;
                DataCopyPadExtParams<DT_X> padParams;
                padParams.isPad = false;
                padParams.leftPadding = 0;
                padParams.rightPadding = 0;
                padParams.paddingValue = static_cast<DT_X>(0);
                DataCopyPad(xLocal, xGm_[offset], copyParams, padParams);
            }
            inQueueX_.EnQue(xLocal);
        }

        // Compute (VEC) — DeQue implicitly waits for CopyIn EnQue
        {
            LocalTensor<DT_X> xLocal = inQueueX_.DeQue<DT_X>();
            LocalTensor<DT_X> yLocal = outQueueY_.AllocTensor<DT_X>();
            LocalTensor<DT_X> tLocal = tmp1_.Get<DT_X>();
            ComputeErfBody(xLocal, yLocal, tLocal, calCount);
            outQueueY_.EnQue(yLocal);
            inQueueX_.FreeTensor(xLocal);
        }

        // CopyOut (MTE3) — DeQue implicitly waits for Compute EnQue
        {
            LocalTensor<DT_X> yLocal = outQueueY_.DeQue<DT_X>();
            if (((offset | static_cast<uint64_t>(calCount)) & (ALIGN_NUM - 1)) == 0) {
                DataCopy(yGm_[offset], yLocal, calCount);
            } else {
                DataCopyExtParams copyParams;
                copyParams.blockCount = 1;
                copyParams.blockLen = calCount * sizeof(DT_X);
                copyParams.srcStride = 0;
                copyParams.dstStride = 0;
                copyParams.rsv = 0;
                DataCopyPad(yGm_[offset], yLocal, copyParams);
            }
            outQueueY_.FreeTensor(yLocal);
        }
    }

private:
    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_X> yGm_;

    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> inQueueX_;
    TQue<QuePosition::VECOUT, 1> outQueueY_;
    TBuf<QuePosition::VECCALC> tmp1_;
};

// ========== DoubleBuffer class: TQue pipeline, no TBuf in/out ==========
template <class DT_X>
class KernelErfDoubleBuffer {
public:
    __aicore__ inline KernelErfDoubleBuffer() {}

    // Init + Process merged into one function to eliminate one call/return overhead.
    __aicore__ inline void InitAndProcess(GM_ADDR x, GM_ADDR y, uint64_t totalLength, uint32_t tileLength) {
        // ---- Init ----
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y));

        pipe_.InitBuffer(inQueueX_, BUFFER_NUM, tileLength * sizeof(DT_X));
        pipe_.InitBuffer(outQueueY_, BUFFER_NUM, tileLength * sizeof(DT_X));
        pipe_.InitBuffer(tmp1_, tileLength * sizeof(DT_X));

        // ---- Process (inlined) ----
        const uint64_t totalLengthLocal = totalLength;
        const uint32_t tileLengthLocal = tileLength;
        const uint64_t tileNum = (totalLengthLocal + tileLengthLocal - 1) / tileLengthLocal;
        const uint64_t tileLenU64 = static_cast<uint64_t>(tileLengthLocal);

        const uint32_t firstLength = static_cast<uint32_t>(
            tileLenU64 < totalLengthLocal ? tileLenU64 : totalLengthLocal);
        CopyIn(0, firstLength);

        uint64_t prevOffset = 0;
        uint32_t prevLength = firstLength;
        uint64_t curOffset = tileLenU64;

        for (uint64_t tileIdx = 1; tileIdx < tileNum; ++tileIdx) {
            const uint64_t remain = totalLengthLocal - curOffset;
            const uint32_t curLength = static_cast<uint32_t>(
                remain < tileLenU64 ? remain : tileLenU64);

            Compute(prevLength);
            CopyIn(curOffset, curLength);
            CopyOut(prevOffset, prevLength);

            prevOffset = curOffset;
            prevLength = curLength;
            curOffset += tileLenU64;
        }

        Compute(prevLength);
        CopyOut(prevOffset, prevLength);
    }

private:
    __aicore__ inline void CopyIn(uint64_t offset, uint32_t calCount) {
        LocalTensor<DT_X> xLocal = inQueueX_.AllocTensor<DT_X>();
        if (((offset | static_cast<uint64_t>(calCount)) & (ALIGN_NUM - 1)) == 0) {
            DataCopy(xLocal, xGm_[offset], calCount);
        } else {
            DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = calCount * sizeof(DT_X);
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            copyParams.rsv = 0;
            DataCopyPadExtParams<DT_X> padParams;
            padParams.isPad = false;
            padParams.leftPadding = 0;
            padParams.rightPadding = 0;
            padParams.paddingValue = static_cast<DT_X>(0);
            DataCopyPad(xLocal, xGm_[offset], copyParams, padParams);
        }
        inQueueX_.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t calCount) {
        LocalTensor<DT_X> xLocal = inQueueX_.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outQueueY_.AllocTensor<DT_X>();
        LocalTensor<DT_X> tLocal = tmp1_.Get<DT_X>();
        ComputeErfBody(xLocal, yLocal, tLocal, calCount);
        outQueueY_.EnQue(yLocal);
        inQueueX_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint64_t offset, uint32_t calCount) {
        LocalTensor<DT_X> yLocal = outQueueY_.DeQue<DT_X>();
        if (((offset | static_cast<uint64_t>(calCount)) & (ALIGN_NUM - 1)) == 0) {
            DataCopy(yGm_[offset], yLocal, calCount);
        } else {
            DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = calCount * sizeof(DT_X);
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            copyParams.rsv = 0;
            DataCopyPad(yGm_[offset], yLocal, copyParams);
        }
        outQueueY_.FreeTensor(yLocal);
    }

private:
    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_X> yGm_;

    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY_;
    TBuf<QuePosition::VECCALC> tmp1_;
};

template <typename DT_X, bool IS_SINGLE_TILE>
__global__ __aicore__ void erf(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    (void)workspace;
    REGISTER_TILING_DEFAULT(ErfTilingData);
    GET_TILING_DATA_WITH_STRUCT(ErfTilingData, tilingData, tiling);

    const uint32_t blockDim = tilingData.usedCoreNum;
    const uint32_t blockIdx = GetBlockIdx();
    if (blockIdx >= blockDim) {
        return;
    }

    // Host pre-computed per-core data distribution (eliminates kernel-side div/mod)
    const uint32_t tailBlockNum = tilingData.tailBlockNum;
    uint64_t offset;
    uint64_t length;
    if (blockIdx < tailBlockNum) {
        length = tilingData.bigCoreDataNum;
        offset = static_cast<uint64_t>(blockIdx) * tilingData.bigCoreDataNum;
    } else {
        length = tilingData.smallCoreDataNum;
        offset = static_cast<uint64_t>(tailBlockNum) * tilingData.bigCoreDataNum
               + static_cast<uint64_t>(blockIdx - tailBlockNum) * tilingData.smallCoreDataNum;
    }

    // Append global trailing elements (< 8) to the last active core.
    // All aligned bodies are 8-element aligned, so only this final tail
    // may need the DataCopyPad slow path.
    if (tilingData.tailNum != 0 && blockIdx == blockDim - 1) {
        length += tilingData.tailNum;
    }

    if (length == 0) {
        return;
    }

    if constexpr (IS_SINGLE_TILE) {
        KernelErfSingleTile<DT_X> op;
        op.InitAndProcess(x + offset * sizeof(DT_X), y + offset * sizeof(DT_X), length, tilingData.tileLength);
    } else {
        KernelErfDoubleBuffer<DT_X> op;
        op.InitAndProcess(x + offset * sizeof(DT_X), y + offset * sizeof(DT_X), length, tilingData.tileLength);
    }
}
