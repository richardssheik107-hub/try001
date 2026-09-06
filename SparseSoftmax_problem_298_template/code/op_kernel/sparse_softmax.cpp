#include "kernel_operator.h"
#include "sparse_softmax_tiling.h"
#include "tiling_key_sparse_softmax.h"

// The included validated base still owns the full ProcessIndex and ProcessPtr
// grouping paths.  This wrapper only intercepts one broadcast-index hot path.
// Keep the previously validated kernel as the base implementation.  Only
// expose its internals to this translation unit so we can add one narrow,
// semantics-preserving hot path without rewriting the stable 1700+ line
// implementation.
#define private public
#define sparse_softmax sparse_softmax_base_impl
#include "sparse_softmax_base.inc"
#undef sparse_softmax
#undef private

template <int DT_MODE>
class KernelSparseSoftmaxSorted final : public KernelSparseSoftmax<DT_MODE> {
public:
    using Base = KernelSparseSoftmax<DT_MODE>;
    using StorageType = typename Base::StorageType;

    __aicore__ inline bool CachedBroadcastIndexIsSorted(
        AscendC::LocalTensor<int32_t> cachedIndex) const {
        if (this->dimSize_ <= 1U) {
            return true;
        }

        int32_t previous = cachedIndex.GetValue(0);
        uint32_t runLength = 1U;
        for (uint64_t pos = 1U; pos < this->dimSize_; ++pos) {
            const int32_t current = cachedIndex.GetValue(
                static_cast<uint32_t>(pos));
            if (current < previous) {
                return false;
            }
            if (current == previous) {
                ++runLength;
                // Existing batched work buffer stores one complete group per
                // lane.  Keep the optimized route inside that proven budget.
                if (runLength > Base::WORK_ELEMS) {
                    return false;
                }
            } else {
                previous = current;
                runLength = 1U;
            }
        }
        return true;
    }

    __aicore__ inline void ProcessSortedBroadcastRuns(
        AscendC::LocalTensor<StorageType> raw,
        AscendC::LocalTensor<int32_t> cachedIndex,
        uint64_t rowStride, uint64_t tileWidth) {
        uint64_t runStart = 0U;
        while (runStart < this->dimSize_) {
            const int32_t group = cachedIndex.GetValue(
                static_cast<uint32_t>(runStart));
            uint64_t runEnd = runStart + 1U;
            while (runEnd < this->dimSize_ &&
                   cachedIndex.GetValue(static_cast<uint32_t>(runEnd)) ==
                       group) {
                ++runEnd;
            }

            const uint32_t groupCount = static_cast<uint32_t>(
                runEnd - runStart);
            uint64_t laneStart = 0U;
            while (laneStart < tileWidth) {
                const uint32_t laneBatch = this->ChooseLaneBatch(
                    tileWidth - laneStart, groupCount);
                this->SoftmaxContiguousBatch(
                    raw, rowStride, laneStart, laneBatch,
                    runStart, runEnd);
                laneStart += laneBatch;
            }
            runStart = runEnd;
        }
    }

    __aicore__ inline void ProcessOneBroadcastTile(
        AscendC::LocalTensor<StorageType> raw,
        AscendC::LocalTensor<int32_t> cachedIndex,
        bool sortedIndex,
        uint64_t outer, uint64_t innerStart,
        uint64_t tileWidth) {
        const uint64_t rowBytes =
            tileWidth * sizeof(StorageType);
        const uint64_t rowStrideBytes = this->Align32(rowBytes);
        const uint64_t rowStride =
            rowStrideBytes / sizeof(StorageType);
        const uint64_t gmBase =
            this->OuterBase(outer) + innerStart;

        this->template CopyGmToLocal<StorageType>(
            raw, this->srcGlobal_[gmBase],
            static_cast<uint16_t>(this->dimSize_),
            static_cast<uint32_t>(rowBytes),
            static_cast<uint32_t>(
                (this->innerSize_ - tileWidth) *
                sizeof(StorageType)),
            0);
        this->SyncMTE2ToS();

        if (sortedIndex) {
            ProcessSortedBroadcastRuns(
                raw, cachedIndex, rowStride, tileWidth);
        } else {
            // Exact original algorithm for arbitrary/unsorted labels.
            this->ProcessBroadcastIndexTile(
                raw, cachedIndex, true,
                rowStride, outer, innerStart, tileWidth);
        }

        this->SyncSToMTE3();
        this->template CopyLocalToGm<StorageType>(
            this->outGlobal_[gmBase], raw,
            static_cast<uint16_t>(this->dimSize_),
            static_cast<uint32_t>(rowBytes),
            0,
            static_cast<uint32_t>(
                (this->innerSize_ - tileWidth) *
                sizeof(StorageType)));
        this->SyncMTE3ToS();
    }

    __aicore__ inline void ProcessBroadcastTiledDma() {
        AscendC::LocalTensor<StorageType> raw =
            this->rawBuf_.template Get<StorageType>();
        AscendC::LocalTensor<int32_t> cachedIndex =
            this->indexBuf_.template Get<int32_t>();

        // The linear-run route is intentionally limited to broadcast index
        // metadata that can be cached once in UB.  Larger metadata takes the
        // original path unchanged rather than paying extra GM classification.
        const bool indexCached = this->LoadBroadcastIndex(cachedIndex);
        if (!indexCached) {
            Base::Process();
            return;
        }
        const bool sortedIndex =
            CachedBroadcastIndexIsSorted(cachedIndex);

        const uint64_t taskCount =
            this->outerSize_ * this->tileCount_;
        const uint64_t blockIdx = static_cast<uint64_t>(
            AscendC::GetBlockIdx());
        uint64_t blockNum = static_cast<uint64_t>(
            AscendC::GetBlockNum());
        if (blockNum == 0U) {
            blockNum = 1U;
        }

        for (uint64_t task = blockIdx;
             task < taskCount; task += blockNum) {
            const uint64_t outer = task / this->tileCount_;
            const uint64_t tile =
                task - outer * this->tileCount_;
            const uint64_t innerStart =
                tile * this->innerTileWidth_;
            if (innerStart >= this->innerSize_) {
                continue;
            }
            uint64_t tileWidth =
                this->innerSize_ - innerStart;
            if (tileWidth > this->innerTileWidth_) {
                tileWidth = this->innerTileWidth_;
            }

            ProcessOneBroadcastTile(
                raw, cachedIndex, sortedIndex,
                outer, innerStart, tileWidth);
        }
    }

    __aicore__ inline void Process() {
        if (this->totalLength_ == 0U ||
            this->dimSize_ == 0U ||
            this->outerSize_ == 0U) {
            return;
        }

        // Only intercept the common PyG layout: one broadcast index vector
        // shared by multiple inner lanes.  All other layouts execute the
        // previously validated implementation byte-for-byte from the base.
        if (this->mode_ == 0U &&
            this->fastPath_ == 1U &&
            this->innerSize_ > 1U &&
            this->indexLength_ != this->totalLength_) {
            ProcessBroadcastTiledDma();
            return;
        }
        Base::Process();
    }
};

template <int DT_MODE>
__global__ __aicore__ void sparse_softmax(
    GM_ADDR src, GM_ADDR index, GM_ADDR ptr,
    GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(SparseSoftmaxTilingData);
    GET_TILING_DATA_WITH_STRUCT(
        SparseSoftmaxTilingData, tilingData, tiling);

    KernelSparseSoftmaxSorted<DT_MODE> op;
    op.Init(src, index, ptr, out, tilingData);
    op.Process();
}
