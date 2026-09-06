#include "kernel_operator.h"

#include "sparse_softmax_tiling.h"
#include "tiling_key_sparse_softmax.h"

template <int DT_MODE>
struct StorageTraits;

template <>
struct StorageTraits<SPARSE_SOFTMAX_FP32> {
    using StorageType = float;
    __aicore__ static inline float ToFloatValue(StorageType value) { return value; }
    __aicore__ static inline StorageType FromFloatValue(float value) { return value; }
};

template <>
struct StorageTraits<SPARSE_SOFTMAX_FP16> {
    using StorageType = half;
    __aicore__ static inline float ToFloatValue(StorageType value) {
        return static_cast<float>(value);
    }
    __aicore__ static inline StorageType FromFloatValue(float value) {
        return static_cast<half>(value);
    }
};

template <>
struct StorageTraits<SPARSE_SOFTMAX_BF16> {
    using StorageType = uint16_t;
    __aicore__ static inline float ToFloatValue(StorageType value) {
        union { uint32_t u; float f; } bits;
        bits.u = static_cast<uint32_t>(value) << 16;
        return bits.f;
    }
    __aicore__ static inline StorageType FromFloatValue(float value) {
        union { float f; uint32_t u; } bits;
        bits.f = value;
        const uint32_t upper = bits.u >> 16;
        const uint32_t lsb = upper & 1U;
        const uint32_t rounded = bits.u + 0x7FFFU + lsb;
        return static_cast<uint16_t>(rounded >> 16);
    }
};

template <int DT_MODE>
class KernelSparseSoftmax {
public:
    using StorageType = typename StorageTraits<DT_MODE>::StorageType;

    static constexpr uint32_t GROUP_BUFFER_ELEMS = 2048;
    static constexpr uint32_t INDEX_CACHE_ELEMS = 4096;
    static constexpr uint32_t REDUCE_WORK_ELEMS = 2048;

    __aicore__ inline KernelSparseSoftmax() {}

    __aicore__ inline void Init(GM_ADDR src, GM_ADDR index, GM_ADDR ptr, GM_ADDR out,
                                uint64_t totalLength, uint64_t outerSize,
                                uint64_t dimSize, uint64_t innerSize,
                                uint64_t indexLength, uint64_t ptrLength,
                                uint32_t mode, float eps) {
        srcGlobal_.SetGlobalBuffer((__gm__ StorageType *)src);
        outGlobal_.SetGlobalBuffer((__gm__ StorageType *)out);
        if (index != nullptr) {
            indexGlobal_.SetGlobalBuffer((__gm__ int32_t *)index);
        }
        if (ptr != nullptr) {
            ptrGlobal_.SetGlobalBuffer((__gm__ int32_t *)ptr);
        }

        totalLength_ = totalLength;
        outerSize_ = outerSize;
        dimSize_ = dimSize;
        innerSize_ = innerSize;
        indexLength_ = indexLength;
        ptrLength_ = ptrLength;
        mode_ = mode;
        eps_ = eps;

        pipe_.InitBuffer(valueBuf_, GROUP_BUFFER_ELEMS * sizeof(float));
        pipe_.InitBuffer(expBuf_, GROUP_BUFFER_ELEMS * sizeof(float));
        pipe_.InitBuffer(indexBuf_, INDEX_CACHE_ELEMS * sizeof(int32_t));
        pipe_.InitBuffer(reduceOutBuf_, 32);
        pipe_.InitBuffer(reduceWorkBuf_, REDUCE_WORK_ELEMS * sizeof(float));

        eventSToV_ = static_cast<int32_t>(pipe_.FetchEventID(AscendC::HardEvent::S_V));
        eventVToS_ = static_cast<int32_t>(pipe_.FetchEventID(AscendC::HardEvent::V_S));
    }

    __aicore__ inline void Process() {
        if (totalLength_ == 0 || dimSize_ == 0 || outerSize_ == 0) {
            return;
        }

        if (mode_ == 1U) {
            ProcessPtr();
        } else {
            ProcessIndex();
        }

        AscendC::DataCacheCleanAndInvalid<StorageType,
            AscendC::CacheLine::ENTIRE_DATA_CACHE,
            AscendC::DcciDst::CACHELINE_OUT>(outGlobal_);
    }

private:
    __aicore__ inline void SyncSToV() {
        AscendC::SetFlag<AscendC::HardEvent::S_V>(eventSToV_);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(eventSToV_);
    }

    __aicore__ inline void SyncVToS() {
        AscendC::SetFlag<AscendC::HardEvent::V_S>(eventVToS_);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(eventVToS_);
    }

    __aicore__ inline uint64_t FlatOffset(uint64_t outer, uint64_t dim,
                                          uint64_t inner) const {
        return (outer * dimSize_ + dim) * innerSize_ + inner;
    }

    __aicore__ inline int32_t ReadIndex(uint64_t outer, uint64_t dim,
                                        uint64_t inner) const {
        if (indexLength_ == totalLength_) {
            return indexGlobal_.GetValue(FlatOffset(outer, dim, inner));
        }
        return indexGlobal_.GetValue(dim);
    }

    __aicore__ inline float ReadSrc(uint64_t offset) const {
        return StorageTraits<DT_MODE>::ToFloatValue(srcGlobal_.GetValue(offset));
    }

    __aicore__ inline void WriteOut(uint64_t offset, float value) {
        outGlobal_.SetValue(offset, StorageTraits<DT_MODE>::FromFloatValue(value));
    }

    __aicore__ inline void VectorNormalize(uint32_t count, float groupMax) {
        AscendC::LocalTensor<float> values = valueBuf_.Get<float>();
        AscendC::LocalTensor<float> exps = expBuf_.Get<float>();
        AscendC::LocalTensor<float> reduceOut = reduceOutBuf_.Get<float>();
        AscendC::LocalTensor<float> reduceWork = reduceWorkBuf_.Get<float>();

        SyncSToV();
        AscendC::Adds(values, values, -groupMax, static_cast<int32_t>(count));
        AscendC::Exp(exps, values, static_cast<int32_t>(count));
        AscendC::ReduceSum(reduceOut, exps, reduceWork, static_cast<int32_t>(count));
        SyncVToS();

        const float groupSum = AscendC::GetAccVal<float>();
        const float invDenom = 1.0f / (groupSum + eps_);

        SyncSToV();
        AscendC::Muls(exps, exps, invDenom, static_cast<int32_t>(count));
        SyncVToS();
    }

    __aicore__ inline void LoadContiguousValues(uint64_t outer, uint64_t inner,
                                                uint64_t start, uint32_t count,
                                                float &groupMax) {
        AscendC::LocalTensor<float> values = valueBuf_.Get<float>();
        groupMax = -3.402823466e+38F;
        for (uint32_t i = 0; i < count; ++i) {
            const float value = ReadSrc(FlatOffset(outer, start + i, inner));
            values.SetValue(i, value);
            if (value > groupMax) {
                groupMax = value;
            }
        }
    }

    __aicore__ inline float SumExpChunk(uint32_t count, float groupMax) {
        AscendC::LocalTensor<float> values = valueBuf_.Get<float>();
        AscendC::LocalTensor<float> exps = expBuf_.Get<float>();
        AscendC::LocalTensor<float> reduceOut = reduceOutBuf_.Get<float>();
        AscendC::LocalTensor<float> reduceWork = reduceWorkBuf_.Get<float>();

        SyncSToV();
        AscendC::Adds(values, values, -groupMax, static_cast<int32_t>(count));
        AscendC::Exp(exps, values, static_cast<int32_t>(count));
        AscendC::ReduceSum(reduceOut, exps, reduceWork, static_cast<int32_t>(count));
        SyncVToS();
        return AscendC::GetAccVal<float>();
    }

    __aicore__ inline void NormalizeExpBuffer(uint32_t count, float invDenom) {
        AscendC::LocalTensor<float> exps = expBuf_.Get<float>();
        SyncSToV();
        AscendC::Muls(exps, exps, invDenom, static_cast<int32_t>(count));
        SyncVToS();
    }

    __aicore__ inline void ProcessContiguousGroup(uint64_t outer, uint64_t inner,
                                                  uint64_t start, uint64_t end) {
        const uint64_t count64 = end - start;
        if (count64 == 0) {
            return;
        }
        if (count64 == 1) {
            WriteOut(FlatOffset(outer, start, inner), 1.0f / (1.0f + eps_));
            return;
        }

        if (count64 <= GROUP_BUFFER_ELEMS) {
            const uint32_t count = static_cast<uint32_t>(count64);
            float groupMax;
            LoadContiguousValues(outer, inner, start, count, groupMax);
            VectorNormalize(count, groupMax);

            AscendC::LocalTensor<float> exps = expBuf_.Get<float>();
            for (uint32_t i = 0; i < count; ++i) {
                WriteOut(FlatOffset(outer, start + i, inner), exps.GetValue(i));
            }
            return;
        }

        ProcessContiguousGroupLarge(outer, inner, start, end);
    }

    __aicore__ inline void ProcessContiguousGroupLarge(uint64_t outer, uint64_t inner,
                                                       uint64_t start, uint64_t end) {
        AscendC::LocalTensor<float> values = valueBuf_.Get<float>();
        AscendC::LocalTensor<float> exps = expBuf_.Get<float>();

        float groupMax = -3.402823466e+38F;
        for (uint64_t k = start; k < end; ++k) {
            const float value = ReadSrc(FlatOffset(outer, k, inner));
            if (value > groupMax) {
                groupMax = value;
            }
        }

        float groupSum = 0.0f;
        uint64_t chunkStart = start;
        while (chunkStart < end) {
            const uint64_t remain = end - chunkStart;
            const uint32_t count = static_cast<uint32_t>(
                remain > GROUP_BUFFER_ELEMS ? GROUP_BUFFER_ELEMS : remain);

            for (uint32_t i = 0; i < count; ++i) {
                values.SetValue(i, ReadSrc(FlatOffset(outer, chunkStart + i, inner)));
            }
            groupSum += SumExpChunk(count, groupMax);
            chunkStart += count;
        }

        const float invDenom = 1.0f / (groupSum + eps_);
        chunkStart = start;
        while (chunkStart < end) {
            const uint64_t remain = end - chunkStart;
            const uint32_t count = static_cast<uint32_t>(
                remain > GROUP_BUFFER_ELEMS ? GROUP_BUFFER_ELEMS : remain);

            for (uint32_t i = 0; i < count; ++i) {
                values.SetValue(i, ReadSrc(FlatOffset(outer, chunkStart + i, inner)));
            }

            SumExpChunk(count, groupMax);
            NormalizeExpBuffer(count, invDenom);
            for (uint32_t i = 0; i < count; ++i) {
                WriteOut(FlatOffset(outer, chunkStart + i, inner), exps.GetValue(i));
            }
            chunkStart += count;
        }
    }

    __aicore__ inline int32_t IndexValue(const AscendC::LocalTensor<int32_t>& cachedIndex,
                                         bool useCache,
                                         uint64_t outer, uint64_t dim,
                                         uint64_t inner) const {
        if (useCache) {
            return cachedIndex.GetValue(static_cast<uint32_t>(dim));
        }
        return ReadIndex(outer, dim, inner);
    }

    __aicore__ inline bool CacheIndexSliceAndCheckSorted(
        uint64_t outer, uint64_t inner,
        AscendC::LocalTensor<int32_t>& cachedIndex) {
        bool sorted = true;
        int32_t prev = 0;
        for (uint64_t dim = 0; dim < dimSize_; ++dim) {
            const int32_t current = ReadIndex(outer, dim, inner);
            cachedIndex.SetValue(static_cast<uint32_t>(dim), current);
            if (dim != 0 && current < prev) {
                sorted = false;
            }
            prev = current;
        }
        return sorted;
    }

    __aicore__ inline bool IsIndexNonDecreasingNoCache(uint64_t outer,
                                                       uint64_t inner) const {
        if (dimSize_ < 2) {
            return true;
        }
        int32_t prev = ReadIndex(outer, 0, inner);
        for (uint64_t dim = 1; dim < dimSize_; ++dim) {
            const int32_t current = ReadIndex(outer, dim, inner);
            if (current < prev) {
                return false;
            }
            prev = current;
        }
        return true;
    }

    __aicore__ inline void ProcessSortedIndexSlice(
        uint64_t outer, uint64_t inner,
        const AscendC::LocalTensor<int32_t>& cachedIndex, bool useCache) {
        uint64_t runStart = 0;
        while (runStart < dimSize_) {
            const int32_t group = IndexValue(cachedIndex, useCache, outer, runStart, inner);
            uint64_t runEnd = runStart + 1;
            while (runEnd < dimSize_ &&
                   IndexValue(cachedIndex, useCache, outer, runEnd, inner) == group) {
                ++runEnd;
            }
            ProcessContiguousGroup(outer, inner, runStart, runEnd);
            runStart = runEnd;
        }
    }

    __aicore__ inline void ProcessIndexGroup(
        uint64_t outer, uint64_t inner, int32_t group,
        const AscendC::LocalTensor<int32_t>& cachedIndex, bool useCache) {
        AscendC::LocalTensor<float> values = valueBuf_.Get<float>();
        AscendC::LocalTensor<float> exps = expBuf_.Get<float>();

        float groupMax = -3.402823466e+38F;
        uint64_t groupCount = 0;
        uint64_t singletonOffset = 0;

        for (uint64_t k = 0; k < dimSize_; ++k) {
            if (IndexValue(cachedIndex, useCache, outer, k, inner) != group) {
                continue;
            }

            const float value = ReadSrc(FlatOffset(outer, k, inner));
            singletonOffset = k;
            if (groupCount < GROUP_BUFFER_ELEMS) {
                values.SetValue(static_cast<uint32_t>(groupCount), value);
            }
            ++groupCount;
            if (value > groupMax) {
                groupMax = value;
            }
        }

        if (groupCount == 0) {
            return;
        }
        if (groupCount == 1) {
            WriteOut(FlatOffset(outer, singletonOffset, inner), 1.0f / (1.0f + eps_));
            return;
        }

        if (groupCount <= GROUP_BUFFER_ELEMS) {
            const uint32_t count = static_cast<uint32_t>(groupCount);
            VectorNormalize(count, groupMax);

            uint32_t pos = 0;
            for (uint64_t k = 0; k < dimSize_; ++k) {
                if (IndexValue(cachedIndex, useCache, outer, k, inner) != group) {
                    continue;
                }
                WriteOut(FlatOffset(outer, k, inner), exps.GetValue(pos));
                ++pos;
            }
            return;
        }

        ProcessIndexGroupLarge(outer, inner, group, groupMax, cachedIndex, useCache);
    }

    __aicore__ inline void ProcessIndexGroupLarge(
        uint64_t outer, uint64_t inner, int32_t group, float groupMax,
        const AscendC::LocalTensor<int32_t>& cachedIndex, bool useCache) {
        AscendC::LocalTensor<float> values = valueBuf_.Get<float>();
        AscendC::LocalTensor<float> exps = expBuf_.Get<float>();

        float groupSum = 0.0f;
        uint64_t scan = 0;
        while (scan < dimSize_) {
            uint32_t count = 0;
            while (scan < dimSize_ && count < GROUP_BUFFER_ELEMS) {
                if (IndexValue(cachedIndex, useCache, outer, scan, inner) == group) {
                    values.SetValue(count, ReadSrc(FlatOffset(outer, scan, inner)));
                    ++count;
                }
                ++scan;
            }
            if (count != 0) {
                groupSum += SumExpChunk(count, groupMax);
            }
        }

        const float invDenom = 1.0f / (groupSum + eps_);
        scan = 0;
        while (scan < dimSize_) {
            const uint64_t chunkScanStart = scan;
            uint32_t count = 0;
            while (scan < dimSize_ && count < GROUP_BUFFER_ELEMS) {
                if (IndexValue(cachedIndex, useCache, outer, scan, inner) == group) {
                    values.SetValue(count, ReadSrc(FlatOffset(outer, scan, inner)));
                    ++count;
                }
                ++scan;
            }
            if (count == 0) {
                continue;
            }

            SumExpChunk(count, groupMax);
            NormalizeExpBuffer(count, invDenom);

            uint32_t pos = 0;
            for (uint64_t k = chunkScanStart; k < scan; ++k) {
                if (IndexValue(cachedIndex, useCache, outer, k, inner) != group) {
                    continue;
                }
                WriteOut(FlatOffset(outer, k, inner), exps.GetValue(pos));
                ++pos;
            }
        }
    }

    __aicore__ inline void ProcessUnsortedIndexSlice(
        uint64_t outer, uint64_t inner,
        const AscendC::LocalTensor<int32_t>& cachedIndex, bool useCache) {
        for (uint64_t dim = 0; dim < dimSize_; ++dim) {
            const int32_t group = IndexValue(cachedIndex, useCache, outer, dim, inner);

            bool seen = false;
            for (uint64_t prev = 0; prev < dim; ++prev) {
                if (IndexValue(cachedIndex, useCache, outer, prev, inner) == group) {
                    seen = true;
                    break;
                }
            }
            if (!seen) {
                ProcessIndexGroup(outer, inner, group, cachedIndex, useCache);
            }
        }
    }

    __aicore__ inline void ProcessIndex() {
        AscendC::LocalTensor<int32_t> cachedIndex = indexBuf_.Get<int32_t>();

        const bool isBroadcastIndex = indexLength_ != totalLength_;
        if (isBroadcastIndex && dimSize_ <= INDEX_CACHE_ELEMS) {
            const bool sorted = CacheIndexSliceAndCheckSorted(0, 0, cachedIndex);
            for (uint64_t outer = 0; outer < outerSize_; ++outer) {
                for (uint64_t inner = 0; inner < innerSize_; ++inner) {
                    if (sorted) {
                        ProcessSortedIndexSlice(outer, inner, cachedIndex, true);
                    } else {
                        ProcessUnsortedIndexSlice(outer, inner, cachedIndex, true);
                    }
                }
            }
            return;
        }

        for (uint64_t outer = 0; outer < outerSize_; ++outer) {
            for (uint64_t inner = 0; inner < innerSize_; ++inner) {
                if (dimSize_ <= INDEX_CACHE_ELEMS) {
                    const bool sorted = CacheIndexSliceAndCheckSorted(outer, inner, cachedIndex);
                    if (sorted) {
                        ProcessSortedIndexSlice(outer, inner, cachedIndex, true);
                    } else {
                        ProcessUnsortedIndexSlice(outer, inner, cachedIndex, true);
                    }
                } else {
                    const bool sorted = IsIndexNonDecreasingNoCache(outer, inner);
                    if (sorted) {
                        ProcessSortedIndexSlice(outer, inner, cachedIndex, false);
                    } else {
                        ProcessUnsortedIndexSlice(outer, inner, cachedIndex, false);
                    }
                }
            }
        }
    }

    __aicore__ inline void ProcessPtr() {
        if (ptrLength_ < 2) {
            return;
        }

        const uint64_t groupCount = ptrLength_ - 1;
        for (uint64_t outer = 0; outer < outerSize_; ++outer) {
            for (uint64_t inner = 0; inner < innerSize_; ++inner) {
                int64_t startRaw = static_cast<int64_t>(ptrGlobal_.GetValue(0));
                for (uint64_t group = 0; group < groupCount; ++group) {
                    const int64_t nextStartRaw = static_cast<int64_t>(ptrGlobal_.GetValue(group + 1));
                    int64_t start = startRaw;
                    int64_t end = nextStartRaw;

                    if (start < 0) {
                        start = 0;
                    }
                    if (end >= start && start <= static_cast<int64_t>(dimSize_)) {
                        if (end > static_cast<int64_t>(dimSize_)) {
                            end = static_cast<int64_t>(dimSize_);
                        }
                        if (end > start) {
                            ProcessContiguousGroup(outer, inner,
                                static_cast<uint64_t>(start),
                                static_cast<uint64_t>(end));
                        }
                    }
                    startRaw = nextStartRaw;
                }
            }
        }
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> valueBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> expBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> indexBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceOutBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceWorkBuf_;

    AscendC::GlobalTensor<StorageType> srcGlobal_;
    AscendC::GlobalTensor<int32_t> indexGlobal_;
    AscendC::GlobalTensor<int32_t> ptrGlobal_;
    AscendC::GlobalTensor<StorageType> outGlobal_;

    uint64_t totalLength_ = 0;
    uint64_t outerSize_ = 0;
    uint64_t dimSize_ = 0;
    uint64_t innerSize_ = 0;
    uint64_t indexLength_ = 0;
    uint64_t ptrLength_ = 0;
    uint32_t mode_ = 0;
    float eps_ = 1e-16f;

    int32_t eventSToV_ = 0;
    int32_t eventVToS_ = 0;
};

template <int DT_MODE>
__global__ __aicore__ void sparse_softmax(GM_ADDR src, GM_ADDR index, GM_ADDR ptr,
                                          GM_ADDR out, GM_ADDR workspace,
                                          GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(SparseSoftmaxTilingData);
    GET_TILING_DATA_WITH_STRUCT(SparseSoftmaxTilingData, tilingData, tiling);

    KernelSparseSoftmax<DT_MODE> op;
    op.Init(src, index, ptr, out,
            tilingData.totalLength,
            tilingData.outerSize,
            tilingData.dimSize,
            tilingData.innerSize,
            tilingData.indexLength,
            tilingData.ptrLength,
            tilingData.mode,
            tilingData.eps);
    op.Process();
}
