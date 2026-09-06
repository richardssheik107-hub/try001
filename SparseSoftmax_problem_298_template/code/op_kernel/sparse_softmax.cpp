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
        union {
            uint32_t u;
            float f;
        } bits;
        bits.u = static_cast<uint32_t>(value) << 16;
        return bits.f;
    }

    __aicore__ static inline StorageType FromFloatValue(float value) {
        union {
            float f;
            uint32_t u;
        } bits;
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

    static constexpr uint32_t GROUP_BUFFER_ELEMS =
        SPARSE_SOFTMAX_GROUP_BUFFER_ELEMS;
    static constexpr uint32_t INDEX_CACHE_ELEMS =
        SPARSE_SOFTMAX_INDEX_BUFFER_BYTES / sizeof(int32_t);
    static constexpr uint64_t CACHE_LINE_BYTES = 64;
    static constexpr uint64_t PRECISE_FLUSH_MAX_BYTES = 2048;

    __aicore__ inline KernelSparseSoftmax() {}

    __aicore__ inline void Init(GM_ADDR src, GM_ADDR index, GM_ADDR ptr, GM_ADDR out,
                                uint64_t totalLength, uint64_t outerSize,
                                uint64_t dimSize, uint64_t innerSize,
                                uint64_t indexLength, uint64_t ptrLength,
                                uint32_t mode, uint32_t fastPath,
                                uint32_t blockDim, float eps) {
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
        fastPath_ = fastPath;
        blockDim_ = blockDim;
        eps_ = eps;
        singletonValue_ = 1.0f / (1.0f + eps_);

        pipe_.InitBuffer(rawBuf_, SPARSE_SOFTMAX_RAW_BUFFER_BYTES);
        pipe_.InitBuffer(valueBuf_, GROUP_BUFFER_ELEMS * sizeof(float));
        pipe_.InitBuffer(expBuf_, GROUP_BUFFER_ELEMS * sizeof(float));
        pipe_.InitBuffer(indexBuf_, SPARSE_SOFTMAX_INDEX_BUFFER_BYTES);

        eventSToV_ = static_cast<int32_t>(
            pipe_.FetchEventID(AscendC::HardEvent::S_V));
        eventVToS_ = static_cast<int32_t>(
            pipe_.FetchEventID(AscendC::HardEvent::V_S));
        eventMTE2ToS_ = static_cast<int32_t>(
            pipe_.FetchEventID(AscendC::HardEvent::MTE2_S));
        eventSToMTE2_ = static_cast<int32_t>(
            pipe_.FetchEventID(AscendC::HardEvent::S_MTE2));
        eventSToMTE3_ = static_cast<int32_t>(
            pipe_.FetchEventID(AscendC::HardEvent::S_MTE3));
        eventMTE3ToS_ = static_cast<int32_t>(
            pipe_.FetchEventID(AscendC::HardEvent::MTE3_S));
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

    __aicore__ inline void SyncMTE2ToS() {
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(eventMTE2ToS_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(eventMTE2ToS_);
    }

    __aicore__ inline void SyncSToMTE2() {
        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(eventSToMTE2_);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(eventSToMTE2_);
    }

    __aicore__ inline void SyncSToMTE3() {
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(eventSToMTE3_);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(eventSToMTE3_);
    }

    __aicore__ inline void SyncMTE3ToS() {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(eventMTE3ToS_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(eventMTE3ToS_);
    }

    template <typename T>
    __aicore__ inline void CopyGmToLocal(
        const AscendC::LocalTensor<T>& dst,
        const AscendC::GlobalTensor<T>& src,
        uint16_t blockCount, uint32_t blockLen,
        uint32_t srcStride, uint32_t dstStride) {
        AscendC::DataCopyExtParams copyParams = {
            blockCount, blockLen, srcStride, dstStride, 0};
        AscendC::DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};
        AscendC::DataCopyPad<T>(dst, src, copyParams, padParams);
    }

    template <typename T>
    __aicore__ inline void CopyLocalToGm(
        const AscendC::GlobalTensor<T>& dst,
        const AscendC::LocalTensor<T>& src,
        uint16_t blockCount, uint32_t blockLen,
        uint32_t srcStride, uint32_t dstStride) {
        AscendC::DataCopyExtParams copyParams = {
            blockCount, blockLen, srcStride, dstStride, 0};
        AscendC::DataCopyPad<T>(dst, src, copyParams);
    }

    __aicore__ inline uint64_t Align32(uint64_t bytes) const {
        return (bytes + 31ULL) & ~31ULL;
    }

    __aicore__ inline uint64_t RawCapacityElems() const {
        return SPARSE_SOFTMAX_RAW_BUFFER_BYTES / sizeof(StorageType);
    }

    __aicore__ inline uint64_t FlatOffset(
        uint64_t outer, uint64_t dim, uint64_t inner) const {
        return (outer * dimSize_ + dim) * innerSize_ + inner;
    }

    __aicore__ inline uint64_t OuterBase(uint64_t outer) const {
        return outer * dimSize_ * innerSize_;
    }

    __aicore__ inline int32_t ReadIndex(
        uint64_t outer, uint64_t dim, uint64_t inner) const {
        if (indexLength_ == totalLength_) {
            return indexGlobal_.GetValue(FlatOffset(outer, dim, inner));
        }
        return indexGlobal_.GetValue(dim);
    }

    __aicore__ inline float ReadSrc(uint64_t offset) const {
        return StorageTraits<DT_MODE>::ToFloatValue(
            srcGlobal_.GetValue(offset));
    }

    __aicore__ inline void WriteOut(uint64_t offset, float value) {
        outGlobal_.SetValue(
            offset, StorageTraits<DT_MODE>::FromFloatValue(value));
    }

    __aicore__ inline float ReadRaw(
        AscendC::LocalTensor<StorageType> raw, uint64_t offset) const {
        return StorageTraits<DT_MODE>::ToFloatValue(
            raw.GetValue(static_cast<uint32_t>(offset)));
    }

    __aicore__ inline void WriteRaw(
        AscendC::LocalTensor<StorageType> raw,
        uint64_t offset, float value) {
        raw.SetValue(static_cast<uint32_t>(offset),
                     StorageTraits<DT_MODE>::FromFloatValue(value));
    }

    __aicore__ inline void ExpBuffered(uint32_t count) {
        AscendC::LocalTensor<float> values = valueBuf_.Get<float>();
        AscendC::LocalTensor<float> exps = expBuf_.Get<float>();

        SyncSToV();
        AscendC::Exp(exps, values, static_cast<int32_t>(count));
        SyncVToS();
    }

    __aicore__ inline float SumExp(
        AscendC::LocalTensor<float> exps, uint32_t count) const {
        float s0 = 0.0f;
        float s1 = 0.0f;
        float s2 = 0.0f;
        float s3 = 0.0f;
        uint32_t i = 0;
        for (; i + 3U < count; i += 4U) {
            s0 += exps.GetValue(i);
            s1 += exps.GetValue(i + 1U);
            s2 += exps.GetValue(i + 2U);
            s3 += exps.GetValue(i + 3U);
        }
        float sum = (s0 + s1) + (s2 + s3);
        for (; i < count; ++i) {
            sum += exps.GetValue(i);
        }
        return sum;
    }

    __aicore__ inline void ProcessLocalContiguousGroup(
        AscendC::LocalTensor<StorageType> raw,
        uint64_t rawRowStride, uint64_t innerLocal,
        uint64_t start, uint64_t end) {
        const uint64_t count64 = end - start;
        if (count64 == 0) {
            return;
        }
        if (count64 == 1) {
            WriteRaw(raw, start * rawRowStride + innerLocal, singletonValue_);
            return;
        }

        if (count64 <= GROUP_BUFFER_ELEMS) {
            AscendC::LocalTensor<float> values = valueBuf_.Get<float>();
            AscendC::LocalTensor<float> exps = expBuf_.Get<float>();
            const uint32_t count = static_cast<uint32_t>(count64);

            float groupMax = -3.402823466e+38F;
            for (uint32_t i = 0; i < count; ++i) {
                const float value = ReadRaw(
                    raw, (start + i) * rawRowStride + innerLocal);
                values.SetValue(i, value);
                if (value > groupMax) {
                    groupMax = value;
                }
            }

            for (uint32_t i = 0; i < count; ++i) {
                values.SetValue(i, values.GetValue(i) - groupMax);
            }
            ExpBuffered(count);

            const float invDenom = 1.0f / (SumExp(exps, count) + eps_);
            for (uint32_t i = 0; i < count; ++i) {
                WriteRaw(raw, (start + i) * rawRowStride + innerLocal,
                         exps.GetValue(i) * invDenom);
            }
            return;
        }

        ProcessLocalContiguousGroupLarge(
            raw, rawRowStride, innerLocal, start, end);
    }

    __aicore__ inline void ProcessLocalContiguousGroupLarge(
        AscendC::LocalTensor<StorageType> raw,
        uint64_t rawRowStride, uint64_t innerLocal,
        uint64_t start, uint64_t end) {
        AscendC::LocalTensor<float> values = valueBuf_.Get<float>();
        AscendC::LocalTensor<float> exps = expBuf_.Get<float>();

        float groupMax = -3.402823466e+38F;
        for (uint64_t k = start; k < end; ++k) {
            const float value =
                ReadRaw(raw, k * rawRowStride + innerLocal);
            if (value > groupMax) {
                groupMax = value;
            }
        }

        float groupSum = 0.0f;
        uint64_t chunkStart = start;
        while (chunkStart < end) {
            const uint64_t remain = end - chunkStart;
            const uint32_t count = static_cast<uint32_t>(
                remain > GROUP_BUFFER_ELEMS ?
                GROUP_BUFFER_ELEMS : remain);

            for (uint32_t i = 0; i < count; ++i) {
                values.SetValue(i,
                    ReadRaw(raw, (chunkStart + i) * rawRowStride + innerLocal)
                    - groupMax);
            }
            ExpBuffered(count);
            groupSum += SumExp(exps, count);
            chunkStart += count;
        }

        const float invDenom = 1.0f / (groupSum + eps_);
        chunkStart = start;
        while (chunkStart < end) {
            const uint64_t remain = end - chunkStart;
            const uint32_t count = static_cast<uint32_t>(
                remain > GROUP_BUFFER_ELEMS ?
                GROUP_BUFFER_ELEMS : remain);

            for (uint32_t i = 0; i < count; ++i) {
                values.SetValue(i,
                    ReadRaw(raw, (chunkStart + i) * rawRowStride + innerLocal)
                    - groupMax);
            }
            ExpBuffered(count);
            for (uint32_t i = 0; i < count; ++i) {
                WriteRaw(raw,
                    (chunkStart + i) * rawRowStride + innerLocal,
                    exps.GetValue(i) * invDenom);
            }
            chunkStart += count;
        }
    }

    __aicore__ inline int32_t IndexAtTile(
        AscendC::LocalTensor<int32_t> cachedIndex,
        bool indexLocal, bool broadcastIndex,
        uint64_t indexRowStride, uint64_t outer,
        uint64_t dim, uint64_t globalInner,
        uint64_t localInner) const {
        if (indexLocal) {
            if (broadcastIndex) {
                return cachedIndex.GetValue(static_cast<uint32_t>(dim));
            }
            return cachedIndex.GetValue(static_cast<uint32_t>(
                dim * indexRowStride + localInner));
        }
        return ReadIndex(outer, dim, globalInner);
    }

    __aicore__ inline bool IsTileIndexSorted(
        AscendC::LocalTensor<int32_t> cachedIndex,
        bool indexLocal, bool broadcastIndex,
        uint64_t indexRowStride, uint64_t outer,
        uint64_t globalInner, uint64_t localInner) const {
        if (dimSize_ < 2) {
            return true;
        }
        int32_t prev = IndexAtTile(
            cachedIndex, indexLocal, broadcastIndex,
            indexRowStride, outer, 0, globalInner, localInner);
        for (uint64_t d = 1; d < dimSize_; ++d) {
            const int32_t current = IndexAtTile(
                cachedIndex, indexLocal, broadcastIndex,
                indexRowStride, outer, d, globalInner, localInner);
            if (current < prev) {
                return false;
            }
            prev = current;
        }
        return true;
    }

    __aicore__ inline void ProcessLocalIndexGroup(
        AscendC::LocalTensor<StorageType> raw,
        AscendC::LocalTensor<int32_t> cachedIndex,
        bool indexLocal, bool broadcastIndex,
        uint64_t rawRowStride, uint64_t indexRowStride,
        uint64_t outer, uint64_t globalInner,
        uint64_t localInner, int32_t group) {
        AscendC::LocalTensor<float> values = valueBuf_.Get<float>();
        AscendC::LocalTensor<float> exps = expBuf_.Get<float>();

        float groupMax = -3.402823466e+38F;
        uint64_t groupCount = 0;
        uint64_t singletonPos = 0;

        for (uint64_t k = 0; k < dimSize_; ++k) {
            if (IndexAtTile(cachedIndex, indexLocal, broadcastIndex,
                            indexRowStride, outer, k,
                            globalInner, localInner) != group) {
                continue;
            }
            const float value =
                ReadRaw(raw, k * rawRowStride + localInner);
            singletonPos = k;
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
            WriteRaw(raw, singletonPos * rawRowStride + localInner,
                     singletonValue_);
            return;
        }

        if (groupCount <= GROUP_BUFFER_ELEMS) {
            const uint32_t count = static_cast<uint32_t>(groupCount);
            for (uint32_t i = 0; i < count; ++i) {
                values.SetValue(i, values.GetValue(i) - groupMax);
            }
            ExpBuffered(count);
            const float invDenom = 1.0f / (SumExp(exps, count) + eps_);

            uint32_t pos = 0;
            for (uint64_t k = 0; k < dimSize_; ++k) {
                if (IndexAtTile(cachedIndex, indexLocal, broadcastIndex,
                                indexRowStride, outer, k,
                                globalInner, localInner) != group) {
                    continue;
                }
                WriteRaw(raw, k * rawRowStride + localInner,
                         exps.GetValue(pos) * invDenom);
                ++pos;
            }
            return;
        }

        ProcessLocalIndexGroupLarge(
            raw, cachedIndex, indexLocal, broadcastIndex,
            rawRowStride, indexRowStride, outer,
            globalInner, localInner, group, groupMax);
    }

    __aicore__ inline void ProcessLocalIndexGroupLarge(
        AscendC::LocalTensor<StorageType> raw,
        AscendC::LocalTensor<int32_t> cachedIndex,
        bool indexLocal, bool broadcastIndex,
        uint64_t rawRowStride, uint64_t indexRowStride,
        uint64_t outer, uint64_t globalInner,
        uint64_t localInner, int32_t group, float groupMax) {
        AscendC::LocalTensor<float> values = valueBuf_.Get<float>();
        AscendC::LocalTensor<float> exps = expBuf_.Get<float>();

        float groupSum = 0.0f;
        uint64_t scan = 0;
        while (scan < dimSize_) {
            uint32_t count = 0;
            while (scan < dimSize_ && count < GROUP_BUFFER_ELEMS) {
                if (IndexAtTile(cachedIndex, indexLocal, broadcastIndex,
                                indexRowStride, outer, scan,
                                globalInner, localInner) == group) {
                    values.SetValue(count,
                        ReadRaw(raw, scan * rawRowStride + localInner)
                        - groupMax);
                    ++count;
                }
                ++scan;
            }
            if (count != 0) {
                ExpBuffered(count);
                groupSum += SumExp(exps, count);
            }
        }

        const float invDenom = 1.0f / (groupSum + eps_);
        scan = 0;
        while (scan < dimSize_) {
            const uint64_t chunkScanStart = scan;
            uint32_t count = 0;
            while (scan < dimSize_ && count < GROUP_BUFFER_ELEMS) {
                if (IndexAtTile(cachedIndex, indexLocal, broadcastIndex,
                                indexRowStride, outer, scan,
                                globalInner, localInner) == group) {
                    values.SetValue(count,
                        ReadRaw(raw, scan * rawRowStride + localInner)
                        - groupMax);
                    ++count;
                }
                ++scan;
            }
            if (count == 0) {
                continue;
            }

            ExpBuffered(count);
            uint32_t pos = 0;
            for (uint64_t k = chunkScanStart; k < scan; ++k) {
                if (IndexAtTile(cachedIndex, indexLocal, broadcastIndex,
                                indexRowStride, outer, k,
                                globalInner, localInner) != group) {
                    continue;
                }
                WriteRaw(raw, k * rawRowStride + localInner,
                         exps.GetValue(pos) * invDenom);
                ++pos;
            }
        }
    }

    __aicore__ inline void ProcessSortedIndexColumn(
        AscendC::LocalTensor<StorageType> raw,
        AscendC::LocalTensor<int32_t> cachedIndex,
        bool indexLocal, bool broadcastIndex,
        uint64_t rawRowStride, uint64_t indexRowStride,
        uint64_t outer, uint64_t globalInner,
        uint64_t localInner) {
        uint64_t runStart = 0;
        while (runStart < dimSize_) {
            const int32_t group = IndexAtTile(
                cachedIndex, indexLocal, broadcastIndex,
                indexRowStride, outer, runStart,
                globalInner, localInner);
            uint64_t runEnd = runStart + 1;
            while (runEnd < dimSize_ &&
                   IndexAtTile(cachedIndex, indexLocal, broadcastIndex,
                               indexRowStride, outer, runEnd,
                               globalInner, localInner) == group) {
                ++runEnd;
            }
            ProcessLocalContiguousGroup(
                raw, rawRowStride, localInner, runStart, runEnd);
            runStart = runEnd;
        }
    }

    __aicore__ inline void ProcessUnsortedIndexColumn(
        AscendC::LocalTensor<StorageType> raw,
        AscendC::LocalTensor<int32_t> cachedIndex,
        bool indexLocal, bool broadcastIndex,
        uint64_t rawRowStride, uint64_t indexRowStride,
        uint64_t outer, uint64_t globalInner,
        uint64_t localInner) {
        for (uint64_t dim = 0; dim < dimSize_; ++dim) {
            const int32_t group = IndexAtTile(
                cachedIndex, indexLocal, broadcastIndex,
                indexRowStride, outer, dim,
                globalInner, localInner);

            bool seen = false;
            for (uint64_t prev = 0; prev < dim; ++prev) {
                if (IndexAtTile(cachedIndex, indexLocal, broadcastIndex,
                                indexRowStride, outer, prev,
                                globalInner, localInner) == group) {
                    seen = true;
                    break;
                }
            }
            if (!seen) {
                ProcessLocalIndexGroup(
                    raw, cachedIndex, indexLocal, broadcastIndex,
                    rawRowStride, indexRowStride, outer,
                    globalInner, localInner, group);
            }
        }
    }

    __aicore__ inline void ProcessBroadcastIndexTile(
        AscendC::LocalTensor<StorageType> raw,
        AscendC::LocalTensor<int32_t> cachedIndex,
        bool indexLocal, bool sorted,
        uint64_t rawRowStride, uint64_t outer,
        uint64_t innerStart, uint64_t tileWidth) {
        if (sorted) {
            uint64_t runStart = 0;
            while (runStart < dimSize_) {
                const int32_t group = IndexAtTile(
                    cachedIndex, indexLocal, true, 0,
                    outer, runStart, innerStart, 0);
                uint64_t runEnd = runStart + 1;
                while (runEnd < dimSize_ &&
                       IndexAtTile(cachedIndex, indexLocal, true, 0,
                                   outer, runEnd, innerStart, 0) == group) {
                    ++runEnd;
                }
                for (uint64_t j = 0; j < tileWidth; ++j) {
                    ProcessLocalContiguousGroup(
                        raw, rawRowStride, j, runStart, runEnd);
                }
                runStart = runEnd;
            }
            return;
        }

        for (uint64_t dim = 0; dim < dimSize_; ++dim) {
            const int32_t group = IndexAtTile(
                cachedIndex, indexLocal, true, 0,
                outer, dim, innerStart, 0);
            bool seen = false;
            for (uint64_t prev = 0; prev < dim; ++prev) {
                if (IndexAtTile(cachedIndex, indexLocal, true, 0,
                                outer, prev, innerStart, 0) == group) {
                    seen = true;
                    break;
                }
            }
            if (seen) {
                continue;
            }
            for (uint64_t j = 0; j < tileWidth; ++j) {
                ProcessLocalIndexGroup(
                    raw, cachedIndex, indexLocal, true,
                    rawRowStride, 0, outer,
                    innerStart + j, j, group);
            }
        }
    }

    __aicore__ inline int64_t ReadPtrValue(
        AscendC::LocalTensor<int32_t> cachedPtr,
        bool ptrCached, uint64_t idx) const {
        if (ptrCached) {
            return static_cast<int64_t>(
                cachedPtr.GetValue(static_cast<uint32_t>(idx)));
        }
        return static_cast<int64_t>(ptrGlobal_.GetValue(idx));
    }

    __aicore__ inline bool ClampPtrRange(
        int64_t &startRaw, int64_t &endRaw) const {
        if (startRaw < 0) {
            startRaw = 0;
        }
        if (endRaw < startRaw) {
            return false;
        }
        if (startRaw > static_cast<int64_t>(dimSize_)) {
            return false;
        }
        if (endRaw > static_cast<int64_t>(dimSize_)) {
            endRaw = static_cast<int64_t>(dimSize_);
        }
        return endRaw > startRaw;
    }

    __aicore__ inline void ProcessPtrTile(
        AscendC::LocalTensor<StorageType> raw,
        AscendC::LocalTensor<int32_t> cachedPtr,
        bool ptrCached, uint64_t rawRowStride,
        uint64_t tileWidth) {
        const uint64_t groupCount = ptrLength_ - 1U;
        int64_t startRaw = ReadPtrValue(cachedPtr, ptrCached, 0);
        for (uint64_t group = 0; group < groupCount; ++group) {
            int64_t endRaw =
                ReadPtrValue(cachedPtr, ptrCached, group + 1U);
            int64_t start = startRaw;
            int64_t end = endRaw;
            startRaw = endRaw;

            if (!ClampPtrRange(start, end)) {
                continue;
            }
            for (uint64_t j = 0; j < tileWidth; ++j) {
                ProcessLocalContiguousGroup(
                    raw, rawRowStride, j,
                    static_cast<uint64_t>(start),
                    static_cast<uint64_t>(end));
            }
        }
    }

    __aicore__ inline bool LoadBroadcastIndex(
        AscendC::LocalTensor<int32_t> cachedIndex) {
        if (dimSize_ * sizeof(int32_t) >
            SPARSE_SOFTMAX_INDEX_BUFFER_BYTES) {
            return false;
        }
        CopyGmToLocal<int32_t>(
            cachedIndex, indexGlobal_, 1,
            static_cast<uint32_t>(dimSize_ * sizeof(int32_t)), 0, 0);
        SyncMTE2ToS();
        return true;
    }

    __aicore__ inline bool LoadPtrCache(
        AscendC::LocalTensor<int32_t> cachedPtr) {
        if (ptrLength_ * sizeof(int32_t) >
            SPARSE_SOFTMAX_INDEX_BUFFER_BYTES) {
            return false;
        }
        CopyGmToLocal<int32_t>(
            cachedPtr, ptrGlobal_, 1,
            static_cast<uint32_t>(ptrLength_ * sizeof(int32_t)), 0, 0);
        SyncMTE2ToS();
        return true;
    }

    __aicore__ inline bool IsBroadcastIndexSorted(
        AscendC::LocalTensor<int32_t> cachedIndex,
        bool indexLocal) const {
        if (dimSize_ < 2) {
            return true;
        }
        int32_t prev = indexLocal ?
            cachedIndex.GetValue(0) : indexGlobal_.GetValue(0);
        for (uint64_t d = 1; d < dimSize_; ++d) {
            const int32_t current = indexLocal ?
                cachedIndex.GetValue(static_cast<uint32_t>(d)) :
                indexGlobal_.GetValue(d);
            if (current < prev) {
                return false;
            }
            prev = current;
        }
        return true;
    }

    __aicore__ inline uint64_t ComputeInnerTileWidth() const {
        if (dimSize_ == 0 ||
            dimSize_ > SPARSE_SOFTMAX_MAX_DMA_BLOCKS) {
            return 0;
        }

        uint64_t bytesPerRowBudget =
            SPARSE_SOFTMAX_RAW_BUFFER_BYTES / dimSize_;
        bytesPerRowBudget =
            (bytesPerRowBudget / 32ULL) * 32ULL;
        if (bytesPerRowBudget < 32ULL) {
            return 0;
        }

        uint64_t width =
            bytesPerRowBudget / sizeof(StorageType);
        if (width > innerSize_) {
            width = innerSize_;
        }
        return width;
    }

    __aicore__ inline void ProcessOuterDma(bool ptrMode) {
        AscendC::LocalTensor<StorageType> raw =
            rawBuf_.Get<StorageType>();
        AscendC::LocalTensor<int32_t> cachedIndex =
            indexBuf_.Get<int32_t>();

        const bool broadcastIndex =
            !ptrMode && indexLength_ != totalLength_;
        bool broadcastIndexLocal = false;
        bool broadcastSorted = false;
        bool ptrCached = false;

        if (ptrMode) {
            ptrCached = LoadPtrCache(cachedIndex);
        } else if (broadcastIndex) {
            broadcastIndexLocal = LoadBroadcastIndex(cachedIndex);
            broadcastSorted =
                IsBroadcastIndexSorted(cachedIndex,
                                       broadcastIndexLocal);
        }

        const uint64_t blockIdx =
            static_cast<uint64_t>(AscendC::GetBlockIdx());
        uint64_t blockNum =
            static_cast<uint64_t>(AscendC::GetBlockNum());
        if (blockNum == 0) {
            blockNum = 1;
        }

        const uint64_t slabElems = dimSize_ * innerSize_;
        const uint64_t slabBytes =
            slabElems * sizeof(StorageType);
        const bool wholeOuterFits =
            slabBytes <= SPARSE_SOFTMAX_RAW_BUFFER_BYTES;

        for (uint64_t outer = blockIdx;
             outer < outerSize_; outer += blockNum) {
            if (wholeOuterFits) {
                ProcessWholeOuterDma(
                    raw, cachedIndex, ptrMode, ptrCached,
                    broadcastIndex, broadcastIndexLocal,
                    broadcastSorted, outer);
            } else {
                ProcessOuter2DTiles(
                    raw, cachedIndex, ptrMode, ptrCached,
                    broadcastIndex, broadcastIndexLocal,
                    broadcastSorted, outer);
            }
        }
    }

    __aicore__ inline void ProcessWholeOuterDma(
        AscendC::LocalTensor<StorageType> raw,
        AscendC::LocalTensor<int32_t> cachedIndex,
        bool ptrMode, bool ptrCached,
        bool broadcastIndex, bool broadcastIndexLocal,
        bool broadcastSorted, uint64_t outer) {
        const uint64_t slabElems = dimSize_ * innerSize_;
        const uint32_t slabBytes = static_cast<uint32_t>(
            slabElems * sizeof(StorageType));
        const uint64_t base = OuterBase(outer);

        CopyGmToLocal<StorageType>(
            raw, srcGlobal_[base], 1, slabBytes, 0, 0);

        bool fullIndexLocal = false;
        if (!ptrMode && !broadcastIndex &&
            slabElems * sizeof(int32_t) <=
                SPARSE_SOFTMAX_INDEX_BUFFER_BYTES) {
            CopyGmToLocal<int32_t>(
                cachedIndex,
                indexGlobal_[base], 1,
                static_cast<uint32_t>(
                    slabElems * sizeof(int32_t)), 0, 0);
            fullIndexLocal = true;
        }

        SyncMTE2ToS();

        if (ptrMode) {
            ProcessPtrTile(
                raw, cachedIndex, ptrCached,
                innerSize_, innerSize_);
        } else if (broadcastIndex) {
            ProcessBroadcastIndexTile(
                raw, cachedIndex, broadcastIndexLocal,
                broadcastSorted, innerSize_,
                outer, 0, innerSize_);
        } else {
            for (uint64_t j = 0; j < innerSize_; ++j) {
                const bool sorted = IsTileIndexSorted(
                    cachedIndex, fullIndexLocal, false,
                    innerSize_, outer, j, j);
                if (sorted) {
                    ProcessSortedIndexColumn(
                        raw, cachedIndex, fullIndexLocal, false,
                        innerSize_, innerSize_,
                        outer, j, j);
                } else {
                    ProcessUnsortedIndexColumn(
                        raw, cachedIndex, fullIndexLocal, false,
                        innerSize_, innerSize_,
                        outer, j, j);
                }
            }
        }

        SyncSToMTE3();
        CopyLocalToGm<StorageType>(
            outGlobal_[base], raw, 1, slabBytes, 0, 0);
        SyncMTE3ToS();
    }

    __aicore__ inline void ProcessOuter2DTiles(
        AscendC::LocalTensor<StorageType> raw,
        AscendC::LocalTensor<int32_t> cachedIndex,
        bool ptrMode, bool ptrCached,
        bool broadcastIndex, bool broadcastIndexLocal,
        bool broadcastSorted, uint64_t outer) {
        const uint64_t maxWidth = ComputeInnerTileWidth();
        if (maxWidth == 0) {
            if (AscendC::GetBlockIdx() == 0) {
                if (ptrMode) {
                    ProcessPtrScalar();
                } else {
                    ProcessIndexScalar();
                }
                FlushOutputScalar();
            }
            return;
        }

        for (uint64_t innerStart = 0;
             innerStart < innerSize_;
             innerStart += maxWidth) {
            uint64_t tileWidth = innerSize_ - innerStart;
            if (tileWidth > maxWidth) {
                tileWidth = maxWidth;
            }

            const uint64_t rawRowBytes =
                tileWidth * sizeof(StorageType);
            const uint64_t rawRowStrideBytes =
                Align32(rawRowBytes);
            const uint64_t rawRowStride =
                rawRowStrideBytes / sizeof(StorageType);

            const uint64_t gmBase =
                OuterBase(outer) + innerStart;
            CopyGmToLocal<StorageType>(
                raw, srcGlobal_[gmBase],
                static_cast<uint16_t>(dimSize_),
                static_cast<uint32_t>(rawRowBytes),
                static_cast<uint32_t>(
                    (innerSize_ - tileWidth) *
                    sizeof(StorageType)),
                0);

            bool fullIndexLocal = false;
            uint64_t indexRowStride = 0;
            if (!ptrMode && !broadcastIndex) {
                const uint64_t indexRowBytes =
                    tileWidth * sizeof(int32_t);
                const uint64_t indexRowStrideBytes =
                    Align32(indexRowBytes);
                if (dimSize_ * indexRowStrideBytes <=
                    SPARSE_SOFTMAX_INDEX_BUFFER_BYTES) {
                    indexRowStride =
                        indexRowStrideBytes / sizeof(int32_t);
                    CopyGmToLocal<int32_t>(
                        cachedIndex,
                        indexGlobal_[gmBase],
                        static_cast<uint16_t>(dimSize_),
                        static_cast<uint32_t>(indexRowBytes),
                        static_cast<uint32_t>(
                            (innerSize_ - tileWidth) *
                            sizeof(int32_t)),
                        0);
                    fullIndexLocal = true;
                }
            }

            SyncMTE2ToS();

            if (ptrMode) {
                ProcessPtrTile(
                    raw, cachedIndex, ptrCached,
                    rawRowStride, tileWidth);
            } else if (broadcastIndex) {
                ProcessBroadcastIndexTile(
                    raw, cachedIndex, broadcastIndexLocal,
                    broadcastSorted, rawRowStride,
                    outer, innerStart, tileWidth);
            } else {
                for (uint64_t j = 0; j < tileWidth; ++j) {
                    const bool sorted = IsTileIndexSorted(
                        cachedIndex, fullIndexLocal, false,
                        indexRowStride, outer,
                        innerStart + j, j);
                    if (sorted) {
                        ProcessSortedIndexColumn(
                            raw, cachedIndex,
                            fullIndexLocal, false,
                            rawRowStride, indexRowStride,
                            outer, innerStart + j, j);
                    } else {
                        ProcessUnsortedIndexColumn(
                            raw, cachedIndex,
                            fullIndexLocal, false,
                            rawRowStride, indexRowStride,
                            outer, innerStart + j, j);
                    }
                }
            }

            SyncSToMTE3();
            CopyLocalToGm<StorageType>(
                outGlobal_[gmBase], raw,
                static_cast<uint16_t>(dimSize_),
                static_cast<uint32_t>(rawRowBytes),
                0,
                static_cast<uint32_t>(
                    (innerSize_ - tileWidth) *
                    sizeof(StorageType)));
            SyncMTE3ToS();
        }
    }

    __aicore__ inline void ProcessPtrDmaTasks() {
        AscendC::LocalTensor<int32_t> cachedPtr =
            indexBuf_.Get<int32_t>();
        const bool ptrCached = LoadPtrCache(cachedPtr);

        const uint64_t groupCount = ptrLength_ - 1U;
        const uint64_t taskCount = outerSize_ * groupCount;
        const uint64_t blockIdx =
            static_cast<uint64_t>(AscendC::GetBlockIdx());
        uint64_t blockNum =
            static_cast<uint64_t>(AscendC::GetBlockNum());
        if (blockNum == 0) {
            blockNum = 1;
        }

        for (uint64_t task = blockIdx;
             task < taskCount; task += blockNum) {
            const uint64_t outer = task / groupCount;
            const uint64_t group = task - outer * groupCount;

            int64_t start = ReadPtrValue(
                cachedPtr, ptrCached, group);
            int64_t end = ReadPtrValue(
                cachedPtr, ptrCached, group + 1U);
            if (!ClampPtrRange(start, end)) {
                continue;
            }
            ProcessGmContiguousGroupDma(
                outer, static_cast<uint64_t>(start),
                static_cast<uint64_t>(end));
        }
    }

    __aicore__ inline void ProcessGmContiguousGroupDma(
        uint64_t outer, uint64_t start, uint64_t end) {
        AscendC::LocalTensor<StorageType> raw =
            rawBuf_.Get<StorageType>();
        AscendC::LocalTensor<float> values =
            valueBuf_.Get<float>();
        AscendC::LocalTensor<float> exps =
            expBuf_.Get<float>();

        const uint64_t count64 = end - start;
        const uint64_t gmOffset =
            outer * dimSize_ + start;

        if (count64 == 1) {
            WriteRaw(raw, 0, singletonValue_);
            SyncSToMTE3();
            CopyLocalToGm<StorageType>(
                outGlobal_[gmOffset], raw, 1,
                static_cast<uint32_t>(sizeof(StorageType)), 0, 0);
            SyncMTE3ToS();
            return;
        }

        if (count64 <= RawCapacityElems()) {
            const uint32_t bytes = static_cast<uint32_t>(
                count64 * sizeof(StorageType));

            CopyGmToLocal<StorageType>(
                raw, srcGlobal_[gmOffset],
                1, bytes, 0, 0);
            SyncMTE2ToS();

            ProcessLocalContiguousGroup(
                raw, 1, 0, 0, count64);

            SyncSToMTE3();
            CopyLocalToGm<StorageType>(
                outGlobal_[gmOffset], raw,
                1, bytes, 0, 0);
            SyncMTE3ToS();
            return;
        }

        const uint64_t rawCap = RawCapacityElems();
        float groupMax = -3.402823466e+38F;

        uint64_t chunkStart = 0;
        while (chunkStart < count64) {
            uint64_t chunkCount = count64 - chunkStart;
            if (chunkCount > rawCap) {
                chunkCount = rawCap;
            }
            CopyGmToLocal<StorageType>(
                raw, srcGlobal_[gmOffset + chunkStart],
                1, static_cast<uint32_t>(
                    chunkCount * sizeof(StorageType)), 0, 0);
            SyncMTE2ToS();

            for (uint64_t i = 0; i < chunkCount; ++i) {
                const float value = ReadRaw(raw, i);
                if (value > groupMax) {
                    groupMax = value;
                }
            }
            SyncSToMTE2();
            chunkStart += chunkCount;
        }

        float groupSum = 0.0f;
        chunkStart = 0;
        while (chunkStart < count64) {
            uint64_t chunkCount64 = count64 - chunkStart;
            if (chunkCount64 > rawCap) {
                chunkCount64 = rawCap;
            }

            CopyGmToLocal<StorageType>(
                raw, srcGlobal_[gmOffset + chunkStart],
                1, static_cast<uint32_t>(
                    chunkCount64 * sizeof(StorageType)), 0, 0);
            SyncMTE2ToS();

            uint64_t innerChunk = 0;
            while (innerChunk < chunkCount64) {
                uint64_t n64 = chunkCount64 - innerChunk;
                if (n64 > GROUP_BUFFER_ELEMS) {
                    n64 = GROUP_BUFFER_ELEMS;
                }
                const uint32_t n =
                    static_cast<uint32_t>(n64);
                for (uint32_t i = 0; i < n; ++i) {
                    values.SetValue(i,
                        ReadRaw(raw, innerChunk + i) - groupMax);
                }
                ExpBuffered(n);
                groupSum += SumExp(exps, n);
                innerChunk += n64;
            }
            SyncSToMTE2();
            chunkStart += chunkCount64;
        }

        const float invDenom = 1.0f / (groupSum + eps_);
        chunkStart = 0;
        while (chunkStart < count64) {
            uint64_t chunkCount64 = count64 - chunkStart;
            if (chunkCount64 > rawCap) {
                chunkCount64 = rawCap;
            }

            CopyGmToLocal<StorageType>(
                raw, srcGlobal_[gmOffset + chunkStart],
                1, static_cast<uint32_t>(
                    chunkCount64 * sizeof(StorageType)), 0, 0);
            SyncMTE2ToS();

            uint64_t innerChunk = 0;
            while (innerChunk < chunkCount64) {
                uint64_t n64 = chunkCount64 - innerChunk;
                if (n64 > GROUP_BUFFER_ELEMS) {
                    n64 = GROUP_BUFFER_ELEMS;
                }
                const uint32_t n =
                    static_cast<uint32_t>(n64);
                for (uint32_t i = 0; i < n; ++i) {
                    values.SetValue(i,
                        ReadRaw(raw, innerChunk + i) - groupMax);
                }
                ExpBuffered(n);
                for (uint32_t i = 0; i < n; ++i) {
                    WriteRaw(raw, innerChunk + i,
                             exps.GetValue(i) * invDenom);
                }
                innerChunk += n64;
            }

            SyncSToMTE3();
            CopyLocalToGm<StorageType>(
                outGlobal_[gmOffset + chunkStart], raw,
                1, static_cast<uint32_t>(
                    chunkCount64 * sizeof(StorageType)), 0, 0);
            SyncMTE3ToS();
            chunkStart += chunkCount64;
        }
    }

    __aicore__ inline int32_t ScalarIndexValue(
        const AscendC::LocalTensor<int32_t>& cachedIndex,
        bool useCache, uint64_t outer,
        uint64_t dim, uint64_t inner) const {
        if (useCache) {
            return cachedIndex.GetValue(static_cast<uint32_t>(dim));
        }
        return ReadIndex(outer, dim, inner);
    }

    __aicore__ inline bool CacheScalarIndexSlice(
        uint64_t outer, uint64_t inner,
        AscendC::LocalTensor<int32_t>& cachedIndex) {
        if (dimSize_ > INDEX_CACHE_ELEMS) {
            return false;
        }
        for (uint64_t dim = 0; dim < dimSize_; ++dim) {
            cachedIndex.SetValue(
                static_cast<uint32_t>(dim),
                ReadIndex(outer, dim, inner));
        }
        return true;
    }

    __aicore__ inline bool IsScalarIndexNonDecreasing(
        const AscendC::LocalTensor<int32_t>& cachedIndex,
        bool useCache, uint64_t outer,
        uint64_t inner) const {
        if (dimSize_ < 2) {
            return true;
        }
        int32_t prev = ScalarIndexValue(
            cachedIndex, useCache, outer, 0, inner);
        for (uint64_t dim = 1; dim < dimSize_; ++dim) {
            const int32_t current = ScalarIndexValue(
                cachedIndex, useCache, outer, dim, inner);
            if (current < prev) {
                return false;
            }
            prev = current;
        }
        return true;
    }

    __aicore__ inline void ProcessScalarContiguousGroup(
        uint64_t outer, uint64_t inner,
        uint64_t start, uint64_t end) {
        const uint64_t count64 = end - start;
        if (count64 == 0) {
            return;
        }
        if (count64 == 1) {
            WriteOut(FlatOffset(outer, start, inner), singletonValue_);
            return;
        }

        AscendC::LocalTensor<float> values = valueBuf_.Get<float>();
        AscendC::LocalTensor<float> exps = expBuf_.Get<float>();

        if (count64 <= GROUP_BUFFER_ELEMS) {
            const uint32_t count = static_cast<uint32_t>(count64);
            float groupMax = -3.402823466e+38F;
            for (uint32_t i = 0; i < count; ++i) {
                const float value =
                    ReadSrc(FlatOffset(outer, start + i, inner));
                values.SetValue(i, value);
                if (value > groupMax) {
                    groupMax = value;
                }
            }
            for (uint32_t i = 0; i < count; ++i) {
                values.SetValue(i, values.GetValue(i) - groupMax);
            }
            ExpBuffered(count);
            const float invDenom = 1.0f / (SumExp(exps, count) + eps_);
            for (uint32_t i = 0; i < count; ++i) {
                WriteOut(FlatOffset(outer, start + i, inner),
                         exps.GetValue(i) * invDenom);
            }
            return;
        }

        float groupMax = -3.402823466e+38F;
        for (uint64_t k = start; k < end; ++k) {
            const float value =
                ReadSrc(FlatOffset(outer, k, inner));
            if (value > groupMax) {
                groupMax = value;
            }
        }

        float groupSum = 0.0f;
        uint64_t chunkStart = start;
        while (chunkStart < end) {
            uint64_t remain = end - chunkStart;
            if (remain > GROUP_BUFFER_ELEMS) {
                remain = GROUP_BUFFER_ELEMS;
            }
            const uint32_t count =
                static_cast<uint32_t>(remain);
            for (uint32_t i = 0; i < count; ++i) {
                values.SetValue(i,
                    ReadSrc(FlatOffset(
                        outer, chunkStart + i, inner)) - groupMax);
            }
            ExpBuffered(count);
            groupSum += SumExp(exps, count);
            chunkStart += count;
        }

        const float invDenom = 1.0f / (groupSum + eps_);
        chunkStart = start;
        while (chunkStart < end) {
            uint64_t remain = end - chunkStart;
            if (remain > GROUP_BUFFER_ELEMS) {
                remain = GROUP_BUFFER_ELEMS;
            }
            const uint32_t count =
                static_cast<uint32_t>(remain);
            for (uint32_t i = 0; i < count; ++i) {
                values.SetValue(i,
                    ReadSrc(FlatOffset(
                        outer, chunkStart + i, inner)) - groupMax);
            }
            ExpBuffered(count);
            for (uint32_t i = 0; i < count; ++i) {
                WriteOut(FlatOffset(
                    outer, chunkStart + i, inner),
                    exps.GetValue(i) * invDenom);
            }
            chunkStart += count;
        }
    }

    __aicore__ inline void ProcessScalarIndexGroup(
        uint64_t outer, uint64_t inner, int32_t group,
        const AscendC::LocalTensor<int32_t>& cachedIndex,
        bool useCache) {
        AscendC::LocalTensor<float> values = valueBuf_.Get<float>();
        AscendC::LocalTensor<float> exps = expBuf_.Get<float>();

        float groupMax = -3.402823466e+38F;
        uint64_t groupCount = 0;
        uint64_t singletonPos = 0;

        for (uint64_t k = 0; k < dimSize_; ++k) {
            if (ScalarIndexValue(
                    cachedIndex, useCache, outer, k, inner) != group) {
                continue;
            }
            const float value =
                ReadSrc(FlatOffset(outer, k, inner));
            singletonPos = k;
            if (groupCount < GROUP_BUFFER_ELEMS) {
                values.SetValue(
                    static_cast<uint32_t>(groupCount), value);
            }
            ++groupCount;
            if (value > groupMax) {
                groupMax = value;
            }
        }

        if (groupCount == 1) {
            WriteOut(FlatOffset(
                outer, singletonPos, inner), singletonValue_);
            return;
        }

        if (groupCount <= GROUP_BUFFER_ELEMS) {
            const uint32_t count =
                static_cast<uint32_t>(groupCount);
            for (uint32_t i = 0; i < count; ++i) {
                values.SetValue(i, values.GetValue(i) - groupMax);
            }
            ExpBuffered(count);
            const float invDenom = 1.0f / (SumExp(exps, count) + eps_);

            uint32_t pos = 0;
            for (uint64_t k = 0; k < dimSize_; ++k) {
                if (ScalarIndexValue(
                        cachedIndex, useCache,
                        outer, k, inner) != group) {
                    continue;
                }
                WriteOut(FlatOffset(outer, k, inner),
                         exps.GetValue(pos) * invDenom);
                ++pos;
            }
            return;
        }

        float groupSum = 0.0f;
        uint64_t scan = 0;
        while (scan < dimSize_) {
            uint32_t count = 0;
            while (scan < dimSize_ &&
                   count < GROUP_BUFFER_ELEMS) {
                if (ScalarIndexValue(
                        cachedIndex, useCache,
                        outer, scan, inner) == group) {
                    values.SetValue(count,
                        ReadSrc(FlatOffset(
                            outer, scan, inner)) - groupMax);
                    ++count;
                }
                ++scan;
            }
            if (count != 0) {
                ExpBuffered(count);
                groupSum += SumExp(exps, count);
            }
        }

        const float invDenom = 1.0f / (groupSum + eps_);
        scan = 0;
        while (scan < dimSize_) {
            const uint64_t chunkStart = scan;
            uint32_t count = 0;
            while (scan < dimSize_ &&
                   count < GROUP_BUFFER_ELEMS) {
                if (ScalarIndexValue(
                        cachedIndex, useCache,
                        outer, scan, inner) == group) {
                    values.SetValue(count,
                        ReadSrc(FlatOffset(
                            outer, scan, inner)) - groupMax);
                    ++count;
                }
                ++scan;
            }
            if (count == 0) {
                continue;
            }
            ExpBuffered(count);
            uint32_t pos = 0;
            for (uint64_t k = chunkStart; k < scan; ++k) {
                if (ScalarIndexValue(
                        cachedIndex, useCache,
                        outer, k, inner) != group) {
                    continue;
                }
                WriteOut(FlatOffset(outer, k, inner),
                         exps.GetValue(pos) * invDenom);
                ++pos;
            }
        }
    }

    __aicore__ inline void ProcessIndexScalar() {
        AscendC::LocalTensor<int32_t> cachedIndex =
            indexBuf_.Get<int32_t>();

        for (uint64_t outer = 0; outer < outerSize_; ++outer) {
            for (uint64_t inner = 0; inner < innerSize_; ++inner) {
                const bool useCache =
                    CacheScalarIndexSlice(
                        outer, inner, cachedIndex);
                const bool sorted =
                    IsScalarIndexNonDecreasing(
                        cachedIndex, useCache,
                        outer, inner);

                if (sorted) {
                    uint64_t runStart = 0;
                    while (runStart < dimSize_) {
                        const int32_t group =
                            ScalarIndexValue(
                                cachedIndex, useCache,
                                outer, runStart, inner);
                        uint64_t runEnd = runStart + 1;
                        while (runEnd < dimSize_ &&
                               ScalarIndexValue(
                                   cachedIndex, useCache,
                                   outer, runEnd, inner) == group) {
                            ++runEnd;
                        }
                        ProcessScalarContiguousGroup(
                            outer, inner, runStart, runEnd);
                        runStart = runEnd;
                    }
                    continue;
                }

                for (uint64_t dim = 0; dim < dimSize_; ++dim) {
                    const int32_t group =
                        ScalarIndexValue(
                            cachedIndex, useCache,
                            outer, dim, inner);
                    bool seen = false;
                    for (uint64_t prev = 0; prev < dim; ++prev) {
                        if (ScalarIndexValue(
                                cachedIndex, useCache,
                                outer, prev, inner) == group) {
                            seen = true;
                            break;
                        }
                    }
                    if (!seen) {
                        ProcessScalarIndexGroup(
                            outer, inner, group,
                            cachedIndex, useCache);
                    }
                }
            }
        }
    }

    __aicore__ inline void ProcessPtrScalar() {
        const uint64_t groupCount = ptrLength_ - 1U;
        for (uint64_t outer = 0; outer < outerSize_; ++outer) {
            for (uint64_t inner = 0; inner < innerSize_; ++inner) {
                int64_t startRaw =
                    static_cast<int64_t>(ptrGlobal_.GetValue(0));
                for (uint64_t group = 0;
                     group < groupCount; ++group) {
                    int64_t endRaw = static_cast<int64_t>(
                        ptrGlobal_.GetValue(group + 1U));
                    int64_t start = startRaw;
                    int64_t end = endRaw;
                    startRaw = endRaw;
                    if (!ClampPtrRange(start, end)) {
                        continue;
                    }
                    ProcessScalarContiguousGroup(
                        outer, inner,
                        static_cast<uint64_t>(start),
                        static_cast<uint64_t>(end));
                }
            }
        }
    }

    __aicore__ inline void FlushOutputScalar() {
        const uint64_t totalBytes =
            totalLength_ * sizeof(StorageType);
        const uint64_t elemsPerCacheLine =
            CACHE_LINE_BYTES / sizeof(StorageType);

        if (totalBytes <= PRECISE_FLUSH_MAX_BYTES) {
            for (uint64_t offset = 0;
                 offset < totalLength_;
                 offset += elemsPerCacheLine) {
                AscendC::DataCacheCleanAndInvalid<
                    StorageType,
                    AscendC::CacheLine::SINGLE_CACHE_LINE,
                    AscendC::DcciDst::CACHELINE_OUT>(
                        outGlobal_[offset]);
            }
            return;
        }

        AscendC::DataCacheCleanAndInvalid<
            StorageType,
            AscendC::CacheLine::ENTIRE_DATA_CACHE,
            AscendC::DcciDst::CACHELINE_OUT>(outGlobal_);
    }

    __aicore__ inline void ProcessIndex() {
        if (fastPath_ == 1U) {
            ProcessOuterDma(false);
            return;
        }

        if (AscendC::GetBlockIdx() == 0) {
            ProcessIndexScalar();
            FlushOutputScalar();
        }
    }

    __aicore__ inline void ProcessPtr() {
        if (fastPath_ == 2U && innerSize_ == 1U) {
            ProcessPtrDmaTasks();
            return;
        }
        if (fastPath_ == 1U) {
            ProcessOuterDma(true);
            return;
        }

        if (AscendC::GetBlockIdx() == 0) {
            ProcessPtrScalar();
            FlushOutputScalar();
        }
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECIN> rawBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> valueBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> expBuf_;
    AscendC::TBuf<AscendC::TPosition::VECIN> indexBuf_;

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
    uint32_t fastPath_ = 0;
    uint32_t blockDim_ = 1;
    float eps_ = 1e-16f;
    float singletonValue_ = 1.0f;

    int32_t eventSToV_ = 0;
    int32_t eventVToS_ = 0;
    int32_t eventMTE2ToS_ = 0;
    int32_t eventSToMTE2_ = 0;
    int32_t eventSToMTE3_ = 0;
    int32_t eventMTE3ToS_ = 0;
};

template <int DT_MODE>
__global__ __aicore__ void sparse_softmax(
    GM_ADDR src, GM_ADDR index, GM_ADDR ptr,
    GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(SparseSoftmaxTilingData);
    GET_TILING_DATA_WITH_STRUCT(
        SparseSoftmaxTilingData, tilingData, tiling);

    KernelSparseSoftmax<DT_MODE> op;
    op.Init(src, index, ptr, out,
            tilingData.totalLength,
            tilingData.outerSize,
            tilingData.dimSize,
            tilingData.innerSize,
            tilingData.indexLength,
            tilingData.ptrLength,
            tilingData.mode,
            tilingData.fastPath,
            tilingData.blockDim,
            tilingData.eps);
    op.Process();
}
