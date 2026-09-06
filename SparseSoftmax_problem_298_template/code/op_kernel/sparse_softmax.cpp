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
    static constexpr uint32_t WORK_ELEMS = SPARSE_SOFTMAX_WORK_BUFFER_ELEMS;
    static constexpr uint32_t INDEX_ELEMS = SPARSE_SOFTMAX_INDEX_BUFFER_BYTES / sizeof(int32_t);
    static constexpr uint32_t LANE_BATCH_MAX = 8;
    static constexpr float NEG_FLOAT_MAX = -3.402823466e+38F;

    __aicore__ inline KernelSparseSoftmax() {}

    __aicore__ inline void Init(
        GM_ADDR src, GM_ADDR index, GM_ADDR ptr, GM_ADDR out,
        const SparseSoftmaxTilingData &tiling) {
        srcGlobal_.SetGlobalBuffer((__gm__ StorageType *)src);
        outGlobal_.SetGlobalBuffer((__gm__ StorageType *)out);
        if (index != nullptr) indexGlobal_.SetGlobalBuffer((__gm__ int32_t *)index);
        if (ptr != nullptr) ptrGlobal_.SetGlobalBuffer((__gm__ int32_t *)ptr);

        totalLength_ = tiling.totalLength;
        outerSize_ = tiling.outerSize;
        dimSize_ = tiling.dimSize;
        innerSize_ = tiling.innerSize;
        indexLength_ = tiling.indexLength;
        ptrLength_ = tiling.ptrLength;
        mode_ = tiling.mode;
        fastPath_ = tiling.fastPath;
        blockDim_ = tiling.blockDim;
        innerTileWidth_ = tiling.innerTileWidth;
        tileCount_ = tiling.tileCount;
        eps_ = tiling.eps;
        singletonValue_ = 1.0F / (1.0F + eps_);

        pipe_.InitBuffer(workBuf_, WORK_ELEMS * sizeof(float));

        if (fastPath_ == 1U || fastPath_ == 2U || fastPath_ == 3U) {
            pipe_.InitBuffer(rawBuf_, SPARSE_SOFTMAX_RAW_BUFFER_BYTES);
            pipe_.InitBuffer(indexBuf_, SPARSE_SOFTMAX_INDEX_BUFFER_BYTES);
            eventMTE2ToS_ = static_cast<int32_t>(
                pipe_.FetchEventID(AscendC::HardEvent::MTE2_S));
        }
        if (fastPath_ == 2U || fastPath_ == 3U) {
            pipe_.InitBuffer(outQueue_, 1, SPARSE_SOFTMAX_AXIS_OUT_QUEUE_BYTES);
        }
        if (fastPath_ == 1U || fastPath_ == 2U) {
            eventSToMTE3_ = static_cast<int32_t>(
                pipe_.FetchEventID(AscendC::HardEvent::S_MTE3));
            eventMTE3ToS_ = static_cast<int32_t>(
                pipe_.FetchEventID(AscendC::HardEvent::MTE3_S));
        }
        if (fastPath_ == 2U) {
            eventSToMTE2_ = static_cast<int32_t>(
                pipe_.FetchEventID(AscendC::HardEvent::S_MTE2));
        }
    }

    __aicore__ inline void Process() {
        if (totalLength_ == 0 || dimSize_ == 0 || outerSize_ == 0) return;
        if (mode_ == 0U) ProcessIndex();
        else ProcessPtr();
    }

    __aicore__ inline void ProcessIndex() {
        if (fastPath_ == 3U && innerSize_ == 1U) {
            if (AxisCacheEligible()) {
                ProcessIndexAxisCached();
            } else if (AscendC::GetBlockIdx() == 0) {
                ProcessIndexScalar();
                FlushScalarOutput();
            }
            return;
        }
        if (fastPath_ == 1U) {
            const bool broadcast = indexLength_ != totalLength_;
            if (broadcast && dimSize_ <= INDEX_ELEMS) {
                ProcessTiledDma(false);
            } else if (AscendC::GetBlockIdx() == 0) {
                ProcessIndexScalar();
                FlushScalarOutput();
            }
            return;
        }
        if (AscendC::GetBlockIdx() == 0) {
            ProcessIndexScalar();
            FlushScalarOutput();
        }
    }

    __aicore__ inline void ProcessPtr() {
        if (fastPath_ == 2U && innerSize_ == 1U) {
            if (ptrLength_ <= INDEX_ELEMS) {
                ProcessPtrAxisDma();
            } else if (AscendC::GetBlockIdx() == 0) {
                ProcessPtrScalar();
                FlushScalarOutput();
            }
            return;
        }
        if (fastPath_ == 1U) {
            if (ptrLength_ <= INDEX_ELEMS) {
                ProcessTiledDma(true);
            } else if (AscendC::GetBlockIdx() == 0) {
                ProcessPtrScalar();
                FlushScalarOutput();
            }
            return;
        }
        if (AscendC::GetBlockIdx() == 0) {
            ProcessPtrScalar();
            FlushScalarOutput();
        }
    }

private:
    __aicore__ inline uint64_t Align32(uint64_t bytes) const {
        return (bytes + 31ULL) & ~31ULL;
    }
    __aicore__ inline uint64_t FlatOffset(
        uint64_t outer, uint64_t dim, uint64_t inner) const {
        return (outer * dimSize_ + dim) * innerSize_ + inner;
    }
    __aicore__ inline uint64_t OuterBase(uint64_t outer) const {
        return outer * dimSize_ * innerSize_;
    }
    __aicore__ inline float ReadSrc(uint64_t offset) const {
        return StorageTraits<DT_MODE>::ToFloatValue(srcGlobal_.GetValue(offset));
    }
    __aicore__ inline void WriteOut(uint64_t offset, float value) {
        outGlobal_.SetValue(offset, StorageTraits<DT_MODE>::FromFloatValue(value));
    }
    __aicore__ inline int32_t ReadIndex(
        uint64_t outer, uint64_t dim, uint64_t inner) const {
        if (indexLength_ == totalLength_) {
            return indexGlobal_.GetValue(FlatOffset(outer, dim, inner));
        }
        return indexGlobal_.GetValue(dim);
    }
    __aicore__ inline float ReadRaw(
        AscendC::LocalTensor<StorageType> raw, uint64_t offset) const {
        return StorageTraits<DT_MODE>::ToFloatValue(
            raw.GetValue(static_cast<uint32_t>(offset)));
    }
    __aicore__ inline void WriteRaw(
        AscendC::LocalTensor<StorageType> raw, uint64_t offset, float value) {
        raw.SetValue(static_cast<uint32_t>(offset),
                     StorageTraits<DT_MODE>::FromFloatValue(value));
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

    __aicore__ inline float SumWorkRange(
        AscendC::LocalTensor<float> work,
        uint32_t offset, uint32_t count) const {
        float s0 = 0.0F, s1 = 0.0F, s2 = 0.0F, s3 = 0.0F;
        uint32_t i = 0;
        for (; i + 3U < count; i += 4U) {
            s0 += work.GetValue(offset + i);
            s1 += work.GetValue(offset + i + 1U);
            s2 += work.GetValue(offset + i + 2U);
            s3 += work.GetValue(offset + i + 3U);
        }
        float sum = (s0 + s1) + (s2 + s3);
        for (; i < count; ++i) sum += work.GetValue(offset + i);
        return sum;
    }
    __aicore__ inline float SumWork(
        AscendC::LocalTensor<float> work, uint32_t count) const {
        return SumWorkRange(work, 0, count);
    }
    __aicore__ inline void ExpWorkOnly(uint32_t count) {
        AscendC::LocalTensor<float> work = workBuf_.Get<float>();
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Exp(work, work, static_cast<int32_t>(count));
        AscendC::PipeBarrier<PIPE_V>();
    }
    __aicore__ inline float ExpAndSumWork(uint32_t count) {
        AscendC::LocalTensor<float> work = workBuf_.Get<float>();
        ExpWorkOnly(count);
        return SumWork(work, count);
    }

    __aicore__ inline uint32_t QueueElems() const {
        return SPARSE_SOFTMAX_AXIS_OUT_QUEUE_BYTES / sizeof(StorageType);
    }

    __aicore__ inline void WriteWorkQueue(
        uint64_t gmOffset, uint32_t workOffset,
        uint32_t count, float inverse) {
        AscendC::LocalTensor<float> work = workBuf_.Get<float>();
        AscendC::LocalTensor<StorageType> outLocal =
            outQueue_.AllocTensor<StorageType>();
        for (uint32_t i = 0; i < count; ++i) {
            outLocal.SetValue(i, StorageTraits<DT_MODE>::FromFloatValue(
                work.GetValue(workOffset + i) * inverse));
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        outQueue_.EnQue(outLocal);
        outLocal = outQueue_.DeQue<StorageType>();
        AscendC::DataCopyExtParams params;
        params.blockCount = 1;
        params.blockLen = static_cast<uint32_t>(count * sizeof(StorageType));
        params.srcStride = 0;
        params.dstStride = 0;
        params.rsv = 0;
        AscendC::DataCopyPad(outGlobal_[gmOffset], outLocal, params);
        outQueue_.FreeTensor(outLocal);
    }

    __aicore__ inline void WriteSingletonQueue(uint64_t gmOffset) {
        AscendC::LocalTensor<StorageType> outLocal =
            outQueue_.AllocTensor<StorageType>();
        outLocal.SetValue(0, StorageTraits<DT_MODE>::FromFloatValue(singletonValue_));
        AscendC::PipeBarrier<PIPE_ALL>();
        outQueue_.EnQue(outLocal);
        outLocal = outQueue_.DeQue<StorageType>();
        AscendC::DataCopyExtParams params;
        params.blockCount = 1;
        params.blockLen = static_cast<uint32_t>(sizeof(StorageType));
        params.srcStride = 0;
        params.dstStride = 0;
        params.rsv = 0;
        AscendC::DataCopyPad(outGlobal_[gmOffset], outLocal, params);
        outQueue_.FreeTensor(outLocal);
    }

    // ------------------------------------------------------------------
    // index + inner==1: cache src/index once per outer on every participating
    // AIV, preserve group-owner parallelism, and eliminate repeated scalar GM
    // reads from the hot group scans.
    // ------------------------------------------------------------------
    __aicore__ inline bool AxisCacheEligible() const {
        const uint64_t rawCapacity = SPARSE_SOFTMAX_RAW_BUFFER_BYTES / sizeof(StorageType);
        return dimSize_ <= rawCapacity && dimSize_ <= INDEX_ELEMS &&
               dimSize_ <= WORK_ELEMS && dimSize_ <= QueueElems();
    }

    __aicore__ inline bool LoadAxisIndexCache(
        AscendC::LocalTensor<int32_t> cachedIndex, uint64_t outer) {
        const uint64_t base = indexLength_ == totalLength_ ? outer * dimSize_ : 0U;
        CopyGmToLocal<int32_t>(
            cachedIndex, indexGlobal_[base], 1,
            static_cast<uint32_t>(dimSize_ * sizeof(int32_t)), 0, 0);
        SyncMTE2ToS();
        return true;
    }

    __aicore__ inline void LoadAxisSrc(
        AscendC::LocalTensor<StorageType> raw, uint64_t outer) {
        CopyGmToLocal<StorageType>(
            raw, srcGlobal_[outer * dimSize_], 1,
            static_cast<uint32_t>(dimSize_ * sizeof(StorageType)), 0, 0);
        SyncMTE2ToS();
    }

    __aicore__ inline bool CachedIndexSorted(
        AscendC::LocalTensor<int32_t> cachedIndex) const {
        if (dimSize_ <= 1U) return true;
        int32_t previous = cachedIndex.GetValue(0);
        for (uint64_t pos = 1; pos < dimSize_; ++pos) {
            const int32_t current = cachedIndex.GetValue(static_cast<uint32_t>(pos));
            if (current < previous) return false;
            previous = current;
        }
        return true;
    }

    __aicore__ inline bool CachedFirstOccurrence(
        AscendC::LocalTensor<int32_t> cachedIndex,
        uint64_t seed, int32_t group) const {
        for (uint64_t pos = 0; pos < seed; ++pos) {
            if (cachedIndex.GetValue(static_cast<uint32_t>(pos)) == group) return false;
        }
        return true;
    }

    __aicore__ inline void ProcessCachedAxisRun(
        AscendC::LocalTensor<StorageType> raw,
        uint64_t outer, uint64_t start, uint64_t end) {
        const uint32_t count = static_cast<uint32_t>(end - start);
        const uint64_t gmBase = outer * dimSize_ + start;
        if (count == 0) return;
        if (count == 1) {
            WriteSingletonQueue(gmBase);
            return;
        }
        float maxValue = NEG_FLOAT_MAX;
        for (uint64_t pos = start; pos < end; ++pos) {
            const float value = ReadRaw(raw, pos);
            if (value > maxValue) maxValue = value;
        }
        AscendC::LocalTensor<float> work = workBuf_.Get<float>();
        uint32_t packed = 0;
        for (uint64_t pos = start; pos < end; ++pos) {
            work.SetValue(packed++, ReadRaw(raw, pos) - maxValue);
        }
        const float sum = ExpAndSumWork(count);
        WriteWorkQueue(gmBase, 0, count, 1.0F / (sum + eps_));
    }

    __aicore__ inline void ProcessCachedAxisGroup(
        AscendC::LocalTensor<StorageType> raw,
        AscendC::LocalTensor<int32_t> cachedIndex,
        uint64_t outer, int32_t group) {
        float maxValue = NEG_FLOAT_MAX;
        uint32_t count = 0;
        for (uint64_t pos = 0; pos < dimSize_; ++pos) {
            if (cachedIndex.GetValue(static_cast<uint32_t>(pos)) != group) continue;
            const float value = ReadRaw(raw, pos);
            if (value > maxValue) maxValue = value;
            ++count;
        }
        if (count == 0) return;
        if (count == 1) {
            for (uint64_t pos = 0; pos < dimSize_; ++pos) {
                if (cachedIndex.GetValue(static_cast<uint32_t>(pos)) == group) {
                    WriteSingletonQueue(outer * dimSize_ + pos);
                    return;
                }
            }
        }
        AscendC::LocalTensor<float> work = workBuf_.Get<float>();
        uint32_t packed = 0;
        for (uint64_t pos = 0; pos < dimSize_; ++pos) {
            if (cachedIndex.GetValue(static_cast<uint32_t>(pos)) == group) {
                work.SetValue(packed++, ReadRaw(raw, pos) - maxValue);
            }
        }
        const float sum = ExpAndSumWork(count);
        const float inverse = 1.0F / (sum + eps_);
        uint64_t pos = 0;
        packed = 0;
        while (pos < dimSize_) {
            if (cachedIndex.GetValue(static_cast<uint32_t>(pos)) != group) {
                ++pos;
                continue;
            }
            const uint64_t runBegin = pos;
            const uint32_t packedBegin = packed;
            do {
                ++pos;
                ++packed;
            } while (pos < dimSize_ &&
                     cachedIndex.GetValue(static_cast<uint32_t>(pos)) == group);
            const uint32_t runLength = static_cast<uint32_t>(pos - runBegin);
            WriteWorkQueue(outer * dimSize_ + runBegin,
                           packedBegin, runLength, inverse);
        }
    }

    __aicore__ inline void ProcessIndexAxisCached() {
        AscendC::LocalTensor<StorageType> raw = rawBuf_.Get<StorageType>();
        AscendC::LocalTensor<int32_t> cachedIndex = indexBuf_.Get<int32_t>();
        const uint64_t blockIdx = static_cast<uint64_t>(AscendC::GetBlockIdx());
        uint64_t blockNum = static_cast<uint64_t>(AscendC::GetBlockNum());
        if (blockNum == 0) blockNum = 1;
        const bool broadcast = indexLength_ != totalLength_;
        if (broadcast) LoadAxisIndexCache(cachedIndex, 0);

        for (uint64_t outer = 0; outer < outerSize_; ++outer) {
            if (!broadcast) LoadAxisIndexCache(cachedIndex, outer);
            LoadAxisSrc(raw, outer);
            const bool sorted = CachedIndexSorted(cachedIndex);
            if (sorted) {
                uint64_t runStart = 0;
                uint64_t runOrdinal = 0;
                while (runStart < dimSize_) {
                    const int32_t group = cachedIndex.GetValue(static_cast<uint32_t>(runStart));
                    uint64_t runEnd = runStart + 1U;
                    while (runEnd < dimSize_ &&
                           cachedIndex.GetValue(static_cast<uint32_t>(runEnd)) == group) {
                        ++runEnd;
                    }
                    if ((runOrdinal % blockNum) == blockIdx) {
                        ProcessCachedAxisRun(raw, outer, runStart, runEnd);
                    }
                    ++runOrdinal;
                    runStart = runEnd;
                }
            } else {
                for (uint64_t seed = blockIdx; seed < dimSize_; seed += blockNum) {
                    const int32_t group = cachedIndex.GetValue(static_cast<uint32_t>(seed));
                    if (CachedFirstOccurrence(cachedIndex, seed, group)) {
                        ProcessCachedAxisGroup(raw, cachedIndex, outer, group);
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // ptr + inner==1. Small/medium groups keep one Exp result and write via
    // VECOUT queue, so the common path avoids raw-buffer MTE3 event pairs.
    // ------------------------------------------------------------------
    __aicore__ inline int64_t ReadPtrValue(
        AscendC::LocalTensor<int32_t> cachedPtr, uint64_t idx) const {
        return static_cast<int64_t>(cachedPtr.GetValue(static_cast<uint32_t>(idx)));
    }
    __aicore__ inline bool ClampPtrRange(int64_t &start, int64_t &end) const {
        if (start < 0) start = 0;
        if (end < start) return false;
        if (start > static_cast<int64_t>(dimSize_)) return false;
        if (end > static_cast<int64_t>(dimSize_)) end = static_cast<int64_t>(dimSize_);
        return end > start;
    }
    __aicore__ inline void LoadPtrCache(AscendC::LocalTensor<int32_t> cachedPtr) {
        CopyGmToLocal<int32_t>(
            cachedPtr, ptrGlobal_, 1,
            static_cast<uint32_t>(ptrLength_ * sizeof(int32_t)), 0, 0);
        SyncMTE2ToS();
    }

    __aicore__ inline void ProcessPtrAxisGroup(
        AscendC::LocalTensor<StorageType> raw,
        uint64_t outer, uint64_t start, uint64_t end) {
        const uint64_t count64 = end - start;
        const uint64_t gmBase = outer * dimSize_ + start;
        const uint64_t rawCapacity = SPARSE_SOFTMAX_RAW_BUFFER_BYTES / sizeof(StorageType);
        const uint64_t queueCapacity = QueueElems();
        if (count64 == 1U) {
            WriteSingletonQueue(gmBase);
            return;
        }
        if (count64 <= rawCapacity && count64 <= WORK_ELEMS && count64 <= queueCapacity) {
            const uint32_t count = static_cast<uint32_t>(count64);
            CopyGmToLocal<StorageType>(
                raw, srcGlobal_[gmBase], 1,
                static_cast<uint32_t>(count64 * sizeof(StorageType)), 0, 0);
            SyncMTE2ToS();
            float maxValue = NEG_FLOAT_MAX;
            for (uint32_t i = 0; i < count; ++i) {
                const float value = ReadRaw(raw, i);
                if (value > maxValue) maxValue = value;
            }
            AscendC::LocalTensor<float> work = workBuf_.Get<float>();
            for (uint32_t i = 0; i < count; ++i) {
                work.SetValue(i, ReadRaw(raw, i) - maxValue);
            }
            const float sum = ExpAndSumWork(count);
            WriteWorkQueue(gmBase, 0, count, 1.0F / (sum + eps_));
            return;
        }

        // General large-group path: three stable passes through raw chunks.
        float maxValue = NEG_FLOAT_MAX;
        uint64_t chunkStart = 0;
        while (chunkStart < count64) {
            uint64_t chunkCount = count64 - chunkStart;
            if (chunkCount > rawCapacity) chunkCount = rawCapacity;
            CopyGmToLocal<StorageType>(
                raw, srcGlobal_[gmBase + chunkStart], 1,
                static_cast<uint32_t>(chunkCount * sizeof(StorageType)), 0, 0);
            SyncMTE2ToS();
            for (uint64_t i = 0; i < chunkCount; ++i) {
                const float value = ReadRaw(raw, i);
                if (value > maxValue) maxValue = value;
            }
            SyncSToMTE2();
            chunkStart += chunkCount;
        }

        AscendC::LocalTensor<float> work = workBuf_.Get<float>();
        float groupSum = 0.0F;
        chunkStart = 0;
        while (chunkStart < count64) {
            uint64_t chunkCount = count64 - chunkStart;
            if (chunkCount > rawCapacity) chunkCount = rawCapacity;
            CopyGmToLocal<StorageType>(
                raw, srcGlobal_[gmBase + chunkStart], 1,
                static_cast<uint32_t>(chunkCount * sizeof(StorageType)), 0, 0);
            SyncMTE2ToS();
            uint64_t local = 0;
            while (local < chunkCount) {
                uint64_t remain = chunkCount - local;
                uint32_t n = static_cast<uint32_t>(remain > WORK_ELEMS ? WORK_ELEMS : remain);
                for (uint32_t i = 0; i < n; ++i) {
                    work.SetValue(i, ReadRaw(raw, local + i) - maxValue);
                }
                groupSum += ExpAndSumWork(n);
                local += n;
            }
            SyncSToMTE2();
            chunkStart += chunkCount;
        }

        const float inverse = 1.0F / (groupSum + eps_);
        chunkStart = 0;
        while (chunkStart < count64) {
            uint64_t chunkCount = count64 - chunkStart;
            if (chunkCount > rawCapacity) chunkCount = rawCapacity;
            CopyGmToLocal<StorageType>(
                raw, srcGlobal_[gmBase + chunkStart], 1,
                static_cast<uint32_t>(chunkCount * sizeof(StorageType)), 0, 0);
            SyncMTE2ToS();
            uint64_t local = 0;
            while (local < chunkCount) {
                uint64_t remain = chunkCount - local;
                uint32_t n = static_cast<uint32_t>(remain > WORK_ELEMS ? WORK_ELEMS : remain);
                for (uint32_t i = 0; i < n; ++i) {
                    work.SetValue(i, ReadRaw(raw, local + i) - maxValue);
                }
                ExpWorkOnly(n);
                for (uint32_t i = 0; i < n; ++i) {
                    WriteRaw(raw, local + i, work.GetValue(i) * inverse);
                }
                local += n;
            }
            SyncSToMTE3();
            CopyLocalToGm<StorageType>(
                outGlobal_[gmBase + chunkStart], raw, 1,
                static_cast<uint32_t>(chunkCount * sizeof(StorageType)), 0, 0);
            SyncMTE3ToS();
            chunkStart += chunkCount;
        }
    }

    __aicore__ inline void ProcessPtrAxisDma() {
        AscendC::LocalTensor<int32_t> cachedPtr = indexBuf_.Get<int32_t>();
        AscendC::LocalTensor<StorageType> raw = rawBuf_.Get<StorageType>();
        LoadPtrCache(cachedPtr);
        const uint64_t groupCount = ptrLength_ - 1U;
        const uint64_t taskCount = outerSize_ * groupCount;
        const uint64_t blockIdx = static_cast<uint64_t>(AscendC::GetBlockIdx());
        uint64_t blockNum = static_cast<uint64_t>(AscendC::GetBlockNum());
        if (blockNum == 0) blockNum = 1;
        for (uint64_t task = blockIdx; task < taskCount; task += blockNum) {
            const uint64_t outer = task / groupCount;
            const uint64_t group = task - outer * groupCount;
            int64_t start = ReadPtrValue(cachedPtr, group);
            int64_t end = ReadPtrValue(cachedPtr, group + 1U);
            if (!ClampPtrRange(start, end)) continue;
            ProcessPtrAxisGroup(raw, outer,
                static_cast<uint64_t>(start), static_cast<uint64_t>(end));
        }
    }

    // ------------------------------------------------------------------
    // inner>1: one DMA tile, then fuse a complete contiguous sparse partition
    // across up to 8 lanes into one vector Exp. This removes one Exp/barrier
    // pair per group while keeping the proven 8-lane/small-UB resource model.
    // ------------------------------------------------------------------
    __aicore__ inline bool LoadBroadcastIndex(
        AscendC::LocalTensor<int32_t> cachedIndex) {
        if (dimSize_ > INDEX_ELEMS) return false;
        CopyGmToLocal<int32_t>(
            cachedIndex, indexGlobal_, 1,
            static_cast<uint32_t>(dimSize_ * sizeof(int32_t)), 0, 0);
        SyncMTE2ToS();
        return true;
    }

    __aicore__ inline bool PtrIsCompletePartition(
        AscendC::LocalTensor<int32_t> cachedPtr) const {
        if (ptrLength_ < 2U) return false;
        if (cachedPtr.GetValue(0) != 0) return false;
        int32_t previous = 0;
        for (uint64_t i = 1; i < ptrLength_; ++i) {
            const int32_t current = cachedPtr.GetValue(static_cast<uint32_t>(i));
            if (current < previous || current < 0 ||
                current > static_cast<int32_t>(dimSize_)) return false;
            previous = current;
        }
        return static_cast<uint64_t>(previous) == dimSize_;
    }

    __aicore__ inline uint32_t PartitionLaneBatch(uint64_t remaining) const {
        uint32_t byWork = dimSize_ == 0 ? 1U : static_cast<uint32_t>(WORK_ELEMS / dimSize_);
        if (byWork == 0) byWork = 1;
        uint32_t batch = static_cast<uint32_t>(
            remaining > LANE_BATCH_MAX ? LANE_BATCH_MAX : remaining);
        if (batch > byWork) batch = byWork;
        return batch == 0 ? 1U : batch;
    }

    __aicore__ inline void FillRunIntoPartitionWork(
        AscendC::LocalTensor<StorageType> raw,
        uint64_t rowStride, uint64_t laneStart, uint32_t laneCount,
        uint64_t start, uint64_t end) {
        float maxValues[LANE_BATCH_MAX];
        for (uint32_t lane = 0; lane < laneCount; ++lane) maxValues[lane] = NEG_FLOAT_MAX;
        for (uint64_t pos = start; pos < end; ++pos) {
            const uint64_t rowBase = pos * rowStride + laneStart;
            for (uint32_t lane = 0; lane < laneCount; ++lane) {
                const float value = ReadRaw(raw, rowBase + lane);
                if (value > maxValues[lane]) maxValues[lane] = value;
            }
        }
        AscendC::LocalTensor<float> work = workBuf_.Get<float>();
        for (uint64_t pos = start; pos < end; ++pos) {
            const uint64_t rowBase = pos * rowStride + laneStart;
            for (uint32_t lane = 0; lane < laneCount; ++lane) {
                work.SetValue(lane * static_cast<uint32_t>(dimSize_) + static_cast<uint32_t>(pos),
                              ReadRaw(raw, rowBase + lane) - maxValues[lane]);
            }
        }
    }

    __aicore__ inline void NormalizeRunFromPartitionWork(
        AscendC::LocalTensor<StorageType> raw,
        uint64_t rowStride, uint64_t laneStart, uint32_t laneCount,
        uint64_t start, uint64_t end) {
        const uint32_t count = static_cast<uint32_t>(end - start);
        AscendC::LocalTensor<float> work = workBuf_.Get<float>();
        for (uint32_t lane = 0; lane < laneCount; ++lane) {
            const uint32_t base = lane * static_cast<uint32_t>(dimSize_) + static_cast<uint32_t>(start);
            const float sum = SumWorkRange(work, base, count);
            const float inverse = 1.0F / (sum + eps_);
            for (uint64_t pos = start; pos < end; ++pos) {
                WriteRaw(raw, pos * rowStride + laneStart + lane,
                         work.GetValue(lane * static_cast<uint32_t>(dimSize_) +
                                       static_cast<uint32_t>(pos)) * inverse);
            }
        }
    }

    __aicore__ inline void ProcessSortedBroadcastPartition(
        AscendC::LocalTensor<StorageType> raw,
        AscendC::LocalTensor<int32_t> cachedIndex,
        uint64_t rowStride, uint64_t tileWidth) {
        uint64_t laneStart = 0;
        while (laneStart < tileWidth) {
            const uint32_t laneCount = PartitionLaneBatch(tileWidth - laneStart);
            uint64_t runStart = 0;
            while (runStart < dimSize_) {
                const int32_t group = cachedIndex.GetValue(static_cast<uint32_t>(runStart));
                uint64_t runEnd = runStart + 1U;
                while (runEnd < dimSize_ &&
                       cachedIndex.GetValue(static_cast<uint32_t>(runEnd)) == group) ++runEnd;
                FillRunIntoPartitionWork(raw, rowStride, laneStart, laneCount, runStart, runEnd);
                runStart = runEnd;
            }
            ExpWorkOnly(laneCount * static_cast<uint32_t>(dimSize_));
            runStart = 0;
            while (runStart < dimSize_) {
                const int32_t group = cachedIndex.GetValue(static_cast<uint32_t>(runStart));
                uint64_t runEnd = runStart + 1U;
                while (runEnd < dimSize_ &&
                       cachedIndex.GetValue(static_cast<uint32_t>(runEnd)) == group) ++runEnd;
                NormalizeRunFromPartitionWork(raw, rowStride, laneStart, laneCount, runStart, runEnd);
                runStart = runEnd;
            }
            laneStart += laneCount;
        }
    }

    __aicore__ inline void ProcessPtrCompletePartition(
        AscendC::LocalTensor<StorageType> raw,
        AscendC::LocalTensor<int32_t> cachedPtr,
        uint64_t rowStride, uint64_t tileWidth) {
        uint64_t laneStart = 0;
        while (laneStart < tileWidth) {
            const uint32_t laneCount = PartitionLaneBatch(tileWidth - laneStart);
            for (uint64_t group = 0; group + 1U < ptrLength_; ++group) {
                const uint64_t start = static_cast<uint64_t>(
                    cachedPtr.GetValue(static_cast<uint32_t>(group)));
                const uint64_t end = static_cast<uint64_t>(
                    cachedPtr.GetValue(static_cast<uint32_t>(group + 1U)));
                if (end > start) {
                    FillRunIntoPartitionWork(raw, rowStride, laneStart, laneCount, start, end);
                }
            }
            ExpWorkOnly(laneCount * static_cast<uint32_t>(dimSize_));
            for (uint64_t group = 0; group + 1U < ptrLength_; ++group) {
                const uint64_t start = static_cast<uint64_t>(
                    cachedPtr.GetValue(static_cast<uint32_t>(group)));
                const uint64_t end = static_cast<uint64_t>(
                    cachedPtr.GetValue(static_cast<uint32_t>(group + 1U)));
                if (end > start) {
                    NormalizeRunFromPartitionWork(raw, rowStride, laneStart, laneCount, start, end);
                }
            }
            laneStart += laneCount;
        }
    }

    // Fallback batch for unsorted broadcast index.
    __aicore__ inline bool FirstBroadcastOccurrence(
        AscendC::LocalTensor<int32_t> cachedIndex,
        uint64_t seed, int32_t group) const {
        for (uint64_t pos = 0; pos < seed; ++pos) {
            if (cachedIndex.GetValue(static_cast<uint32_t>(pos)) == group) return false;
        }
        return true;
    }
    __aicore__ inline uint32_t CountBroadcastGroup(
        AscendC::LocalTensor<int32_t> cachedIndex,
        int32_t group, uint64_t &singletonPos) const {
        uint32_t count = 0;
        singletonPos = 0;
        for (uint64_t pos = 0; pos < dimSize_; ++pos) {
            if (cachedIndex.GetValue(static_cast<uint32_t>(pos)) == group) {
                singletonPos = pos;
                ++count;
            }
        }
        return count;
    }
    __aicore__ inline uint32_t ChooseLaneBatch(
        uint64_t remaining, uint32_t groupCount) const {
        uint32_t byWork = groupCount == 0 ? 1U : WORK_ELEMS / groupCount;
        if (byWork == 0) byWork = 1;
        uint32_t batch = static_cast<uint32_t>(
            remaining > LANE_BATCH_MAX ? LANE_BATCH_MAX : remaining);
        if (batch > byWork) batch = byWork;
        return batch == 0 ? 1U : batch;
    }

    __aicore__ inline void SoftmaxBroadcastGroupBatch(
        AscendC::LocalTensor<StorageType> raw,
        AscendC::LocalTensor<int32_t> cachedIndex,
        uint64_t rowStride, uint64_t laneStart, uint32_t laneCount,
        int32_t group, uint32_t groupCount, uint64_t singletonPos) {
        if (groupCount == 0) return;
        if (groupCount == 1) {
            for (uint32_t lane = 0; lane < laneCount; ++lane) {
                WriteRaw(raw, singletonPos * rowStride + laneStart + lane, singletonValue_);
            }
            return;
        }
        float maxValues[LANE_BATCH_MAX];
        float inverseValues[LANE_BATCH_MAX];
        for (uint32_t lane = 0; lane < laneCount; ++lane) maxValues[lane] = NEG_FLOAT_MAX;
        for (uint64_t pos = 0; pos < dimSize_; ++pos) {
            if (cachedIndex.GetValue(static_cast<uint32_t>(pos)) != group) continue;
            const uint64_t rowBase = pos * rowStride + laneStart;
            for (uint32_t lane = 0; lane < laneCount; ++lane) {
                const float value = ReadRaw(raw, rowBase + lane);
                if (value > maxValues[lane]) maxValues[lane] = value;
            }
        }
        AscendC::LocalTensor<float> work = workBuf_.Get<float>();
        uint32_t packed = 0;
        for (uint64_t pos = 0; pos < dimSize_; ++pos) {
            if (cachedIndex.GetValue(static_cast<uint32_t>(pos)) != group) continue;
            const uint64_t rowBase = pos * rowStride + laneStart;
            for (uint32_t lane = 0; lane < laneCount; ++lane) {
                work.SetValue(lane * groupCount + packed,
                              ReadRaw(raw, rowBase + lane) - maxValues[lane]);
            }
            ++packed;
        }
        ExpWorkOnly(laneCount * groupCount);
        for (uint32_t lane = 0; lane < laneCount; ++lane) {
            inverseValues[lane] = 1.0F /
                (SumWorkRange(work, lane * groupCount, groupCount) + eps_);
        }
        packed = 0;
        for (uint64_t pos = 0; pos < dimSize_; ++pos) {
            if (cachedIndex.GetValue(static_cast<uint32_t>(pos)) != group) continue;
            const uint64_t rowBase = pos * rowStride + laneStart;
            for (uint32_t lane = 0; lane < laneCount; ++lane) {
                WriteRaw(raw, rowBase + lane,
                         work.GetValue(lane * groupCount + packed) * inverseValues[lane]);
            }
            ++packed;
        }
    }

    __aicore__ inline void SoftmaxContiguousBatch(
        AscendC::LocalTensor<StorageType> raw,
        uint64_t rowStride, uint64_t laneStart, uint32_t laneCount,
        uint64_t start, uint64_t end) {
        const uint32_t groupCount = static_cast<uint32_t>(end - start);
        if (groupCount == 0) return;
        if (groupCount == 1) {
            for (uint32_t lane = 0; lane < laneCount; ++lane) {
                WriteRaw(raw, start * rowStride + laneStart + lane, singletonValue_);
            }
            return;
        }
        float maxValues[LANE_BATCH_MAX];
        float inverseValues[LANE_BATCH_MAX];
        for (uint32_t lane = 0; lane < laneCount; ++lane) maxValues[lane] = NEG_FLOAT_MAX;
        for (uint64_t pos = start; pos < end; ++pos) {
            const uint64_t rowBase = pos * rowStride + laneStart;
            for (uint32_t lane = 0; lane < laneCount; ++lane) {
                const float value = ReadRaw(raw, rowBase + lane);
                if (value > maxValues[lane]) maxValues[lane] = value;
            }
        }
        AscendC::LocalTensor<float> work = workBuf_.Get<float>();
        for (uint32_t lane = 0; lane < laneCount; ++lane) {
            for (uint32_t i = 0; i < groupCount; ++i) {
                work.SetValue(lane * groupCount + i,
                    ReadRaw(raw, (start + i) * rowStride + laneStart + lane) - maxValues[lane]);
            }
        }
        ExpWorkOnly(laneCount * groupCount);
        for (uint32_t lane = 0; lane < laneCount; ++lane) {
            inverseValues[lane] = 1.0F /
                (SumWorkRange(work, lane * groupCount, groupCount) + eps_);
        }
        for (uint32_t i = 0; i < groupCount; ++i) {
            const uint64_t rowBase = (start + i) * rowStride + laneStart;
            for (uint32_t lane = 0; lane < laneCount; ++lane) {
                WriteRaw(raw, rowBase + lane,
                         work.GetValue(lane * groupCount + i) * inverseValues[lane]);
            }
        }
    }

    __aicore__ inline void ProcessBroadcastFallback(
        AscendC::LocalTensor<StorageType> raw,
        AscendC::LocalTensor<int32_t> cachedIndex,
        uint64_t rowStride, uint64_t tileWidth) {
        for (uint64_t seed = 0; seed < dimSize_; ++seed) {
            const int32_t group = cachedIndex.GetValue(static_cast<uint32_t>(seed));
            if (!FirstBroadcastOccurrence(cachedIndex, seed, group)) continue;
            uint64_t singletonPos = 0;
            const uint32_t groupCount = CountBroadcastGroup(cachedIndex, group, singletonPos);
            uint64_t laneStart = 0;
            while (laneStart < tileWidth) {
                const uint32_t laneCount = ChooseLaneBatch(tileWidth - laneStart, groupCount);
                SoftmaxBroadcastGroupBatch(raw, cachedIndex, rowStride, laneStart,
                                           laneCount, group, groupCount, singletonPos);
                laneStart += laneCount;
            }
        }
    }

    __aicore__ inline void ProcessPtrFallback(
        AscendC::LocalTensor<StorageType> raw,
        AscendC::LocalTensor<int32_t> cachedPtr,
        uint64_t rowStride, uint64_t tileWidth) {
        for (uint64_t group = 0; group + 1U < ptrLength_; ++group) {
            int64_t start = ReadPtrValue(cachedPtr, group);
            int64_t end = ReadPtrValue(cachedPtr, group + 1U);
            if (!ClampPtrRange(start, end)) continue;
            const uint32_t groupCount = static_cast<uint32_t>(end - start);
            uint64_t laneStart = 0;
            while (laneStart < tileWidth) {
                const uint32_t laneCount = ChooseLaneBatch(tileWidth - laneStart, groupCount);
                SoftmaxContiguousBatch(raw, rowStride, laneStart, laneCount,
                                       static_cast<uint64_t>(start), static_cast<uint64_t>(end));
                laneStart += laneCount;
            }
        }
    }

    __aicore__ inline void ProcessOneTile(
        AscendC::LocalTensor<StorageType> raw,
        AscendC::LocalTensor<int32_t> cachedMeta,
        bool ptrMode, bool sortedBroadcast, bool completePtr,
        uint64_t outer, uint64_t innerStart, uint64_t tileWidth) {
        const uint64_t rowBytes = tileWidth * sizeof(StorageType);
        const uint64_t rowStrideBytes = Align32(rowBytes);
        const uint64_t rowStride = rowStrideBytes / sizeof(StorageType);
        const uint64_t gmBase = OuterBase(outer) + innerStart;
        CopyGmToLocal<StorageType>(
            raw, srcGlobal_[gmBase], static_cast<uint16_t>(dimSize_),
            static_cast<uint32_t>(rowBytes),
            static_cast<uint32_t>((innerSize_ - tileWidth) * sizeof(StorageType)), 0);
        SyncMTE2ToS();

        const bool partitionFits = dimSize_ <= WORK_ELEMS;
        if (ptrMode) {
            if (completePtr && partitionFits) {
                ProcessPtrCompletePartition(raw, cachedMeta, rowStride, tileWidth);
            } else {
                ProcessPtrFallback(raw, cachedMeta, rowStride, tileWidth);
            }
        } else {
            if (sortedBroadcast && partitionFits) {
                ProcessSortedBroadcastPartition(raw, cachedMeta, rowStride, tileWidth);
            } else {
                ProcessBroadcastFallback(raw, cachedMeta, rowStride, tileWidth);
            }
        }

        SyncSToMTE3();
        CopyLocalToGm<StorageType>(
            outGlobal_[gmBase], raw, static_cast<uint16_t>(dimSize_),
            static_cast<uint32_t>(rowBytes), 0,
            static_cast<uint32_t>((innerSize_ - tileWidth) * sizeof(StorageType)));
        SyncMTE3ToS();
    }

    __aicore__ inline void ProcessTiledDma(bool ptrMode) {
        AscendC::LocalTensor<StorageType> raw = rawBuf_.Get<StorageType>();
        AscendC::LocalTensor<int32_t> cachedMeta = indexBuf_.Get<int32_t>();
        bool sortedBroadcast = false;
        bool completePtr = false;
        if (ptrMode) {
            LoadPtrCache(cachedMeta);
            completePtr = PtrIsCompletePartition(cachedMeta);
        } else {
            LoadBroadcastIndex(cachedMeta);
            sortedBroadcast = CachedIndexSorted(cachedMeta);
        }

        const uint64_t taskCount = outerSize_ * tileCount_;
        const uint64_t blockIdx = static_cast<uint64_t>(AscendC::GetBlockIdx());
        uint64_t blockNum = static_cast<uint64_t>(AscendC::GetBlockNum());
        if (blockNum == 0) blockNum = 1;
        for (uint64_t task = blockIdx; task < taskCount; task += blockNum) {
            const uint64_t outer = task / tileCount_;
            const uint64_t tile = task - outer * tileCount_;
            const uint64_t innerStart = tile * innerTileWidth_;
            if (innerStart >= innerSize_) continue;
            uint64_t tileWidth = innerSize_ - innerStart;
            if (tileWidth > innerTileWidth_) tileWidth = innerTileWidth_;
            ProcessOneTile(raw, cachedMeta, ptrMode, sortedBroadcast, completePtr,
                           outer, innerStart, tileWidth);
        }
    }

    // ------------------------------------------------------------------
    // Fully generic scalar fallback for layouts outside the optimized cached
    // paths. Correctness is preserved for arbitrary supported rank/dim/index.
    // ------------------------------------------------------------------
    __aicore__ inline bool IsFirstScalarOccurrence(
        uint64_t outer, uint64_t inner, uint64_t seed, int32_t group) const {
        for (uint64_t pos = 0; pos < seed; ++pos) {
            if (ReadIndex(outer, pos, inner) == group) return false;
        }
        return true;
    }

    __aicore__ inline float ComputeScalarIndexGroupSum(
        uint64_t outer, uint64_t inner, int32_t group, float maxValue) {
        AscendC::LocalTensor<float> work = workBuf_.Get<float>();
        float sum = 0.0F;
        uint32_t count = 0;
        for (uint64_t pos = 0; pos < dimSize_; ++pos) {
            if (ReadIndex(outer, pos, inner) != group) continue;
            work.SetValue(count++, ReadSrc(FlatOffset(outer, pos, inner)) - maxValue);
            if (count == WORK_ELEMS) {
                sum += ExpAndSumWork(count);
                count = 0;
            }
        }
        if (count != 0) sum += ExpAndSumWork(count);
        return sum;
    }

    __aicore__ inline void ProcessScalarIndexGroup(
        uint64_t outer, uint64_t inner, int32_t group) {
        uint64_t count = 0, singletonPos = 0;
        float maxValue = NEG_FLOAT_MAX;
        for (uint64_t pos = 0; pos < dimSize_; ++pos) {
            if (ReadIndex(outer, pos, inner) != group) continue;
            const float value = ReadSrc(FlatOffset(outer, pos, inner));
            if (value > maxValue) maxValue = value;
            singletonPos = pos;
            ++count;
        }
        if (count == 0) return;
        if (count == 1) {
            WriteOut(FlatOffset(outer, singletonPos, inner), singletonValue_);
            return;
        }
        const float inverse = 1.0F /
            (ComputeScalarIndexGroupSum(outer, inner, group, maxValue) + eps_);
        AscendC::LocalTensor<float> work = workBuf_.Get<float>();
        uint64_t scan = 0;
        while (scan < dimSize_) {
            const uint64_t chunkStart = scan;
            uint32_t packed = 0;
            while (scan < dimSize_ && packed < WORK_ELEMS) {
                if (ReadIndex(outer, scan, inner) == group) {
                    work.SetValue(packed++,
                        ReadSrc(FlatOffset(outer, scan, inner)) - maxValue);
                }
                ++scan;
            }
            if (packed == 0) continue;
            ExpWorkOnly(packed);
            uint32_t outputPos = 0;
            for (uint64_t pos = chunkStart; pos < scan; ++pos) {
                if (ReadIndex(outer, pos, inner) == group) {
                    WriteOut(FlatOffset(outer, pos, inner),
                             work.GetValue(outputPos++) * inverse);
                }
            }
        }
    }

    __aicore__ inline void ProcessIndexScalar() {
        for (uint64_t outer = 0; outer < outerSize_; ++outer) {
            for (uint64_t inner = 0; inner < innerSize_; ++inner) {
                for (uint64_t seed = 0; seed < dimSize_; ++seed) {
                    const int32_t group = ReadIndex(outer, seed, inner);
                    if (IsFirstScalarOccurrence(outer, inner, seed, group)) {
                        ProcessScalarIndexGroup(outer, inner, group);
                    }
                }
            }
        }
    }

    __aicore__ inline void ProcessScalarContiguous(
        uint64_t outer, uint64_t inner, uint64_t start, uint64_t end) {
        const uint64_t count64 = end - start;
        if (count64 == 0) return;
        if (count64 == 1) {
            WriteOut(FlatOffset(outer, start, inner), singletonValue_);
            return;
        }
        float maxValue = NEG_FLOAT_MAX;
        for (uint64_t pos = start; pos < end; ++pos) {
            const float value = ReadSrc(FlatOffset(outer, pos, inner));
            if (value > maxValue) maxValue = value;
        }
        AscendC::LocalTensor<float> work = workBuf_.Get<float>();
        float sum = 0.0F;
        uint64_t chunkStart = start;
        while (chunkStart < end) {
            uint64_t remain = end - chunkStart;
            uint32_t n = static_cast<uint32_t>(remain > WORK_ELEMS ? WORK_ELEMS : remain);
            for (uint32_t i = 0; i < n; ++i) {
                work.SetValue(i, ReadSrc(FlatOffset(outer, chunkStart + i, inner)) - maxValue);
            }
            sum += ExpAndSumWork(n);
            chunkStart += n;
        }
        const float inverse = 1.0F / (sum + eps_);
        chunkStart = start;
        while (chunkStart < end) {
            uint64_t remain = end - chunkStart;
            uint32_t n = static_cast<uint32_t>(remain > WORK_ELEMS ? WORK_ELEMS : remain);
            for (uint32_t i = 0; i < n; ++i) {
                work.SetValue(i, ReadSrc(FlatOffset(outer, chunkStart + i, inner)) - maxValue);
            }
            ExpWorkOnly(n);
            for (uint32_t i = 0; i < n; ++i) {
                WriteOut(FlatOffset(outer, chunkStart + i, inner),
                         work.GetValue(i) * inverse);
            }
            chunkStart += n;
        }
    }

    __aicore__ inline void ProcessPtrScalar() {
        const uint64_t groupCount = ptrLength_ - 1U;
        for (uint64_t outer = 0; outer < outerSize_; ++outer) {
            for (uint64_t inner = 0; inner < innerSize_; ++inner) {
                int64_t startRaw = static_cast<int64_t>(ptrGlobal_.GetValue(0));
                for (uint64_t group = 0; group < groupCount; ++group) {
                    int64_t endRaw = static_cast<int64_t>(ptrGlobal_.GetValue(group + 1U));
                    int64_t start = startRaw;
                    int64_t end = endRaw;
                    startRaw = endRaw;
                    if (!ClampPtrRange(start, end)) continue;
                    ProcessScalarContiguous(outer, inner,
                        static_cast<uint64_t>(start), static_cast<uint64_t>(end));
                }
            }
        }
    }

    __aicore__ inline void FlushScalarOutput() {
        AscendC::DataCacheCleanAndInvalid<
            StorageType, AscendC::CacheLine::ENTIRE_DATA_CACHE,
            AscendC::DcciDst::CACHELINE_OUT>(outGlobal_);
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> workBuf_;
    AscendC::TBuf<AscendC::TPosition::VECIN> rawBuf_;
    AscendC::TBuf<AscendC::TPosition::VECIN> indexBuf_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueue_;

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
    uint64_t innerTileWidth_ = 1;
    uint64_t tileCount_ = 1;
    float eps_ = 1e-16F;
    float singletonValue_ = 1.0F;

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
    GET_TILING_DATA_WITH_STRUCT(SparseSoftmaxTilingData, tilingData, tiling);
    KernelSparseSoftmax<DT_MODE> op;
    op.Init(src, index, ptr, out, tilingData);
    op.Process();
}
