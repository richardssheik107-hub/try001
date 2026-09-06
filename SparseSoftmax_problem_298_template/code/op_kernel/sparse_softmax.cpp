#include "kernel_operator.h"

#include "sparse_softmax_tiling.h"
#include "tiling_key_sparse_softmax.h"

template <int DT_MODE>
struct StorageTraits;

template <>
struct StorageTraits<SPARSE_SOFTMAX_FP32> {
    using StorageType = float;
    __aicore__ static inline float ToFloatValue(StorageType value) {
        return value;
    }
    __aicore__ static inline StorageType FromFloatValue(float value) {
        return value;
    }
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

    static constexpr uint32_t WORK_ELEMS =
        SPARSE_SOFTMAX_WORK_BUFFER_ELEMS;
    static constexpr uint32_t INDEX_ELEMS =
        SPARSE_SOFTMAX_INDEX_BUFFER_BYTES / sizeof(int32_t);
    static constexpr uint32_t LANE_BATCH_MAX = 8;
    static constexpr float NEG_FLOAT_MAX = -3.402823466e+38F;

    __aicore__ inline KernelSparseSoftmax() {}

    __aicore__ inline void Init(
        GM_ADDR src, GM_ADDR index, GM_ADDR ptr, GM_ADDR out,
        const SparseSoftmaxTilingData &tiling) {
        srcGlobal_.SetGlobalBuffer((__gm__ StorageType *)src);
        outGlobal_.SetGlobalBuffer((__gm__ StorageType *)out);
        if (index != nullptr) {
            indexGlobal_.SetGlobalBuffer((__gm__ int32_t *)index);
        }
        if (ptr != nullptr) {
            ptrGlobal_.SetGlobalBuffer((__gm__ int32_t *)ptr);
        }

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

        pipe_.InitBuffer(
            workBuf_, SPARSE_SOFTMAX_WORK_BUFFER_ELEMS * sizeof(float));

        if (fastPath_ == 3U) {
            pipe_.InitBuffer(
                outQueue_, 1, SPARSE_SOFTMAX_AXIS_OUT_QUEUE_BYTES);
            return;
        }

        if (fastPath_ == 1U || fastPath_ == 2U) {
            pipe_.InitBuffer(rawBuf_, SPARSE_SOFTMAX_RAW_BUFFER_BYTES);
            pipe_.InitBuffer(indexBuf_, SPARSE_SOFTMAX_INDEX_BUFFER_BYTES);

            eventMTE2ToS_ = static_cast<int32_t>(
                pipe_.FetchEventID(AscendC::HardEvent::MTE2_S));
            eventSToMTE2_ = static_cast<int32_t>(
                pipe_.FetchEventID(AscendC::HardEvent::S_MTE2));
            eventSToMTE3_ = static_cast<int32_t>(
                pipe_.FetchEventID(AscendC::HardEvent::S_MTE3));
            eventMTE3ToS_ = static_cast<int32_t>(
                pipe_.FetchEventID(AscendC::HardEvent::MTE3_S));
        }
    }

    __aicore__ inline void Process() {
        if (totalLength_ == 0 || dimSize_ == 0 || outerSize_ == 0) {
            return;
        }

        if (mode_ == 0U) {
            if (fastPath_ == 3U && innerSize_ == 1U) {
                ProcessIndexAxisMte3();
                return;
            }
            if (fastPath_ == 1U) {
                ProcessTiledDma(false);
                return;
            }
            if (AscendC::GetBlockIdx() == 0) {
                ProcessIndexScalar();
                FlushScalarOutput();
            }
            return;
        }

        if (fastPath_ == 2U && innerSize_ == 1U) {
            ProcessPtrAxisDma();
            return;
        }
        if (fastPath_ == 1U) {
            ProcessTiledDma(true);
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
        return StorageTraits<DT_MODE>::ToFloatValue(
            srcGlobal_.GetValue(offset));
    }

    __aicore__ inline void WriteOut(uint64_t offset, float value) {
        outGlobal_.SetValue(
            offset, StorageTraits<DT_MODE>::FromFloatValue(value));
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
        AscendC::LocalTensor<StorageType> raw,
        uint64_t offset, float value) {
        raw.SetValue(
            static_cast<uint32_t>(offset),
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
        AscendC::DataCopyPadExtParams<T> padParams = {
            false, 0, 0, 0};
        AscendC::DataCopyPad<T>(
            dst, src, copyParams, padParams);
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

    __aicore__ inline float SumWork(
        AscendC::LocalTensor<float> work, uint32_t count) const {
        return SumWorkRange(work, 0, count);
    }

    __aicore__ inline float SumWorkRange(
        AscendC::LocalTensor<float> work,
        uint32_t offset, uint32_t count) const {
        float s0 = 0.0F;
        float s1 = 0.0F;
        float s2 = 0.0F;
        float s3 = 0.0F;
        uint32_t i = 0;
        for (; i + 3U < count; i += 4U) {
            s0 += work.GetValue(offset + i);
            s1 += work.GetValue(offset + i + 1U);
            s2 += work.GetValue(offset + i + 2U);
            s3 += work.GetValue(offset + i + 3U);
        }
        float sum = (s0 + s1) + (s2 + s3);
        for (; i < count; ++i) {
            sum += work.GetValue(offset + i);
        }
        return sum;
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

    __aicore__ inline void ExpAndScaleWork(
        uint32_t count, float inverse) {
        AscendC::LocalTensor<float> work = workBuf_.Get<float>();
        ExpWorkOnly(count);
        AscendC::Muls(
            work, work, inverse, static_cast<int32_t>(count));
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline float ExpNormalizeWork(uint32_t count) {
        AscendC::LocalTensor<float> work = workBuf_.Get<float>();
        ExpWorkOnly(count);
        const float inverse = 1.0F / (SumWork(work, count) + eps_);
        AscendC::Muls(
            work, work, inverse, static_cast<int32_t>(count));
        AscendC::PipeBarrier<PIPE_V>();
        return inverse;
    }

    // ---------------------------------------------------------------------
    // Fast path 3: index + contiguous axis (inner == 1).
    // ---------------------------------------------------------------------

    __aicore__ inline int32_t ReadAxisIndex(
        uint64_t outer, uint64_t pos) const {
        if (indexLength_ == totalLength_) {
            return indexGlobal_.GetValue(outer * dimSize_ + pos);
        }
        return indexGlobal_.GetValue(pos);
    }

    __aicore__ inline bool IsFirstAxisOccurrence(
        uint64_t outer, uint64_t seed, int32_t group) const {
        for (uint64_t pos = 0; pos < seed; ++pos) {
            if (ReadAxisIndex(outer, pos) == group) {
                return false;
            }
        }
        return true;
    }

    __aicore__ inline void ScanAxisGroupStats(
        uint64_t outer, int32_t group,
        uint64_t &count, float &maxValue,
        uint64_t &alignedRunBytes) const {
        count = 0;
        maxValue = NEG_FLOAT_MAX;
        alignedRunBytes = 0;

        bool inRun = false;
        uint64_t runLength = 0;
        const uint64_t base = outer * dimSize_;

        for (uint64_t pos = 0; pos < dimSize_; ++pos) {
            const bool member = ReadAxisIndex(outer, pos) == group;
            if (member) {
                const float value = ReadSrc(base + pos);
                if (value > maxValue) {
                    maxValue = value;
                }
                ++count;
                if (inRun) {
                    ++runLength;
                } else {
                    inRun = true;
                    runLength = 1;
                }
            } else if (inRun) {
                alignedRunBytes +=
                    Align32(runLength * sizeof(StorageType));
                inRun = false;
                runLength = 0;
            }
        }
        if (inRun) {
            alignedRunBytes +=
                Align32(runLength * sizeof(StorageType));
        }
    }

    __aicore__ inline void FillAxisPackedWork(
        uint64_t outer, int32_t group, float maxValue) {
        AscendC::LocalTensor<float> work = workBuf_.Get<float>();
        uint32_t packed = 0;
        const uint64_t base = outer * dimSize_;
        for (uint64_t pos = 0; pos < dimSize_; ++pos) {
            if (ReadAxisIndex(outer, pos) == group) {
                work.SetValue(
                    packed++, ReadSrc(base + pos) - maxValue);
            }
        }
    }

    __aicore__ inline void WriteAxisPackedRuns(
        uint64_t outer, int32_t group) {
        AscendC::LocalTensor<float> work = workBuf_.Get<float>();
        AscendC::LocalTensor<StorageType> outLocal =
            outQueue_.AllocTensor<StorageType>();

        uint64_t pos = 0;
        uint32_t packed = 0;
        uint32_t slotBytes = 0;

        while (pos < dimSize_) {
            if (ReadAxisIndex(outer, pos) != group) {
                ++pos;
                continue;
            }

            const uint64_t runBegin = pos;
            const uint32_t packedBegin = packed;
            do {
                ++pos;
                ++packed;
            } while (pos < dimSize_ &&
                     ReadAxisIndex(outer, pos) == group);

            const uint32_t runLength =
                static_cast<uint32_t>(pos - runBegin);
            const uint32_t slotElement =
                slotBytes / sizeof(StorageType);

            for (uint32_t i = 0; i < runLength; ++i) {
                outLocal.SetValue(
                    slotElement + i,
                    StorageTraits<DT_MODE>::FromFloatValue(
                        work.GetValue(packedBegin + i)));
            }
            slotBytes += static_cast<uint32_t>(
                Align32(
                    static_cast<uint64_t>(runLength) *
                    sizeof(StorageType)));
        }

        AscendC::PipeBarrier<PIPE_ALL>();
        outQueue_.EnQue(outLocal);
        outLocal = outQueue_.DeQue<StorageType>();

        pos = 0;
        slotBytes = 0;
        const uint64_t base = outer * dimSize_;
        while (pos < dimSize_) {
            if (ReadAxisIndex(outer, pos) != group) {
                ++pos;
                continue;
            }

            const uint64_t runBegin = pos;
            do {
                ++pos;
            } while (pos < dimSize_ &&
                     ReadAxisIndex(outer, pos) == group);

            const uint32_t runLength =
                static_cast<uint32_t>(pos - runBegin);
            const uint32_t slotElement =
                slotBytes / sizeof(StorageType);

            AscendC::DataCopyExtParams params;
            params.blockCount = 1;
            params.blockLen = runLength * sizeof(StorageType);
            params.srcStride = 0;
            params.dstStride = 0;
            params.rsv = 0;
            AscendC::DataCopyPad(
                outGlobal_[base + runBegin],
                outLocal[slotElement], params);

            slotBytes += static_cast<uint32_t>(
                Align32(
                    static_cast<uint64_t>(runLength) *
                    sizeof(StorageType)));
        }
        outQueue_.FreeTensor(outLocal);
    }

    __aicore__ inline float ComputeAxisGroupSum(
        uint64_t outer, int32_t group, float maxValue) {
        AscendC::LocalTensor<float> work = workBuf_.Get<float>();
        const uint64_t base = outer * dimSize_;
        float sum = 0.0F;
        uint32_t count = 0;

        for (uint64_t pos = 0; pos < dimSize_; ++pos) {
            if (ReadAxisIndex(outer, pos) != group) {
                continue;
            }
            work.SetValue(
                count++, ReadSrc(base + pos) - maxValue);
            if (count == WORK_ELEMS) {
                sum += ExpAndSumWork(count);
                count = 0;
            }
        }
        if (count != 0) {
            sum += ExpAndSumWork(count);
        }
        return sum;
    }

    __aicore__ inline void WriteOneAxisChunk(
        uint64_t gmOffset, uint32_t count,
        float maxValue, float inverse) {
        AscendC::LocalTensor<float> work = workBuf_.Get<float>();
        for (uint32_t i = 0; i < count; ++i) {
            work.SetValue(
                i, ReadSrc(gmOffset + i) - maxValue);
        }
        ExpAndScaleWork(count, inverse);

        AscendC::LocalTensor<StorageType> outLocal =
            outQueue_.AllocTensor<StorageType>();
        for (uint32_t i = 0; i < count; ++i) {
            outLocal.SetValue(
                i, StorageTraits<DT_MODE>::FromFloatValue(
                    work.GetValue(i)));
        }

        AscendC::PipeBarrier<PIPE_ALL>();
        outQueue_.EnQue(outLocal);
        outLocal = outQueue_.DeQue<StorageType>();

        AscendC::DataCopyExtParams params;
        params.blockCount = 1;
        params.blockLen = count * sizeof(StorageType);
        params.srcStride = 0;
        params.dstStride = 0;
        params.rsv = 0;
        AscendC::DataCopyPad(
            outGlobal_[gmOffset], outLocal, params);
        outQueue_.FreeTensor(outLocal);
    }

    __aicore__ inline void WriteAxisGroupChunked(
        uint64_t outer, int32_t group,
        float maxValue, float inverse) {
        const uint32_t queueElems =
            SPARSE_SOFTMAX_AXIS_OUT_QUEUE_BYTES /
            sizeof(StorageType);
        uint32_t chunkLimit = WORK_ELEMS;
        if (chunkLimit > queueElems) {
            chunkLimit = queueElems;
        }

        const uint64_t base = outer * dimSize_;
        uint64_t pos = 0;
        while (pos < dimSize_) {
            if (ReadAxisIndex(outer, pos) != group) {
                ++pos;
                continue;
            }

            const uint64_t runBegin = pos;
            do {
                ++pos;
            } while (pos < dimSize_ &&
                     ReadAxisIndex(outer, pos) == group);
            const uint64_t runEnd = pos;

            uint64_t chunkStart = runBegin;
            while (chunkStart < runEnd) {
                uint64_t remain = runEnd - chunkStart;
                uint32_t count = static_cast<uint32_t>(
                    remain > chunkLimit ? chunkLimit : remain);
                WriteOneAxisChunk(
                    base + chunkStart, count,
                    maxValue, inverse);
                chunkStart += count;
            }
        }
    }

    __aicore__ inline void ProcessAxisGroupMte3(
        uint64_t outer, int32_t group) {
        uint64_t count = 0;
        uint64_t alignedRunBytes = 0;
        float maxValue = NEG_FLOAT_MAX;
        ScanAxisGroupStats(
            outer, group, count, maxValue, alignedRunBytes);
        if (count == 0) {
            return;
        }

        if (count == 1) {
            AscendC::LocalTensor<float> work = workBuf_.Get<float>();
            work.SetValue(0, singletonValue_);
            if (alignedRunBytes <= SPARSE_SOFTMAX_AXIS_OUT_QUEUE_BYTES) {
                WriteAxisPackedRuns(outer, group);
            } else {
                WriteAxisGroupChunked(
                    outer, group, maxValue, singletonValue_);
            }
            return;
        }

        if (count <= WORK_ELEMS &&
            alignedRunBytes <= SPARSE_SOFTMAX_AXIS_OUT_QUEUE_BYTES) {
            FillAxisPackedWork(outer, group, maxValue);
            ExpNormalizeWork(static_cast<uint32_t>(count));
            WriteAxisPackedRuns(outer, group);
            return;
        }

        const float groupSum =
            ComputeAxisGroupSum(outer, group, maxValue);
        const float inverse = 1.0F / (groupSum + eps_);
        WriteAxisGroupChunked(
            outer, group, maxValue, inverse);
    }

    __aicore__ inline void ProcessIndexAxisMte3() {
        const uint64_t taskCount = outerSize_ * dimSize_;
        const uint64_t blockIdx =
            static_cast<uint64_t>(AscendC::GetBlockIdx());
        uint64_t blockNum =
            static_cast<uint64_t>(AscendC::GetBlockNum());
        if (blockNum == 0) {
            blockNum = 1;
        }

        for (uint64_t task = blockIdx;
             task < taskCount; task += blockNum) {
            const uint64_t outer = task / dimSize_;
            const uint64_t seed = task - outer * dimSize_;
            const int32_t group = ReadAxisIndex(outer, seed);

            if (IsFirstAxisOccurrence(outer, seed, group)) {
                ProcessAxisGroupMte3(outer, group);
            }
        }
    }

    // ---------------------------------------------------------------------
    // Fast path 2: ptr + contiguous axis.
    // ---------------------------------------------------------------------

    __aicore__ inline int64_t ReadPtrValue(
        AscendC::LocalTensor<int32_t> cachedPtr,
        bool cached, uint64_t idx) const {
        if (cached) {
            return static_cast<int64_t>(
                cachedPtr.GetValue(static_cast<uint32_t>(idx)));
        }
        return static_cast<int64_t>(ptrGlobal_.GetValue(idx));
    }

    __aicore__ inline bool ClampPtrRange(
        int64_t &start, int64_t &end) const {
        if (start < 0) {
            start = 0;
        }
        if (end < start) {
            return false;
        }
        if (start > static_cast<int64_t>(dimSize_)) {
            return false;
        }
        if (end > static_cast<int64_t>(dimSize_)) {
            end = static_cast<int64_t>(dimSize_);
        }
        return end > start;
    }

    __aicore__ inline bool LoadPtrCache(
        AscendC::LocalTensor<int32_t> cachedPtr) {
        if (ptrLength_ * sizeof(int32_t) >
            SPARSE_SOFTMAX_INDEX_BUFFER_BYTES) {
            return false;
        }
        CopyGmToLocal<int32_t>(
            cachedPtr, ptrGlobal_, 1,
            static_cast<uint32_t>(
                ptrLength_ * sizeof(int32_t)),
            0, 0);
        SyncMTE2ToS();
        return true;
    }

    __aicore__ inline void NormalizeRawContiguous(
        AscendC::LocalTensor<StorageType> raw,
        uint64_t count) {
        AscendC::LocalTensor<float> work = workBuf_.Get<float>();

        float maxValue = NEG_FLOAT_MAX;
        for (uint64_t i = 0; i < count; ++i) {
            const float value = ReadRaw(raw, i);
            if (value > maxValue) {
                maxValue = value;
            }
        }

        // Common small/medium groups fit the work buffer.  Keep the first Exp
        // result and normalize while writing raw, instead of recomputing Exp.
        if (count <= WORK_ELEMS) {
            const uint32_t n = static_cast<uint32_t>(count);
            for (uint32_t i = 0; i < n; ++i) {
                work.SetValue(i, ReadRaw(raw, i) - maxValue);
            }
            const float sum = ExpAndSumWork(n);
            const float inverse = 1.0F / (sum + eps_);
            for (uint32_t i = 0; i < n; ++i) {
                WriteRaw(raw, i, work.GetValue(i) * inverse);
            }
            return;
        }

        float groupSum = 0.0F;
        uint64_t start = 0;
        while (start < count) {
            uint64_t remain = count - start;
            uint32_t n = static_cast<uint32_t>(
                remain > WORK_ELEMS ? WORK_ELEMS : remain);
            for (uint32_t i = 0; i < n; ++i) {
                work.SetValue(
                    i, ReadRaw(raw, start + i) - maxValue);
            }
            groupSum += ExpAndSumWork(n);
            start += n;
        }

        const float inverse = 1.0F / (groupSum + eps_);
        start = 0;
        while (start < count) {
            uint64_t remain = count - start;
            uint32_t n = static_cast<uint32_t>(
                remain > WORK_ELEMS ? WORK_ELEMS : remain);
            for (uint32_t i = 0; i < n; ++i) {
                work.SetValue(
                    i, ReadRaw(raw, start + i) - maxValue);
            }
            ExpWorkOnly(n);
            for (uint32_t i = 0; i < n; ++i) {
                WriteRaw(raw, start + i, work.GetValue(i) * inverse);
            }
            start += n;
        }
    }

    __aicore__ inline void ProcessPtrAxisGroup(
        uint64_t outer, uint64_t start, uint64_t end) {
        AscendC::LocalTensor<StorageType> raw =
            rawBuf_.Get<StorageType>();
        AscendC::LocalTensor<float> work = workBuf_.Get<float>();

        const uint64_t count = end - start;
        const uint64_t gmBase = outer * dimSize_ + start;
        const uint64_t rawCapacity =
            SPARSE_SOFTMAX_RAW_BUFFER_BYTES / sizeof(StorageType);

        if (count == 1) {
            WriteRaw(raw, 0, singletonValue_);
            SyncSToMTE3();
            CopyLocalToGm<StorageType>(
                outGlobal_[gmBase], raw, 1,
                static_cast<uint32_t>(sizeof(StorageType)),
                0, 0);
            SyncMTE3ToS();
            return;
        }

        if (count <= rawCapacity) {
            CopyGmToLocal<StorageType>(
                raw, srcGlobal_[gmBase], 1,
                static_cast<uint32_t>(
                    count * sizeof(StorageType)),
                0, 0);
            SyncMTE2ToS();

            NormalizeRawContiguous(raw, count);

            SyncSToMTE3();
            CopyLocalToGm<StorageType>(
                outGlobal_[gmBase], raw, 1,
                static_cast<uint32_t>(
                    count * sizeof(StorageType)),
                0, 0);
            SyncMTE3ToS();
            return;
        }

        float maxValue = NEG_FLOAT_MAX;
        uint64_t chunkStart = 0;
        while (chunkStart < count) {
            uint64_t chunkCount = count - chunkStart;
            if (chunkCount > rawCapacity) {
                chunkCount = rawCapacity;
            }

            CopyGmToLocal<StorageType>(
                raw, srcGlobal_[gmBase + chunkStart], 1,
                static_cast<uint32_t>(
                    chunkCount * sizeof(StorageType)),
                0, 0);
            SyncMTE2ToS();

            for (uint64_t i = 0; i < chunkCount; ++i) {
                const float value = ReadRaw(raw, i);
                if (value > maxValue) {
                    maxValue = value;
                }
            }
            SyncSToMTE2();
            chunkStart += chunkCount;
        }

        float groupSum = 0.0F;
        chunkStart = 0;
        while (chunkStart < count) {
            uint64_t chunkCount = count - chunkStart;
            if (chunkCount > rawCapacity) {
                chunkCount = rawCapacity;
            }

            CopyGmToLocal<StorageType>(
                raw, srcGlobal_[gmBase + chunkStart], 1,
                static_cast<uint32_t>(
                    chunkCount * sizeof(StorageType)),
                0, 0);
            SyncMTE2ToS();

            uint64_t localStart = 0;
            while (localStart < chunkCount) {
                uint64_t remain = chunkCount - localStart;
                uint32_t n = static_cast<uint32_t>(
                    remain > WORK_ELEMS ? WORK_ELEMS : remain);
                for (uint32_t i = 0; i < n; ++i) {
                    work.SetValue(
                        i, ReadRaw(raw, localStart + i) - maxValue);
                }
                groupSum += ExpAndSumWork(n);
                localStart += n;
            }
            SyncSToMTE2();
            chunkStart += chunkCount;
        }

        const float inverse = 1.0F / (groupSum + eps_);
        chunkStart = 0;
        while (chunkStart < count) {
            uint64_t chunkCount = count - chunkStart;
            if (chunkCount > rawCapacity) {
                chunkCount = rawCapacity;
            }

            CopyGmToLocal<StorageType>(
                raw, srcGlobal_[gmBase + chunkStart], 1,
                static_cast<uint32_t>(
                    chunkCount * sizeof(StorageType)),
                0, 0);
            SyncMTE2ToS();

            uint64_t localStart = 0;
            while (localStart < chunkCount) {
                uint64_t remain = chunkCount - localStart;
                uint32_t n = static_cast<uint32_t>(
                    remain > WORK_ELEMS ? WORK_ELEMS : remain);
                for (uint32_t i = 0; i < n; ++i) {
                    work.SetValue(
                        i, ReadRaw(raw, localStart + i) - maxValue);
                }
                ExpWorkOnly(n);
                for (uint32_t i = 0; i < n; ++i) {
                    WriteRaw(
                        raw, localStart + i,
                        work.GetValue(i) * inverse);
                }
                localStart += n;
            }

            SyncSToMTE3();
            CopyLocalToGm<StorageType>(
                outGlobal_[gmBase + chunkStart], raw, 1,
                static_cast<uint32_t>(
                    chunkCount * sizeof(StorageType)),
                0, 0);
            SyncMTE3ToS();
            chunkStart += chunkCount;
        }
    }

    __aicore__ inline void ProcessPtrAxisDma() {
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

            int64_t start =
                ReadPtrValue(cachedPtr, ptrCached, group);
            int64_t end =
                ReadPtrValue(cachedPtr, ptrCached, group + 1U);
            if (!ClampPtrRange(start, end)) {
                continue;
            }
            ProcessPtrAxisGroup(
                outer,
                static_cast<uint64_t>(start),
                static_cast<uint64_t>(end));
        }
    }

    // ---------------------------------------------------------------------
    // Fast path 1: arbitrary-rank tiled DMA.
    // Broadcast index and ptr share group membership across inner lanes, so
    // lanes are batched into one vector Exp to amortize fixed vector barriers.
    // ---------------------------------------------------------------------

    __aicore__ inline bool LoadBroadcastIndex(
        AscendC::LocalTensor<int32_t> cachedIndex) {
        if (dimSize_ * sizeof(int32_t) >
            SPARSE_SOFTMAX_INDEX_BUFFER_BYTES) {
            return false;
        }
        CopyGmToLocal<int32_t>(
            cachedIndex, indexGlobal_, 1,
            static_cast<uint32_t>(
                dimSize_ * sizeof(int32_t)),
            0, 0);
        SyncMTE2ToS();
        return true;
    }

    __aicore__ inline int32_t IndexAtTile(
        AscendC::LocalTensor<int32_t> cachedIndex,
        bool localIndex, bool broadcastIndex,
        uint64_t indexRowStride,
        uint64_t outer, uint64_t dim,
        uint64_t globalInner, uint64_t localInner) const {
        if (localIndex) {
            if (broadcastIndex) {
                return cachedIndex.GetValue(
                    static_cast<uint32_t>(dim));
            }
            return cachedIndex.GetValue(
                static_cast<uint32_t>(
                    dim * indexRowStride + localInner));
        }
        return ReadIndex(outer, dim, globalInner);
    }

    __aicore__ inline void SoftmaxLocalContiguous(
        AscendC::LocalTensor<StorageType> raw,
        uint64_t rowStride, uint64_t lane,
        uint64_t start, uint64_t end) {
        const uint64_t count = end - start;
        if (count == 0) {
            return;
        }
        if (count == 1) {
            WriteRaw(
                raw, start * rowStride + lane,
                singletonValue_);
            return;
        }

        AscendC::LocalTensor<float> work = workBuf_.Get<float>();
        float maxValue = NEG_FLOAT_MAX;
        for (uint64_t pos = start; pos < end; ++pos) {
            const float value =
                ReadRaw(raw, pos * rowStride + lane);
            if (value > maxValue) {
                maxValue = value;
            }
        }

        uint32_t packed = 0;
        for (uint64_t pos = start; pos < end; ++pos) {
            work.SetValue(
                packed++,
                ReadRaw(raw, pos * rowStride + lane) - maxValue);
        }
        const float sum = ExpAndSumWork(packed);
        const float inverse = 1.0F / (sum + eps_);

        packed = 0;
        for (uint64_t pos = start; pos < end; ++pos) {
            WriteRaw(
                raw, pos * rowStride + lane,
                work.GetValue(packed++) * inverse);
        }
    }

    __aicore__ inline void SoftmaxLocalIndexGroup(
        AscendC::LocalTensor<StorageType> raw,
        AscendC::LocalTensor<int32_t> cachedIndex,
        bool localIndex, bool broadcastIndex,
        uint64_t rowStride, uint64_t indexRowStride,
        uint64_t outer, uint64_t globalInner,
        uint64_t localInner, int32_t group) {
        AscendC::LocalTensor<float> work = workBuf_.Get<float>();

        float maxValue = NEG_FLOAT_MAX;
        uint32_t count = 0;
        uint64_t singletonPos = 0;

        for (uint64_t pos = 0; pos < dimSize_; ++pos) {
            if (IndexAtTile(
                    cachedIndex, localIndex, broadcastIndex,
                    indexRowStride, outer, pos,
                    globalInner, localInner) != group) {
                continue;
            }
            const float value =
                ReadRaw(raw, pos * rowStride + localInner);
            if (value > maxValue) {
                maxValue = value;
            }
            singletonPos = pos;
            ++count;
        }

        if (count == 0) {
            return;
        }
        if (count == 1) {
            WriteRaw(
                raw, singletonPos * rowStride + localInner,
                singletonValue_);
            return;
        }

        uint32_t packed = 0;
        for (uint64_t pos = 0; pos < dimSize_; ++pos) {
            if (IndexAtTile(
                    cachedIndex, localIndex, broadcastIndex,
                    indexRowStride, outer, pos,
                    globalInner, localInner) == group) {
                work.SetValue(
                    packed++,
                    ReadRaw(raw, pos * rowStride + localInner) -
                    maxValue);
            }
        }
        const float sum = ExpAndSumWork(packed);
        const float inverse = 1.0F / (sum + eps_);

        packed = 0;
        for (uint64_t pos = 0; pos < dimSize_; ++pos) {
            if (IndexAtTile(
                    cachedIndex, localIndex, broadcastIndex,
                    indexRowStride, outer, pos,
                    globalInner, localInner) == group) {
                WriteRaw(
                    raw, pos * rowStride + localInner,
                    work.GetValue(packed++) * inverse);
            }
        }
    }

    __aicore__ inline bool IsFirstTileOccurrence(
        AscendC::LocalTensor<int32_t> cachedIndex,
        bool localIndex, bool broadcastIndex,
        uint64_t indexRowStride,
        uint64_t outer, uint64_t seed,
        uint64_t globalInner, uint64_t localInner,
        int32_t group) const {
        for (uint64_t pos = 0; pos < seed; ++pos) {
            if (IndexAtTile(
                    cachedIndex, localIndex, broadcastIndex,
                    indexRowStride, outer, pos,
                    globalInner, localInner) == group) {
                return false;
            }
        }
        return true;
    }

    __aicore__ inline uint32_t CountBroadcastGroup(
        AscendC::LocalTensor<int32_t> cachedIndex,
        bool localIndex, uint64_t outer,
        uint64_t innerStart, int32_t group,
        uint64_t &singletonPos) const {
        uint32_t count = 0;
        singletonPos = 0;
        for (uint64_t pos = 0; pos < dimSize_; ++pos) {
            if (IndexAtTile(
                    cachedIndex, localIndex, true, 0,
                    outer, pos, innerStart, 0) == group) {
                singletonPos = pos;
                ++count;
            }
        }
        return count;
    }

    __aicore__ inline uint32_t ChooseLaneBatch(
        uint64_t remainingLanes, uint32_t groupCount) const {
        if (groupCount == 0) {
            return 1;
        }
        uint32_t byWork = WORK_ELEMS / groupCount;
        if (byWork == 0) {
            byWork = 1;
        }
        uint32_t batch = static_cast<uint32_t>(
            remainingLanes > LANE_BATCH_MAX ?
            LANE_BATCH_MAX : remainingLanes);
        if (batch > byWork) {
            batch = byWork;
        }
        return batch == 0 ? 1 : batch;
    }

    __aicore__ inline void SoftmaxBroadcastGroupBatch(
        AscendC::LocalTensor<StorageType> raw,
        AscendC::LocalTensor<int32_t> cachedIndex,
        bool localIndex, uint64_t rowStride,
        uint64_t outer, uint64_t innerStart,
        uint64_t laneStart, uint32_t laneCount,
        int32_t group, uint32_t groupCount,
        uint64_t singletonPos) {
        if (groupCount == 0) {
            return;
        }
        if (groupCount == 1) {
            for (uint32_t lane = 0; lane < laneCount; ++lane) {
                WriteRaw(
                    raw,
                    singletonPos * rowStride + laneStart + lane,
                    singletonValue_);
            }
            return;
        }

        float maxValues[LANE_BATCH_MAX];
        float inverseValues[LANE_BATCH_MAX];
        for (uint32_t lane = 0; lane < laneCount; ++lane) {
            maxValues[lane] = NEG_FLOAT_MAX;
        }

        for (uint64_t pos = 0; pos < dimSize_; ++pos) {
            if (IndexAtTile(
                    cachedIndex, localIndex, true, 0,
                    outer, pos, innerStart, 0) != group) {
                continue;
            }
            const uint64_t rowBase = pos * rowStride + laneStart;
            for (uint32_t lane = 0; lane < laneCount; ++lane) {
                const float value = ReadRaw(raw, rowBase + lane);
                if (value > maxValues[lane]) {
                    maxValues[lane] = value;
                }
            }
        }

        AscendC::LocalTensor<float> work = workBuf_.Get<float>();
        uint32_t packed = 0;
        for (uint64_t pos = 0; pos < dimSize_; ++pos) {
            if (IndexAtTile(
                    cachedIndex, localIndex, true, 0,
                    outer, pos, innerStart, 0) != group) {
                continue;
            }
            const uint64_t rowBase = pos * rowStride + laneStart;
            for (uint32_t lane = 0; lane < laneCount; ++lane) {
                work.SetValue(
                    lane * groupCount + packed,
                    ReadRaw(raw, rowBase + lane) - maxValues[lane]);
            }
            ++packed;
        }

        ExpWorkOnly(laneCount * groupCount);
        for (uint32_t lane = 0; lane < laneCount; ++lane) {
            const float sum = SumWorkRange(
                work, lane * groupCount, groupCount);
            inverseValues[lane] = 1.0F / (sum + eps_);
        }

        packed = 0;
        for (uint64_t pos = 0; pos < dimSize_; ++pos) {
            if (IndexAtTile(
                    cachedIndex, localIndex, true, 0,
                    outer, pos, innerStart, 0) != group) {
                continue;
            }
            const uint64_t rowBase = pos * rowStride + laneStart;
            for (uint32_t lane = 0; lane < laneCount; ++lane) {
                WriteRaw(
                    raw, rowBase + lane,
                    work.GetValue(lane * groupCount + packed) *
                    inverseValues[lane]);
            }
            ++packed;
        }
    }

    __aicore__ inline void SoftmaxContiguousBatch(
        AscendC::LocalTensor<StorageType> raw,
        uint64_t rowStride, uint64_t laneStart,
        uint32_t laneCount, uint64_t start, uint64_t end) {
        const uint32_t groupCount =
            static_cast<uint32_t>(end - start);
        if (groupCount == 0) {
            return;
        }
        if (groupCount == 1) {
            for (uint32_t lane = 0; lane < laneCount; ++lane) {
                WriteRaw(
                    raw, start * rowStride + laneStart + lane,
                    singletonValue_);
            }
            return;
        }

        float maxValues[LANE_BATCH_MAX];
        float inverseValues[LANE_BATCH_MAX];
        for (uint32_t lane = 0; lane < laneCount; ++lane) {
            maxValues[lane] = NEG_FLOAT_MAX;
        }

        for (uint64_t pos = start; pos < end; ++pos) {
            const uint64_t rowBase = pos * rowStride + laneStart;
            for (uint32_t lane = 0; lane < laneCount; ++lane) {
                const float value = ReadRaw(raw, rowBase + lane);
                if (value > maxValues[lane]) {
                    maxValues[lane] = value;
                }
            }
        }

        AscendC::LocalTensor<float> work = workBuf_.Get<float>();
        for (uint32_t lane = 0; lane < laneCount; ++lane) {
            for (uint32_t i = 0; i < groupCount; ++i) {
                work.SetValue(
                    lane * groupCount + i,
                    ReadRaw(
                        raw,
                        (start + i) * rowStride + laneStart + lane) -
                    maxValues[lane]);
            }
        }

        ExpWorkOnly(laneCount * groupCount);
        for (uint32_t lane = 0; lane < laneCount; ++lane) {
            const float sum = SumWorkRange(
                work, lane * groupCount, groupCount);
            inverseValues[lane] = 1.0F / (sum + eps_);
        }

        for (uint32_t i = 0; i < groupCount; ++i) {
            const uint64_t rowBase =
                (start + i) * rowStride + laneStart;
            for (uint32_t lane = 0; lane < laneCount; ++lane) {
                WriteRaw(
                    raw, rowBase + lane,
                    work.GetValue(lane * groupCount + i) *
                    inverseValues[lane]);
            }
        }
    }

    __aicore__ inline void ProcessBroadcastIndexTile(
        AscendC::LocalTensor<StorageType> raw,
        AscendC::LocalTensor<int32_t> cachedIndex,
        bool localIndex, uint64_t rowStride,
        uint64_t outer, uint64_t innerStart,
        uint64_t tileWidth) {
        for (uint64_t seed = 0; seed < dimSize_; ++seed) {
            const int32_t group = IndexAtTile(
                cachedIndex, localIndex, true, 0,
                outer, seed, innerStart, 0);
            if (!IsFirstTileOccurrence(
                    cachedIndex, localIndex, true, 0,
                    outer, seed, innerStart, 0, group)) {
                continue;
            }

            uint64_t singletonPos = 0;
            const uint32_t groupCount = CountBroadcastGroup(
                cachedIndex, localIndex, outer,
                innerStart, group, singletonPos);

            uint64_t laneStart = 0;
            while (laneStart < tileWidth) {
                const uint32_t laneBatch = ChooseLaneBatch(
                    tileWidth - laneStart, groupCount);
                SoftmaxBroadcastGroupBatch(
                    raw, cachedIndex, localIndex,
                    rowStride, outer, innerStart,
                    laneStart, laneBatch, group,
                    groupCount, singletonPos);
                laneStart += laneBatch;
            }
        }
    }

    __aicore__ inline void ProcessFullIndexTile(
        AscendC::LocalTensor<StorageType> raw,
        AscendC::LocalTensor<int32_t> cachedIndex,
        bool localIndex, uint64_t rowStride,
        uint64_t indexRowStride, uint64_t outer,
        uint64_t innerStart, uint64_t tileWidth) {
        for (uint64_t lane = 0; lane < tileWidth; ++lane) {
            const uint64_t globalInner = innerStart + lane;
            for (uint64_t seed = 0; seed < dimSize_; ++seed) {
                const int32_t group = IndexAtTile(
                    cachedIndex, localIndex, false,
                    indexRowStride, outer, seed,
                    globalInner, lane);
                if (!IsFirstTileOccurrence(
                        cachedIndex, localIndex, false,
                        indexRowStride, outer, seed,
                        globalInner, lane, group)) {
                    continue;
                }

                SoftmaxLocalIndexGroup(
                    raw, cachedIndex, localIndex, false,
                    rowStride, indexRowStride, outer,
                    globalInner, lane, group);
            }
        }
    }

    __aicore__ inline void ProcessPtrTile(
        AscendC::LocalTensor<StorageType> raw,
        AscendC::LocalTensor<int32_t> cachedPtr,
        bool ptrCached, uint64_t rowStride,
        uint64_t tileWidth) {
        const uint64_t groupCountAll = ptrLength_ - 1U;
        int64_t startRaw =
            ReadPtrValue(cachedPtr, ptrCached, 0);

        for (uint64_t group = 0; group < groupCountAll; ++group) {
            int64_t endRaw =
                ReadPtrValue(cachedPtr, ptrCached, group + 1U);
            int64_t start = startRaw;
            int64_t end = endRaw;
            startRaw = endRaw;

            if (!ClampPtrRange(start, end)) {
                continue;
            }

            const uint32_t groupCount = static_cast<uint32_t>(end - start);
            uint64_t laneStart = 0;
            while (laneStart < tileWidth) {
                const uint32_t laneBatch = ChooseLaneBatch(
                    tileWidth - laneStart, groupCount);
                SoftmaxContiguousBatch(
                    raw, rowStride, laneStart, laneBatch,
                    static_cast<uint64_t>(start),
                    static_cast<uint64_t>(end));
                laneStart += laneBatch;
            }
        }
    }

    __aicore__ inline void ProcessOneTile(
        AscendC::LocalTensor<StorageType> raw,
        AscendC::LocalTensor<int32_t> cachedMeta,
        bool ptrMode, bool ptrCached,
        bool broadcastIndex, bool broadcastIndexLocal,
        uint64_t outer, uint64_t innerStart,
        uint64_t tileWidth) {
        const uint64_t rowBytes =
            tileWidth * sizeof(StorageType);
        const uint64_t rowStrideBytes = Align32(rowBytes);
        const uint64_t rowStride =
            rowStrideBytes / sizeof(StorageType);
        const uint64_t gmBase =
            OuterBase(outer) + innerStart;

        CopyGmToLocal<StorageType>(
            raw, srcGlobal_[gmBase],
            static_cast<uint16_t>(dimSize_),
            static_cast<uint32_t>(rowBytes),
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
                    cachedMeta,
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
                raw, cachedMeta, ptrCached,
                rowStride, tileWidth);
        } else if (broadcastIndex) {
            ProcessBroadcastIndexTile(
                raw, cachedMeta, broadcastIndexLocal,
                rowStride, outer, innerStart, tileWidth);
        } else {
            ProcessFullIndexTile(
                raw, cachedMeta, fullIndexLocal,
                rowStride, indexRowStride,
                outer, innerStart, tileWidth);
        }

        SyncSToMTE3();
        CopyLocalToGm<StorageType>(
            outGlobal_[gmBase], raw,
            static_cast<uint16_t>(dimSize_),
            static_cast<uint32_t>(rowBytes),
            0,
            static_cast<uint32_t>(
                (innerSize_ - tileWidth) *
                sizeof(StorageType)));
        SyncMTE3ToS();
    }

    __aicore__ inline void ProcessTiledDma(bool ptrMode) {
        AscendC::LocalTensor<StorageType> raw =
            rawBuf_.Get<StorageType>();
        AscendC::LocalTensor<int32_t> cachedMeta =
            indexBuf_.Get<int32_t>();

        const bool broadcastIndex =
            !ptrMode && indexLength_ != totalLength_;
        bool broadcastIndexLocal = false;
        bool ptrCached = false;

        if (ptrMode) {
            ptrCached = LoadPtrCache(cachedMeta);
        } else if (broadcastIndex) {
            broadcastIndexLocal =
                LoadBroadcastIndex(cachedMeta);
        }

        const uint64_t taskCount = outerSize_ * tileCount_;
        const uint64_t blockIdx =
            static_cast<uint64_t>(AscendC::GetBlockIdx());
        uint64_t blockNum =
            static_cast<uint64_t>(AscendC::GetBlockNum());
        if (blockNum == 0) {
            blockNum = 1;
        }

        for (uint64_t task = blockIdx;
             task < taskCount; task += blockNum) {
            const uint64_t outer = task / tileCount_;
            const uint64_t tile = task - outer * tileCount_;
            const uint64_t innerStart =
                tile * innerTileWidth_;
            if (innerStart >= innerSize_) {
                continue;
            }
            uint64_t tileWidth =
                innerSize_ - innerStart;
            if (tileWidth > innerTileWidth_) {
                tileWidth = innerTileWidth_;
            }

            ProcessOneTile(
                raw, cachedMeta,
                ptrMode, ptrCached,
                broadcastIndex, broadcastIndexLocal,
                outer, innerStart, tileWidth);
        }
    }

    // ---------------------------------------------------------------------
    // Fully generic single-core scalar fallback.
    // ---------------------------------------------------------------------

    __aicore__ inline bool IsFirstScalarOccurrence(
        uint64_t outer, uint64_t inner,
        uint64_t seed, int32_t group) const {
        for (uint64_t pos = 0; pos < seed; ++pos) {
            if (ReadIndex(outer, pos, inner) == group) {
                return false;
            }
        }
        return true;
    }

    __aicore__ inline float ComputeScalarIndexGroupSum(
        uint64_t outer, uint64_t inner,
        int32_t group, float maxValue) {
        AscendC::LocalTensor<float> work = workBuf_.Get<float>();
        float sum = 0.0F;
        uint32_t count = 0;

        for (uint64_t pos = 0; pos < dimSize_; ++pos) {
            if (ReadIndex(outer, pos, inner) != group) {
                continue;
            }
            work.SetValue(
                count++,
                ReadSrc(FlatOffset(outer, pos, inner)) -
                maxValue);
            if (count == WORK_ELEMS) {
                sum += ExpAndSumWork(count);
                count = 0;
            }
        }
        if (count != 0) {
            sum += ExpAndSumWork(count);
        }
        return sum;
    }

    __aicore__ inline void ProcessScalarIndexGroup(
        uint64_t outer, uint64_t inner, int32_t group) {
        uint64_t count = 0;
        uint64_t singletonPos = 0;
        float maxValue = NEG_FLOAT_MAX;

        for (uint64_t pos = 0; pos < dimSize_; ++pos) {
            if (ReadIndex(outer, pos, inner) != group) {
                continue;
            }
            const float value =
                ReadSrc(FlatOffset(outer, pos, inner));
            if (value > maxValue) {
                maxValue = value;
            }
            singletonPos = pos;
            ++count;
        }

        if (count == 0) {
            return;
        }
        if (count == 1) {
            WriteOut(
                FlatOffset(outer, singletonPos, inner),
                singletonValue_);
            return;
        }

        const float groupSum =
            ComputeScalarIndexGroupSum(
                outer, inner, group, maxValue);
        const float inverse =
            1.0F / (groupSum + eps_);

        AscendC::LocalTensor<float> work = workBuf_.Get<float>();
        uint64_t scan = 0;
        while (scan < dimSize_) {
            const uint64_t chunkStart = scan;
            uint32_t packed = 0;

            while (scan < dimSize_ && packed < WORK_ELEMS) {
                if (ReadIndex(outer, scan, inner) == group) {
                    work.SetValue(
                        packed++,
                        ReadSrc(
                            FlatOffset(outer, scan, inner)) -
                        maxValue);
                }
                ++scan;
            }
            if (packed == 0) {
                continue;
            }

            ExpWorkOnly(packed);
            uint32_t outputPos = 0;
            for (uint64_t pos = chunkStart;
                 pos < scan; ++pos) {
                if (ReadIndex(outer, pos, inner) == group) {
                    WriteOut(
                        FlatOffset(outer, pos, inner),
                        work.GetValue(outputPos++) * inverse);
                }
            }
        }
    }

    __aicore__ inline void ProcessIndexScalar() {
        for (uint64_t outer = 0; outer < outerSize_; ++outer) {
            for (uint64_t inner = 0; inner < innerSize_; ++inner) {
                for (uint64_t seed = 0; seed < dimSize_; ++seed) {
                    const int32_t group =
                        ReadIndex(outer, seed, inner);
                    if (IsFirstScalarOccurrence(
                            outer, inner, seed, group)) {
                        ProcessScalarIndexGroup(
                            outer, inner, group);
                    }
                }
            }
        }
    }

    __aicore__ inline void ProcessScalarContiguous(
        uint64_t outer, uint64_t inner,
        uint64_t start, uint64_t end) {
        const uint64_t count = end - start;
        if (count == 0) {
            return;
        }
        if (count == 1) {
            WriteOut(
                FlatOffset(outer, start, inner),
                singletonValue_);
            return;
        }

        float maxValue = NEG_FLOAT_MAX;
        for (uint64_t pos = start; pos < end; ++pos) {
            const float value =
                ReadSrc(FlatOffset(outer, pos, inner));
            if (value > maxValue) {
                maxValue = value;
            }
        }

        AscendC::LocalTensor<float> work = workBuf_.Get<float>();
        float groupSum = 0.0F;
        uint64_t chunkStart = start;
        while (chunkStart < end) {
            uint64_t remain = end - chunkStart;
            uint32_t n = static_cast<uint32_t>(
                remain > WORK_ELEMS ? WORK_ELEMS : remain);
            for (uint32_t i = 0; i < n; ++i) {
                work.SetValue(
                    i,
                    ReadSrc(
                        FlatOffset(
                            outer, chunkStart + i, inner)) -
                    maxValue);
            }
            groupSum += ExpAndSumWork(n);
            chunkStart += n;
        }

        const float inverse =
            1.0F / (groupSum + eps_);
        chunkStart = start;
        while (chunkStart < end) {
            uint64_t remain = end - chunkStart;
            uint32_t n = static_cast<uint32_t>(
                remain > WORK_ELEMS ? WORK_ELEMS : remain);
            for (uint32_t i = 0; i < n; ++i) {
                work.SetValue(
                    i,
                    ReadSrc(
                        FlatOffset(
                            outer, chunkStart + i, inner)) -
                    maxValue);
            }
            ExpWorkOnly(n);
            for (uint32_t i = 0; i < n; ++i) {
                WriteOut(
                    FlatOffset(
                        outer, chunkStart + i, inner),
                    work.GetValue(i) * inverse);
            }
            chunkStart += n;
        }
    }

    __aicore__ inline void ProcessPtrScalar() {
        const uint64_t groupCount = ptrLength_ - 1U;
        for (uint64_t outer = 0; outer < outerSize_; ++outer) {
            for (uint64_t inner = 0; inner < innerSize_; ++inner) {
                int64_t startRaw =
                    static_cast<int64_t>(
                        ptrGlobal_.GetValue(0));
                for (uint64_t group = 0;
                     group < groupCount; ++group) {
                    int64_t endRaw =
                        static_cast<int64_t>(
                            ptrGlobal_.GetValue(group + 1U));
                    int64_t start = startRaw;
                    int64_t end = endRaw;
                    startRaw = endRaw;

                    if (!ClampPtrRange(start, end)) {
                        continue;
                    }
                    ProcessScalarContiguous(
                        outer, inner,
                        static_cast<uint64_t>(start),
                        static_cast<uint64_t>(end));
                }
            }
        }
    }

    __aicore__ inline void FlushScalarOutput() {
        AscendC::DataCacheCleanAndInvalid<
            StorageType,
            AscendC::CacheLine::ENTIRE_DATA_CACHE,
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
    GET_TILING_DATA_WITH_STRUCT(
        SparseSoftmaxTilingData, tilingData, tiling);

    KernelSparseSoftmax<DT_MODE> op;
    op.Init(src, index, ptr, out, tilingData);
    op.Process();
}
