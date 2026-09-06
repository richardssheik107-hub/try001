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
    // Keep BF16 storage as raw 16-bit payloads to avoid unsupported scalar casts.
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
    static constexpr uint32_t LANE_BATCH_MAX = 32;
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
        blockDim_ = tiling.blockDim == 0 ? 1U : tiling.blockDim;
        innerTileWidth_ = tiling.innerTileWidth;
        tileCount_ = tiling.tileCount;
        eps_ = tiling.eps;
        singletonValue_ = 1.0F / (1.0F + eps_);

        pipe_.InitBuffer(workBuf_, WORK_ELEMS * sizeof(float));

        // Champion implementations repeatedly avoid broad PIPE_ALL barriers.
        // Fetch each scalar/vector dependency once and reuse the pair.
        eventSToV_ = static_cast<int32_t>(
            pipe_.FetchEventID(AscendC::HardEvent::S_V));
        eventVToS_ = static_cast<int32_t>(
            pipe_.FetchEventID(AscendC::HardEvent::V_S));

        if (fastPath_ == 3U) {
            pipe_.InitBuffer(indexBuf_, SPARSE_SOFTMAX_INDEX_BUFFER_BYTES);
            pipe_.InitBuffer(outQueue_, 1, SPARSE_SOFTMAX_AXIS_OUT_QUEUE_BYTES);
            eventMTE2ToS_ = static_cast<int32_t>(
                pipe_.FetchEventID(AscendC::HardEvent::MTE2_S));
            eventSToMTE3_ = static_cast<int32_t>(
                pipe_.FetchEventID(AscendC::HardEvent::S_MTE3));
            eventMTE3ToS_ = static_cast<int32_t>(
                pipe_.FetchEventID(AscendC::HardEvent::MTE3_S));
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
            } else if (fastPath_ == 1U) {
                ProcessTiledDma(false);
            } else if (AscendC::GetBlockIdx() == 0) {
                ProcessIndexScalar();
                FlushScalarOutput();
            }
            return;
        }

        if (fastPath_ == 2U && innerSize_ == 1U) {
            ProcessPtrAxisDma();
        } else if (fastPath_ == 1U) {
            ProcessTiledDma(true);
        } else if (AscendC::GetBlockIdx() == 0) {
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
        AscendC::LocalTensor<StorageType> raw,
        uint64_t offset, float value) {
        raw.SetValue(static_cast<uint32_t>(offset),
                     StorageTraits<DT_MODE>::FromFloatValue(value));
    }

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
        AscendC::DataCopyExtParams params = {
            blockCount, blockLen, srcStride, dstStride, 0};
        AscendC::DataCopyPadExtParams<T> pad = {false, 0, 0, 0};
        AscendC::DataCopyPad<T>(dst, src, params, pad);
    }

    template <typename T>
    __aicore__ inline void CopyLocalToGm(
        const AscendC::GlobalTensor<T>& dst,
        const AscendC::LocalTensor<T>& src,
        uint16_t blockCount, uint32_t blockLen,
        uint32_t srcStride, uint32_t dstStride) {
        AscendC::DataCopyExtParams params = {
            blockCount, blockLen, srcStride, dstStride, 0};
        AscendC::DataCopyPad<T>(dst, src, params);
    }

    // Host-side champions precompute work ownership. Here the only remaining
    // division/modulo is once per core, not once per task.
    __aicore__ inline void GetTaskRange(
        uint64_t totalTasks, uint64_t &begin, uint64_t &count) const {
        const uint64_t blockIdx = static_cast<uint64_t>(AscendC::GetBlockIdx());
        const uint64_t blocks = static_cast<uint64_t>(blockDim_);
        if (blockIdx >= blocks || totalTasks == 0) {
            begin = 0;
            count = 0;
            return;
        }
        const uint64_t base = totalTasks / blocks;
        const uint64_t extra = totalTasks - base * blocks;
        count = base + (blockIdx < extra ? 1ULL : 0ULL);
        begin = blockIdx * base + (blockIdx < extra ? blockIdx : extra);
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

    __aicore__ inline float SumWork(
        AscendC::LocalTensor<float> work, uint32_t count) const {
        return SumWorkRange(work, 0, count);
    }

    __aicore__ inline void ExpWorkOnly(uint32_t count) {
        if (count == 0) {
            return;
        }
        AscendC::LocalTensor<float> work = workBuf_.Get<float>();
        SyncSToV();
        AscendC::Exp(work, work, static_cast<int32_t>(count));
        SyncVToS();
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
        AscendC::Muls(work, work, inverse, static_cast<int32_t>(count));
        SyncVToS();
    }

    // ------------------------------------------------------------------
    // index + inner==1: group ownership. Broadcast index is cached once;
    // sorted labels switch from O(dim^2) discovery to linear run discovery.
    // ------------------------------------------------------------------

    __aicore__ inline bool LoadAxisIndexCache() {
        if (indexLength_ != dimSize_ ||
            dimSize_ * sizeof(int32_t) > SPARSE_SOFTMAX_INDEX_BUFFER_BYTES) {
            return false;
        }
        AscendC::LocalTensor<int32_t> local = indexBuf_.Get<int32_t>();
        CopyGmToLocal<int32_t>(
            local, indexGlobal_, 1,
            static_cast<uint32_t>(dimSize_ * sizeof(int32_t)), 0, 0);
        SyncMTE2ToS();
        axisIndexCached_ = true;
        return true;
    }

    __aicore__ inline int32_t ReadAxisIndex(
        uint64_t outer, uint64_t pos) const {
        if (axisIndexCached_) {
            return indexBuf_.Get<int32_t>().GetValue(static_cast<uint32_t>(pos));
        }
        if (indexLength_ == totalLength_) {
            return indexGlobal_.GetValue(outer * dimSize_ + pos);
        }
        return indexGlobal_.GetValue(pos);
    }

    __aicore__ inline bool IsAxisSorted(uint64_t outer) const {
        if (dimSize_ < 2) {
            return true;
        }
        int32_t prev = ReadAxisIndex(outer, 0);
        for (uint64_t pos = 1; pos < dimSize_; ++pos) {
            const int32_t cur = ReadAxisIndex(outer, pos);
            if (cur < prev) {
                return false;
            }
            prev = cur;
        }
        return true;
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

    __aicore__ inline void WriteAxisContiguousWork(
        uint64_t gmOffset, uint32_t count, float inverse) {
        const uint32_t queueElems =
            SPARSE_SOFTMAX_AXIS_OUT_QUEUE_BYTES / sizeof(StorageType);
        AscendC::LocalTensor<float> work = workBuf_.Get<float>();
        uint32_t start = 0;
        while (start < count) {
            uint32_t n = count - start;
            if (n > queueElems) {
                n = queueElems;
            }
            AscendC::LocalTensor<StorageType> outLocal =
                outQueue_.AllocTensor<StorageType>();
            for (uint32_t i = 0; i < n; ++i) {
                outLocal.SetValue(
                    i, StorageTraits<DT_MODE>::FromFloatValue(
                        work.GetValue(start + i) * inverse));
            }
            SyncSToMTE3();
            outQueue_.EnQue(outLocal);
            outLocal = outQueue_.DeQue<StorageType>();
            AscendC::DataCopyExtParams params = {
                1, n * sizeof(StorageType), 0, 0, 0};
            AscendC::DataCopyPad(outGlobal_[gmOffset + start], outLocal, params);
            outQueue_.FreeTensor(outLocal);
            start += n;
        }
    }

    __aicore__ inline void ProcessAxisSortedRun(
        uint64_t outer, uint64_t start, uint64_t end) {
        const uint64_t count = end - start;
        if (count == 0) {
            return;
        }
        const uint64_t gmBase = outer * dimSize_ + start;
        AscendC::LocalTensor<float> work = workBuf_.Get<float>();

        if (count == 1) {
            work.SetValue(0, 1.0F);
            WriteAxisContiguousWork(gmBase, 1, singletonValue_);
            return;
        }

        float maxValue = NEG_FLOAT_MAX;
        for (uint64_t i = 0; i < count; ++i) {
            const float value = ReadSrc(gmBase + i);
            if (value > maxValue) {
                maxValue = value;
            }
        }

        if (count <= WORK_ELEMS) {
            const uint32_t n = static_cast<uint32_t>(count);
            for (uint32_t i = 0; i < n; ++i) {
                work.SetValue(i, ReadSrc(gmBase + i) - maxValue);
            }
            const float sum = ExpAndSumWork(n);
            WriteAxisContiguousWork(gmBase, n, 1.0F / (sum + eps_));
            return;
        }

        float sum = 0.0F;
        uint64_t offset = 0;
        while (offset < count) {
            uint32_t n = static_cast<uint32_t>(
                count - offset > WORK_ELEMS ? WORK_ELEMS : count - offset);
            for (uint32_t i = 0; i < n; ++i) {
                work.SetValue(i, ReadSrc(gmBase + offset + i) - maxValue);
            }
            sum += ExpAndSumWork(n);
            offset += n;
        }
        const float inverse = 1.0F / (sum + eps_);
        const uint32_t queueElems =
            SPARSE_SOFTMAX_AXIS_OUT_QUEUE_BYTES / sizeof(StorageType);
        uint32_t chunkLimit = WORK_ELEMS < queueElems ? WORK_ELEMS : queueElems;
        offset = 0;
        while (offset < count) {
            uint32_t n = static_cast<uint32_t>(
                count - offset > chunkLimit ? chunkLimit : count - offset);
            for (uint32_t i = 0; i < n; ++i) {
                work.SetValue(i, ReadSrc(gmBase + offset + i) - maxValue);
            }
            ExpWorkOnly(n);
            WriteAxisContiguousWork(gmBase + offset, n, inverse);
            offset += n;
        }
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
                alignedRunBytes += Align32(runLength * sizeof(StorageType));
                inRun = false;
                runLength = 0;
            }
        }
        if (inRun) {
            alignedRunBytes += Align32(runLength * sizeof(StorageType));
        }
    }

    __aicore__ inline void FillAxisPackedWork(
        uint64_t outer, int32_t group, float maxValue) {
        AscendC::LocalTensor<float> work = workBuf_.Get<float>();
        uint32_t packed = 0;
        const uint64_t base = outer * dimSize_;
        for (uint64_t pos = 0; pos < dimSize_; ++pos) {
            if (ReadAxisIndex(outer, pos) == group) {
                work.SetValue(packed++, ReadSrc(base + pos) - maxValue);
            }
        }
    }

    __aicore__ inline void WriteAxisPackedRuns(
        uint64_t outer, int32_t group, float inverse) {
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
            } while (pos < dimSize_ && ReadAxisIndex(outer, pos) == group);
            const uint32_t runLength = static_cast<uint32_t>(pos - runBegin);
            const uint32_t slotElement = slotBytes / sizeof(StorageType);
            for (uint32_t i = 0; i < runLength; ++i) {
                outLocal.SetValue(
                    slotElement + i,
                    StorageTraits<DT_MODE>::FromFloatValue(
                        work.GetValue(packedBegin + i) * inverse));
            }
            slotBytes += static_cast<uint32_t>(
                Align32(static_cast<uint64_t>(runLength) * sizeof(StorageType)));
        }

        SyncSToMTE3();
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
            } while (pos < dimSize_ && ReadAxisIndex(outer, pos) == group);
            const uint32_t runLength = static_cast<uint32_t>(pos - runBegin);
            const uint32_t slotElement = slotBytes / sizeof(StorageType);
            AscendC::DataCopyExtParams params = {
                1, runLength * sizeof(StorageType), 0, 0, 0};
            AscendC::DataCopyPad(
                outGlobal_[base + runBegin], outLocal[slotElement], params);
            slotBytes += static_cast<uint32_t>(
                Align32(static_cast<uint64_t>(runLength) * sizeof(StorageType)));
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
            work.SetValue(count++, ReadSrc(base + pos) - maxValue);
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
            work.SetValue(i, ReadSrc(gmOffset + i) - maxValue);
        }
        ExpWorkOnly(count);
        WriteAxisContiguousWork(gmOffset, count, inverse);
    }

    __aicore__ inline void WriteAxisGroupChunked(
        uint64_t outer, int32_t group,
        float maxValue, float inverse) {
        const uint32_t queueElems =
            SPARSE_SOFTMAX_AXIS_OUT_QUEUE_BYTES / sizeof(StorageType);
        const uint32_t chunkLimit = WORK_ELEMS < queueElems ? WORK_ELEMS : queueElems;
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
            } while (pos < dimSize_ && ReadAxisIndex(outer, pos) == group);
            uint64_t chunkStart = runBegin;
            while (chunkStart < pos) {
                const uint64_t remain = pos - chunkStart;
                const uint32_t count = static_cast<uint32_t>(
                    remain > chunkLimit ? chunkLimit : remain);
                WriteOneAxisChunk(base + chunkStart, count, maxValue, inverse);
                chunkStart += count;
            }
        }
    }

    __aicore__ inline void ProcessAxisGroupMte3(
        uint64_t outer, int32_t group) {
        uint64_t count = 0;
        uint64_t alignedRunBytes = 0;
        float maxValue = NEG_FLOAT_MAX;
        ScanAxisGroupStats(outer, group, count, maxValue, alignedRunBytes);
        if (count == 0) {
            return;
        }

        if (count <= WORK_ELEMS &&
            alignedRunBytes <= SPARSE_SOFTMAX_AXIS_OUT_QUEUE_BYTES) {
            AscendC::LocalTensor<float> work = workBuf_.Get<float>();
            float inverse = singletonValue_;
            if (count == 1) {
                work.SetValue(0, 1.0F);
            } else {
                FillAxisPackedWork(outer, group, maxValue);
                const float sum = ExpAndSumWork(static_cast<uint32_t>(count));
                inverse = 1.0F / (sum + eps_);
            }
            WriteAxisPackedRuns(outer, group, inverse);
            return;
        }

        const float sum = ComputeAxisGroupSum(outer, group, maxValue);
        WriteAxisGroupChunked(outer, group, maxValue, 1.0F / (sum + eps_));
    }

    __aicore__ inline void ProcessIndexAxisMte3() {
        LoadAxisIndexCache();
        const bool broadcast = indexLength_ == dimSize_;
        const bool broadcastSorted = broadcast ? IsAxisSorted(0) : false;
        const uint64_t totalTasks = outerSize_ * dimSize_;
        uint64_t begin = 0;
        uint64_t count = 0;
        GetTaskRange(totalTasks, begin, count);
        if (count == 0) {
            return;
        }

        uint64_t outer = begin / dimSize_;
        uint64_t seed = begin - outer * dimSize_;
        uint64_t sortedOuter = 0;
        bool hasSortedOuter = false;
        bool currentSorted = broadcastSorted;

        for (uint64_t n = 0; n < count; ++n) {
            if (!broadcast && (!hasSortedOuter || sortedOuter != outer)) {
                sortedOuter = outer;
                currentSorted = IsAxisSorted(outer);
                hasSortedOuter = true;
            }
            const int32_t group = ReadAxisIndex(outer, seed);
            if (currentSorted) {
                if (seed == 0 || ReadAxisIndex(outer, seed - 1U) != group) {
                    uint64_t end = seed + 1U;
                    while (end < dimSize_ && ReadAxisIndex(outer, end) == group) {
                        ++end;
                    }
                    ProcessAxisSortedRun(outer, seed, end);
                }
            } else if (IsFirstAxisOccurrence(outer, seed, group)) {
                ProcessAxisGroupMte3(outer, group);
            }

            ++seed;
            if (seed == dimSize_) {
                seed = 0;
                ++outer;
            }
        }
    }

    // ------------------------------------------------------------------
    // ptr + inner==1: independent group owners, cached metadata and one Exp
    // for all small/medium groups that fit the work buffer.
    // ------------------------------------------------------------------

    __aicore__ inline int64_t ReadPtrValue(
        AscendC::LocalTensor<int32_t> cachedPtr,
        bool cached, uint64_t idx) const {
        return cached
            ? static_cast<int64_t>(cachedPtr.GetValue(static_cast<uint32_t>(idx)))
            : static_cast<int64_t>(ptrGlobal_.GetValue(idx));
    }

    __aicore__ inline bool ClampPtrRange(int64_t &start, int64_t &end) const {
        if (start < 0) start = 0;
        if (end < start) return false;
        if (start > static_cast<int64_t>(dimSize_)) return false;
        if (end > static_cast<int64_t>(dimSize_)) end = static_cast<int64_t>(dimSize_);
        return end > start;
    }

    __aicore__ inline bool LoadPtrCache(
        AscendC::LocalTensor<int32_t> cachedPtr) {
        if (ptrLength_ * sizeof(int32_t) > SPARSE_SOFTMAX_INDEX_BUFFER_BYTES) {
            return false;
        }
        CopyGmToLocal<int32_t>(
            cachedPtr, ptrGlobal_, 1,
            static_cast<uint32_t>(ptrLength_ * sizeof(int32_t)), 0, 0);
        SyncMTE2ToS();
        return true;
    }

    __aicore__ inline bool IsPtrFullPartition(
        AscendC::LocalTensor<int32_t> cachedPtr, bool cached) const {
        if (ptrLength_ < 2) {
            return false;
        }
        int64_t prev = ReadPtrValue(cachedPtr, cached, 0);
        if (prev != 0) {
            return false;
        }
        for (uint64_t i = 1; i < ptrLength_; ++i) {
            const int64_t cur = ReadPtrValue(cachedPtr, cached, i);
            if (cur < prev || cur < 0 || cur > static_cast<int64_t>(dimSize_)) {
                return false;
            }
            prev = cur;
        }
        return prev == static_cast<int64_t>(dimSize_);
    }

    __aicore__ inline void NormalizeRawContiguous(
        AscendC::LocalTensor<StorageType> raw, uint64_t count) {
        AscendC::LocalTensor<float> work = workBuf_.Get<float>();
        float maxValue = NEG_FLOAT_MAX;
        for (uint64_t i = 0; i < count; ++i) {
            const float value = ReadRaw(raw, i);
            if (value > maxValue) maxValue = value;
        }

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

        float sum = 0.0F;
        uint64_t start = 0;
        while (start < count) {
            const uint32_t n = static_cast<uint32_t>(
                count - start > WORK_ELEMS ? WORK_ELEMS : count - start);
            for (uint32_t i = 0; i < n; ++i) {
                work.SetValue(i, ReadRaw(raw, start + i) - maxValue);
            }
            sum += ExpAndSumWork(n);
            start += n;
        }
        const float inverse = 1.0F / (sum + eps_);
        start = 0;
        while (start < count) {
            const uint32_t n = static_cast<uint32_t>(
                count - start > WORK_ELEMS ? WORK_ELEMS : count - start);
            for (uint32_t i = 0; i < n; ++i) {
                work.SetValue(i, ReadRaw(raw, start + i) - maxValue);
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
        AscendC::LocalTensor<StorageType> raw = rawBuf_.Get<StorageType>();
        AscendC::LocalTensor<float> work = workBuf_.Get<float>();
        const uint64_t count = end - start;
        const uint64_t gmBase = outer * dimSize_ + start;
        const uint64_t rawCapacity =
            SPARSE_SOFTMAX_RAW_BUFFER_BYTES / sizeof(StorageType);

        if (count == 1) {
            WriteRaw(raw, 0, singletonValue_);
            SyncSToMTE3();
            CopyLocalToGm<StorageType>(
                outGlobal_[gmBase], raw, 1, sizeof(StorageType), 0, 0);
            SyncMTE3ToS();
            return;
        }

        if (count <= rawCapacity) {
            CopyGmToLocal<StorageType>(
                raw, srcGlobal_[gmBase], 1,
                static_cast<uint32_t>(count * sizeof(StorageType)), 0, 0);
            SyncMTE2ToS();
            NormalizeRawContiguous(raw, count);
            SyncSToMTE3();
            CopyLocalToGm<StorageType>(
                outGlobal_[gmBase], raw, 1,
                static_cast<uint32_t>(count * sizeof(StorageType)), 0, 0);
            SyncMTE3ToS();
            return;
        }

        float maxValue = NEG_FLOAT_MAX;
        uint64_t chunkStart = 0;
        while (chunkStart < count) {
            uint64_t chunkCount = count - chunkStart;
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

        float sum = 0.0F;
        chunkStart = 0;
        while (chunkStart < count) {
            uint64_t chunkCount = count - chunkStart;
            if (chunkCount > rawCapacity) chunkCount = rawCapacity;
            CopyGmToLocal<StorageType>(
                raw, srcGlobal_[gmBase + chunkStart], 1,
                static_cast<uint32_t>(chunkCount * sizeof(StorageType)), 0, 0);
            SyncMTE2ToS();
            uint64_t localStart = 0;
            while (localStart < chunkCount) {
                const uint32_t n = static_cast<uint32_t>(
                    chunkCount - localStart > WORK_ELEMS
                        ? WORK_ELEMS : chunkCount - localStart);
                for (uint32_t i = 0; i < n; ++i) {
                    work.SetValue(i, ReadRaw(raw, localStart + i) - maxValue);
                }
                sum += ExpAndSumWork(n);
                localStart += n;
            }
            SyncSToMTE2();
            chunkStart += chunkCount;
        }

        const float inverse = 1.0F / (sum + eps_);
        chunkStart = 0;
        while (chunkStart < count) {
            uint64_t chunkCount = count - chunkStart;
            if (chunkCount > rawCapacity) chunkCount = rawCapacity;
            CopyGmToLocal<StorageType>(
                raw, srcGlobal_[gmBase + chunkStart], 1,
                static_cast<uint32_t>(chunkCount * sizeof(StorageType)), 0, 0);
            SyncMTE2ToS();
            uint64_t localStart = 0;
            while (localStart < chunkCount) {
                const uint32_t n = static_cast<uint32_t>(
                    chunkCount - localStart > WORK_ELEMS
                        ? WORK_ELEMS : chunkCount - localStart);
                for (uint32_t i = 0; i < n; ++i) {
                    work.SetValue(i, ReadRaw(raw, localStart + i) - maxValue);
                }
                ExpWorkOnly(n);
                for (uint32_t i = 0; i < n; ++i) {
                    WriteRaw(raw, localStart + i, work.GetValue(i) * inverse);
                }
                localStart += n;
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
        const bool ptrCached = LoadPtrCache(cachedPtr);
        const uint64_t groupCount = ptrLength_ - 1U;
        const uint64_t totalTasks = outerSize_ * groupCount;
        uint64_t begin = 0;
        uint64_t count = 0;
        GetTaskRange(totalTasks, begin, count);
        if (count == 0) return;

        uint64_t outer = begin / groupCount;
        uint64_t group = begin - outer * groupCount;
        for (uint64_t n = 0; n < count; ++n) {
            int64_t start = ReadPtrValue(cachedPtr, ptrCached, group);
            int64_t end = ReadPtrValue(cachedPtr, ptrCached, group + 1U);
            if (ClampPtrRange(start, end)) {
                ProcessPtrAxisGroup(
                    outer, static_cast<uint64_t>(start), static_cast<uint64_t>(end));
            }
            ++group;
            if (group == groupCount) {
                group = 0;
                ++outer;
            }
        }
    }

    // ------------------------------------------------------------------
    // arbitrary rank: one DMA tile owns [dim, inner-tile]. For broadcast
    // index/ptr partitions, all groups in up to 32 lanes share ONE Exp call.
    // ------------------------------------------------------------------

    __aicore__ inline bool LoadBroadcastIndex(
        AscendC::LocalTensor<int32_t> cachedIndex) {
        if (dimSize_ * sizeof(int32_t) > SPARSE_SOFTMAX_INDEX_BUFFER_BYTES) {
            return false;
        }
        CopyGmToLocal<int32_t>(
            cachedIndex, indexGlobal_, 1,
            static_cast<uint32_t>(dimSize_ * sizeof(int32_t)), 0, 0);
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
                return cachedIndex.GetValue(static_cast<uint32_t>(dim));
            }
            return cachedIndex.GetValue(static_cast<uint32_t>(
                dim * indexRowStride + localInner));
        }
        return ReadIndex(outer, dim, globalInner);
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

    __aicore__ inline bool IsBroadcastSorted(
        AscendC::LocalTensor<int32_t> cachedIndex,
        bool localIndex, uint64_t outer, uint64_t innerStart) const {
        if (dimSize_ < 2) return true;
        int32_t prev = IndexAtTile(
            cachedIndex, localIndex, true, 0,
            outer, 0, innerStart, 0);
        for (uint64_t pos = 1; pos < dimSize_; ++pos) {
            const int32_t cur = IndexAtTile(
                cachedIndex, localIndex, true, 0,
                outer, pos, innerStart, 0);
            if (cur < prev) return false;
            prev = cur;
        }
        return true;
    }

    __aicore__ inline uint32_t ChooseLaneBatch(
        uint64_t remainingLanes, uint32_t valuesPerLane) const {
        if (valuesPerLane == 0) return 1;
        uint32_t byWork = WORK_ELEMS / valuesPerLane;
        if (byWork == 0) byWork = 1;
        uint32_t batch = static_cast<uint32_t>(
            remainingLanes > LANE_BATCH_MAX ? LANE_BATCH_MAX : remainingLanes);
        if (batch > byWork) batch = byWork;
        return batch == 0 ? 1 : batch;
    }

    __aicore__ inline void BroadcastPartitionBatch(
        AscendC::LocalTensor<StorageType> raw,
        AscendC::LocalTensor<int32_t> index,
        bool localIndex, bool sorted,
        uint64_t rowStride, uint64_t outer,
        uint64_t innerStart, uint64_t laneStart,
        uint32_t laneCount) {
        AscendC::LocalTensor<float> work = workBuf_.Get<float>();
        float maxValues[LANE_BATCH_MAX];
        float inverseValues[LANE_BATCH_MAX];

        if (sorted) {
            uint64_t start = 0;
            while (start < dimSize_) {
                const int32_t group = IndexAtTile(
                    index, localIndex, true, 0,
                    outer, start, innerStart, 0);
                uint64_t end = start + 1;
                while (end < dimSize_ &&
                       IndexAtTile(index, localIndex, true, 0,
                                   outer, end, innerStart, 0) == group) {
                    ++end;
                }
                for (uint32_t lane = 0; lane < laneCount; ++lane) {
                    maxValues[lane] = NEG_FLOAT_MAX;
                }
                for (uint64_t pos = start; pos < end; ++pos) {
                    const uint64_t base = pos * rowStride + laneStart;
                    for (uint32_t lane = 0; lane < laneCount; ++lane) {
                        const float v = ReadRaw(raw, base + lane);
                        if (v > maxValues[lane]) maxValues[lane] = v;
                    }
                }
                for (uint64_t pos = start; pos < end; ++pos) {
                    const uint64_t base = pos * rowStride + laneStart;
                    for (uint32_t lane = 0; lane < laneCount; ++lane) {
                        work.SetValue(
                            lane * dimSize_ + pos,
                            ReadRaw(raw, base + lane) - maxValues[lane]);
                    }
                }
                start = end;
            }
        } else {
            for (uint64_t seed = 0; seed < dimSize_; ++seed) {
                const int32_t group = IndexAtTile(
                    index, localIndex, true, 0,
                    outer, seed, innerStart, 0);
                if (!IsFirstTileOccurrence(
                        index, localIndex, true, 0,
                        outer, seed, innerStart, 0, group)) {
                    continue;
                }
                for (uint32_t lane = 0; lane < laneCount; ++lane) {
                    maxValues[lane] = NEG_FLOAT_MAX;
                }
                for (uint64_t pos = 0; pos < dimSize_; ++pos) {
                    if (IndexAtTile(index, localIndex, true, 0,
                                    outer, pos, innerStart, 0) != group) {
                        continue;
                    }
                    const uint64_t base = pos * rowStride + laneStart;
                    for (uint32_t lane = 0; lane < laneCount; ++lane) {
                        const float v = ReadRaw(raw, base + lane);
                        if (v > maxValues[lane]) maxValues[lane] = v;
                    }
                }
                for (uint64_t pos = 0; pos < dimSize_; ++pos) {
                    if (IndexAtTile(index, localIndex, true, 0,
                                    outer, pos, innerStart, 0) != group) {
                        continue;
                    }
                    const uint64_t base = pos * rowStride + laneStart;
                    for (uint32_t lane = 0; lane < laneCount; ++lane) {
                        work.SetValue(
                            lane * dimSize_ + pos,
                            ReadRaw(raw, base + lane) - maxValues[lane]);
                    }
                }
            }
        }

        ExpWorkOnly(static_cast<uint32_t>(laneCount * dimSize_));

        if (sorted) {
            uint64_t start = 0;
            while (start < dimSize_) {
                const int32_t group = IndexAtTile(
                    index, localIndex, true, 0,
                    outer, start, innerStart, 0);
                uint64_t end = start + 1;
                while (end < dimSize_ &&
                       IndexAtTile(index, localIndex, true, 0,
                                   outer, end, innerStart, 0) == group) {
                    ++end;
                }
                for (uint32_t lane = 0; lane < laneCount; ++lane) {
                    float sum = 0.0F;
                    for (uint64_t pos = start; pos < end; ++pos) {
                        sum += work.GetValue(static_cast<uint32_t>(lane * dimSize_ + pos));
                    }
                    inverseValues[lane] = 1.0F / (sum + eps_);
                }
                for (uint64_t pos = start; pos < end; ++pos) {
                    const uint64_t base = pos * rowStride + laneStart;
                    for (uint32_t lane = 0; lane < laneCount; ++lane) {
                        WriteRaw(
                            raw, base + lane,
                            work.GetValue(static_cast<uint32_t>(lane * dimSize_ + pos)) *
                                inverseValues[lane]);
                    }
                }
                start = end;
            }
        } else {
            for (uint64_t seed = 0; seed < dimSize_; ++seed) {
                const int32_t group = IndexAtTile(
                    index, localIndex, true, 0,
                    outer, seed, innerStart, 0);
                if (!IsFirstTileOccurrence(
                        index, localIndex, true, 0,
                        outer, seed, innerStart, 0, group)) {
                    continue;
                }
                for (uint32_t lane = 0; lane < laneCount; ++lane) {
                    float sum = 0.0F;
                    for (uint64_t pos = 0; pos < dimSize_; ++pos) {
                        if (IndexAtTile(index, localIndex, true, 0,
                                        outer, pos, innerStart, 0) == group) {
                            sum += work.GetValue(static_cast<uint32_t>(lane * dimSize_ + pos));
                        }
                    }
                    inverseValues[lane] = 1.0F / (sum + eps_);
                }
                for (uint64_t pos = 0; pos < dimSize_; ++pos) {
                    if (IndexAtTile(index, localIndex, true, 0,
                                    outer, pos, innerStart, 0) != group) {
                        continue;
                    }
                    const uint64_t base = pos * rowStride + laneStart;
                    for (uint32_t lane = 0; lane < laneCount; ++lane) {
                        WriteRaw(
                            raw, base + lane,
                            work.GetValue(static_cast<uint32_t>(lane * dimSize_ + pos)) *
                                inverseValues[lane]);
                    }
                }
            }
        }
    }

    __aicore__ inline void PtrPartitionBatch(
        AscendC::LocalTensor<StorageType> raw,
        AscendC::LocalTensor<int32_t> ptr, bool ptrCached,
        uint64_t rowStride, uint64_t laneStart,
        uint32_t laneCount) {
        AscendC::LocalTensor<float> work = workBuf_.Get<float>();
        float maxValues[LANE_BATCH_MAX];
        float inverseValues[LANE_BATCH_MAX];
        const uint64_t groups = ptrLength_ - 1U;

        for (uint64_t group = 0; group < groups; ++group) {
            const uint64_t start = static_cast<uint64_t>(ReadPtrValue(ptr, ptrCached, group));
            const uint64_t end = static_cast<uint64_t>(ReadPtrValue(ptr, ptrCached, group + 1U));
            if (end <= start) continue;
            for (uint32_t lane = 0; lane < laneCount; ++lane) {
                maxValues[lane] = NEG_FLOAT_MAX;
            }
            for (uint64_t pos = start; pos < end; ++pos) {
                const uint64_t base = pos * rowStride + laneStart;
                for (uint32_t lane = 0; lane < laneCount; ++lane) {
                    const float v = ReadRaw(raw, base + lane);
                    if (v > maxValues[lane]) maxValues[lane] = v;
                }
            }
            for (uint64_t pos = start; pos < end; ++pos) {
                const uint64_t base = pos * rowStride + laneStart;
                for (uint32_t lane = 0; lane < laneCount; ++lane) {
                    work.SetValue(
                        lane * dimSize_ + pos,
                        ReadRaw(raw, base + lane) - maxValues[lane]);
                }
            }
        }

        ExpWorkOnly(static_cast<uint32_t>(laneCount * dimSize_));

        for (uint64_t group = 0; group < groups; ++group) {
            const uint64_t start = static_cast<uint64_t>(ReadPtrValue(ptr, ptrCached, group));
            const uint64_t end = static_cast<uint64_t>(ReadPtrValue(ptr, ptrCached, group + 1U));
            if (end <= start) continue;
            for (uint32_t lane = 0; lane < laneCount; ++lane) {
                float sum = 0.0F;
                for (uint64_t pos = start; pos < end; ++pos) {
                    sum += work.GetValue(static_cast<uint32_t>(lane * dimSize_ + pos));
                }
                inverseValues[lane] = 1.0F / (sum + eps_);
            }
            for (uint64_t pos = start; pos < end; ++pos) {
                const uint64_t base = pos * rowStride + laneStart;
                for (uint32_t lane = 0; lane < laneCount; ++lane) {
                    WriteRaw(
                        raw, base + lane,
                        work.GetValue(static_cast<uint32_t>(lane * dimSize_ + pos)) *
                            inverseValues[lane]);
                }
            }
        }
    }

    __aicore__ inline void SoftmaxBroadcastGroupBatch(
        AscendC::LocalTensor<StorageType> raw,
        AscendC::LocalTensor<int32_t> index,
        bool localIndex, uint64_t rowStride,
        uint64_t outer, uint64_t innerStart,
        uint64_t laneStart, uint32_t laneCount,
        int32_t group, uint32_t groupCount,
        uint64_t singletonPos) {
        if (groupCount == 0) return;
        if (groupCount == 1) {
            for (uint32_t lane = 0; lane < laneCount; ++lane) {
                WriteRaw(raw, singletonPos * rowStride + laneStart + lane,
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
            if (IndexAtTile(index, localIndex, true, 0,
                            outer, pos, innerStart, 0) != group) continue;
            const uint64_t base = pos * rowStride + laneStart;
            for (uint32_t lane = 0; lane < laneCount; ++lane) {
                const float v = ReadRaw(raw, base + lane);
                if (v > maxValues[lane]) maxValues[lane] = v;
            }
        }
        AscendC::LocalTensor<float> work = workBuf_.Get<float>();
        uint32_t packed = 0;
        for (uint64_t pos = 0; pos < dimSize_; ++pos) {
            if (IndexAtTile(index, localIndex, true, 0,
                            outer, pos, innerStart, 0) != group) continue;
            const uint64_t base = pos * rowStride + laneStart;
            for (uint32_t lane = 0; lane < laneCount; ++lane) {
                work.SetValue(lane * groupCount + packed,
                              ReadRaw(raw, base + lane) - maxValues[lane]);
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
            if (IndexAtTile(index, localIndex, true, 0,
                            outer, pos, innerStart, 0) != group) continue;
            const uint64_t base = pos * rowStride + laneStart;
            for (uint32_t lane = 0; lane < laneCount; ++lane) {
                WriteRaw(raw, base + lane,
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
        const uint32_t groupCount = static_cast<uint32_t>(end - start);
        if (groupCount == 0) return;
        if (groupCount == 1) {
            for (uint32_t lane = 0; lane < laneCount; ++lane) {
                WriteRaw(raw, start * rowStride + laneStart + lane,
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
            const uint64_t base = pos * rowStride + laneStart;
            for (uint32_t lane = 0; lane < laneCount; ++lane) {
                const float v = ReadRaw(raw, base + lane);
                if (v > maxValues[lane]) maxValues[lane] = v;
            }
        }
        AscendC::LocalTensor<float> work = workBuf_.Get<float>();
        for (uint32_t lane = 0; lane < laneCount; ++lane) {
            for (uint32_t i = 0; i < groupCount; ++i) {
                work.SetValue(
                    lane * groupCount + i,
                    ReadRaw(raw, (start + i) * rowStride + laneStart + lane) -
                        maxValues[lane]);
            }
        }
        ExpWorkOnly(laneCount * groupCount);
        for (uint32_t lane = 0; lane < laneCount; ++lane) {
            inverseValues[lane] = 1.0F /
                (SumWorkRange(work, lane * groupCount, groupCount) + eps_);
        }
        for (uint32_t i = 0; i < groupCount; ++i) {
            const uint64_t base = (start + i) * rowStride + laneStart;
            for (uint32_t lane = 0; lane < laneCount; ++lane) {
                WriteRaw(raw, base + lane,
                         work.GetValue(lane * groupCount + i) *
                             inverseValues[lane]);
            }
        }
    }

    __aicore__ inline uint32_t CountBroadcastGroup(
        AscendC::LocalTensor<int32_t> index,
        bool localIndex, uint64_t outer,
        uint64_t innerStart, int32_t group,
        uint64_t &singletonPos) const {
        uint32_t count = 0;
        singletonPos = 0;
        for (uint64_t pos = 0; pos < dimSize_; ++pos) {
            if (IndexAtTile(index, localIndex, true, 0,
                            outer, pos, innerStart, 0) == group) {
                singletonPos = pos;
                ++count;
            }
        }
        return count;
    }

    __aicore__ inline void ProcessBroadcastIndexTile(
        AscendC::LocalTensor<StorageType> raw,
        AscendC::LocalTensor<int32_t> index,
        bool localIndex, uint64_t rowStride,
        uint64_t outer, uint64_t innerStart,
        uint64_t tileWidth) {
        // Coarse-grain champion principle: process the largest safe unit.
        // When a complete dim partition fits, all groups share one Exp call.
        if (dimSize_ <= WORK_ELEMS) {
            const bool sorted = IsBroadcastSorted(
                index, localIndex, outer, innerStart);
            uint64_t laneStart = 0;
            while (laneStart < tileWidth) {
                const uint32_t batch = ChooseLaneBatch(
                    tileWidth - laneStart, static_cast<uint32_t>(dimSize_));
                BroadcastPartitionBatch(
                    raw, index, localIndex, sorted,
                    rowStride, outer, innerStart,
                    laneStart, batch);
                laneStart += batch;
            }
            return;
        }

        for (uint64_t seed = 0; seed < dimSize_; ++seed) {
            const int32_t group = IndexAtTile(
                index, localIndex, true, 0,
                outer, seed, innerStart, 0);
            if (!IsFirstTileOccurrence(
                    index, localIndex, true, 0,
                    outer, seed, innerStart, 0, group)) continue;
            uint64_t singletonPos = 0;
            const uint32_t groupCount = CountBroadcastGroup(
                index, localIndex, outer, innerStart, group, singletonPos);
            uint64_t laneStart = 0;
            while (laneStart < tileWidth) {
                const uint32_t batch = ChooseLaneBatch(
                    tileWidth - laneStart, groupCount);
                SoftmaxBroadcastGroupBatch(
                    raw, index, localIndex, rowStride,
                    outer, innerStart, laneStart, batch,
                    group, groupCount, singletonPos);
                laneStart += batch;
            }
        }
    }

    __aicore__ inline void ProcessFullIndexTile(
        AscendC::LocalTensor<StorageType> raw,
        AscendC::LocalTensor<int32_t> index,
        bool localIndex, uint64_t rowStride,
        uint64_t indexRowStride, uint64_t outer,
        uint64_t innerStart, uint64_t tileWidth) {
        // Full-shape index may differ per lane; retain the generic owner path.
        for (uint64_t lane = 0; lane < tileWidth; ++lane) {
            const uint64_t globalInner = innerStart + lane;
            for (uint64_t seed = 0; seed < dimSize_; ++seed) {
                const int32_t group = IndexAtTile(
                    index, localIndex, false, indexRowStride,
                    outer, seed, globalInner, lane);
                if (!IsFirstTileOccurrence(
                        index, localIndex, false, indexRowStride,
                        outer, seed, globalInner, lane, group)) continue;

                float maxValue = NEG_FLOAT_MAX;
                uint32_t count = 0;
                uint64_t singletonPos = 0;
                for (uint64_t pos = 0; pos < dimSize_; ++pos) {
                    if (IndexAtTile(index, localIndex, false, indexRowStride,
                                    outer, pos, globalInner, lane) != group) continue;
                    const float v = ReadRaw(raw, pos * rowStride + lane);
                    if (v > maxValue) maxValue = v;
                    singletonPos = pos;
                    ++count;
                }
                if (count == 1) {
                    WriteRaw(raw, singletonPos * rowStride + lane, singletonValue_);
                    continue;
                }
                AscendC::LocalTensor<float> work = workBuf_.Get<float>();
                uint32_t packed = 0;
                for (uint64_t pos = 0; pos < dimSize_; ++pos) {
                    if (IndexAtTile(index, localIndex, false, indexRowStride,
                                    outer, pos, globalInner, lane) == group) {
                        work.SetValue(packed++,
                            ReadRaw(raw, pos * rowStride + lane) - maxValue);
                    }
                }
                const float sum = ExpAndSumWork(packed);
                const float inverse = 1.0F / (sum + eps_);
                packed = 0;
                for (uint64_t pos = 0; pos < dimSize_; ++pos) {
                    if (IndexAtTile(index, localIndex, false, indexRowStride,
                                    outer, pos, globalInner, lane) == group) {
                        WriteRaw(raw, pos * rowStride + lane,
                                 work.GetValue(packed++) * inverse);
                    }
                }
            }
        }
    }

    __aicore__ inline void ProcessPtrTile(
        AscendC::LocalTensor<StorageType> raw,
        AscendC::LocalTensor<int32_t> ptr,
        bool ptrCached, bool fullPartition,
        uint64_t rowStride, uint64_t tileWidth) {
        if (fullPartition && dimSize_ <= WORK_ELEMS) {
            uint64_t laneStart = 0;
            while (laneStart < tileWidth) {
                const uint32_t batch = ChooseLaneBatch(
                    tileWidth - laneStart, static_cast<uint32_t>(dimSize_));
                PtrPartitionBatch(
                    raw, ptr, ptrCached, rowStride,
                    laneStart, batch);
                laneStart += batch;
            }
            return;
        }

        const uint64_t groups = ptrLength_ - 1U;
        int64_t startRaw = ReadPtrValue(ptr, ptrCached, 0);
        for (uint64_t group = 0; group < groups; ++group) {
            int64_t endRaw = ReadPtrValue(ptr, ptrCached, group + 1U);
            int64_t start = startRaw;
            int64_t end = endRaw;
            startRaw = endRaw;
            if (!ClampPtrRange(start, end)) continue;
            const uint32_t groupCount = static_cast<uint32_t>(end - start);
            uint64_t laneStart = 0;
            while (laneStart < tileWidth) {
                const uint32_t batch = ChooseLaneBatch(
                    tileWidth - laneStart, groupCount);
                SoftmaxContiguousBatch(
                    raw, rowStride, laneStart, batch,
                    static_cast<uint64_t>(start), static_cast<uint64_t>(end));
                laneStart += batch;
            }
        }
    }

    __aicore__ inline void ProcessOneTile(
        AscendC::LocalTensor<StorageType> raw,
        AscendC::LocalTensor<int32_t> meta,
        bool ptrMode, bool ptrCached, bool ptrFullPartition,
        bool broadcastIndex, bool broadcastIndexLocal,
        uint64_t outer, uint64_t innerStart, uint64_t tileWidth) {
        const uint64_t rowBytes = tileWidth * sizeof(StorageType);
        const uint64_t rowStrideBytes = Align32(rowBytes);
        const uint64_t rowStride = rowStrideBytes / sizeof(StorageType);
        const uint64_t gmBase = OuterBase(outer) + innerStart;

        CopyGmToLocal<StorageType>(
            raw, srcGlobal_[gmBase], static_cast<uint16_t>(dimSize_),
            static_cast<uint32_t>(rowBytes),
            static_cast<uint32_t>((innerSize_ - tileWidth) * sizeof(StorageType)),
            0);

        bool fullIndexLocal = false;
        uint64_t indexRowStride = 0;
        if (!ptrMode && !broadcastIndex) {
            const uint64_t indexRowBytes = tileWidth * sizeof(int32_t);
            const uint64_t indexRowStrideBytes = Align32(indexRowBytes);
            if (dimSize_ * indexRowStrideBytes <= SPARSE_SOFTMAX_INDEX_BUFFER_BYTES) {
                indexRowStride = indexRowStrideBytes / sizeof(int32_t);
                CopyGmToLocal<int32_t>(
                    meta, indexGlobal_[gmBase], static_cast<uint16_t>(dimSize_),
                    static_cast<uint32_t>(indexRowBytes),
                    static_cast<uint32_t>((innerSize_ - tileWidth) * sizeof(int32_t)),
                    0);
                fullIndexLocal = true;
            }
        }
        SyncMTE2ToS();

        if (ptrMode) {
            ProcessPtrTile(raw, meta, ptrCached, ptrFullPartition,
                           rowStride, tileWidth);
        } else if (broadcastIndex) {
            ProcessBroadcastIndexTile(
                raw, meta, broadcastIndexLocal, rowStride,
                outer, innerStart, tileWidth);
        } else {
            ProcessFullIndexTile(
                raw, meta, fullIndexLocal, rowStride, indexRowStride,
                outer, innerStart, tileWidth);
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
        AscendC::LocalTensor<int32_t> meta = indexBuf_.Get<int32_t>();
        const bool broadcastIndex = !ptrMode && indexLength_ != totalLength_;
        bool broadcastIndexLocal = false;
        bool ptrCached = false;
        bool ptrFullPartition = false;
        if (ptrMode) {
            ptrCached = LoadPtrCache(meta);
            ptrFullPartition = IsPtrFullPartition(meta, ptrCached);
        } else if (broadcastIndex) {
            broadcastIndexLocal = LoadBroadcastIndex(meta);
        }

        const uint64_t totalTasks = outerSize_ * tileCount_;
        uint64_t begin = 0;
        uint64_t count = 0;
        GetTaskRange(totalTasks, begin, count);
        if (count == 0) return;
        uint64_t outer = begin / tileCount_;
        uint64_t tile = begin - outer * tileCount_;

        for (uint64_t n = 0; n < count; ++n) {
            const uint64_t innerStart = tile * innerTileWidth_;
            if (innerStart < innerSize_) {
                uint64_t tileWidth = innerSize_ - innerStart;
                if (tileWidth > innerTileWidth_) tileWidth = innerTileWidth_;
                ProcessOneTile(
                    raw, meta, ptrMode, ptrCached, ptrFullPartition,
                    broadcastIndex, broadcastIndexLocal,
                    outer, innerStart, tileWidth);
            }
            ++tile;
            if (tile == tileCount_) {
                tile = 0;
                ++outer;
            }
        }
    }

    // ------------------------------------------------------------------
    // Fully generic single-core correctness fallback.
    // ------------------------------------------------------------------

    __aicore__ inline bool IsFirstScalarOccurrence(
        uint64_t outer, uint64_t inner,
        uint64_t seed, int32_t group) const {
        for (uint64_t pos = 0; pos < seed; ++pos) {
            if (ReadIndex(outer, pos, inner) == group) return false;
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
        uint64_t count = 0;
        uint64_t singletonPos = 0;
        float maxValue = NEG_FLOAT_MAX;
        for (uint64_t pos = 0; pos < dimSize_; ++pos) {
            if (ReadIndex(outer, pos, inner) != group) continue;
            const float v = ReadSrc(FlatOffset(outer, pos, inner));
            if (v > maxValue) maxValue = v;
            singletonPos = pos;
            ++count;
        }
        if (count == 0) return;
        if (count == 1) {
            WriteOut(FlatOffset(outer, singletonPos, inner), singletonValue_);
            return;
        }
        const float sum = ComputeScalarIndexGroupSum(outer, inner, group, maxValue);
        const float inverse = 1.0F / (sum + eps_);
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
        uint64_t outer, uint64_t inner,
        uint64_t start, uint64_t end) {
        const uint64_t count = end - start;
        if (count == 0) return;
        if (count == 1) {
            WriteOut(FlatOffset(outer, start, inner), singletonValue_);
            return;
        }
        float maxValue = NEG_FLOAT_MAX;
        for (uint64_t pos = start; pos < end; ++pos) {
            const float v = ReadSrc(FlatOffset(outer, pos, inner));
            if (v > maxValue) maxValue = v;
        }
        AscendC::LocalTensor<float> work = workBuf_.Get<float>();
        float sum = 0.0F;
        uint64_t chunkStart = start;
        while (chunkStart < end) {
            const uint32_t n = static_cast<uint32_t>(
                end - chunkStart > WORK_ELEMS ? WORK_ELEMS : end - chunkStart);
            for (uint32_t i = 0; i < n; ++i) {
                work.SetValue(i,
                    ReadSrc(FlatOffset(outer, chunkStart + i, inner)) - maxValue);
            }
            sum += ExpAndSumWork(n);
            chunkStart += n;
        }
        const float inverse = 1.0F / (sum + eps_);
        chunkStart = start;
        while (chunkStart < end) {
            const uint32_t n = static_cast<uint32_t>(
                end - chunkStart > WORK_ELEMS ? WORK_ELEMS : end - chunkStart);
            for (uint32_t i = 0; i < n; ++i) {
                work.SetValue(i,
                    ReadSrc(FlatOffset(outer, chunkStart + i, inner)) - maxValue);
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
        const uint64_t groups = ptrLength_ - 1U;
        for (uint64_t outer = 0; outer < outerSize_; ++outer) {
            for (uint64_t inner = 0; inner < innerSize_; ++inner) {
                int64_t startRaw = static_cast<int64_t>(ptrGlobal_.GetValue(0));
                for (uint64_t group = 0; group < groups; ++group) {
                    int64_t endRaw = static_cast<int64_t>(ptrGlobal_.GetValue(group + 1U));
                    int64_t start = startRaw;
                    int64_t end = endRaw;
                    startRaw = endRaw;
                    if (ClampPtrRange(start, end)) {
                        ProcessScalarContiguous(
                            outer, inner,
                            static_cast<uint64_t>(start),
                            static_cast<uint64_t>(end));
                    }
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
    mutable AscendC::TBuf<AscendC::TPosition::VECIN> indexBuf_;
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
    bool axisIndexCached_ = false;

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
    GET_TILING_DATA_WITH_STRUCT(SparseSoftmaxTilingData, tilingData, tiling);
    KernelSparseSoftmax<DT_MODE> op;
    op.Init(src, index, ptr, out, tilingData);
    op.Process();
}
