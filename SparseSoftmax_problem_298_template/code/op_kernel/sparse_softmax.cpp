#include "kernel_operator.h"

#include "sparse_softmax_tiling.h"
#include "tiling_key_sparse_softmax.h"

template <typename T>
struct StorageTraits {
    using StorageType = T;

    __aicore__ static inline float ToFloatValue(StorageType value) {
        return static_cast<float>(value);
    }

    __aicore__ static inline StorageType FromFloatValue(float value) {
        return static_cast<StorageType>(value);
    }
};

template <>
struct StorageTraits<bfloat16_t> {
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

        const uint32_t lsb = (bits.u >> 16) & 1U;
        const uint32_t rounded = bits.u + 0x7FFFU + lsb;
        return static_cast<uint16_t>(rounded >> 16);
    }
};

template <typename T>
class KernelSparseSoftmax {
public:
    using StorageType = typename StorageTraits<T>::StorageType;

    __aicore__ inline KernelSparseSoftmax() {}

    __aicore__ inline void Init(GM_ADDR src, GM_ADDR index, GM_ADDR ptr, GM_ADDR out,
                                uint64_t totalLength, uint64_t outerSize,
                                uint64_t dimSize, uint64_t innerSize,
                                uint64_t indexLength, uint64_t ptrLength,
                                uint32_t mode, float eps) {
        srcGlobal_.SetGlobalBuffer((__gm__ StorageType *)src);
        outGlobal_.SetGlobalBuffer((__gm__ StorageType *)out);
        if (index != nullptr) {
            indexGlobal_.SetGlobalBuffer((__gm__ int64_t *)index);
        }
        if (ptr != nullptr) {
            ptrGlobal_.SetGlobalBuffer((__gm__ int64_t *)ptr);
        }

        totalLength_ = totalLength;
        outerSize_ = outerSize;
        dimSize_ = dimSize;
        innerSize_ = innerSize;
        indexLength_ = indexLength;
        ptrLength_ = ptrLength;
        mode_ = mode;
        eps_ = eps;

        pipe_.InitBuffer(expInputBuf_, 32);
        pipe_.InitBuffer(expOutputBuf_, 32);
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
    __aicore__ inline uint64_t FlatOffset(uint64_t outer, uint64_t dim,
                                          uint64_t inner) const {
        return (outer * dimSize_ + dim) * innerSize_ + inner;
    }

    __aicore__ inline int64_t ReadIndex(uint64_t outer, uint64_t dim,
                                        uint64_t inner) const {
        if (indexLength_ == totalLength_) {
            return indexGlobal_.GetValue(FlatOffset(outer, dim, inner));
        }
        return indexGlobal_.GetValue(dim);
    }

    __aicore__ inline float ReadSrc(uint64_t offset) const {
        return StorageTraits<T>::ToFloatValue(srcGlobal_.GetValue(offset));
    }

    __aicore__ inline void WriteOut(uint64_t offset, float value) {
        outGlobal_.SetValue(offset, StorageTraits<T>::FromFloatValue(value));
    }

    __aicore__ inline float ExpScalar(float value) {
        AscendC::LocalTensor<float> in = expInputBuf_.Get<float>();
        AscendC::LocalTensor<float> out = expOutputBuf_.Get<float>();

        in.SetValue(0, value);
        auto eventSToV = pipe_.FetchEventID(AscendC::HardEvent::S_V);
        AscendC::SetFlag<AscendC::HardEvent::S_V>(eventSToV);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(eventSToV);

        AscendC::Exp(out, in, 1);
        auto eventVToS = pipe_.FetchEventID(AscendC::HardEvent::V_S);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(eventVToS);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(eventVToS);

        return out.GetValue(0);
    }

    __aicore__ inline void ProcessIndex() {
        for (uint64_t outer = 0; outer < outerSize_; ++outer) {
            for (uint64_t inner = 0; inner < innerSize_; ++inner) {
                for (uint64_t dim = 0; dim < dimSize_; ++dim) {
                    const int64_t group = ReadIndex(outer, dim, inner);

                    bool seen = false;
                    for (uint64_t prev = 0; prev < dim; ++prev) {
                        if (ReadIndex(outer, prev, inner) == group) {
                            seen = true;
                            break;
                        }
                    }
                    if (seen) {
                        continue;
                    }

                    float groupMax = -3.402823466e+38F;
                    for (uint64_t k = 0; k < dimSize_; ++k) {
                        if (ReadIndex(outer, k, inner) != group) {
                            continue;
                        }
                        const float value = ReadSrc(FlatOffset(outer, k, inner));
                        if (value > groupMax) {
                            groupMax = value;
                        }
                    }

                    float groupSum = 0.0f;
                    for (uint64_t k = 0; k < dimSize_; ++k) {
                        if (ReadIndex(outer, k, inner) != group) {
                            continue;
                        }
                        const float value = ReadSrc(FlatOffset(outer, k, inner));
                        groupSum += ExpScalar(value - groupMax);
                    }

                    const float denom = groupSum + eps_;
                    for (uint64_t k = 0; k < dimSize_; ++k) {
                        if (ReadIndex(outer, k, inner) != group) {
                            continue;
                        }
                        const uint64_t offset = FlatOffset(outer, k, inner);
                        const float value = ReadSrc(offset);
                        WriteOut(offset, ExpScalar(value - groupMax) / denom);
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
                for (uint64_t group = 0; group < groupCount; ++group) {
                    int64_t startRaw = ptrGlobal_.GetValue(group);
                    int64_t endRaw = ptrGlobal_.GetValue(group + 1);

                    if (startRaw < 0) {
                        startRaw = 0;
                    }
                    if (endRaw < startRaw) {
                        continue;
                    }
                    if (startRaw > static_cast<int64_t>(dimSize_)) {
                        continue;
                    }
                    if (endRaw > static_cast<int64_t>(dimSize_)) {
                        endRaw = static_cast<int64_t>(dimSize_);
                    }
                    if (endRaw <= startRaw) {
                        continue;
                    }

                    const uint64_t start = static_cast<uint64_t>(startRaw);
                    const uint64_t end = static_cast<uint64_t>(endRaw);

                    float groupMax = -3.402823466e+38F;
                    for (uint64_t k = start; k < end; ++k) {
                        const float value = ReadSrc(FlatOffset(outer, k, inner));
                        if (value > groupMax) {
                            groupMax = value;
                        }
                    }

                    float groupSum = 0.0f;
                    for (uint64_t k = start; k < end; ++k) {
                        const float value = ReadSrc(FlatOffset(outer, k, inner));
                        groupSum += ExpScalar(value - groupMax);
                    }

                    const float denom = groupSum + eps_;
                    for (uint64_t k = start; k < end; ++k) {
                        const uint64_t offset = FlatOffset(outer, k, inner);
                        const float value = ReadSrc(offset);
                        WriteOut(offset, ExpScalar(value - groupMax) / denom);
                    }
                }
            }
        }
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> expInputBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> expOutputBuf_;
    AscendC::GlobalTensor<StorageType> srcGlobal_;
    AscendC::GlobalTensor<int64_t> indexGlobal_;
    AscendC::GlobalTensor<int64_t> ptrGlobal_;
    AscendC::GlobalTensor<StorageType> outGlobal_;

    uint64_t totalLength_ = 0;
    uint64_t outerSize_ = 0;
    uint64_t dimSize_ = 0;
    uint64_t innerSize_ = 0;
    uint64_t indexLength_ = 0;
    uint64_t ptrLength_ = 0;
    uint32_t mode_ = 0;
    float eps_ = 1e-16f;
};

template <typename DT_SRC>
__global__ __aicore__ void sparse_softmax(GM_ADDR src, GM_ADDR index, GM_ADDR ptr,
                                          GM_ADDR out, GM_ADDR workspace,
                                          GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(SparseSoftmaxTilingData);
    GET_TILING_DATA_WITH_STRUCT(SparseSoftmaxTilingData, tilingData, tiling);

    KernelSparseSoftmax<DT_SRC> op;
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
