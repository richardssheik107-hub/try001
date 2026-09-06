/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_CUSTOM_H
#define OPS_HCCL_CUSTOM_H

#include <cstdint>
#include <cstring>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>
#include <hccl/hccl_types.h>
#include <hccl/hccl_res.h>

#include "common.h"

typedef struct {
    void *addr;
    uint64_t size;
} CommBuffer;

struct CcuKernelArgBase {
    ChannelHandle channels[MAX_RANK_SIZE]{};
    uint32_t channelCount{0};
};

enum class FinalTopology : uint32_t {
    TOPOLOGY_4X1 = 0,
    TOPOLOGY_8X4 = 1,
    TOPOLOGY_2X8 = 2,
};

enum class OwnerRsagPhase : uint64_t {
    PARTIAL_REDUCE = 0,
    COMBINE = 1,
    BROADCAST = 2,
};

enum class Algorithm2x8 : uint32_t {
    OWNER_RSAG = 0,
    BFLY16_CLOS_4STEP = 1,
    CROSS_LANE_512M = 2,
    CROSS_LANE_400M4B = 3,
    P4_DIRECT_RSAG_512K = 4,
    P4_ROTATE3_512M = 5,
    P4_ROTATE3_400M4B = 6,
    P12_DIRECT_RSAG_512K = 7,
    P12_HIER8_512M = 8,
    P12_HIER8_400M4B = 9,
    P12_ALLPAIRS_512M = 10,
    P12_ALLPAIRS_400M4B = 11,
    P4_DIRECT_SLOT3_512M = 12,
    P4_DIRECT_SLOT3_400M4B = 13,
    P4_ROTATE3_UNIQUE_NOTIFY_512M = 14,
    P4_ROTATE3_UNIQUE_NOTIFY_400M4B = 15,
    P12_ALLPAIRS_BALANCED_512M = 16,
    P12_ALLPAIRS_BALANCED_400M4B = 17,
    P4_ROTATE3_DIRECT_OUTPUT_512M = 18,
    P4_ROTATE3_DIRECT_OUTPUT_400M4B = 19,
    CROSS_LANE_DIRECT_OUTPUT_400M4B = 24,
    P12_DIRECT_RSAG_PRESYNC_ONCE_512K = 26,
    P12_ALLPAIRS_BALANCED_PRESYNC_ONCE_512M = 27,
    P12_ALLPAIRS_BALANCED_PRESYNC_ONCE_400M4B = 28,
    P4_PULL_R3_512M = 29,
    P4_PULL_R3_400M4B = 30,
    P12_DIRECT_RSAG_EARLY_INIT_512K = 31,
    P12_ALLPAIRS_EARLY_INIT_512M = 32,
    P12_ALLPAIRS_1SEG_400M4B = 33,
    P16_MESH_LANE_512M = 34,
    P16_MESH_LANE_400M4B = 35,
    P4_OUTPUT_ACCUMULATE_512M = 36,
    P4_OUTPUT_ACCUMULATE_400M4B = 37,
    P12_CLOS_ACCUMULATE_512M = 38,
    P12_CLOS_ACCUMULATE_400M4B = 39,
    P4_OUTPUT_ACCUMULATE_CHAIN_512M = 40,
    P4_OUTPUT_ACCUMULATE_CHAIN_400M4B = 41,
    V101_DIRECT_MESH_512K = 42,
    P4_OUTPUT_ACCUMULATE_WAVE2_512M = 43,
    P4_OUTPUT_ACCUMULATE_WAVE2_400M4B = 44,
    P12_ALLPAIRS_1SEG_OUTPUT_TREE_512M = 45,
    P12_ALLPAIRS_1SEG_OUTPUT_TREE_400M4B = 46,
    V109_DIRECT_MESH_R16_512K = 47,
};

constexpr uint32_t V101_MAX_RANK_SIZE = 16U;
constexpr uint32_t V101_INVALID_CHANNEL_INDEX = 0xFFFFFFFFU;

enum class V101DirectStage : uint64_t {
    PARTIAL = 0,
    MERGE = 1,
    ALLGATHER = 2,
};

enum class ButterflyPhase : uint64_t {
    PHASE_0 = 0,
    PHASE_1 = 1,
    PHASE_2 = 2,
    PHASE_3 = 3,
};

enum class CrossLanePhase : uint64_t {
    REDUCE_SCATTER = 0,
    LOCAL_REDUCE = 1,
    BROADCAST = 2,
    MESH_LANE_0 = 3,
    MESH_LANE_1 = 4,
    MESH_LANE_2 = 5,
    MESH_LANE_3 = 6,
};

enum class DirectRsagPhase : uint64_t {
    SOURCE_PUSH = 0,
    LOCAL_REDUCE = 1,
    BROADCAST = 2,
    FINAL_MERGE = 3,
};

enum class P12HierPhase : uint64_t {
    LOCAL_DATA_0 = 0,
    LOCAL_RELEASE_0 = 1,
    LOCAL_DATA_1 = 2,
    LOCAL_RELEASE_1 = 3,
    LOCAL_DATA_2 = 4,
    LOCAL_RELEASE_2 = 5,
    LOCAL_DATA_3 = 6,
    LOCAL_RELEASE_3 = 7,
    LOCAL_DATA_4 = 8,
    LOCAL_RELEASE_4 = 9,
    LOCAL_DATA_5 = 10,
    LOCAL_RELEASE_5 = 11,
    LOCAL_DATA_6 = 12,
    LOCAL_RELEASE_6 = 13,
    LOCAL_REDUCE = 14,
    CROSS_REDUCE = 15,
    LOCAL_ALLGATHER = 16,
};

constexpr uint32_t KERNEL_ROLE_PRIMARY = 1U;
constexpr uint32_t KERNEL_ROLE_BFLY_PHASE_SHIFT = 8U;
constexpr uint32_t KERNEL_ROLE_LANE_COUNT_SHIFT = 16U;
constexpr uint32_t CROSS_LANE_SAME_SLOT_COUNT = 7U;
constexpr uint32_t CROSS_LANE_LEGACY_MAX_COUNT = 3U;
constexpr uint32_t CROSS_LANE_MAX_COUNT = 4U;
constexpr uint32_t CROSS_LANE_SHARD_COUNT = 4U;
constexpr uint32_t CROSS_LANE_MODE_PARALLEL_LOCAL = 2U;
constexpr uint32_t MESH_LANE_MAX_COUNT = 5U;
constexpr uint32_t P4_ROTATE_SHARD_COUNT = 3U;
constexpr uint32_t HIER_GLOBAL_SHARD_COUNT = 8U;
constexpr uint32_t HIER_MAX_OWNED_SHARD_COUNT = 2U;
constexpr uint32_t HIER_LANE_SLOT_COUNT = 2U;
constexpr uint32_t HIER_TOTAL_SLOT_COUNT = HIER_MAX_OWNED_SHARD_COUNT * HIER_LANE_SLOT_COUNT;
constexpr uint32_t HIER_LOCAL_PHASE_COUNT = 7U;
constexpr uint32_t HIER_LOCAL_TASK_PHASE_COUNT = 2U * HIER_LOCAL_PHASE_COUNT;
constexpr uint32_t P12_ALLPAIRS_SEGMENT_COUNT = 2U;
constexpr uint32_t P12_ALLPAIRS_A_RANK_COUNT = 8U;
constexpr uint32_t P12_ALLPAIRS_B_RANK_COUNT = 4U;

struct CcuKernelArgOwnerRsag : public CcuKernelArgBase {
    uint32_t rankId{0};
    uint32_t rankSize{0};
    uint32_t kernelIndex{0};
    uint32_t kernelCount{0};
    uint32_t actualDieId{0};
    uint32_t isPrimary{0};
    uint32_t peerRanks[MAX_RANK_SIZE]{};
    uint32_t peerIsLocal[MAX_RANK_SIZE]{};
    HcclDataType dataType{HCCL_DATA_TYPE_RESERVED};
    HcclReduceOp reduceType{HCCL_REDUCE_RESERVED};
};

struct CcuKernelArgButterfly2x8 : public CcuKernelArgBase {
    uint32_t rankId{0};
    uint32_t kernelIndex{0};
    uint32_t kernelCount{0};
    uint32_t actualDieId{0};
    uint32_t phaseByChannel[4]{};
    uint32_t phaseOwnerKernel[4]{};
    uint64_t totalBytes{0};
    uint64_t scratchStride{0};
    uint64_t scratchAddr{0};
    uint64_t scratchToken{0};
    HcclDataType dataType{HCCL_DATA_TYPE_RESERVED};
    HcclReduceOp reduceType{HCCL_REDUCE_RESERVED};
};

struct CcuKernelArgCrossLane2x8 : public CcuKernelArgBase {
    uint32_t rankId{0};
    uint32_t kernelIndex{0};
    uint32_t kernelCount{0};
    uint32_t actualDieId{0};
    uint32_t localShardMask{0};
    uint32_t localLaneCount{0};
    uint32_t meshLaneMode{0};
    uint32_t directOutputAccumulator{0};
    uint32_t mode{0};
    uint32_t peerRanks[MAX_RANK_SIZE]{};
    uint32_t peerIsLocal[MAX_RANK_SIZE]{};
    uint32_t targetSlot[MAX_RANK_SIZE]{};
    uint32_t targetLane[MAX_RANK_SIZE]{};
    uint32_t targetLanePosition[MAX_RANK_SIZE]{};
    uint32_t targetLaneSize[MAX_RANK_SIZE]{};
    uint32_t sourceLane[MAX_RANK_SIZE]{};
    uint32_t sourceLanePosition[MAX_RANK_SIZE]{};
    uint32_t sourceLaneSize[MAX_RANK_SIZE]{};
    uint64_t targetOwnerOffset[MAX_RANK_SIZE]{};
    uint64_t targetOwnerBytes[MAX_RANK_SIZE]{};
    uint64_t targetShardOffset[MAX_RANK_SIZE][CROSS_LANE_SHARD_COUNT]{};
    uint64_t targetShardBytes[MAX_RANK_SIZE][CROSS_LANE_SHARD_COUNT]{};
    uint64_t ownerOffset{0};
    uint64_t ownerBytes{0};
    uint64_t ownerShardOffset[CROSS_LANE_SHARD_COUNT]{};
    uint64_t ownerShardBytes[CROSS_LANE_SHARD_COUNT]{};
    uint64_t scratchStride{0};
    HcclDataType dataType{HCCL_DATA_TYPE_RESERVED};
    HcclReduceOp reduceType{HCCL_REDUCE_RESERVED};
};

struct CcuKernelArgDirectRsag : public CcuKernelArgBase {
    uint32_t rankId{0};
    uint32_t rankSize{0};
    uint32_t kernelIndex{0};
    uint32_t kernelCount{0};
    uint32_t actualDieId{0};
    uint32_t isPrimary{0};
    uint32_t preSyncOnce{0};
    uint32_t earlyInit{0};
    uint32_t peerRanks[MAX_RANK_SIZE]{};
    uint32_t peerIsLocal[MAX_RANK_SIZE]{};
    uint32_t targetSlot[MAX_RANK_SIZE]{};
    uint64_t targetOwnerOffset[MAX_RANK_SIZE]{};
    uint64_t targetOwnerBytes[MAX_RANK_SIZE]{};
    uint64_t ownerOffset{0};
    uint64_t ownerBytes{0};
    uint64_t scratchStride{0};
    uint64_t scratchAddr{0};
    uint64_t scratchToken{0};
    HcclDataType dataType{HCCL_DATA_TYPE_RESERVED};
    HcclReduceOp reduceType{HCCL_REDUCE_RESERVED};
};

struct CcuKernelArgP4Rotate3 : public CcuKernelArgBase {
    uint32_t rankId{0};
    uint32_t kernelIndex{0};
    uint32_t kernelCount{0};
    uint32_t actualDieId{0};
    uint32_t isPrimary{0};
    uint32_t uniquePhaseNotify{0};
    uint32_t directOutputAccumulator{0};
    uint32_t ownerPull{0};
    uint32_t fullSliceAccumulate{0};
    uint32_t phaseChain{0};
    uint32_t wave2Accumulate{0};
    uint32_t peerRanks[MAX_RANK_SIZE]{};
    uint32_t targetPosition[MAX_RANK_SIZE]{};
    uint64_t targetOwnerOffset[MAX_RANK_SIZE]{};
    uint64_t targetOwnerBytes[MAX_RANK_SIZE]{};
    uint64_t ownerOffset{0};
    uint64_t ownerBytes{0};
    uint64_t scratchStride{0};
    uint64_t scratchAddr{0};
    uint64_t scratchToken{0};
    HcclDataType dataType{HCCL_DATA_TYPE_RESERVED};
    HcclReduceOp reduceType{HCCL_REDUCE_RESERVED};
};

struct CcuKernelArgP12Hier8 : public CcuKernelArgBase {
    uint32_t rankId{0};
    uint32_t kernelIndex{0};
    uint32_t kernelCount{0};
    uint32_t actualDieId{0};
    uint32_t isPrimary{0};
    uint32_t localSize{0};
    uint32_t localIndex{0};
    uint32_t localLaneCount{0};
    uint32_t ownedShardCount{0};
    uint32_t ownedShardId[HIER_MAX_OWNED_SHARD_COUNT]{};
    uint32_t peerRanks[MAX_RANK_SIZE]{};
    uint32_t peerIsLocal[MAX_RANK_SIZE]{};
    uint32_t peerLocalIndex[MAX_RANK_SIZE]{};
    uint32_t targetLane[MAX_RANK_SIZE]{};
    uint32_t targetLanePosition[MAX_RANK_SIZE]{};
    uint32_t targetLaneSize[MAX_RANK_SIZE]{};
    uint32_t sourceLane[MAX_RANK_SIZE]{};
    uint32_t sourceLaneSize[MAX_RANK_SIZE]{};
    uint32_t crossOwnedIndex[MAX_RANK_SIZE]{};
    uint64_t shardOffset[HIER_GLOBAL_SHARD_COUNT]{};
    uint64_t shardBytes[HIER_GLOBAL_SHARD_COUNT]{};
    uint64_t scratchStride{0};
    uint64_t scratchAddr{0};
    uint64_t scratchToken{0};
    HcclDataType dataType{HCCL_DATA_TYPE_RESERVED};
    HcclReduceOp reduceType{HCCL_REDUCE_RESERVED};
};

// 8+4 大消息：按全局 segment 重新做 12-owner 切分，复用 v2.2 Direct-RSAG
// 的大块 HBM primitive，避免完整 512 MiB 对应的 11-slot scratch 超限。
struct CcuKernelArgP12AllPairs : public CcuKernelArgBase {
    uint32_t rankId{0};
    uint32_t rankSize{0};
    uint32_t kernelIndex{0};
    uint32_t kernelCount{0};
    uint32_t actualDieId{0};
    uint32_t isPrimary{0};
    uint32_t balancedReduce{0};
    uint32_t preSyncOnce{0};
    uint32_t segmentCount{P12_ALLPAIRS_SEGMENT_COUNT};
    uint32_t earlyInit{0};
    uint32_t closAccumulate{0};
    uint32_t isClosOwner{0};
    uint32_t treeRanks[MAX_RANK_SIZE]{};
    uint32_t peerRanks[MAX_RANK_SIZE]{};
    uint32_t peerIsLocal[MAX_RANK_SIZE]{};
    uint32_t peerClosPhase[MAX_RANK_SIZE]{};
    uint32_t targetSlot[MAX_RANK_SIZE]{};
    uint64_t targetOwnerOffset[P12_ALLPAIRS_SEGMENT_COUNT][MAX_RANK_SIZE]{};
    uint64_t targetOwnerBytes[P12_ALLPAIRS_SEGMENT_COUNT][MAX_RANK_SIZE]{};
    uint64_t ownerOffset[P12_ALLPAIRS_SEGMENT_COUNT]{};
    uint64_t ownerBytes[P12_ALLPAIRS_SEGMENT_COUNT]{};
    uint64_t segmentBytes[P12_ALLPAIRS_SEGMENT_COUNT]{};
    uint64_t scratchStride{0};
    uint64_t scratchAddr{0};
    uint64_t scratchToken{0};
    HcclDataType dataType{HCCL_DATA_TYPE_RESERVED};
    HcclReduceOp reduceType{HCCL_REDUCE_RESERVED};
};

// v112 / 8+4 大消息：双 die 旋转 ReadReduce，主 die 合并后广播。
// 独立参数结构避免改变已有 P12 All-Pairs kernel 的注册布局。
struct CcuKernelArgP12OutputTree : public CcuKernelArgBase {
    uint32_t rankId{0};
    uint32_t rankSize{0};
    uint32_t kernelIndex{0};
    uint32_t kernelCount{0};
    uint32_t actualDieId{0};
    uint32_t isPrimary{0};
    uint32_t balancedReduce{0};
    uint32_t preSyncOnce{0};
    uint32_t segmentCount{P12_ALLPAIRS_SEGMENT_COUNT};
    uint32_t earlyInit{0};
    uint32_t compactOutputTree{0};
    uint32_t dualGroupReduce{0};
    uint32_t rotatingReadReduce{0};
    uint32_t groupOutputRoot{0};
    uint32_t groupRootRank{0};
    uint32_t directOutputSourceRank{0};
    uint32_t tempOutputSourceRank{0};
    uint32_t treeRanks[MAX_RANK_SIZE]{};
    uint32_t peerRanks[MAX_RANK_SIZE]{};
    uint32_t targetSlot[MAX_RANK_SIZE]{};
    uint64_t targetOwnerOffset[P12_ALLPAIRS_SEGMENT_COUNT][MAX_RANK_SIZE]{};
    uint64_t targetOwnerBytes[P12_ALLPAIRS_SEGMENT_COUNT][MAX_RANK_SIZE]{};
    uint64_t targetTempOutputOffset[MAX_RANK_SIZE]{};
    uint64_t ownerOffset[P12_ALLPAIRS_SEGMENT_COUNT]{};
    uint64_t ownerBytes[P12_ALLPAIRS_SEGMENT_COUNT]{};
    uint64_t tempOutputOffset{0};
    uint64_t segmentBytes[P12_ALLPAIRS_SEGMENT_COUNT]{};
    uint64_t scratchStride{0};
    uint64_t scratchAddr{0};
    uint64_t scratchToken{0};
    HcclDataType dataType{HCCL_DATA_TYPE_RESERVED};
    HcclReduceOp reduceType{HCCL_REDUCE_RESERVED};
};

// v101 / 512 KiB 专用参数。该结构保持参考实现的字段布局与语义，
// 只由 V101_DIRECT_MESH_512K 注册和执行路径使用。
struct CcuKernelArgV101Small final : public CcuKernelArgBase {
    uint32_t rankSize{0};
    uint32_t rankId{0};
    uint32_t channelIndexByRank[V101_MAX_RANK_SIZE]{};
    uint64_t directSliceOffset{0};
    uint64_t directSliceBytes{0};
    uint64_t directTileBytes{0};
    uint64_t directTailBytes{0};
    uint32_t directTileCount{0};
    uint32_t directTwoDie{0};
    uint32_t directDieId{0};
    uint32_t directIncludeLocalInput{0};
    uint32_t directUseMsGroup{0};
    uint32_t directUseR4MsDirect{0};
    uint32_t directUseCompactArgs{0};
    uint32_t directPartialZeroInOutput{0};
    uint32_t directReplicatedMerge{0};
    uint32_t directInputInOutput{0};
    uint32_t directMsParallel{0};
    uint64_t directScratchOffset{0};
    uint64_t directMergeOffset{0};
    uint64_t directMergeBytes{0};
    HcclDataType dataType{HCCL_DATA_TYPE_FP32};
    HcclReduceOp reduceType{HCCL_REDUCE_SUM};
};

// ccu kernel register所需信息
struct CcuKernelInfo {
    // kernel名称
    char kernelFuncName[64];
    // kernel函数
    void *kernelFunc;
    // KernelArg实例指针
    void *kernelArg;

private:
    std::shared_ptr<CcuKernelArgBase> kernelArgSmartPtr;

public:
    template <typename T> void setKernelArg(std::shared_ptr<T> arg)
    {
        kernelArgSmartPtr = std::static_pointer_cast<CcuKernelArgBase>(arg);
        kernelArg = static_cast<void *>(arg.get());
    }
};

struct KernelLaunchMeta {
    uint32_t actualDieId{0};
    uint32_t channelCount{0};
    // bit 0: primary；bit 8..11: 当前 kernel 持有的 butterfly phase。
    uint32_t roleFlags{0};
};

struct AlgResourceCtx {
    static constexpr uint32_t MAGIC = 0x41525253; // "ARRS"
    static constexpr uint32_t OWNER_VERSION = 6;
    static constexpr uint32_t CROSS_LANE_VERSION = 7;
    static constexpr uint32_t CROSS_LANE_V82_512M_VERSION = 24;
    static constexpr uint32_t BFLY_FUSED_VERSION = 8;
    static constexpr uint32_t DIRECT_RSAG_VERSION = 9;
    static constexpr uint32_t P4_ROTATE3_VERSION = 10;
    static constexpr uint32_t P12_HIER8_VERSION = 11;
    static constexpr uint32_t P12_ALLPAIRS_VERSION = 15;
    static constexpr uint32_t P4_DIRECT_SLOT3_VERSION = 16;
    static constexpr uint32_t P4_ROTATE3_UNIQUE_NOTIFY_VERSION = 17;
    static constexpr uint32_t P12_ALLPAIRS_BALANCED_VERSION = 18;
    static constexpr uint32_t P4_ROTATE3_DIRECT_OUTPUT_VERSION = 21;
    static constexpr uint32_t CROSS_LANE_DIRECT_OUTPUT_VERSION = 23;
    static constexpr uint32_t P12_PRESYNC_ONCE_VERSION = 25;
    static constexpr uint32_t P4_PULL_R3_VERSION = 26;
    static constexpr uint32_t P12_ALLPAIRS_1SEG_400M4B_VERSION = 27;
    static constexpr uint32_t P12_ALLPAIRS_EARLY_INIT_512M_VERSION = 28;
    static constexpr uint32_t P12_DIRECT_RSAG_EARLY_INIT_512K_VERSION = 31;
    static constexpr uint32_t P4_OUTPUT_ACCUMULATE_VERSION = 32;
    static constexpr uint32_t P16_MESH_LANE_VERSION = 33;
    static constexpr uint32_t P12_CLOS_ACCUMULATE_VERSION = 34;
    static constexpr uint32_t P4_OUTPUT_ACCUMULATE_CHAIN_VERSION = 35;
    static constexpr uint32_t V101_DIRECT_MESH_512K_VERSION = 36;
    static constexpr uint32_t P4_OUTPUT_ACCUMULATE_WAVE2_VERSION = 37;
    static constexpr uint32_t P4_OUTPUT_ACCUMULATE_WAVE2_400M4B_VERSION = 38;
    static constexpr uint32_t P12_ALLPAIRS_1SEG_OUTPUT_TREE_512M_VERSION = 39;
    static constexpr uint32_t P12_ALLPAIRS_1SEG_OUTPUT_TREE_400M4B_VERSION = 40;
    static constexpr uint32_t V109_DIRECT_MESH_R16_512K_VERSION = 41;
    static constexpr uint64_t MAX_SERIALIZED_SIZE = 1024;
    static constexpr uint32_t MAX_EXTRA_THREAD_COUNT = 1;
    static constexpr uint32_t MAX_KERNEL_COUNT = 2;

    uint32_t magic{MAGIC};
    uint32_t version{OWNER_VERSION};
    uint32_t rankSize{0};
    FinalTopology topology{FinalTopology::TOPOLOGY_4X1};
    CommBuffer localBuffer{}; ///< 本端HCCL通信内存
    // 用户 stream 对应的主 thread 每次调用重新获取，EngineCtx 只缓存额外 thread。
    std::vector<ThreadHandle> extraThreads;
    std::vector<CcuKernelHandle> ccuKernels;
    std::vector<KernelLaunchMeta> kernelMeta;

    // 序列化
    std::vector<char> Serialize() const
    {
        std::vector<char> result;
        result.reserve(MAX_SERIALIZED_SIZE);
        AppendPod(result, magic);
        AppendPod(result, version);
        AppendPod(result, rankSize);
        AppendPod(result, topology);
        AppendPod(result, localBuffer);
        AppendVector(result, extraThreads);
        AppendVector(result, ccuKernels);
        AppendVector(result, kernelMeta);
        return result;
    }

    // EngineCtx 可能因创建/拷贝中断而不完整，反序列化必须先做长度与数量上界校验。
    bool DeSerialize(const std::vector<char> &data)
    {
        if (data.empty() || data.size() > MAX_SERIALIZED_SIZE) {
            return false;
        }

        size_t offset = 0;
        uint32_t parsedMagic = 0;
        uint32_t parsedVersion = 0;
        uint32_t parsedRankSize = 0;
        FinalTopology parsedTopology = FinalTopology::TOPOLOGY_4X1;
        CommBuffer parsedLocalBuffer{};
        std::vector<ThreadHandle> parsedExtraThreads;
        std::vector<CcuKernelHandle> parsedKernels;
        std::vector<KernelLaunchMeta> parsedKernelMeta;
        if (!ReadPod(data, offset, parsedMagic) || !ReadPod(data, offset, parsedVersion)
            || !ReadPod(data, offset, parsedRankSize) || !ReadPod(data, offset, parsedTopology)
            || !ReadPod(data, offset, parsedLocalBuffer)
            || !ReadVector(data, offset, MAX_EXTRA_THREAD_COUNT, parsedExtraThreads)
            || !ReadVector(data, offset, MAX_KERNEL_COUNT, parsedKernels)
            || !ReadVector(data, offset, MAX_KERNEL_COUNT, parsedKernelMeta) || offset != data.size()) {
            return false;
        }

        magic = parsedMagic;
        version = parsedVersion;
        rankSize = parsedRankSize;
        topology = parsedTopology;
        localBuffer = parsedLocalBuffer;
        extraThreads = std::move(parsedExtraThreads);
        ccuKernels = std::move(parsedKernels);
        kernelMeta = std::move(parsedKernelMeta);
        return true;
    }

private:
    template <typename T> static void AppendPod(std::vector<char> &data, const T &value)
    {
        static_assert(std::is_trivially_copyable<T>::value, "EngineCtx fields must be trivially copyable");
        const char *begin = reinterpret_cast<const char *>(&value);
        data.insert(data.end(), begin, begin + sizeof(T));
    }

    template <typename T> static void AppendVector(std::vector<char> &data, const std::vector<T> &values)
    {
        static_assert(std::is_trivially_copyable<T>::value, "EngineCtx fields must be trivially copyable");
        const uint32_t count = static_cast<uint32_t>(values.size());
        AppendPod(data, count);
        if (!values.empty()) {
            const char *begin = reinterpret_cast<const char *>(values.data());
            data.insert(data.end(), begin, begin + values.size() * sizeof(T));
        }
    }

    template <typename T> static bool ReadPod(const std::vector<char> &data, size_t &offset, T &value)
    {
        static_assert(std::is_trivially_copyable<T>::value, "EngineCtx fields must be trivially copyable");
        if (offset > data.size() || sizeof(T) > data.size() - offset) {
            return false;
        }
        std::memcpy(&value, data.data() + offset, sizeof(T));
        offset += sizeof(T);
        return true;
    }

    template <typename T>
    static bool ReadVector(const std::vector<char> &data, size_t &offset, uint32_t maxCount, std::vector<T> &values)
    {
        static_assert(std::is_trivially_copyable<T>::value, "EngineCtx fields must be trivially copyable");
        uint32_t count = 0;
        if (!ReadPod(data, offset, count) || count > maxCount || offset > data.size()
            || static_cast<size_t>(count) > (data.size() - offset) / sizeof(T)) {
            return false;
        }
        values.resize(count);
        if (count != 0) {
            const size_t byteCount = static_cast<size_t>(count) * sizeof(T);
            std::memcpy(values.data(), data.data() + offset, byteCount);
            offset += byteCount;
        }
        return true;
    }
};

#endif // OPS_HCCL_CUSTOM_H
