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

#include <algorithm>
#include <array>
#include <vector>
#include <hccl/hccl_types.h>
#include <hccl/hccl_res.h>

#include "binary_stream.h"
#include "common.h"

typedef struct {
    void *addr;
    uint64_t size;
} CommBuffer;

struct CcuKernelArgBase {
    ChannelHandle channels[MAX_RANK_SIZE]{};
    uint32_t channelCount = 0;
};

namespace custom_rs {

constexpr uint64_t SLOT_ALIGNMENT_BYTES = 4096;
constexpr uint32_t PIPELINE_BANK_NUM = 2;

// 4x1 recursive-halving segmentation. Round 1 exchanges two output blocks in
// eight pieces (four pieces per block). Round 2 exchanges one already reduced
// block with the fade-out profile below; its last segment absorbs alignment
// and message-size remainder.
constexpr uint32_t ROUND1_4X1_TILES_PER_BLOCK = 4;
constexpr uint32_t ROUND1_4X1_SEGMENT_NUM = 2U * ROUND1_4X1_TILES_PER_BLOCK;
constexpr uint32_t ROUND1_4X1_MAX_OUTSTANDING = 4;
constexpr std::array<uint32_t, 8> SEGMENT_4X1_WEIGHTS = {
    12, 12, 10, 8,
    7, 6, 5, 4,
};
constexpr uint32_t SEGMENT_4X1_NUM = SEGMENT_4X1_WEIGHTS.size();
constexpr uint32_t SEGMENT_4X1_MAX_OUTSTANDING = 4;

constexpr uint32_t Sum4X1SegmentWeights()
{
    uint32_t sum = 0;
    for (uint32_t weight : SEGMENT_4X1_WEIGHTS) sum += weight;
    return sum;
}

constexpr bool HasPositive4X1SegmentWeights()
{
    for (uint32_t weight : SEGMENT_4X1_WEIGHTS) {
        if (weight == 0) return false;
    }
    return true;
}

constexpr uint32_t SEGMENT_4X1_WEIGHT_SUM = Sum4X1SegmentWeights();
static_assert(SEGMENT_4X1_NUM > 0 && SEGMENT_4X1_NUM <= 16,
    "4x1 named readiness Event supports 1..16 segments");
static_assert(SEGMENT_4X1_WEIGHT_SUM > 0,
    "4x1 segment weights must contain a positive total");
static_assert(HasPositive4X1SegmentWeights(),
    "4x1 segment weights must all be positive");

inline uint64_t Get4X1SegmentUnitBytes(uint64_t totalBytes)
{
    return totalBytes / SEGMENT_4X1_WEIGHT_SUM /
        SLOT_ALIGNMENT_BYTES * SLOT_ALIGNMENT_BYTES;
}

inline std::array<uint64_t, SEGMENT_4X1_NUM> Get4X1SegmentBytes(
    uint64_t totalBytes)
{
    std::array<uint64_t, SEGMENT_4X1_NUM> segments{};
    const uint64_t unitBytes = Get4X1SegmentUnitBytes(totalBytes);
    uint64_t assigned = 0;
    for (uint32_t index = 0; index + 1 < SEGMENT_4X1_NUM; ++index) {
        segments[index] = unitBytes * SEGMENT_4X1_WEIGHTS[index];
        assigned += segments[index];
    }
    segments.back() = totalBytes - assigned;
    return segments;
}

struct RoutePlan {
    uint32_t srcServerSize = 0;
    uint32_t dstServerSize = 0;
    uint32_t aggregateNumerator = 0;
    uint32_t aggregateDenominator = 1;
};

struct MixedScratchLayout {
    bool enabled = false;
    bool asymmetricTwoPass = false;
    uint64_t receiveAggregateBytes = 0;
    uint64_t directBytes = 0;
    uint64_t directTileBytes = 0;
    uint64_t directTileCount = 0;
    std::array<uint64_t, PIPELINE_BANK_NUM> directBankBytes{};
    std::array<uint64_t, PIPELINE_BANK_NUM> closBankStride{};
    std::array<uint64_t, PIPELINE_BANK_NUM> meshBankStride{};
    std::array<uint64_t, PIPELINE_BANK_NUM> meshBankOffset{};
    std::array<uint64_t, PIPELINE_BANK_NUM> closBankOffset{};
    uint64_t outboundOffset = 0;
    uint64_t totalBytes = 0;
};

struct AggregateTask {
    uint32_t rank = 0;
    uint64_t offset = 0;
    uint64_t bytes = 0;
};

struct AggregateSourceTask {
    uint32_t rank = 0;
    uint64_t remoteOffset = 0;
    uint64_t localOffset = 0;
    uint64_t bytes = 0;
};

// Let a/b be the source/destination server sizes and x be the MRCW fraction:
//   mesh(x) = a(a-1)S + b(a-1)xS
//   clos(x) = abS - b(a-1)xS,  Cclos = 4*min(a,b)V
// Solving mesh(x)/(a(a-1)V) = clos(x)/Cclos gives:
//   8->8: x=4/11, 8->4: x=4/9, 4->8: x=2/7.
// The aligned prefix uses MRCW; the remaining suffix uses direct CMR/A2A.
inline RoutePlan GetRoutePlan(uint32_t srcServerSize, uint32_t dstServerSize)
{
    if (srcServerSize == 8 && dstServerSize == 8) {
        return RoutePlan{8, 8, 4, 11};
    }
    if (srcServerSize == 8 && dstServerSize == 4) {
        return RoutePlan{8, 4, 4, 9};
    }
    if (srcServerSize == 4 && dstServerSize == 8) {
        return RoutePlan{4, 8, 2, 7};
    }
    // 4x1 uses its dedicated segmented path; unknown topologies have no plan.
    return RoutePlan{srcServerSize, dstServerSize, 0, 1};
}

inline uint64_t GetAggregateBytes(uint64_t bytes, const RoutePlan &plan)
{
    constexpr uint64_t ALIGN_BYTES = 4096;
    if (plan.aggregateNumerator == 0 || plan.aggregateDenominator == 0) {
        return 0;
    }
    uint64_t aggregateBytes = bytes / plan.aggregateDenominator * plan.aggregateNumerator;
    aggregateBytes += (bytes % plan.aggregateDenominator) * plan.aggregateNumerator /
        plan.aggregateDenominator;
    return aggregateBytes / ALIGN_BYTES * ALIGN_BYTES;
}

inline MixedScratchLayout GetMixedScratchLayout(uint64_t recvBytes, uint64_t bufferBytes,
    uint32_t localCount, uint32_t remoteCount, uint64_t outboundBytes,
    const RoutePlan &sendPlan, const RoutePlan &receivePlan)
{
    MixedScratchLayout layout;
    if (localCount <= 1 || remoteCount == 0) {
        return layout;
    }
    const uint64_t sendAggregateBytes = GetAggregateBytes(recvBytes, sendPlan);
    const uint64_t receiveAggregateBytes = GetAggregateBytes(recvBytes, receivePlan);
    layout.receiveAggregateBytes = receiveAggregateBytes;
    layout.directBytes = recvBytes - receiveAggregateBytes;
    const uint64_t meshStride = std::max(sendAggregateBytes, receiveAggregateBytes);
    // The output is initialized directly from the self slice. Mesh scratch
    // therefore stores only the other local ranks, and the received A partial
    // is consumed with ReadReduce instead of occupying a permanent inbox.
    const uint64_t fixedBytes = outboundBytes;
    const uint64_t meshBankBytes = PIPELINE_BANK_NUM * (localCount - 1U) * meshStride;
    if (sendAggregateBytes == 0 || receiveAggregateBytes == 0 ||
        fixedBytes + meshBankBytes >= bufferBytes) {
        return layout;
    }
    layout.directTileBytes = std::min(meshStride,
        (bufferBytes - fixedBytes - meshBankBytes) /
            (PIPELINE_BANK_NUM * remoteCount));
    layout.directTileBytes -= layout.directTileBytes % SLOT_ALIGNMENT_BYTES;
    if (layout.directTileBytes == 0) {
        return layout;
    }

    const uint64_t directBytes = layout.directBytes;
    layout.directTileCount =
        (directBytes + layout.directTileBytes - 1U) / layout.directTileBytes;
    layout.directBankBytes.fill(layout.directTileBytes);
    layout.closBankStride.fill(layout.directTileBytes);
    layout.meshBankStride.fill(meshStride);

    // For 2x8 there is one outbound-A task, one local-A task, then exactly two
    // B tasks. Equal banks reserve 2*directTileBytes even though the second B
    // pass is shorter. Try a 4:1 B split under the exact bank layout:
    //   fixed + (local-1)*(max(sendA,B0)+max(recvA,B1))
    //         + remote*(B0+B1) <= CCL buffer.
    // This preserves two banks and two passes, spends stranded capacity on B0,
    // and keeps B1 at a multi-MiB transfer size instead of maximizing B0 until
    // the tail degenerates into an inefficient 4 KiB network operation.
    if (sendPlan.srcServerSize == 8 && sendPlan.dstServerSize == 8 &&
        receivePlan.srcServerSize == 8 && receivePlan.dstServerSize == 8) {
        const uint64_t first =
            (directBytes / 5U * 4U + (directBytes % 5U) * 4U / 5U) /
            SLOT_ALIGNMENT_BYTES * SLOT_ALIGNMENT_BYTES;
        const uint64_t second = directBytes - first;
        const uint64_t secondCapacity =
            (second + SLOT_ALIGNMENT_BYTES - 1U) /
            SLOT_ALIGNMENT_BYTES * SLOT_ALIGNMENT_BYTES;
        const uint64_t mesh0 = std::max(sendAggregateBytes, first);
        const uint64_t mesh1 = std::max(receiveAggregateBytes, secondCapacity);
        const uint64_t usedBytes = fixedBytes +
            (localCount - 1U) * (mesh0 + mesh1) +
            remoteCount * (first + secondCapacity);
        if (first > layout.directTileBytes && second != 0 && usedBytes <= bufferBytes) {
            layout.asymmetricTwoPass = true;
            layout.directTileCount = 2;
            layout.directBankBytes = {first, second};
            layout.closBankStride = {first, secondCapacity};
            layout.meshBankStride = {mesh0, mesh1};
        }
    }

    uint64_t offset = 0;
    for (uint32_t bank = 0; bank < PIPELINE_BANK_NUM; ++bank) {
        layout.meshBankOffset[bank] = offset;
        offset += (localCount - 1U) * layout.meshBankStride[bank];
    }
    for (uint32_t bank = 0; bank < PIPELINE_BANK_NUM; ++bank) {
        layout.closBankOffset[bank] = offset;
        offset += remoteCount * layout.closBankStride[bank];
    }
    layout.outboundOffset = offset;
    layout.totalBytes = offset + outboundBytes;
    layout.enabled = layout.totalBytes <= bufferBytes;
    return layout;
}

constexpr uint64_t SMALL_TOTAL_DATA_BYTES = 1024ULL * 1024ULL;
// Keep tiny inputs on the fused small-message path; the competition's two
// large cases are far above this mixed-route threshold.
constexpr uint64_t MIXED_ROUTE_MIN_RECV_BYTES = 1024ULL * 1024ULL;
constexpr uint32_t CHANNEL_NOTIFY_NUM = 1;
constexpr uint32_t IO_DIE_NUM = 2;
constexpr uint32_t COMM_LIFECYCLE_NOTIFY = 0;
// Mixed-route notifications are local to their destination Thread, so the
// same numeric ids can be reused by different Threads.
constexpr uint32_t MIXED_OUT_READY_BASE = 1;
constexpr uint32_t MIXED_REMOTE_B_READY_BASE = 1;
constexpr uint32_t MIXED_REMOTE_PARTIAL_READY_BASE = 6;
constexpr uint32_t MIXED_FINAL_READY_NOTIFY = 9;
constexpr uint32_t FUSED_READY_BASE = 1;
constexpr uint32_t MIXED_BANK_REUSE_BASE = 1;
constexpr uint32_t MIXED_LOCAL_A_READY_NOTIFY = 5;
constexpr uint32_t MAIN_LOCAL_DONE_BASE = 12;
constexpr uint32_t MAIN_THREAD_NOTIFY_NUM = MAIN_LOCAL_DONE_BASE + 4;
constexpr uint32_t WORKER_THREAD_NOTIFY_NUM = 12;
constexpr uint32_t REQUIRED_THREAD_NUM = 4;
constexpr uint32_t COMM_KERNEL_BASE = 0;
constexpr uint32_t REDUCE_KERNEL_BASE = COMM_KERNEL_BASE + IO_DIE_NUM;
constexpr uint32_t KERNEL_SLOT_NUM = REDUCE_KERNEL_BASE + IO_DIE_NUM;
constexpr uint32_t INVALID_IO_DIE = 0xFFFFFFFFU;

constexpr uint64_t CCU_MS_BYTES = 4096ULL;
constexpr uint32_t CCU_BUFFER_REDUCE_MAX_INPUTS = 8;
constexpr uint32_t CCU_SCHED_MS_BLOCKS_PER_DIE = 128;

enum class CommKernelMode : uint64_t {
    // Publish the local input address/token to every peer. On the Clos die,
    // also publish the outbound partial address to the peers that consume it.
    EXCHANGE_ADDRESSES = 0,
    // MixedRoute Clos phase: fetch the current result range from every rank
    // on the remote server into contiguous CCL-buffer slots.
    FETCH_CLOS_REMOTE_SLOTS = 1,
    // 4x1 recursive halving: exchange eight first-round pieces with rank^2,
    // then, only after all first-round network reads finish, exchange the
    // nine fade-out partial pieces with rank^1.
    RECURSIVE_HALVING_4X1 = 3,
    // Cross-rank lifetime barrier after the result is complete. Unlike address
    // exchange, this does not republish any address/token.
    FINISH_BARRIER = 4,
    // MixedRoute Mesh phase: fetch all other ranks in the local server into
    // contiguous slots; self was copied or initialized separately.
    FETCH_MESH_REMOTE_SLOTS = 5,
    // Small-data path: fetch this rank's output slice from every source
    // directly into per-source CCU Buffer ranges and reduce one tile there.
    SMALL_BUFFER_TILE = 6,
};

enum class ReduceKernelMode : uint64_t {
    // Copy one local HBM range to another local HBM range.
    COPY_RANGE = 0,
    // Add one local HBM source range into an existing destination range.
    REDUCE_RANGE = 1,
    // Fold N contiguous source slots into slot 0 in place.
    FOLD_SLOTS_IN_PLACE = 2,
    // Add every Mesh remote-rank slot into an already initialized destination.
    REDUCE_MESH_SLOTS_TO_DST = 3,
    // Read server-level partials through Clos channels and reduce them into
    // their final offsets in output.
    READ_REDUCE_CLOS_PARTIALS = 4,
};

inline uint32_t GetCommKernelIndex(uint32_t ioDieId)
{
    return COMM_KERNEL_BASE + ioDieId;
}

inline uint32_t GetReduceKernelIndex(uint32_t ioDieId)
{
    return REDUCE_KERNEL_BASE + ioDieId;
}

struct SmallBufferTilePlan {
    bool enabled = false;
    uint32_t capacityBlocksPerSource = 0;
    uint64_t tileBytes = 0;
};

inline uint32_t GetSmallBufferCapacityBlocksPerSource(
    uint32_t rankSize, uint32_t maxMembersPerDie)
{
    if (rankSize == 0 || maxMembersPerDie == 0 ||
        maxMembersPerDie > CCU_BUFFER_REDUCE_MAX_INPUTS) {
        return 0;
    }
    const uint64_t maxRecvBytes = SMALL_TOTAL_DATA_BYTES / rankSize;
    const uint32_t topologyBlocks = static_cast<uint32_t>(
        (maxRecvBytes + CCU_MS_BYTES - 1U) / CCU_MS_BYTES);
    const uint32_t resourceBlocks =
        CCU_SCHED_MS_BLOCKS_PER_DIE / maxMembersPerDie;
    return std::min(topologyBlocks, resourceBlocks);
}

inline SmallBufferTilePlan GetSmallBufferTilePlan(
    uint32_t rankSize, uint64_t recvBytes, uint32_t maxMembersPerDie)
{
    SmallBufferTilePlan plan;
    if (rankSize == 0 || recvBytes == 0 || recvBytes % sizeof(float) != 0 ||
        recvBytes > SMALL_TOTAL_DATA_BYTES / rankSize || maxMembersPerDie == 0 ||
        maxMembersPerDie > CCU_BUFFER_REDUCE_MAX_INPUTS) {
        return plan;
    }
    plan.capacityBlocksPerSource =
        GetSmallBufferCapacityBlocksPerSource(rankSize, maxMembersPerDie);
    if (plan.capacityBlocksPerSource == 0) {
        return SmallBufferTilePlan{};
    }
    plan.enabled = true;
    plan.tileBytes = std::min(recvBytes,
        static_cast<uint64_t>(plan.capacityBlocksPerSource) * CCU_MS_BYTES);
    return plan;
}

} // namespace custom_rs

struct AlgResourceCtx {
    CommBuffer localBuffer{};          ///< 本端HCCL通信内存
    std::array<ThreadHandle, custom_rs::REQUIRED_THREAD_NUM> threads{};
    std::array<CcuKernelHandle, custom_rs::KERNEL_SLOT_NUM> ccuKernels{};
    std::array<uint32_t, custom_rs::IO_DIE_NUM> directMemberCounts{};
    std::array<uint32_t, custom_rs::IO_DIE_NUM> fusedMemberCounts{};
    std::array<uint32_t, custom_rs::IO_DIE_NUM> localMemberCounts{};
    std::vector<uint32_t> localServerRanks;
    std::vector<uint32_t> remoteServerRanks;
    std::vector<custom_rs::AggregateTask> aggregateTargets;
    std::vector<custom_rs::AggregateSourceTask> aggregateSources;
    std::vector<uint32_t> peerIoDies;
    custom_rs::RoutePlan sendRoutePlan;
    custom_rs::RoutePlan receiveRoutePlan;
    uint32_t activeIoDieMask = 0;
    uint32_t resultIoDie = 0;
    uint32_t meshIoDie = custom_rs::INVALID_IO_DIE;
    uint32_t closIoDie = custom_rs::INVALID_IO_DIE;
    uint32_t ioDie0ThreadIndex = custom_rs::INVALID_IO_DIE;
    uint32_t ioDie1ThreadIndex = custom_rs::INVALID_IO_DIE;
    uint32_t ioDie0ReduceThreadIndex = custom_rs::INVALID_IO_DIE;
    uint32_t ioDie1ReduceThreadIndex = custom_rs::INVALID_IO_DIE;

    // 序列化
    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;
        binaryStream << localBuffer;
        binaryStream << threads;
        binaryStream << ccuKernels;
        binaryStream << directMemberCounts;
        binaryStream << fusedMemberCounts;
        binaryStream << localMemberCounts;
        // Server membership, aggregateSources and peerIoDies are registration-
        // only topology data.  Registered kernels already contain the derived
        // tables, so carrying those containers into every ExecOp is redundant.
        binaryStream << aggregateTargets;
        binaryStream << sendRoutePlan;
        binaryStream << receiveRoutePlan;
        binaryStream << activeIoDieMask;
        binaryStream << resultIoDie;
        binaryStream << meshIoDie;
        binaryStream << closIoDie;
        binaryStream << ioDie0ThreadIndex;
        binaryStream << ioDie1ThreadIndex;
        binaryStream << ioDie0ReduceThreadIndex;
        binaryStream << ioDie1ReduceThreadIndex;
        std::vector<char> result;
        binaryStream.Dump(result);
        return result;
    }

    // 反序列化
    void DeSerialize(std::vector<char> &data)
    {
        BinaryStream binaryStream(data);
        binaryStream >> localBuffer;
        binaryStream >> threads;
        binaryStream >> ccuKernels;
        binaryStream >> directMemberCounts;
        binaryStream >> fusedMemberCounts;
        binaryStream >> localMemberCounts;
        binaryStream >> aggregateTargets;
        binaryStream >> sendRoutePlan;
        binaryStream >> receiveRoutePlan;
        binaryStream >> activeIoDieMask;
        binaryStream >> resultIoDie;
        binaryStream >> meshIoDie;
        binaryStream >> closIoDie;
        binaryStream >> ioDie0ThreadIndex;
        binaryStream >> ioDie1ThreadIndex;
        binaryStream >> ioDie0ReduceThreadIndex;
        binaryStream >> ioDie1ReduceThreadIndex;
    }
};

#endif // OPS_HCCL_CUSTOM_H
