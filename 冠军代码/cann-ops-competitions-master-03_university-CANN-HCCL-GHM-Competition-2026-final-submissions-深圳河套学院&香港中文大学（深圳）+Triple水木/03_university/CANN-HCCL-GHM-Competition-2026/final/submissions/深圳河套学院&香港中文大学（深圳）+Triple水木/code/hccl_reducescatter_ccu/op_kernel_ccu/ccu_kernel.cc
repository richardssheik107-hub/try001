/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#include <algorithm>
#include <array>
#include <vector>

#include <ccu/ccu_primitives.hpp>

#include "ccu_kernel.h"
#include "log.h"

namespace ccu = ::AscendC::ccu;

namespace ops_hccl {
namespace {

constexpr uint32_t INPUT_XN_ID = 0;
constexpr uint32_t INPUT_TOKEN_XN_ID = 1;
constexpr uint32_t PARTIAL_XN_ID = 2;
constexpr uint32_t PARTIAL_TOKEN_XN_ID = 3;
constexpr uint32_t CHANNEL_CKE_INDEX = 0;
constexpr uint16_t PARTIAL_READY_MASK = 1U << 4;
constexpr uint16_t FINISH_MASK = 1U << 5;
constexpr uint16_t SMALL_POST_SYNC_MASK = 1U << 4;

#define RS_CCU_CHK_RET(call)                                                                    \
    do {                                                                                        \
        CcuResult ccuRet = (call);                                                              \
        if (ccuRet != CCU_SUCCESS) {                                                            \
            HCCL_ERROR("[%s] CCU call failed, ret[%d]", __func__, static_cast<int32_t>(ccuRet)); \
            return ccuRet;                                                                      \
        }                                                                                       \
    } while (0)

uint32_t FindChannelIndex(const ReduceScatterKernelArg &arg, uint32_t remoteRank)
{
    for (uint32_t i = 0; i < arg.channelCount; ++i) {
        if (arg.peerRanks[i] == remoteRank) {
            return i;
        }
    }
    return arg.channelCount;
}

bool IsPartialPeer(const ReduceScatterKernelArg &arg, uint32_t remoteRank)
{
    for (uint32_t i = 0; i < arg.publishTargetCount; ++i) {
        if (arg.publishTargetRanks[i] == remoteRank) return true;
    }
    for (uint32_t i = 0; i < arg.aggregateSourceCount; ++i) {
        if (arg.aggregateSourceRanks[i] == remoteRank) return true;
    }
    return false;
}

void SetRemoteInput(ccu::RemoteAddr &remote, const std::vector<ccu::Variable> &remoteInput,
    const std::vector<ccu::Variable> &remoteInputToken, uint32_t channelIdx, ccu::Variable offset)
{
    remote.addr = remoteInput[channelIdx];
    remote.addr += offset;
    remote.token = remoteInputToken[channelIdx];
}

CcuResult Record4X1Round1Ready(uint32_t segment)
{
    RS_CCU_CHK_RET(ccu::EventRecord(
        "reduce_scatter_4x1_rh_round1", static_cast<uint16_t>(1U << segment)));
    return CCU_SUCCESS;
}

CcuResult Wait4X1Round1Ready(uint32_t segment)
{
    RS_CCU_CHK_RET(ccu::EventWait(
        "reduce_scatter_4x1_rh_round1", static_cast<uint16_t>(1U << segment)));
    return CCU_SUCCESS;
}

CcuResult Record4X1OutgoingReady()
{
    RS_CCU_CHK_RET(ccu::EventRecord(
        "reduce_scatter_4x1_rh_outgoing", 1));
    return CCU_SUCCESS;
}

CcuResult Wait4X1OutgoingReady()
{
    RS_CCU_CHK_RET(ccu::EventWait(
        "reduce_scatter_4x1_rh_outgoing", 1));
    return CCU_SUCCESS;
}

CcuResult Record4X1RetainedReady()
{
    RS_CCU_CHK_RET(ccu::EventRecord(
        "reduce_scatter_4x1_rh_retained", 1));
    return CCU_SUCCESS;
}

CcuResult Wait4X1RetainedReady()
{
    RS_CCU_CHK_RET(ccu::EventWait(
        "reduce_scatter_4x1_rh_retained", 1));
    return CCU_SUCCESS;
}

std::array<ccu::Variable, custom_rs::SEGMENT_4X1_NUM>
Build4X1SegmentLengths(ccu::Variable unitBytes, ccu::Variable tailBytes)
{
    std::array<ccu::Variable, custom_rs::SEGMENT_4X1_NUM> lengths;
    for (uint32_t segment = 0; segment + 1 < custom_rs::SEGMENT_4X1_NUM; ++segment) {
        lengths[segment] = unitBytes;
        for (uint32_t weight = 1;
            weight < custom_rs::SEGMENT_4X1_WEIGHTS[segment]; ++weight) {
            lengths[segment] += unitBytes;
        }
    }
    lengths.back() = tailBytes;
    return lengths;
}

} // namespace

CcuResult CcuReduceScatterKernel_Comm_General(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<ReduceScatterKernelArg *>(arg);

    // All modes share one static resource set.  Ascend 950 exposes only two Mission
    // resources per IO die, so separate Sync/Direct/Aggregate/Publish kernels cannot coexist.
    ccu::Variable mode;
    ccu::Variable primary;
    ccu::Variable primaryToken;
    ccu::Variable localInput;
    ccu::Variable localInputToken;
    uint32_t argId = 0;
    RS_CCU_CHK_RET(ccu::LoadArg(mode, argId++));
    RS_CCU_CHK_RET(ccu::LoadArg(primary, argId++));
    RS_CCU_CHK_RET(ccu::LoadArg(primaryToken, argId++));
    RS_CCU_CHK_RET(ccu::LoadArg(localInput, argId++));
    RS_CCU_CHK_RET(ccu::LoadArg(localInputToken, argId++));
    ccu::Variable sourceOffset;
    ccu::Variable bytes;
    RS_CCU_CHK_RET(ccu::LoadArg(sourceOffset, argId++));
    RS_CCU_CHK_RET(ccu::LoadArg(bytes, argId++));

    std::vector<ccu::Variable> remoteInput(kernelArg->channelCount);
    std::vector<ccu::Variable> remoteInputToken(kernelArg->channelCount);
    std::vector<ccu::RemoteAddr> remoteSources(kernelArg->channelCount);
    for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
        remoteInput[i] = ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[i], INPUT_XN_ID);
        remoteInputToken[i] =
            ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[i], INPUT_TOKEN_XN_ID);
    }
    ccu::LocalAddr dst;
    ccu::Event event;

    CCU_IF(mode == static_cast<uint64_t>(custom_rs::CommKernelMode::EXCHANGE_ADDRESSES)) {
        const uint16_t inputMask = 1U << INPUT_XN_ID;
        const uint16_t tokenMask = 1U << INPUT_TOKEN_XN_ID;
        for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
            RS_CCU_CHK_RET(ccu::WriteVariableWithNotify(kernelArg->channels[i], localInput,
                INPUT_XN_ID, CHANNEL_CKE_INDEX, inputMask));
            RS_CCU_CHK_RET(ccu::WriteVariableWithNotify(kernelArg->channels[i], localInputToken,
                INPUT_TOKEN_XN_ID, CHANNEL_CKE_INDEX, tokenMask));
            if (IsPartialPeer(*kernelArg, kernelArg->peerRanks[i])) {
                constexpr uint16_t partialMask = 1U << PARTIAL_XN_ID;
                constexpr uint16_t partialTokenMask = 1U << PARTIAL_TOKEN_XN_ID;
                RS_CCU_CHK_RET(ccu::WriteVariableWithNotify(kernelArg->channels[i], primary,
                    PARTIAL_XN_ID, CHANNEL_CKE_INDEX, partialMask));
                RS_CCU_CHK_RET(ccu::WriteVariableWithNotify(kernelArg->channels[i], primaryToken,
                    PARTIAL_TOKEN_XN_ID, CHANNEL_CKE_INDEX, partialTokenMask));
            }
        }
        for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
            uint16_t waitMask = static_cast<uint16_t>(
                (1U << INPUT_XN_ID) | (1U << INPUT_TOKEN_XN_ID));
            if (IsPartialPeer(*kernelArg, kernelArg->peerRanks[i])) {
                waitMask = static_cast<uint16_t>(waitMask |
                    (1U << PARTIAL_XN_ID) | (1U << PARTIAL_TOKEN_XN_ID));
            }
            RS_CCU_CHK_RET(ccu::NotifyWait(
                kernelArg->channels[i], CHANNEL_CKE_INDEX, waitMask));
        }
    }

    CCU_IF(mode == static_cast<uint64_t>(custom_rs::CommKernelMode::FINISH_BARRIER)) {
        for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
            RS_CCU_CHK_RET(ccu::NotifyRecord(
                kernelArg->channels[i], CHANNEL_CKE_INDEX, FINISH_MASK));
        }
        for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
            RS_CCU_CHK_RET(ccu::NotifyWait(
                kernelArg->channels[i], CHANNEL_CKE_INDEX, FINISH_MASK));
        }
    }

    if (kernelArg->commRole == CommKernelRole::GENERAL ||
        kernelArg->commRole == CommKernelRole::CLOS) {
        // This is the remote-server half of MixedRoute. It is launched only on
        // the Clos die; the local-server half uses FETCH_MESH_REMOTE_SLOTS.
        CCU_IF(mode == static_cast<uint64_t>(
            custom_rs::CommKernelMode::FETCH_CLOS_REMOTE_SLOTS)) {
            uint16_t readyMask = 0;
            dst.addr = primary;
            dst.token = primaryToken;
            for (uint32_t memberIdx = 0;
                memberIdx < kernelArg->directMemberCount; ++memberIdx) {
                const uint16_t mask = static_cast<uint16_t>(1U << memberIdx);
                readyMask = static_cast<uint16_t>(readyMask | mask);
                const uint32_t sourceRank = kernelArg->directMemberRanks[memberIdx];
                const uint32_t channelIdx = FindChannelIndex(*kernelArg, sourceRank);
                if (channelIdx >= kernelArg->channelCount) return CCU_E_INTERNAL;
                SetRemoteInput(remoteSources[channelIdx], remoteInput, remoteInputToken,
                    channelIdx, sourceOffset);
                RS_CCU_CHK_RET(ccu::Read(kernelArg->channels[channelIdx], dst,
                    remoteSources[channelIdx], bytes, event, mask));
                if (memberIdx + 1U < kernelArg->directMemberCount) {
                    dst.addr += bytes;
                }
            }
            if (readyMask != 0) RS_CCU_CHK_RET(ccu::EventWait(event, readyMask));
        }
    }

    // Fetch one target's large A range from every member of the local server.
    // Each source has a private HBM slot; all Mesh reads are issued before the
    // single EventWait so the full-mesh links can run concurrently.
    if (kernelArg->remoteLocalMemberCount != 0 &&
        (kernelArg->commRole == CommKernelRole::GENERAL ||
            kernelArg->commRole == CommKernelRole::MESH)) {
        CCU_IF(mode == static_cast<uint64_t>(
            custom_rs::CommKernelMode::FETCH_MESH_REMOTE_SLOTS)) {
            uint16_t readyMask = 0;
            dst.addr = primary;
            dst.token = primaryToken;
            for (uint32_t memberIdx = 0;
                memberIdx < kernelArg->remoteLocalMemberCount; ++memberIdx) {
                const uint16_t mask = static_cast<uint16_t>(1U << memberIdx);
                readyMask = static_cast<uint16_t>(readyMask | mask);
                const uint32_t sourceRank = kernelArg->remoteLocalMemberRanks[memberIdx];
                const uint32_t channelIdx = FindChannelIndex(*kernelArg, sourceRank);
                if (channelIdx >= kernelArg->channelCount) return CCU_E_INTERNAL;
                SetRemoteInput(remoteSources[channelIdx], remoteInput, remoteInputToken,
                    channelIdx, sourceOffset);
                RS_CCU_CHK_RET(ccu::Read(kernelArg->channels[channelIdx], dst,
                    remoteSources[channelIdx], bytes, event, mask));
                if (memberIdx + 1U < kernelArg->remoteLocalMemberCount) {
                    dst.addr += bytes;
                }
            }
            if (readyMask != 0) RS_CCU_CHK_RET(ccu::EventWait(event, readyMask));
        }
    }

    return CCU_SUCCESS;
}

CcuResult CcuReduceScatterKernel_Comm_4X1(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<ReduceScatterKernelArg *>(arg);
    if (kernelArg->fusedMemberCount != 4U) {
        return CCU_E_PARA;
    }

    ccu::Variable mode;
    ccu::Variable scratch;
    ccu::Variable scratchToken;
    ccu::Variable localInput;
    ccu::Variable localInputToken;
    ccu::Variable recvBytes;
    ccu::Variable round1TileBytes;
    ccu::Variable round1TailBytes;
    ccu::Variable round2UnitBytes;
    ccu::Variable round2TailBytes;
    uint32_t argId = 0;
    RS_CCU_CHK_RET(ccu::LoadArg(mode, argId++));
    RS_CCU_CHK_RET(ccu::LoadArg(scratch, argId++));
    RS_CCU_CHK_RET(ccu::LoadArg(scratchToken, argId++));
    RS_CCU_CHK_RET(ccu::LoadArg(localInput, argId++));
    RS_CCU_CHK_RET(ccu::LoadArg(localInputToken, argId++));
    RS_CCU_CHK_RET(ccu::LoadArg(recvBytes, argId++));
    RS_CCU_CHK_RET(ccu::LoadArg(round1TileBytes, argId++));
    RS_CCU_CHK_RET(ccu::LoadArg(round1TailBytes, argId++));
    RS_CCU_CHK_RET(ccu::LoadArg(round2UnitBytes, argId++));
    RS_CCU_CHK_RET(ccu::LoadArg(round2TailBytes, argId++));

    std::vector<ccu::Variable> remoteInput(kernelArg->channelCount);
    std::vector<ccu::Variable> remoteInputToken(kernelArg->channelCount);
    std::vector<ccu::Variable> remotePartial(kernelArg->channelCount);
    std::vector<ccu::Variable> remotePartialToken(kernelArg->channelCount);
    for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
        remoteInput[i] = ccu::GetResByChannel<ccu::Variable>(
            kernelArg->channels[i], INPUT_XN_ID);
        remoteInputToken[i] = ccu::GetResByChannel<ccu::Variable>(
            kernelArg->channels[i], INPUT_TOKEN_XN_ID);
        remotePartial[i] = ccu::GetResByChannel<ccu::Variable>(
            kernelArg->channels[i], PARTIAL_XN_ID);
        remotePartialToken[i] = ccu::GetResByChannel<ccu::Variable>(
            kernelArg->channels[i], PARTIAL_TOKEN_XN_ID);
    }

    CCU_IF(mode == static_cast<uint64_t>(custom_rs::CommKernelMode::EXCHANGE_ADDRESSES)) {
        const uint16_t inputMask = 1U << INPUT_XN_ID;
        const uint16_t tokenMask = 1U << INPUT_TOKEN_XN_ID;
        const uint16_t partialMask = 1U << PARTIAL_XN_ID;
        const uint16_t partialTokenMask = 1U << PARTIAL_TOKEN_XN_ID;
        const uint16_t allMask = inputMask | tokenMask | partialMask | partialTokenMask;
        for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
            RS_CCU_CHK_RET(ccu::WriteVariableWithNotify(kernelArg->channels[i], localInput,
                INPUT_XN_ID, CHANNEL_CKE_INDEX, inputMask));
            RS_CCU_CHK_RET(ccu::WriteVariableWithNotify(kernelArg->channels[i],
                localInputToken, INPUT_TOKEN_XN_ID, CHANNEL_CKE_INDEX, tokenMask));
            RS_CCU_CHK_RET(ccu::WriteVariableWithNotify(kernelArg->channels[i], scratch,
                PARTIAL_XN_ID, CHANNEL_CKE_INDEX, partialMask));
            RS_CCU_CHK_RET(ccu::WriteVariableWithNotify(kernelArg->channels[i],
                scratchToken, PARTIAL_TOKEN_XN_ID, CHANNEL_CKE_INDEX, partialTokenMask));
        }
        for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
            RS_CCU_CHK_RET(ccu::NotifyWait(
                kernelArg->channels[i], CHANNEL_CKE_INDEX, allMask));
        }
    }

    CCU_IF(mode == static_cast<uint64_t>(custom_rs::CommKernelMode::FINISH_BARRIER)) {
        for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
            RS_CCU_CHK_RET(ccu::NotifyRecord(
                kernelArg->channels[i], CHANNEL_CKE_INDEX, FINISH_MASK));
        }
        for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
            RS_CCU_CHK_RET(ccu::NotifyWait(
                kernelArg->channels[i], CHANNEL_CKE_INDEX, FINISH_MASK));
        }
    }

    CCU_IF(mode == static_cast<uint64_t>(
        custom_rs::CommKernelMode::RECURSIVE_HALVING_4X1)) {
        const uint32_t round1Partner = kernelArg->rankId ^ 2U;
        const uint32_t round2Partner = kernelArg->rankId ^ 1U;
        const uint32_t round1Channel = FindChannelIndex(*kernelArg, round1Partner);
        const uint32_t round2Channel = FindChannelIndex(*kernelArg, round2Partner);
        if (round1Channel >= kernelArg->channelCount ||
            round2Channel >= kernelArg->channelCount) {
            return CCU_E_INTERNAL;
        }

        std::array<ccu::Variable, custom_rs::ROUND1_4X1_TILES_PER_BLOCK>
            round1Lengths;
        for (uint32_t tile = 0;
            tile < custom_rs::ROUND1_4X1_TILES_PER_BLOCK; ++tile) {
            round1Lengths[tile] = round1TileBytes;
        }
        round1Lengths.back() = round1TailBytes;

        // The outgoing block is generated first. It is the block consumed by
        // rank^1 in round 2. The retained block follows in the same four lanes.
        const uint32_t sendBlock = kernelArg->rankId ^ 1U;
        const uint32_t keepBlock = kernelArg->rankId;
        std::array<ccu::LocalAddr, custom_rs::ROUND1_4X1_SEGMENT_NUM>
            round1Destinations;
        std::array<ccu::RemoteAddr, custom_rs::ROUND1_4X1_SEGMENT_NUM>
            round1Sources;
        ccu::Variable sendSourceBase;
        ccu::Variable keepSourceBase;
        sendSourceBase = remoteInput[round1Channel];
        keepSourceBase = remoteInput[round1Channel];
        for (uint32_t block = 0; block < sendBlock; ++block) sendSourceBase += recvBytes;
        for (uint32_t block = 0; block < keepBlock; ++block) keepSourceBase += recvBytes;
        ccu::Variable sendDstBase;
        ccu::Variable keepDstBase;
        sendDstBase = scratch;
        keepDstBase = localInput;
        for (uint32_t tile = 0; tile < custom_rs::ROUND1_4X1_TILES_PER_BLOCK; ++tile) {
            const uint32_t sendSegment = tile;
            const uint32_t keepSegment = tile + custom_rs::ROUND1_4X1_TILES_PER_BLOCK;
            round1Destinations[sendSegment].addr = sendDstBase;
            round1Destinations[sendSegment].token = scratchToken;
            round1Destinations[keepSegment].addr = keepDstBase;
            round1Destinations[keepSegment].token = localInputToken;
            round1Sources[sendSegment].addr = sendSourceBase;
            round1Sources[sendSegment].token = remoteInputToken[round1Channel];
            round1Sources[keepSegment].addr = keepSourceBase;
            round1Sources[keepSegment].token = remoteInputToken[round1Channel];
            if (tile + 1U < custom_rs::ROUND1_4X1_TILES_PER_BLOCK) {
                sendDstBase += round1Lengths[tile];
                keepDstBase += round1Lengths[tile];
                sendSourceBase += round1Lengths[tile];
                keepSourceBase += round1Lengths[tile];
            }
        }

        std::array<ccu::Event, custom_rs::ROUND1_4X1_MAX_OUTSTANDING>
            round1Events;
        auto issueRound1 = [&](uint32_t segment, uint32_t lane) -> CcuResult {
            const uint32_t tile = segment % custom_rs::ROUND1_4X1_TILES_PER_BLOCK;
            RS_CCU_CHK_RET(ccu::Read(kernelArg->channels[round1Channel],
                round1Destinations[segment], round1Sources[segment],
                round1Lengths[tile], round1Events[lane], 1));
            return CCU_SUCCESS;
        };
        for (uint32_t lane = 0; lane < custom_rs::ROUND1_4X1_MAX_OUTSTANDING; ++lane) {
            RS_CCU_CHK_RET(issueRound1(lane, lane));
        }
        for (uint32_t segment = 0; segment < custom_rs::ROUND1_4X1_SEGMENT_NUM; ++segment) {
            const uint32_t lane = segment % custom_rs::ROUND1_4X1_MAX_OUTSTANDING;
            RS_CCU_CHK_RET(ccu::EventWait(round1Events[lane], 1));
            RS_CCU_CHK_RET(Record4X1Round1Ready(segment));
            const uint32_t next = segment + custom_rs::ROUND1_4X1_MAX_OUTSTANDING;
            if (next < custom_rs::ROUND1_4X1_SEGMENT_NUM) {
                RS_CCU_CHK_RET(issueRound1(next, lane));
            }
            if (segment + 1U == custom_rs::ROUND1_4X1_TILES_PER_BLOCK) {
                // All four retained reads have now been issued. Hide the one
                // outgoing-group dependency and partner handshake under those
                // in-flight reads; round 2 still starts only after segments
                // 4..7 have completed below.
                RS_CCU_CHK_RET(Wait4X1OutgoingReady());
                RS_CCU_CHK_RET(ccu::NotifyRecord(kernelArg->channels[round2Channel],
                    CHANNEL_CKE_INDEX, PARTIAL_READY_MASK));
                RS_CCU_CHK_RET(ccu::NotifyWait(kernelArg->channels[round2Channel],
                    CHANNEL_CKE_INDEX, PARTIAL_READY_MASK));
            }
        }

        // No round-2 ReadReduce is issued until every round-1 network read,
        // both outgoing partials, and this rank's retained output partial are
        // ready. The remote source is rank^1's in-place scratch block.
        RS_CCU_CHK_RET(Wait4X1RetainedReady());
        auto round2Lengths = Build4X1SegmentLengths(
            round2UnitBytes, round2TailBytes);
        std::array<ccu::LocalAddr, custom_rs::SEGMENT_4X1_NUM>
            round2Destinations;
        std::array<ccu::RemoteAddr, custom_rs::SEGMENT_4X1_NUM>
            round2Sources;
        ccu::Variable round2Dst;
        round2Dst = localInput;
        // The Host passes output as localInput for the recursive-halving mode.
        ccu::Variable round2Src;
        round2Src = remotePartial[round2Channel];
        for (uint32_t segment = 0; segment < custom_rs::SEGMENT_4X1_NUM; ++segment) {
            round2Destinations[segment].addr = round2Dst;
            round2Destinations[segment].token = localInputToken;
            round2Sources[segment].addr = round2Src;
            round2Sources[segment].token = remotePartialToken[round2Channel];
            if (segment + 1U < custom_rs::SEGMENT_4X1_NUM) {
                round2Dst += round2Lengths[segment];
                round2Src += round2Lengths[segment];
            }
        }
        std::array<ccu::Event, custom_rs::SEGMENT_4X1_MAX_OUTSTANDING>
            round2Events;
        auto issueRound2 = [&](uint32_t segment, uint32_t lane) -> CcuResult {
            RS_CCU_CHK_RET(ccu::ReadReduce(kernelArg->channels[round2Channel],
                round2Destinations[segment], round2Sources[segment],
                round2Lengths[segment], HCCL_DATA_TYPE_FP32,
                HCCL_REDUCE_SUM, round2Events[lane], 1));
            return CCU_SUCCESS;
        };
        for (uint32_t lane = 0; lane < custom_rs::SEGMENT_4X1_MAX_OUTSTANDING; ++lane) {
            RS_CCU_CHK_RET(issueRound2(lane, lane));
        }
        for (uint32_t segment = 0; segment < custom_rs::SEGMENT_4X1_NUM; ++segment) {
            const uint32_t lane = segment % custom_rs::SEGMENT_4X1_MAX_OUTSTANDING;
            RS_CCU_CHK_RET(ccu::EventWait(round2Events[lane], 1));
            const uint32_t next = segment + custom_rs::SEGMENT_4X1_MAX_OUTSTANDING;
            if (next < custom_rs::SEGMENT_4X1_NUM) {
                RS_CCU_CHK_RET(issueRound2(next, lane));
            }
        }
    }
    return CCU_SUCCESS;
}

CcuResult CcuReduceScatterKernel_Comm_Small(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<ReduceScatterKernelArg *>(arg);

    // This variant is registered only for the small-data route.  It intentionally
    // contains no large-route fetch, partial-publication, or finish branches.
    ccu::Variable mode;
    ccu::Variable primary;
    ccu::Variable primaryToken;
    ccu::Variable localInput;
    ccu::Variable localInputToken;
    ccu::Variable value0;
    ccu::Variable value1;
    ccu::Variable value3;
    uint32_t argId = 0;
    RS_CCU_CHK_RET(ccu::LoadArg(mode, argId++));
    RS_CCU_CHK_RET(ccu::LoadArg(primary, argId++));
    RS_CCU_CHK_RET(ccu::LoadArg(primaryToken, argId++));
    RS_CCU_CHK_RET(ccu::LoadArg(localInput, argId++));
    RS_CCU_CHK_RET(ccu::LoadArg(localInputToken, argId++));
    RS_CCU_CHK_RET(ccu::LoadArg(value0, argId++));
    RS_CCU_CHK_RET(ccu::LoadArg(value1, argId++));
    RS_CCU_CHK_RET(ccu::LoadArg(value3, argId++));

    std::vector<ccu::Variable> remoteInput(kernelArg->channelCount);
    std::vector<ccu::Variable> remoteInputToken(kernelArg->channelCount);
    for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
        remoteInput[i] = ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[i], INPUT_XN_ID);
        remoteInputToken[i] =
            ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[i], INPUT_TOKEN_XN_ID);
    }

    ccu::Variable offset;

    const uint32_t smallBufferMemberCount = kernelArg->fusedMemberCount;
    const uint32_t smallBufferCapacityBlocksPerSource =
        kernelArg->smallBufferCapacityBlocksPerSource;
    if (smallBufferMemberCount == 0 ||
        smallBufferMemberCount > custom_rs::CCU_BUFFER_REDUCE_MAX_INPUTS ||
        smallBufferCapacityBlocksPerSource == 0) {
        return CCU_E_PARA;
    }
    ccu::Array<ccu::CcuBuffer> smallBuffers(
        smallBufferMemberCount * smallBufferCapacityBlocksPerSource);
    ccu::Array<ccu::Event> smallBufferEvents(1);
    ccu::LocalAddr smallBufferLocalSource;
    ccu::LocalAddr smallBufferOutput;
    std::vector<ccu::RemoteAddr> smallBufferRemoteSources(smallBufferMemberCount);
    std::vector<ccu::CcuBuffer> smallBufferReduceInputs;
    smallBufferReduceInputs.reserve(smallBufferMemberCount);
    for (uint32_t memberIdx = 0; memberIdx < smallBufferMemberCount; ++memberIdx) {
        smallBufferReduceInputs.push_back(
            smallBuffers[memberIdx * smallBufferCapacityBlocksPerSource]);
    }

    // The asymmetric 8+4 small path keeps address exchange outside the useful
    // Buffer launch; symmetric and single-die layouts fuse it below.
    CCU_IF(mode == static_cast<uint64_t>(custom_rs::CommKernelMode::EXCHANGE_ADDRESSES)) {
        const uint16_t inputMask = 1U << INPUT_XN_ID;
        const uint16_t tokenMask = 1U << INPUT_TOKEN_XN_ID;
        const uint16_t allMask = inputMask | tokenMask;
        for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
            RS_CCU_CHK_RET(ccu::WriteVariableWithNotify(kernelArg->channels[i], localInput,
                INPUT_XN_ID, CHANNEL_CKE_INDEX, inputMask));
            RS_CCU_CHK_RET(ccu::WriteVariableWithNotify(kernelArg->channels[i], localInputToken,
                INPUT_TOKEN_XN_ID, CHANNEL_CKE_INDEX, tokenMask));
        }
        for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
            RS_CCU_CHK_RET(ccu::NotifyWait(kernelArg->channels[i], CHANNEL_CKE_INDEX, allMask));
        }
    }

    CCU_IF(mode == static_cast<uint64_t>(custom_rs::CommKernelMode::SMALL_BUFFER_TILE)) {
        if (kernelArg->smallBufferFuseSync) {
            const uint16_t inputMask = 1U << INPUT_XN_ID;
            const uint16_t tokenMask = 1U << INPUT_TOKEN_XN_ID;
            const uint16_t allMask = inputMask | tokenMask;
            for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
                RS_CCU_CHK_RET(ccu::WriteVariableWithNotify(kernelArg->channels[i], localInput,
                    INPUT_XN_ID, CHANNEL_CKE_INDEX, inputMask));
                RS_CCU_CHK_RET(ccu::WriteVariableWithNotify(kernelArg->channels[i],
                    localInputToken, INPUT_TOKEN_XN_ID, CHANNEL_CKE_INDEX, tokenMask));
            }
            for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
                RS_CCU_CHK_RET(ccu::NotifyWait(kernelArg->channels[i],
                    CHANNEL_CKE_INDEX, allMask));
            }
        }

        offset = 0;
        for (uint32_t slice = 0; slice < kernelArg->rankId; ++slice) offset += value3;
        offset += value0;
        ccu::Variable tileBytes;
        tileBytes = value1;
        ccu::Event smallBufferEvent = smallBufferEvents[0];

        for (uint32_t memberIdx = 0; memberIdx < smallBufferMemberCount; ++memberIdx) {
            const uint16_t mask = static_cast<uint16_t>(1U << memberIdx);
            const uint32_t bufferBase =
                memberIdx * smallBufferCapacityBlocksPerSource;
            const uint32_t sourceRank = kernelArg->fusedMemberRanks[memberIdx];
            if (sourceRank == kernelArg->rankId) {
                smallBufferLocalSource.addr = localInput;
                smallBufferLocalSource.addr += offset;
                smallBufferLocalSource.token = localInputToken;
                RS_CCU_CHK_RET(ccu::LocalCopy(smallBuffers[bufferBase],
                    smallBufferLocalSource, tileBytes, smallBufferEvent, mask));
            } else {
                const uint32_t channelIdx = FindChannelIndex(*kernelArg, sourceRank);
                if (channelIdx >= kernelArg->channelCount) return CCU_E_INTERNAL;
                SetRemoteInput(smallBufferRemoteSources[memberIdx],
                    remoteInput, remoteInputToken,
                    channelIdx, offset);
                RS_CCU_CHK_RET(ccu::Read(kernelArg->channels[channelIdx],
                    smallBuffers[bufferBase], smallBufferRemoteSources[memberIdx],
                    tileBytes, smallBufferEvent, mask));
            }
        }

        const uint16_t readyMask =
            static_cast<uint16_t>((1U << smallBufferMemberCount) - 1U);
        RS_CCU_CHK_RET(ccu::EventWait(smallBufferEvent, readyMask));
        if (smallBufferMemberCount > 1) {
            RS_CCU_CHK_RET(ccu::LocalReduce(smallBufferReduceInputs.data(),
                smallBufferMemberCount,
                HCCL_DATA_TYPE_FP32, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM,
                tileBytes, smallBufferEvent, 1));
            RS_CCU_CHK_RET(ccu::EventWait(smallBufferEvent, 1));
        }
        smallBufferOutput.addr = primary;
        smallBufferOutput.addr += value0;
        smallBufferOutput.token = primaryToken;
        RS_CCU_CHK_RET(ccu::LocalCopy(smallBufferOutput, smallBufferReduceInputs[0],
            tileBytes, smallBufferEvent, 1));
        RS_CCU_CHK_RET(ccu::EventWait(smallBufferEvent, 1));

        if (kernelArg->smallBufferFuseSync) {
            for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
                RS_CCU_CHK_RET(ccu::NotifyRecord(kernelArg->channels[i],
                    CHANNEL_CKE_INDEX, SMALL_POST_SYNC_MASK));
            }
            for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
                RS_CCU_CHK_RET(ccu::NotifyWait(kernelArg->channels[i],
                    CHANNEL_CKE_INDEX, SMALL_POST_SYNC_MASK));
            }
        }
    }
    return CCU_SUCCESS;
}

CcuResult CcuReduceScatterKernel_Comm_Small_2X8(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<ReduceScatterKernelArg *>(arg);
    const uint32_t memberCount = kernelArg->fusedMemberCount;
    const uint32_t blocksPerSource = kernelArg->smallBufferCapacityBlocksPerSource;
    if (!kernelArg->smallBufferFuseSync || memberCount == 0U ||
        memberCount > custom_rs::CCU_BUFFER_REDUCE_MAX_INPUTS ||
        blocksPerSource == 0U ||
        memberCount * blocksPerSource > custom_rs::CCU_SCHED_MS_BLOCKS_PER_DIE) {
        return CCU_E_PARA;
    }

    // This kernel is registered only for 2x8 small data, so its ABI and
    // instruction stream contain no runtime mode or fallback fields.
    ccu::Variable outputAddr;
    ccu::Variable outputToken;
    ccu::Variable inputAddr;
    ccu::Variable inputToken;
    ccu::Variable sourceOffset;
    ccu::Variable tileBytes;
    uint32_t argId = 0;
    RS_CCU_CHK_RET(ccu::LoadArg(outputAddr, argId++));
    RS_CCU_CHK_RET(ccu::LoadArg(outputToken, argId++));
    RS_CCU_CHK_RET(ccu::LoadArg(inputAddr, argId++));
    RS_CCU_CHK_RET(ccu::LoadArg(inputToken, argId++));
    RS_CCU_CHK_RET(ccu::LoadArg(sourceOffset, argId++));
    RS_CCU_CHK_RET(ccu::LoadArg(tileBytes, argId++));

    std::vector<ccu::Variable> remoteInput(kernelArg->channelCount);
    std::vector<ccu::Variable> remoteInputToken(kernelArg->channelCount);
    for (uint32_t channel = 0; channel < kernelArg->channelCount; ++channel) {
        remoteInput[channel] = ccu::GetResByChannel<ccu::Variable>(
            kernelArg->channels[channel], INPUT_XN_ID);
        remoteInputToken[channel] = ccu::GetResByChannel<ccu::Variable>(
            kernelArg->channels[channel], INPUT_TOKEN_XN_ID);
    }

    const uint16_t inputMask = 1U << INPUT_XN_ID;
    const uint16_t tokenMask = 1U << INPUT_TOKEN_XN_ID;
    const uint16_t addressMask = inputMask | tokenMask;
    for (uint32_t channel = 0; channel < kernelArg->channelCount; ++channel) {
        RS_CCU_CHK_RET(ccu::WriteVariableWithNotify(kernelArg->channels[channel],
            inputAddr, INPUT_XN_ID, CHANNEL_CKE_INDEX, inputMask));
        RS_CCU_CHK_RET(ccu::WriteVariableWithNotify(kernelArg->channels[channel],
            inputToken, INPUT_TOKEN_XN_ID, CHANNEL_CKE_INDEX, tokenMask));
    }

    ccu::Array<ccu::CcuBuffer> buffers(memberCount * blocksPerSource);
    ccu::Array<ccu::Event> events(1);
    std::vector<ccu::CcuBuffer> reduceInputs;
    reduceInputs.reserve(memberCount);
    for (uint32_t member = 0; member < memberCount; ++member) {
        reduceInputs.push_back(buffers[member * blocksPerSource]);
    }

    ccu::Event event = events[0];
    ccu::LocalAddr localSource;
    std::vector<ccu::RemoteAddr> remoteSources(memberCount);
    uint32_t selfMember = memberCount;
    for (uint32_t member = 0; member < memberCount; ++member) {
        if (kernelArg->fusedMemberRanks[member] == kernelArg->rankId) {
            selfMember = member;
            break;
        }
    }

    // Self has no remote-address dependency. Start its GM-to-MS transfer before
    // waiting for peer address publication so the local copy covers part of
    // the fused PreSync latency. The other 2x8 die simply skips this block.
    if (selfMember != memberCount) {
        localSource.addr = inputAddr;
        localSource.addr += sourceOffset;
        localSource.token = inputToken;
        RS_CCU_CHK_RET(ccu::LocalCopy(buffers[selfMember * blocksPerSource],
            localSource, tileBytes, event, 1U << selfMember));
    }

    for (uint32_t channel = 0; channel < kernelArg->channelCount; ++channel) {
        RS_CCU_CHK_RET(ccu::NotifyWait(kernelArg->channels[channel],
            CHANNEL_CKE_INDEX, addressMask));
    }

    for (uint32_t member = 0; member < memberCount; ++member) {
        if (member == selfMember) continue;
        const uint16_t mask = static_cast<uint16_t>(1U << member);
        const uint32_t bufferBase = member * blocksPerSource;
        const uint32_t sourceRank = kernelArg->fusedMemberRanks[member];
        const uint32_t channelIndex = FindChannelIndex(*kernelArg, sourceRank);
        if (channelIndex >= kernelArg->channelCount) return CCU_E_INTERNAL;
        remoteSources[member].addr = remoteInput[channelIndex];
        remoteSources[member].addr += sourceOffset;
        remoteSources[member].token = remoteInputToken[channelIndex];
        RS_CCU_CHK_RET(ccu::Read(kernelArg->channels[channelIndex],
            buffers[bufferBase], remoteSources[member],
            tileBytes, event, mask));
    }

    const uint16_t readyMask = static_cast<uint16_t>((1U << memberCount) - 1U);
    RS_CCU_CHK_RET(ccu::EventWait(event, readyMask));
    if (memberCount > 1U) {
        RS_CCU_CHK_RET(ccu::LocalReduce(reduceInputs.data(), memberCount,
            HCCL_DATA_TYPE_FP32, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM,
            tileBytes, event, 1));
        RS_CCU_CHK_RET(ccu::EventWait(event, 1));
    }

    ccu::LocalAddr output;
    output.addr = outputAddr;
    output.token = outputToken;
    RS_CCU_CHK_RET(ccu::LocalCopy(output, reduceInputs[0], tileBytes, event, 1));

    // The peer lifetime barrier is independent of the local MS->GM write.
    // Submit it while the copy is in flight, then join the copy before return.
    for (uint32_t channel = 0; channel < kernelArg->channelCount; ++channel) {
        RS_CCU_CHK_RET(ccu::NotifyRecord(kernelArg->channels[channel],
            CHANNEL_CKE_INDEX, SMALL_POST_SYNC_MASK));
    }
    for (uint32_t channel = 0; channel < kernelArg->channelCount; ++channel) {
        RS_CCU_CHK_RET(ccu::NotifyWait(kernelArg->channels[channel],
            CHANNEL_CKE_INDEX, SMALL_POST_SYNC_MASK));
    }
    RS_CCU_CHK_RET(ccu::EventWait(event, 1));
    return CCU_SUCCESS;
}

CcuResult CcuReduceScatterKernel_Reduce_Small(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<ReduceScatterKernelArg *>(arg);
    if (kernelArg->channelCount == 0) return CCU_E_PARA;
    const ccu::Variable dieAnchor =
        ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[0], INPUT_XN_ID);
    (void)dieAnchor;
    ccu::Variable mode;
    ccu::Variable dstAddr;
    ccu::Variable dstToken;
    ccu::Variable srcAddr;
    ccu::Variable srcToken;
    ccu::Variable bytes;
    uint32_t argId = 0;
    RS_CCU_CHK_RET(ccu::LoadArg(mode, argId++));
    RS_CCU_CHK_RET(ccu::LoadArg(dstAddr, argId++));
    RS_CCU_CHK_RET(ccu::LoadArg(dstToken, argId++));
    RS_CCU_CHK_RET(ccu::LoadArg(srcAddr, argId++));
    RS_CCU_CHK_RET(ccu::LoadArg(srcToken, argId++));
    RS_CCU_CHK_RET(ccu::LoadArg(bytes, argId++));
    ccu::LocalAddr dst;
    ccu::LocalAddr src;
    ccu::Event event;
    CCU_IF(mode == static_cast<uint64_t>(custom_rs::ReduceKernelMode::REDUCE_RANGE)) {
        dst.addr = dstAddr;
        dst.token = dstToken;
        src.addr = srcAddr;
        src.token = srcToken;
        RS_CCU_CHK_RET(ccu::LocalReduce(dst, src, bytes,
            HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, event));
        RS_CCU_CHK_RET(ccu::EventWait(event));
    }
    return CCU_SUCCESS;
}

CcuResult CcuReduceScatterKernel_Reduce_Small_2X8(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<ReduceScatterKernelArg *>(arg);
    if (kernelArg->channelCount == 0U) return CCU_E_PARA;
    const ccu::Variable dieAnchor =
        ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[0], INPUT_XN_ID);
    (void)dieAnchor;

    ccu::Variable dstAddr;
    ccu::Variable dstToken;
    ccu::Variable srcAddr;
    ccu::Variable srcToken;
    ccu::Variable bytes;
    uint32_t argId = 0;
    RS_CCU_CHK_RET(ccu::LoadArg(dstAddr, argId++));
    RS_CCU_CHK_RET(ccu::LoadArg(dstToken, argId++));
    RS_CCU_CHK_RET(ccu::LoadArg(srcAddr, argId++));
    RS_CCU_CHK_RET(ccu::LoadArg(srcToken, argId++));
    RS_CCU_CHK_RET(ccu::LoadArg(bytes, argId++));

    ccu::LocalAddr dst;
    dst.addr = dstAddr;
    dst.token = dstToken;
    ccu::LocalAddr src;
    src.addr = srcAddr;
    src.token = srcToken;
    ccu::Event event;
    RS_CCU_CHK_RET(ccu::LocalReduce(dst, src, bytes,
        HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, event));
    RS_CCU_CHK_RET(ccu::EventWait(event));
    return CCU_SUCCESS;
}

CcuResult CcuReduceScatterKernel_Reduce_4X1(CcuKernelArg arg)
{
    (void)arg;
    ccu::Variable outputAddr;
    ccu::Variable outputToken;
    ccu::Variable scratchAddr;
    ccu::Variable scratchToken;
    ccu::Variable sendInputAddr;
    ccu::Variable keepInputAddr;
    ccu::Variable inputToken;
    ccu::Variable recvBytes;
    ccu::Variable round1TileBytes;
    ccu::Variable round1TailBytes;
    uint32_t argId = 0;
    RS_CCU_CHK_RET(ccu::LoadArg(outputAddr, argId++));
    RS_CCU_CHK_RET(ccu::LoadArg(outputToken, argId++));
    RS_CCU_CHK_RET(ccu::LoadArg(scratchAddr, argId++));
    RS_CCU_CHK_RET(ccu::LoadArg(scratchToken, argId++));
    RS_CCU_CHK_RET(ccu::LoadArg(sendInputAddr, argId++));
    RS_CCU_CHK_RET(ccu::LoadArg(keepInputAddr, argId++));
    RS_CCU_CHK_RET(ccu::LoadArg(inputToken, argId++));
    RS_CCU_CHK_RET(ccu::LoadArg(recvBytes, argId++));
    RS_CCU_CHK_RET(ccu::LoadArg(round1TileBytes, argId++));
    RS_CCU_CHK_RET(ccu::LoadArg(round1TailBytes, argId++));

    std::array<ccu::Variable, custom_rs::ROUND1_4X1_TILES_PER_BLOCK>
        round1Lengths;
    for (uint32_t tile = 0;
        tile < custom_rs::ROUND1_4X1_TILES_PER_BLOCK; ++tile) {
        round1Lengths[tile] = round1TileBytes;
    }
    round1Lengths.back() = round1TailBytes;

    // The outgoing partial is reduced in-place in scratch. The retained
    // partial is reduced directly in output for round-2 ReadReduce fusion.
    ccu::LocalAddr remoteSend;
    ccu::LocalAddr remoteKeep;
    ccu::LocalAddr selfSend;
    ccu::LocalAddr selfKeep;
    remoteSend.addr = scratchAddr;
    remoteSend.token = scratchToken;
    remoteKeep.addr = outputAddr;
    remoteKeep.token = outputToken;
    selfSend.addr = sendInputAddr;
    selfSend.token = inputToken;
    selfKeep.addr = keepInputAddr;
    selfKeep.token = inputToken;

    ccu::Event reduceEvent;
    for (uint32_t segment = 0; segment < custom_rs::ROUND1_4X1_SEGMENT_NUM; ++segment) {
        const bool outgoing = segment < custom_rs::ROUND1_4X1_TILES_PER_BLOCK;
        const uint32_t tile = segment % custom_rs::ROUND1_4X1_TILES_PER_BLOCK;
        ccu::LocalAddr dst = outgoing ? remoteSend : remoteKeep;
        ccu::LocalAddr src = outgoing ? selfSend : selfKeep;
        RS_CCU_CHK_RET(Wait4X1Round1Ready(segment));
        RS_CCU_CHK_RET(ccu::LocalReduce(dst, src, round1Lengths[tile],
            HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, reduceEvent));
        RS_CCU_CHK_RET(ccu::EventWait(reduceEvent));
        if (segment + 1U == custom_rs::ROUND1_4X1_TILES_PER_BLOCK) {
            RS_CCU_CHK_RET(Record4X1OutgoingReady());
        }
        if (segment + 1U == custom_rs::ROUND1_4X1_SEGMENT_NUM) {
            RS_CCU_CHK_RET(Record4X1RetainedReady());
        }
        if (tile + 1U < custom_rs::ROUND1_4X1_TILES_PER_BLOCK) {
            if (outgoing) {
                selfSend.addr += round1Lengths[tile];
                remoteSend.addr += round1Lengths[tile];
            } else {
                selfKeep.addr += round1Lengths[tile];
                remoteKeep.addr += round1Lengths[tile];
            }
        }
    }

    return CCU_SUCCESS;
}

CcuResult CcuReduceScatterKernel_Reduce_General(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<ReduceScatterKernelArg *>(arg);
    if (kernelArg->channelCount == 0) return CCU_E_PARA;
    const ccu::Variable dieAnchor =
        ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[0], INPUT_XN_ID);
    (void)dieAnchor;
    ccu::Variable mode;
    ccu::Variable dstAddr;
    ccu::Variable dstToken;
    ccu::Variable srcAddr;
    ccu::Variable srcToken;
    ccu::Variable bytes;
    uint32_t argId = 0;
    RS_CCU_CHK_RET(ccu::LoadArg(mode, argId++));
    RS_CCU_CHK_RET(ccu::LoadArg(dstAddr, argId++));
    RS_CCU_CHK_RET(ccu::LoadArg(dstToken, argId++));
    RS_CCU_CHK_RET(ccu::LoadArg(srcAddr, argId++));
    RS_CCU_CHK_RET(ccu::LoadArg(srcToken, argId++));
    RS_CCU_CHK_RET(ccu::LoadArg(bytes, argId++));
    ccu::LocalAddr dst;
    ccu::LocalAddr src;
    ccu::Event event;
    if (kernelArg->reduceRole == ReduceKernelRole::GENERAL ||
        kernelArg->reduceRole == ReduceKernelRole::MESH) {
        CCU_IF(mode == static_cast<uint64_t>(custom_rs::ReduceKernelMode::COPY_RANGE)) {
            dst.addr = dstAddr;
            dst.token = dstToken;
            src.addr = srcAddr;
            src.token = srcToken;
            RS_CCU_CHK_RET(ccu::LocalCopy(dst, src, bytes, event));
            RS_CCU_CHK_RET(ccu::EventWait(event));
        }
        CCU_IF(mode == static_cast<uint64_t>(custom_rs::ReduceKernelMode::REDUCE_RANGE)) {
            dst.addr = dstAddr;
            dst.token = dstToken;
            src.addr = srcAddr;
            src.token = srcToken;
            RS_CCU_CHK_RET(ccu::LocalReduce(dst, src, bytes,
                HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, event));
            RS_CCU_CHK_RET(ccu::EventWait(event));
        }
    }
    if (kernelArg->remoteLocalMemberCount != 0 &&
        (kernelArg->reduceRole == ReduceKernelRole::GENERAL ||
            kernelArg->reduceRole == ReduceKernelRole::MESH)) {
        CCU_IF(mode == static_cast<uint64_t>(
            custom_rs::ReduceKernelMode::REDUCE_MESH_SLOTS_TO_DST)) {
            dst.addr = dstAddr;
            dst.token = dstToken;
            src.addr = srcAddr;
            src.token = srcToken;
            for (uint32_t memberIdx = 0;
                memberIdx < kernelArg->remoteLocalMemberCount; ++memberIdx) {
                RS_CCU_CHK_RET(ccu::LocalReduce(dst, src, bytes,
                    HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, event));
                RS_CCU_CHK_RET(ccu::EventWait(event));
                if (memberIdx + 1U < kernelArg->remoteLocalMemberCount) src.addr += bytes;
            }
        }
    }
    if (kernelArg->reduceRole == ReduceKernelRole::GENERAL ||
        kernelArg->reduceRole == ReduceKernelRole::CLOS) {
        CCU_IF(mode == static_cast<uint64_t>(custom_rs::ReduceKernelMode::FOLD_SLOTS_IN_PLACE)) {
            dst.addr = srcAddr;
            dst.token = srcToken;
            src.addr = srcAddr;
            src.addr += bytes;
            src.token = srcToken;
            for (uint32_t memberIdx = 1; memberIdx < kernelArg->localMemberCount; ++memberIdx) {
                RS_CCU_CHK_RET(ccu::LocalReduce(dst, src, bytes,
                    HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, event));
                RS_CCU_CHK_RET(ccu::EventWait(event));
                if (memberIdx + 1U < kernelArg->localMemberCount) src.addr += bytes;
            }
        }
        if (kernelArg->publishTargetCount != 0 || kernelArg->aggregateSourceCount != 0) {
            ccu::RemoteAddr remotePartial;
            ccu::Variable partValue;
            CCU_IF(mode == static_cast<uint64_t>(
                custom_rs::ReduceKernelMode::READ_REDUCE_CLOS_PARTIALS)) {
                for (uint32_t targetIdx = 0;
                    targetIdx < kernelArg->publishTargetCount; ++targetIdx) {
                    const uint32_t channelIdx = FindChannelIndex(
                        *kernelArg, kernelArg->publishTargetRanks[targetIdx]);
                    if (channelIdx >= kernelArg->channelCount) return CCU_E_INTERNAL;
                    RS_CCU_CHK_RET(ccu::NotifyRecord(kernelArg->channels[channelIdx],
                        CHANNEL_CKE_INDEX, PARTIAL_READY_MASK));
                }
                for (uint32_t sourceIdx = 0;
                    sourceIdx < kernelArg->aggregateSourceCount; ++sourceIdx) {
                    const uint32_t channelIdx = FindChannelIndex(
                        *kernelArg, kernelArg->aggregateSourceRanks[sourceIdx]);
                    if (channelIdx >= kernelArg->channelCount) return CCU_E_INTERNAL;
                    RS_CCU_CHK_RET(ccu::NotifyWait(kernelArg->channels[channelIdx],
                        CHANNEL_CKE_INDEX, PARTIAL_READY_MASK));
                    remotePartial.addr = ccu::GetResByChannel<ccu::Variable>(
                        kernelArg->channels[channelIdx], PARTIAL_XN_ID);
                    partValue = kernelArg->aggregateSourceRemoteOffsets[sourceIdx];
                    remotePartial.addr += partValue;
                    remotePartial.token = ccu::GetResByChannel<ccu::Variable>(
                        kernelArg->channels[channelIdx], PARTIAL_TOKEN_XN_ID);
                    dst.addr = dstAddr;
                    partValue = kernelArg->aggregateSourceLocalOffsets[sourceIdx];
                    dst.addr += partValue;
                    dst.token = dstToken;
                    partValue = kernelArg->aggregateSourceBytes[sourceIdx];
                    RS_CCU_CHK_RET(ccu::ReadReduce(kernelArg->channels[channelIdx], dst,
                        remotePartial, partValue, HCCL_DATA_TYPE_FP32,
                        HCCL_REDUCE_SUM, event));
                    RS_CCU_CHK_RET(ccu::EventWait(event));
                }
            }
        }
    }
    return CCU_SUCCESS;
}

} // namespace ops_hccl
