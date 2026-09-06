/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Integrated CCU kernels for the Broadcast scheme matrix.
 */
#include <cstdint>
#include <vector>
#include <hcomm/hcomm_primitives.h>
#include "ccu_kernel.h"
#ifndef CCU_CHK_RET
#define CCU_CHK_RET(call) do { CcuResult ccuRet=(call); if (ccuRet!=CCU_SUCCESS) return ccuRet; } while (0)
#endif

namespace ops_hccl {
namespace impl_2x8_small {

constexpr uint32_t BUFFER_XN_ID = 1;
constexpr uint32_t TOKEN_XN_ID = 2;
constexpr uint32_t DONE_SYNC_ID = 3;
constexpr uint32_t CKE_INDEX = 0;

struct Broadcast2x8SmallSpecializedContext {
    Broadcast2x8SmallSpecializedKernelArg *arg = nullptr;

    // Task arguments local to this rank. Keep them separate from peer resources;
    // the local rank has no self-channel and therefore no channel-derived entry.
    ccu::Variable localBuffer;
    ccu::Variable localToken;
    ccu::Variable dataSize;

    // Channel-derived remote resources, indexed by global peer rank.
    std::vector<ccu::Variable> remoteBuffer;
    std::vector<ccu::Variable> remoteToken;

    ccu::Event transferEvent;
};

uint16_t RankBit(uint32_t rank)
{
    return static_cast<uint16_t>(uint32_t{1} << rank);
}

uint16_t BuildPeerMask(const Broadcast2x8SmallSpecializedKernelArg &arg)
{
    uint16_t mask = 0;
    for (uint32_t channelIndex = 0;
         channelIndex < arg.channelCount;
         ++channelIndex) {
        mask = static_cast<uint16_t>(
            mask | RankBit(arg.peerRanks[channelIndex]));
    }
    return mask;
}

CcuResult ValidateKernelArg(
    const Broadcast2x8SmallSpecializedKernelArg *arg)
{
    if (arg == nullptr ||
        arg->rankSize != BCAST_2X8_RANK_SIZE ||
        arg->rankId >= BCAST_2X8_RANK_SIZE ||
        arg->rootRank >= BCAST_2X8_RANK_SIZE) {
        return CcuResult::CCU_E_PARA;
    }

    if (arg->role != BCAST_2X8_ROLE_SENDER &&
        arg->role != BCAST_2X8_ROLE_RECEIVER &&
        arg->role != BCAST_2X8_ROLE_IDLE) {
        return CcuResult::CCU_E_PARA;
    }

    if (arg->role == BCAST_2X8_ROLE_IDLE) {
        return CcuResult::CCU_SUCCESS;
    }

    if (arg->channelCount == 0 ||
        arg->channelCount >= BCAST_2X8_RANK_SIZE) {
        return CcuResult::CCU_E_PARA;
    }

    if (arg->role == BCAST_2X8_ROLE_RECEIVER &&
        arg->channelCount != 1) {
        return CcuResult::CCU_E_PARA;
    }

    if (arg->role == BCAST_2X8_ROLE_SENDER &&
        arg->rankId != arg->rootRank) {
        return CcuResult::CCU_E_PARA;
    }

    if (arg->role == BCAST_2X8_ROLE_RECEIVER &&
        arg->rankId == arg->rootRank) {
        return CcuResult::CCU_E_PARA;
    }

    return CcuResult::CCU_SUCCESS;
}

CcuResult InitPeerResources(Broadcast2x8SmallSpecializedContext &ctx)
{
    CCU_CHK_RET(ValidateKernelArg(ctx.arg));

    if (ctx.arg->role == BCAST_2X8_ROLE_IDLE) {
        return CcuResult::CCU_SUCCESS;
    }

    ctx.remoteBuffer.resize(BCAST_2X8_RANK_SIZE);
    ctx.remoteToken.resize(BCAST_2X8_RANK_SIZE);

    for (uint32_t channelIndex = 0;
         channelIndex < ctx.arg->channelCount;
         ++channelIndex) {
        const uint32_t peer = ctx.arg->peerRanks[channelIndex];
        if (peer >= BCAST_2X8_RANK_SIZE || peer == ctx.arg->rankId) {
            return CcuResult::CCU_E_INTERNAL;
        }

        const ChannelHandle channel = ctx.arg->channels[channelIndex];
        ctx.remoteBuffer[peer] =
            ccu::GetResByChannel<ccu::Variable>(channel, BUFFER_XN_ID);
        ctx.remoteToken[peer] =
            ccu::GetResByChannel<ccu::Variable>(channel, TOKEN_XN_ID);
    }

    return CcuResult::CCU_SUCCESS;
}

CcuResult LoadTaskArgs(Broadcast2x8SmallSpecializedContext &ctx)
{
    uint32_t argId = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.localBuffer, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.localToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.dataSize, argId++));
    return CcuResult::CCU_SUCCESS;
}

CcuResult PublishLocalAddressToRoot(
    Broadcast2x8SmallSpecializedContext &ctx)
{
    const ChannelHandle rootChannel = ctx.arg->channels[0];

    CCU_CHK_RET(ccu::WriteVariableWithNotify(
        rootChannel,
        ctx.localBuffer,
        BUFFER_XN_ID,
        CKE_INDEX,
        1U << BUFFER_XN_ID));

    CCU_CHK_RET(ccu::WriteVariableWithNotify(
        rootChannel,
        ctx.localToken,
        TOKEN_XN_ID,
        CKE_INDEX,
        1U << TOKEN_XN_ID));

    return CcuResult::CCU_SUCCESS;
}

CcuResult RunReceiver(Broadcast2x8SmallSpecializedContext &ctx)
{
    // The root-specialized receiver kernel owns exactly one channel: root.
    CCU_CHK_RET(PublishLocalAddressToRoot(ctx));
    CCU_CHK_RET(ccu::NotifyWait(
        ctx.arg->channels[0],
        CKE_INDEX,
        1U << DONE_SYNC_ID));
    return CcuResult::CCU_SUCCESS;
}

CcuResult WaitAllReceiverAddresses(
    Broadcast2x8SmallSpecializedContext &ctx)
{
    constexpr uint32_t addressReadyMask =
        (1U << BUFFER_XN_ID) | (1U << TOKEN_XN_ID);

    for (uint32_t channelIndex = 0;
         channelIndex < ctx.arg->channelCount;
         ++channelIndex) {
        CCU_CHK_RET(ccu::NotifyWait(
            ctx.arg->channels[channelIndex],
            CKE_INDEX,
            addressReadyMask));
    }
    return CcuResult::CCU_SUCCESS;
}

CcuResult IssueAllWrites(Broadcast2x8SmallSpecializedContext &ctx)
{
    for (uint32_t channelIndex = 0;
         channelIndex < ctx.arg->channelCount;
         ++channelIndex) {
        const uint32_t peer = ctx.arg->peerRanks[channelIndex];

        ccu::LocalAddr src;
        src.addr = ctx.localBuffer;
        src.token = ctx.localToken;

        ccu::RemoteAddr dst;
        dst.addr = ctx.remoteBuffer[peer];
        dst.token = ctx.remoteToken[peer];

        CCU_CHK_RET(ccu::Write(
            ctx.arg->channels[channelIndex],
            dst,
            src,
            ctx.dataSize,
            ctx.transferEvent,
            RankBit(peer)));
    }

    CCU_CHK_RET(ccu::EventWait(
        ctx.transferEvent,
        BuildPeerMask(*ctx.arg)));
    return CcuResult::CCU_SUCCESS;
}

CcuResult RecordAllDone(Broadcast2x8SmallSpecializedContext &ctx)
{
    for (uint32_t channelIndex = 0;
         channelIndex < ctx.arg->channelCount;
         ++channelIndex) {
        CCU_CHK_RET(ccu::NotifyRecord(
            ctx.arg->channels[channelIndex],
            CKE_INDEX,
            1U << DONE_SYNC_ID));
    }
    return CcuResult::CCU_SUCCESS;
}

CcuResult RunSender(Broadcast2x8SmallSpecializedContext &ctx)
{
    CCU_CHK_RET(WaitAllReceiverAddresses(ctx));

    // All peer Writes are issued first and share one Event. There is no
    // per-peer EventWait, so transfers on distinct channels may overlap.
    CCU_CHK_RET(IssueAllWrites(ctx));

    CCU_CHK_RET(RecordAllDone(ctx));
    return CcuResult::CCU_SUCCESS;
}

CcuResult RunSpecializedKernel(CcuKernelArg kernelArg)
{
    auto *arg = static_cast<Broadcast2x8SmallSpecializedKernelArg *>(kernelArg);
    CCU_CHK_RET(ValidateKernelArg(arg));

    // Idle kernels exist only to give the translator an unambiguous IO-Die
    // channel binding. The host never launches them on the data path.
    if (arg->role == BCAST_2X8_ROLE_IDLE) {
        return CcuResult::CCU_SUCCESS;
    }

    Broadcast2x8SmallSpecializedContext ctx{};
    ctx.arg = arg;

    CCU_CHK_RET(InitPeerResources(ctx));
    CCU_CHK_RET(LoadTaskArgs(ctx));

    if (arg->role == BCAST_2X8_ROLE_RECEIVER) {
        return RunReceiver(ctx);
    }
    return RunSender(ctx);
}

} // namespace impl_2x8_small

CcuResult Ccu2x8SmallRootSpecializedIntraKernel(CcuKernelArg arg)
{
    using namespace impl_2x8_small;
    return RunSpecializedKernel(arg);
}

CcuResult Ccu2x8SmallRootSpecializedInterKernel(CcuKernelArg arg)
{
    using namespace impl_2x8_small;
    return RunSpecializedKernel(arg);
}

} // namespace ops_hccl


namespace ops_hccl {
namespace impl_2x8_large {

// Five resources only. A bit is reused by a later stage only after the earlier
// stage has completed its full Record -> Wait lifecycle on that same channel.
constexpr uint32_t BUFFER_XN_ID = 1;
constexpr uint32_t TOKEN_XN_ID = 2;
constexpr uint32_t SCATTER_DONE_ID = 3;
constexpr uint32_t FANOUT_DONE_ID = 4;
constexpr uint32_t CKE_INDEX = 0;

constexpr uint64_t PIPE_STAGE_SCATTER_R0 = 0;
constexpr uint64_t PIPE_STAGE_SCATTER_R1_FANOUT_R0 = 1;
constexpr uint64_t PIPE_STAGE_FANOUT_R1 = 2;
constexpr uint32_t ROUND0 = 0;
constexpr uint32_t ROUND1 = 1;

struct PipelineTaskLayout {
    ccu::Variable localBuffer;
    ccu::Variable localToken;
    ccu::Variable baseSliceBytes;
    ccu::Variable baseRound0Bytes;
    ccu::Variable baseRound1Bytes;
    ccu::Variable lastRound0Bytes;
    ccu::Variable lastRound1Bytes;
    ccu::Variable stageId;
};

struct Broadcast2x8PipelineContext {
    Broadcast2x8OwnerPipelineKernelArg *arg = nullptr;
    PipelineTaskLayout task;
    std::vector<ccu::Variable> peerBuffer;
    std::vector<ccu::Variable> peerToken;
    ccu::Event transferEvent;
};

uint16_t RankBit(uint32_t rank)
{
    return static_cast<uint16_t>(1U << rank);
}

uint32_t OwnerSliceIndex(uint32_t ownerRank, uint32_t rootRank)
{
    return ownerRank < rootRank ? ownerRank : ownerRank - 1U;
}

CcuResult ValidateKernelArg(const Broadcast2x8OwnerPipelineKernelArg *arg)
{
    if (arg == nullptr || arg->rankSize != BCAST_2X8_RANK_SIZE ||
        arg->rankId >= arg->rankSize || arg->rootRank >= arg->rankSize) {
        return CcuResult::CCU_E_PARA;
    }
    if (arg->channelCount == 0 || arg->channelCount >= arg->rankSize) {
        return CcuResult::CCU_E_INTERNAL;
    }
    for (uint32_t channelIndex = 0; channelIndex < arg->channelCount; ++channelIndex) {
        const uint32_t peer = arg->peerRanks[channelIndex];
        if (peer >= arg->rankSize || peer == arg->rankId) {
            return CcuResult::CCU_E_INTERNAL;
        }
    }
    return CcuResult::CCU_SUCCESS;
}

CcuResult InitPeerResources(Broadcast2x8PipelineContext &ctx)
{
    ctx.peerBuffer.resize(ctx.arg->rankSize);
    ctx.peerToken.resize(ctx.arg->rankSize);
    for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
        const uint32_t peer = ctx.arg->peerRanks[channelIndex];
        ChannelHandle channel = ctx.arg->channels[channelIndex];
        ctx.peerBuffer[peer] = ccu::GetResByChannel<ccu::Variable>(channel, BUFFER_XN_ID);
        ctx.peerToken[peer] = ccu::GetResByChannel<ccu::Variable>(channel, TOKEN_XN_ID);
    }
    return CcuResult::CCU_SUCCESS;
}

CcuResult LoadTaskArgs(Broadcast2x8PipelineContext &ctx)
{
    uint32_t argId = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.task.localBuffer, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.task.localToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.task.baseSliceBytes, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.task.baseRound0Bytes, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.task.baseRound1Bytes, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.task.lastRound0Bytes, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.task.lastRound1Bytes, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.task.stageId, argId++));
    return CcuResult::CCU_SUCCESS;
}

CcuResult PublishAddrToPeer(Broadcast2x8PipelineContext &ctx, uint32_t channelIndex)
{
    ChannelHandle channel = ctx.arg->channels[channelIndex];
    CCU_CHK_RET(ccu::WriteVariableWithNotify(
        channel, ctx.task.localBuffer, BUFFER_XN_ID,
        CKE_INDEX, 1U << BUFFER_XN_ID));
    CCU_CHK_RET(ccu::WriteVariableWithNotify(
        channel, ctx.task.localToken, TOKEN_XN_ID,
        CKE_INDEX, 1U << TOKEN_XN_ID));
    return CcuResult::CCU_SUCCESS;
}

CcuResult WaitPeerAddr(Broadcast2x8PipelineContext &ctx, uint32_t channelIndex)
{
    constexpr uint32_t readyMask =
        (1U << BUFFER_XN_ID) | (1U << TOKEN_XN_ID);
    CCU_CHK_RET(ccu::NotifyWait(
        ctx.arg->channels[channelIndex], CKE_INDEX, readyMask));
    return CcuResult::CCU_SUCCESS;
}

CcuResult RecordSync(Broadcast2x8PipelineContext &ctx,
    uint32_t channelIndex, uint32_t syncId)
{
    CCU_CHK_RET(ccu::NotifyRecord(
        ctx.arg->channels[channelIndex], CKE_INDEX, 1U << syncId));
    return CcuResult::CCU_SUCCESS;
}

CcuResult WaitSync(Broadcast2x8PipelineContext &ctx,
    uint32_t channelIndex, uint32_t syncId)
{
    CCU_CHK_RET(ccu::NotifyWait(
        ctx.arg->channels[channelIndex], CKE_INDEX, 1U << syncId));
    return CcuResult::CCU_SUCCESS;
}

ccu::Variable MakeSliceOffsetBytes(uint32_t sliceIndex,
    ccu::Variable &baseSliceBytes)
{
    ccu::Variable offset;
    offset = uint64_t{0};
    for (uint32_t index = 0; index < sliceIndex; ++index) {
        offset += baseSliceBytes;
    }
    return offset;
}

ccu::Variable MakeRound0Bytes(uint32_t sliceIndex, PipelineTaskLayout &task)
{
    ccu::Variable bytes;
    bytes = task.baseRound0Bytes;
    if (sliceIndex == BCAST_2X8_OWNER_NUM - 1U) {
        bytes = task.lastRound0Bytes;
    }
    return bytes;
}

ccu::Variable MakeRound1Bytes(uint32_t sliceIndex, PipelineTaskLayout &task)
{
    ccu::Variable bytes;
    bytes = task.baseRound1Bytes;
    if (sliceIndex == BCAST_2X8_OWNER_NUM - 1U) {
        bytes = task.lastRound1Bytes;
    }
    return bytes;
}

ccu::Variable MakeRoundOffsetBytes(uint32_t sliceIndex, uint32_t round,
    PipelineTaskLayout &task)
{
    ccu::Variable offset = MakeSliceOffsetBytes(sliceIndex, task.baseSliceBytes);
    if (round == ROUND1) {
        ccu::Variable round0Bytes = MakeRound0Bytes(sliceIndex, task);
        offset += round0Bytes;
    }
    return offset;
}

ccu::Variable MakeRoundBytes(uint32_t sliceIndex, uint32_t round,
    PipelineTaskLayout &task)
{
    if (round == ROUND0) {
        return MakeRound0Bytes(sliceIndex, task);
    }
    return MakeRound1Bytes(sliceIndex, task);
}

CcuResult WriteRoundToPeer(Broadcast2x8PipelineContext &ctx,
    uint32_t channelIndex, uint32_t sliceIndex, uint32_t round,
    uint16_t eventMask)
{
    const uint32_t peer = ctx.arg->peerRanks[channelIndex];
    ccu::Variable offset = MakeRoundOffsetBytes(sliceIndex, round, ctx.task);
    ccu::Variable bytes = MakeRoundBytes(sliceIndex, round, ctx.task);

    ccu::LocalAddr src;
    src.addr = ctx.task.localBuffer;
    src.addr += offset;
    src.token = ctx.task.localToken;

    ccu::RemoteAddr dst;
    dst.addr = ctx.peerBuffer[peer];
    dst.addr += offset;
    dst.token = ctx.peerToken[peer];

    CCU_CHK_RET(ccu::Write(ctx.arg->channels[channelIndex], dst, src, bytes,
        ctx.transferEvent, eventMask));
    return CcuResult::CCU_SUCCESS;
}

uint16_t AllPeerMask(const Broadcast2x8OwnerPipelineKernelArg &arg)
{
    uint16_t mask = 0;
    for (uint32_t channelIndex = 0; channelIndex < arg.channelCount; ++channelIndex) {
        mask = static_cast<uint16_t>(mask | RankBit(arg.peerRanks[channelIndex]));
    }
    return mask;
}

uint16_t FanoutPeerMask(const Broadcast2x8OwnerPipelineKernelArg &arg)
{
    uint16_t mask = 0;
    for (uint32_t channelIndex = 0; channelIndex < arg.channelCount; ++channelIndex) {
        const uint32_t peer = arg.peerRanks[channelIndex];
        if (peer != arg.rootRank) {
            mask = static_cast<uint16_t>(mask | RankBit(peer));
        }
    }
    return mask;
}

CcuResult RootScatterRound(Broadcast2x8PipelineContext &ctx, uint32_t round)
{
    for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
        CCU_CHK_RET(WaitPeerAddr(ctx, channelIndex));
    }

    // All writes in one layer are issued before one shared EventWait.
    for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
        const uint32_t ownerRank = ctx.arg->peerRanks[channelIndex];
        const uint32_t sliceIndex = OwnerSliceIndex(ownerRank, ctx.arg->rootRank);
        CCU_CHK_RET(WriteRoundToPeer(
            ctx, channelIndex, sliceIndex, round, RankBit(ownerRank)));
    }

    CCU_CHK_RET(ccu::EventWait(ctx.transferEvent, AllPeerMask(*ctx.arg)));
    for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
        CCU_CHK_RET(RecordSync(ctx, channelIndex, SCATTER_DONE_ID));
    }
    return CcuResult::CCU_SUCCESS;
}

CcuResult OwnerScatterReceive(Broadcast2x8PipelineContext &ctx)
{
    // Only one of the two layer kernels contains the root channel.
    const uint32_t rootChannelIndex =
        ctx.arg->channelIndexByRank[ctx.arg->rootRank];
    if (rootChannelIndex != INVALID_VALUE_RANKID &&
        rootChannelIndex < ctx.arg->channelCount) {
        CCU_CHK_RET(PublishAddrToPeer(ctx, rootChannelIndex));
        CCU_CHK_RET(WaitSync(ctx, rootChannelIndex, SCATTER_DONE_ID));
    }
    return CcuResult::CCU_SUCCESS;
}

CcuResult OwnerFanoutRound(Broadcast2x8PipelineContext &ctx, uint32_t round)
{
    // Root is excluded because it already owns the complete source buffer.
    for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
        if (ctx.arg->peerRanks[channelIndex] != ctx.arg->rootRank) {
            CCU_CHK_RET(PublishAddrToPeer(ctx, channelIndex));
        }
    }
    for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
        if (ctx.arg->peerRanks[channelIndex] != ctx.arg->rootRank) {
            CCU_CHK_RET(WaitPeerAddr(ctx, channelIndex));
        }
    }

    const uint32_t ownSliceIndex =
        OwnerSliceIndex(ctx.arg->rankId, ctx.arg->rootRank);
    for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
        const uint32_t peer = ctx.arg->peerRanks[channelIndex];
        if (peer != ctx.arg->rootRank) {
            CCU_CHK_RET(WriteRoundToPeer(
                ctx, channelIndex, ownSliceIndex, round, RankBit(peer)));
        }
    }

    const uint16_t activeMask = FanoutPeerMask(*ctx.arg);
    if (activeMask != 0) {
        CCU_CHK_RET(ccu::EventWait(ctx.transferEvent, activeMask));
    }

    // The symmetric handshake guarantees that all pieces of this round have
    // arrived before the notify resource is reused by a later stage.
    for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
        if (ctx.arg->peerRanks[channelIndex] != ctx.arg->rootRank) {
            CCU_CHK_RET(RecordSync(ctx, channelIndex, FANOUT_DONE_ID));
        }
    }
    for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
        if (ctx.arg->peerRanks[channelIndex] != ctx.arg->rootRank) {
            CCU_CHK_RET(WaitSync(ctx, channelIndex, FANOUT_DONE_ID));
        }
    }
    return CcuResult::CCU_SUCCESS;
}

CcuResult RunStage0(Broadcast2x8PipelineContext &ctx)
{
    if (ctx.arg->rankId == ctx.arg->rootRank) {
        return RootScatterRound(ctx, ROUND0);
    }
    return OwnerScatterReceive(ctx);
}

CcuResult RunStage1(Broadcast2x8PipelineContext &ctx)
{
    if (ctx.arg->rankId == ctx.arg->rootRank) {
        return RootScatterRound(ctx, ROUND1);
    }

    // In the one layer that contains root, publish the R1 destination first.
    // Do not wait for R1 yet: fanout R0 runs while root writes R1 into the
    // disjoint second half of this owner's slice.
    const uint32_t rootChannelIndex =
        ctx.arg->channelIndexByRank[ctx.arg->rootRank];
    if (rootChannelIndex != INVALID_VALUE_RANKID &&
        rootChannelIndex < ctx.arg->channelCount) {
        CCU_CHK_RET(PublishAddrToPeer(ctx, rootChannelIndex));
    }

    CCU_CHK_RET(OwnerFanoutRound(ctx, ROUND0));

    if (rootChannelIndex != INVALID_VALUE_RANKID &&
        rootChannelIndex < ctx.arg->channelCount) {
        CCU_CHK_RET(WaitSync(ctx, rootChannelIndex, SCATTER_DONE_ID));
    }
    return CcuResult::CCU_SUCCESS;
}

CcuResult RunStage2(Broadcast2x8PipelineContext &ctx)
{
    if (ctx.arg->rankId == ctx.arg->rootRank) {
        return CcuResult::CCU_SUCCESS;
    }
    return OwnerFanoutRound(ctx, ROUND1);
}

CcuResult RunCompactPipelineKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<Broadcast2x8OwnerPipelineKernelArg *>(arg);
    CCU_CHK_RET(ValidateKernelArg(kernelArg));

    Broadcast2x8PipelineContext ctx{};
    ctx.arg = kernelArg;
    CCU_CHK_RET(InitPeerResources(ctx));
    CCU_CHK_RET(LoadTaskArgs(ctx));

    CCU_IF(ctx.task.stageId == PIPE_STAGE_SCATTER_R0)
    {
        CCU_CHK_RET(RunStage0(ctx));
    }

    CCU_IF(ctx.task.stageId == PIPE_STAGE_SCATTER_R1_FANOUT_R0)
    {
        CCU_CHK_RET(RunStage1(ctx));
    }

    CCU_IF(ctx.task.stageId == PIPE_STAGE_FANOUT_R1)
    {
        CCU_CHK_RET(RunStage2(ctx));
    }

    return CcuResult::CCU_SUCCESS;
}

} // namespace impl_2x8_large

CcuResult Ccu2x8OwnerPipeline2CompactIntraKernel(CcuKernelArg arg)
{
    using namespace impl_2x8_large;
    return RunCompactPipelineKernel(arg);
}

CcuResult Ccu2x8OwnerPipeline2CompactInterKernel(CcuKernelArg arg)
{
    using namespace impl_2x8_large;
    return RunCompactPipelineKernel(arg);
}

} // namespace ops_hccl


namespace ops_hccl {
namespace impl_4x1_small {

constexpr uint32_t BUFFER_XN_ID = 1;
constexpr uint32_t TOKEN_XN_ID = 2;
constexpr uint32_t DONE_SYNC_ID = 3;
constexpr uint32_t CKE_INDEX = 0;

struct Broadcast4x1SmallSpecializedContext {
    Broadcast4x1SmallSpecializedKernelArg *arg = nullptr;

    // Local task arguments. The local rank has no self-channel, so local
    // resources must not be stored in a peer-indexed channel resource array.
    ccu::Variable localBuffer;
    ccu::Variable localToken;
    ccu::Variable dataSize;

    // Channel-derived remote resources, indexed by global peer rank.
    std::vector<ccu::Variable> remoteBuffer;
    std::vector<ccu::Variable> remoteToken;

    ccu::Event transferEvent;
};

uint16_t RankBit(uint32_t rank)
{
    return static_cast<uint16_t>(uint32_t{1} << rank);
}

CcuResult ValidateKernelArg(const Broadcast4x1SmallSpecializedKernelArg *arg)
{
    if (arg == nullptr ||
        arg->rankSize != BCAST_4X1_RANK_SIZE ||
        arg->rankId >= BCAST_4X1_RANK_SIZE ||
        arg->rootRank >= BCAST_4X1_RANK_SIZE ||
        arg->netLayer == INVALID_VALUE_RANKID) {
        return CcuResult::CCU_E_PARA;
    }

    if (arg->role == BCAST_4X1_ROLE_SENDER) {
        if (arg->rankId != arg->rootRank ||
            arg->channelCount != BCAST_4X1_RANK_SIZE - 1U) {
            return CcuResult::CCU_E_PARA;
        }
        return CcuResult::CCU_SUCCESS;
    }

    if (arg->role == BCAST_4X1_ROLE_RECEIVER) {
        if (arg->rankId == arg->rootRank || arg->channelCount != 1U) {
            return CcuResult::CCU_E_PARA;
        }
        return CcuResult::CCU_SUCCESS;
    }

    return CcuResult::CCU_E_PARA;
}

CcuResult InitPeerResources(Broadcast4x1SmallSpecializedContext &ctx)
{
    CCU_CHK_RET(ValidateKernelArg(ctx.arg));

    ctx.remoteBuffer.resize(BCAST_4X1_RANK_SIZE);
    ctx.remoteToken.resize(BCAST_4X1_RANK_SIZE);

    for (uint32_t channelIndex = 0;
         channelIndex < ctx.arg->channelCount;
         ++channelIndex) {
        const uint32_t peer = ctx.arg->peerRanks[channelIndex];
        if (peer >= BCAST_4X1_RANK_SIZE || peer == ctx.arg->rankId) {
            return CcuResult::CCU_E_INTERNAL;
        }

        const ChannelHandle channel = ctx.arg->channels[channelIndex];
        ctx.remoteBuffer[peer] =
            ccu::GetResByChannel<ccu::Variable>(channel, BUFFER_XN_ID);
        ctx.remoteToken[peer] =
            ccu::GetResByChannel<ccu::Variable>(channel, TOKEN_XN_ID);
    }
    return CcuResult::CCU_SUCCESS;
}

CcuResult LoadTaskArgs(Broadcast4x1SmallSpecializedContext &ctx)
{
    uint32_t argId = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.localBuffer, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.localToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.dataSize, argId++));
    return CcuResult::CCU_SUCCESS;
}

CcuResult PublishLocalAddressToRoot(Broadcast4x1SmallSpecializedContext &ctx)
{
    const ChannelHandle rootChannel = ctx.arg->channels[0];

    CCU_CHK_RET(ccu::WriteVariableWithNotify(
        rootChannel,
        ctx.localBuffer,
        BUFFER_XN_ID,
        CKE_INDEX,
        1U << BUFFER_XN_ID));

    CCU_CHK_RET(ccu::WriteVariableWithNotify(
        rootChannel,
        ctx.localToken,
        TOKEN_XN_ID,
        CKE_INDEX,
        1U << TOKEN_XN_ID));
    return CcuResult::CCU_SUCCESS;
}

CcuResult RunReceiver(Broadcast4x1SmallSpecializedContext &ctx)
{
    // A root-specialized receiver owns exactly one channel: the root channel.
    CCU_CHK_RET(PublishLocalAddressToRoot(ctx));
    CCU_CHK_RET(ccu::NotifyWait(
        ctx.arg->channels[0],
        CKE_INDEX,
        1U << DONE_SYNC_ID));
    return CcuResult::CCU_SUCCESS;
}

CcuResult WaitReceiverReady(Broadcast4x1SmallSpecializedContext &ctx,
    uint32_t channelIndex)
{
    constexpr uint32_t readyMask =
        (1U << BUFFER_XN_ID) | (1U << TOKEN_XN_ID);
    return ccu::NotifyWait(
        ctx.arg->channels[channelIndex], CKE_INDEX, readyMask);
}

CcuResult IssueWrite(Broadcast4x1SmallSpecializedContext &ctx,
    uint32_t channelIndex)
{
    const uint32_t peer = ctx.arg->peerRanks[channelIndex];

    ccu::LocalAddr src;
    src.addr = ctx.localBuffer;
    src.token = ctx.localToken;

    ccu::RemoteAddr dst;
    dst.addr = ctx.remoteBuffer[peer];
    dst.token = ctx.remoteToken[peer];

    return ccu::Write(
        ctx.arg->channels[channelIndex],
        dst,
        src,
        ctx.dataSize,
        ctx.transferEvent,
        RankBit(peer));
}

CcuResult RecordDone(Broadcast4x1SmallSpecializedContext &ctx,
    uint32_t channelIndex)
{
    return ccu::NotifyRecord(
        ctx.arg->channels[channelIndex],
        CKE_INDEX,
        1U << DONE_SYNC_ID);
}

CcuResult RunSender(Broadcast4x1SmallSpecializedContext &ctx)
{
    // Sender channelCount is fixed to three when this root-specific kernel is
    // registered. Keep the data path explicit: three ready waits, three Writes,
    // one EventWait, and three completion records.
    CCU_CHK_RET(WaitReceiverReady(ctx, 0));
    CCU_CHK_RET(WaitReceiverReady(ctx, 1));
    CCU_CHK_RET(WaitReceiverReady(ctx, 2));

    CCU_CHK_RET(IssueWrite(ctx, 0));
    CCU_CHK_RET(IssueWrite(ctx, 1));
    CCU_CHK_RET(IssueWrite(ctx, 2));

    const uint16_t peerMask = static_cast<uint16_t>(
        RankBit(ctx.arg->peerRanks[0]) |
        RankBit(ctx.arg->peerRanks[1]) |
        RankBit(ctx.arg->peerRanks[2]));
    CCU_CHK_RET(ccu::EventWait(ctx.transferEvent, peerMask));

    CCU_CHK_RET(RecordDone(ctx, 0));
    CCU_CHK_RET(RecordDone(ctx, 1));
    CCU_CHK_RET(RecordDone(ctx, 2));
    return CcuResult::CCU_SUCCESS;
}

} // namespace impl_4x1_small

CcuResult Ccu4x1SmallRootSpecializedKernel(CcuKernelArg kernelArg)
{
    using namespace impl_4x1_small;
    auto *arg = static_cast<Broadcast4x1SmallSpecializedKernelArg *>(kernelArg);
    CCU_CHK_RET(ValidateKernelArg(arg));

    Broadcast4x1SmallSpecializedContext ctx{};
    ctx.arg = arg;
    CCU_CHK_RET(InitPeerResources(ctx));
    CCU_CHK_RET(LoadTaskArgs(ctx));

    // role is fixed in the kernel argument at registration time. The translator
    // therefore emits only the sender or receiver path for this rank/root pair.
    if (arg->role == BCAST_4X1_ROLE_SENDER) {
        return RunSender(ctx);
    }
    return RunReceiver(ctx);
}

} // namespace ops_hccl


namespace ops_hccl {
namespace impl_4x1_large_pipeline15 {

constexpr uint32_t BUFFER_XN_ID = 1;
constexpr uint32_t TOKEN_XN_ID = 2;
constexpr uint32_t CHUNK_DONE_BASE_ID = 3;
constexpr uint32_t CKE_INDEX = 0;
constexpr uint32_t RANK_SIZE_4X1 = BCAST_4X1_RANK_SIZE;
constexpr uint32_t PIPELINE_CHUNK_NUM = BCAST_4X1_PIPELINE_CHUNK_NUM;
constexpr uint32_t PIPELINE_STAGE_NUM = BCAST_4X1_PIPELINE_STAGE_NUM;

struct Broadcast4x1Pipeline15Context {
    Broadcast4x1LargePipeline15KernelArg *arg = nullptr;
    std::vector<ccu::Variable> buffer;
    std::vector<ccu::Variable> token;
    ccu::Event transferEvent;
};

uint16_t RankBit(uint32_t rank)
{
    return static_cast<uint16_t>(uint32_t{1} << rank);
}

uint32_t NextRank(uint32_t rank)
{
    return (rank + 1U) % RANK_SIZE_4X1;
}

uint32_t PrevRank(uint32_t rank)
{
    return (rank + RANK_SIZE_4X1 - 1U) % RANK_SIZE_4X1;
}

uint32_t RootForCurrentRankAtPos(uint32_t rank, uint32_t pos)
{
    return (rank + RANK_SIZE_4X1 - pos) % RANK_SIZE_4X1;
}

CcuResult GetChannelIndexToRank(const Broadcast4x1Pipeline15Context &ctx,
    uint32_t peerRank, uint32_t &channelIndex)
{
    if (ctx.arg == nullptr || peerRank >= ctx.arg->rankSize) {
        return CcuResult::CCU_E_PARA;
    }
    channelIndex = ctx.arg->channelIndexByRank[peerRank];
    if (channelIndex >= ctx.arg->channelCount) {
        return CcuResult::CCU_E_INTERNAL;
    }
    return CcuResult::CCU_SUCCESS;
}

CcuResult InitPeerResources(Broadcast4x1Pipeline15Context &ctx)
{
    if (ctx.arg == nullptr ||
        ctx.arg->rankSize != RANK_SIZE_4X1 ||
        ctx.arg->rankId >= ctx.arg->rankSize ||
        ctx.arg->rankSize > MAX_RANK_SIZE) {
        return CcuResult::CCU_E_PARA;
    }
    if (ctx.arg->channelCount != ctx.arg->rankSize - 1U) {
        return CcuResult::CCU_E_INTERNAL;
    }

    bool seen[RANK_SIZE_4X1] = {false, false, false, false};
    ctx.buffer.resize(ctx.arg->rankSize);
    ctx.token.resize(ctx.arg->rankSize);
    for (uint32_t channelIndex = 0;
         channelIndex < ctx.arg->channelCount; ++channelIndex) {
        const uint32_t peer = ctx.arg->peerRanks[channelIndex];
        if (peer >= ctx.arg->rankSize || peer == ctx.arg->rankId || seen[peer]) {
            return CcuResult::CCU_E_INTERNAL;
        }
        if (ctx.arg->channelIndexByRank[peer] != channelIndex) {
            return CcuResult::CCU_E_INTERNAL;
        }
        seen[peer] = true;
        const ChannelHandle channel = ctx.arg->channels[channelIndex];
        ctx.buffer[peer] =
            ccu::GetResByChannel<ccu::Variable>(channel, BUFFER_XN_ID);
        ctx.token[peer] =
            ccu::GetResByChannel<ccu::Variable>(channel, TOKEN_XN_ID);
    }
    return CcuResult::CCU_SUCCESS;
}

CcuResult PublishAddrToPeer(Broadcast4x1Pipeline15Context &ctx,
    uint32_t channelIndex)
{
    const ChannelHandle channel = ctx.arg->channels[channelIndex];
    CCU_CHK_RET(ccu::WriteVariableWithNotify(
        channel, ctx.buffer[ctx.arg->rankId], BUFFER_XN_ID,
        CKE_INDEX, 1U << BUFFER_XN_ID));
    CCU_CHK_RET(ccu::WriteVariableWithNotify(
        channel, ctx.token[ctx.arg->rankId], TOKEN_XN_ID,
        CKE_INDEX, 1U << TOKEN_XN_ID));
    return CcuResult::CCU_SUCCESS;
}

CcuResult WaitPeerAddr(Broadcast4x1Pipeline15Context &ctx,
    uint32_t channelIndex)
{
    constexpr uint32_t addrReadyMask =
        (1U << BUFFER_XN_ID) | (1U << TOKEN_XN_ID);
    CCU_CHK_RET(ccu::NotifyWait(
        ctx.arg->channels[channelIndex], CKE_INDEX, addrReadyMask));
    return CcuResult::CCU_SUCCESS;
}

uint32_t ChunkDoneMask(uint32_t chunkIndex)
{
    // Chunks 0..12 use the thirteen completion bits 3..15.
    // The address exchange is fully completed before ExecutePipeline15 starts,
    // so chunks 13 and 14 safely reuse bits 1 and 2 exactly once.
    // This preserves a strict Record -> Wait -> Record -> Wait lifecycle on
    // every channel and avoids any reuse while a previous notification is still in flight.
    if (chunkIndex < 13U) {
        return 1U << (CHUNK_DONE_BASE_ID + chunkIndex);
    }
    if (chunkIndex == 13U) {
        return 1U << BUFFER_XN_ID;
    }
    return 1U << TOKEN_XN_ID;
}

CcuResult RecordChunkDone(Broadcast4x1Pipeline15Context &ctx,
    uint32_t channelIndex, uint32_t chunkIndex)
{
    CCU_CHK_RET(ccu::NotifyRecord(
        ctx.arg->channels[channelIndex], CKE_INDEX,
        ChunkDoneMask(chunkIndex)));
    return CcuResult::CCU_SUCCESS;
}

CcuResult WaitChunkDone(Broadcast4x1Pipeline15Context &ctx,
    uint32_t channelIndex, uint32_t chunkIndex)
{
    CCU_CHK_RET(ccu::NotifyWait(
        ctx.arg->channels[channelIndex], CKE_INDEX,
        ChunkDoneMask(chunkIndex)));
    return CcuResult::CCU_SUCCESS;
}

void MakeChunkOffset(ccu::Variable &offset, uint32_t chunkIndex,
    ccu::Variable &baseChunkBytes)
{
    offset = 0;
    for (uint32_t index = 0; index < chunkIndex; ++index) {
        offset += baseChunkBytes;
    }
}

void MakeChunkBytes(ccu::Variable &chunkBytes, uint32_t chunkIndex,
    ccu::Variable &baseChunkBytes, ccu::Variable &lastChunkBytes)
{
    chunkBytes = baseChunkBytes;
    if (chunkIndex == PIPELINE_CHUNK_NUM - 1U) {
        chunkBytes = lastChunkBytes;
    }
}

CcuResult WriteChunkToPeer(Broadcast4x1Pipeline15Context &ctx,
    uint32_t channelIndex, uint32_t chunkIndex,
    ccu::Variable &baseChunkBytes, ccu::Variable &lastChunkBytes)
{
    const uint32_t peer = ctx.arg->peerRanks[channelIndex];

    ccu::Variable offset;
    ccu::Variable chunkBytes;
    MakeChunkOffset(offset, chunkIndex, baseChunkBytes);
    MakeChunkBytes(chunkBytes, chunkIndex, baseChunkBytes, lastChunkBytes);

    ccu::LocalAddr src;
    src.addr = ctx.buffer[ctx.arg->rankId];
    src.addr += offset;
    src.token = ctx.token[ctx.arg->rankId];

    ccu::RemoteAddr dst;
    dst.addr = ctx.buffer[peer];
    dst.addr += offset;
    dst.token = ctx.token[peer];

    CCU_CHK_RET(ccu::Write(ctx.arg->channels[channelIndex],
        dst, src, chunkBytes, ctx.transferEvent, RankBit(peer)));
    CCU_CHK_RET(ccu::EventWait(ctx.transferEvent, RankBit(peer)));
    CCU_CHK_RET(RecordChunkDone(ctx, channelIndex, chunkIndex));
    return CcuResult::CCU_SUCCESS;
}

CcuResult ExchangeAddrForPipeline(Broadcast4x1Pipeline15Context &ctx,
    ccu::Variable &rootRank)
{
    const uint32_t rank = ctx.arg->rankId;
    const uint32_t upstream = PrevRank(rank);
    const uint32_t downstream = NextRank(rank);
    const uint32_t rootForPos0 = RootForCurrentRankAtPos(rank, 0);
    const uint32_t rootForPos1 = RootForCurrentRankAtPos(rank, 1);
    const uint32_t rootForPos2 = RootForCurrentRankAtPos(rank, 2);
    const uint32_t rootForPos3 = RootForCurrentRankAtPos(rank, 3);
    uint32_t upstreamChannel = INVALID_VALUE_RANKID;
    uint32_t downstreamChannel = INVALID_VALUE_RANKID;
    CCU_CHK_RET(GetChannelIndexToRank(ctx, upstream, upstreamChannel));
    CCU_CHK_RET(GetChannelIndexToRank(ctx, downstream, downstreamChannel));

    CCU_IF(rootRank == rootForPos0)
    {
        CCU_CHK_RET(WaitPeerAddr(ctx, downstreamChannel));
    }

    CCU_IF(rootRank == rootForPos1)
    {
        CCU_CHK_RET(PublishAddrToPeer(ctx, upstreamChannel));
        CCU_CHK_RET(WaitPeerAddr(ctx, downstreamChannel));
    }

    CCU_IF(rootRank == rootForPos2)
    {
        CCU_CHK_RET(PublishAddrToPeer(ctx, upstreamChannel));
        CCU_CHK_RET(WaitPeerAddr(ctx, downstreamChannel));
    }

    CCU_IF(rootRank == rootForPos3)
    {
        CCU_CHK_RET(PublishAddrToPeer(ctx, upstreamChannel));
    }

    return CcuResult::CCU_SUCCESS;
}

CcuResult ExecutePipeline15(Broadcast4x1Pipeline15Context &ctx,
    ccu::Variable &rootRank, ccu::Variable &baseChunkBytes,
    ccu::Variable &lastChunkBytes)
{
    const uint32_t rank = ctx.arg->rankId;
    const uint32_t upstream = PrevRank(rank);
    const uint32_t downstream = NextRank(rank);
    const uint32_t rootForPos0 = RootForCurrentRankAtPos(rank, 0);
    const uint32_t rootForPos1 = RootForCurrentRankAtPos(rank, 1);
    const uint32_t rootForPos2 = RootForCurrentRankAtPos(rank, 2);
    const uint32_t rootForPos3 = RootForCurrentRankAtPos(rank, 3);
    uint32_t upstreamChannel = INVALID_VALUE_RANKID;
    uint32_t downstreamChannel = INVALID_VALUE_RANKID;
    CCU_CHK_RET(GetChannelIndexToRank(ctx, upstream, upstreamChannel));
    CCU_CHK_RET(GetChannelIndexToRank(ctx, downstream, downstreamChannel));

    for (uint32_t stage = 0; stage < PIPELINE_STAGE_NUM; ++stage) {
        if (stage < PIPELINE_CHUNK_NUM) {
            CCU_IF(rootRank == rootForPos0)
            {
                CCU_CHK_RET(WriteChunkToPeer(ctx, downstreamChannel,
                    stage, baseChunkBytes, lastChunkBytes));
            }
        }

        if (stage >= 1U && stage <= PIPELINE_CHUNK_NUM) {
            const uint32_t chunkIndex = stage - 1U;
            CCU_IF(rootRank == rootForPos1)
            {
                CCU_CHK_RET(WaitChunkDone(
                    ctx, upstreamChannel, chunkIndex));
                CCU_CHK_RET(WriteChunkToPeer(ctx, downstreamChannel,
                    chunkIndex, baseChunkBytes, lastChunkBytes));
            }
        }

        if (stage >= 2U && stage <= PIPELINE_CHUNK_NUM + 1U) {
            const uint32_t chunkIndex = stage - 2U;
            CCU_IF(rootRank == rootForPos2)
            {
                CCU_CHK_RET(WaitChunkDone(
                    ctx, upstreamChannel, chunkIndex));
                CCU_CHK_RET(WriteChunkToPeer(ctx, downstreamChannel,
                    chunkIndex, baseChunkBytes, lastChunkBytes));
            }
        }

        if (stage >= 3U && stage <= PIPELINE_CHUNK_NUM + 2U) {
            const uint32_t chunkIndex = stage - 3U;
            CCU_IF(rootRank == rootForPos3)
            {
                CCU_CHK_RET(WaitChunkDone(
                    ctx, upstreamChannel, chunkIndex));
            }
        }
    }

    return CcuResult::CCU_SUCCESS;
}

CcuResult RunPipeline15Kernel(CcuKernelArg arg)
{
    auto *kernelArg =
        static_cast<Broadcast4x1LargePipeline15KernelArg *>(arg);
    Broadcast4x1Pipeline15Context ctx{};
    ctx.arg = kernelArg;
    CCU_CHK_RET(InitPeerResources(ctx));

    ccu::Variable rootRank;
    ccu::Variable baseChunkBytes;
    ccu::Variable lastChunkBytes;
    uint32_t argId = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.buffer[ctx.arg->rankId], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[ctx.arg->rankId], argId++));
    CCU_CHK_RET(ccu::LoadArg(rootRank, argId++));
    CCU_CHK_RET(ccu::LoadArg(baseChunkBytes, argId++));
    CCU_CHK_RET(ccu::LoadArg(lastChunkBytes, argId++));

    CCU_CHK_RET(ExchangeAddrForPipeline(ctx, rootRank));
    CCU_CHK_RET(ExecutePipeline15(
        ctx, rootRank, baseChunkBytes, lastChunkBytes));
    return CcuResult::CCU_SUCCESS;
}

} // namespace impl_4x1_large_pipeline15

CcuResult Ccu4x1LargePipeline15Kernel(CcuKernelArg arg)
{
    using namespace impl_4x1_large_pipeline15;
    return RunPipeline15Kernel(arg);
}

} // namespace ops_hccl


namespace ops_hccl {
namespace impl_8p4_small {

constexpr uint32_t BUFFER_XN_ID = 1;
constexpr uint32_t TOKEN_XN_ID = 2;
constexpr uint32_t DONE_SYNC_ID = 3;
constexpr uint32_t CKE_INDEX = 0;

struct Broadcast8p4SmallSpecializedContext {
    Broadcast8p4SmallSpecializedKernelArg *arg = nullptr;

    // Task arguments local to this rank. Keep them separate from peer resources;
    // the local rank has no self-channel and therefore no channel-derived entry.
    ccu::Variable localBuffer;
    ccu::Variable localToken;
    ccu::Variable dataSize;

    // Channel-derived remote resources, indexed by global peer rank.
    std::vector<ccu::Variable> remoteBuffer;
    std::vector<ccu::Variable> remoteToken;

    ccu::Event transferEvent;
};

uint16_t RankBit(uint32_t rank)
{
    return static_cast<uint16_t>(uint32_t{1} << rank);
}

uint16_t BuildPeerMask(const Broadcast8p4SmallSpecializedKernelArg &arg)
{
    uint16_t mask = 0;
    for (uint32_t channelIndex = 0;
         channelIndex < arg.channelCount;
         ++channelIndex) {
        mask = static_cast<uint16_t>(
            mask | RankBit(arg.peerRanks[channelIndex]));
    }
    return mask;
}

CcuResult ValidateKernelArg(
    const Broadcast8p4SmallSpecializedKernelArg *arg)
{
    if (arg == nullptr ||
        arg->rankSize != BCAST_8P4_RANK_SIZE ||
        arg->rankId >= BCAST_8P4_RANK_SIZE ||
        arg->rootRank >= BCAST_8P4_RANK_SIZE) {
        return CcuResult::CCU_E_PARA;
    }

    if (arg->role != BCAST_8P4_ROLE_SENDER &&
        arg->role != BCAST_8P4_ROLE_RECEIVER &&
        arg->role != BCAST_8P4_ROLE_IDLE) {
        return CcuResult::CCU_E_PARA;
    }

    if (arg->role == BCAST_8P4_ROLE_IDLE) {
        return CcuResult::CCU_SUCCESS;
    }

    if (arg->channelCount == 0 ||
        arg->channelCount >= BCAST_8P4_RANK_SIZE) {
        return CcuResult::CCU_E_PARA;
    }

    if (arg->role == BCAST_8P4_ROLE_RECEIVER &&
        arg->channelCount != 1) {
        return CcuResult::CCU_E_PARA;
    }

    if (arg->role == BCAST_8P4_ROLE_SENDER &&
        arg->rankId != arg->rootRank) {
        return CcuResult::CCU_E_PARA;
    }

    if (arg->role == BCAST_8P4_ROLE_RECEIVER &&
        arg->rankId == arg->rootRank) {
        return CcuResult::CCU_E_PARA;
    }

    return CcuResult::CCU_SUCCESS;
}

CcuResult InitPeerResources(Broadcast8p4SmallSpecializedContext &ctx)
{
    CCU_CHK_RET(ValidateKernelArg(ctx.arg));

    if (ctx.arg->role == BCAST_8P4_ROLE_IDLE) {
        return CcuResult::CCU_SUCCESS;
    }

    ctx.remoteBuffer.resize(BCAST_8P4_RANK_SIZE);
    ctx.remoteToken.resize(BCAST_8P4_RANK_SIZE);

    for (uint32_t channelIndex = 0;
         channelIndex < ctx.arg->channelCount;
         ++channelIndex) {
        const uint32_t peer = ctx.arg->peerRanks[channelIndex];
        if (peer >= BCAST_8P4_RANK_SIZE || peer == ctx.arg->rankId) {
            return CcuResult::CCU_E_INTERNAL;
        }

        const ChannelHandle channel = ctx.arg->channels[channelIndex];
        ctx.remoteBuffer[peer] =
            ccu::GetResByChannel<ccu::Variable>(channel, BUFFER_XN_ID);
        ctx.remoteToken[peer] =
            ccu::GetResByChannel<ccu::Variable>(channel, TOKEN_XN_ID);
    }

    return CcuResult::CCU_SUCCESS;
}

CcuResult LoadTaskArgs(Broadcast8p4SmallSpecializedContext &ctx)
{
    uint32_t argId = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.localBuffer, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.localToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.dataSize, argId++));
    return CcuResult::CCU_SUCCESS;
}

CcuResult PublishLocalAddressToRoot(
    Broadcast8p4SmallSpecializedContext &ctx)
{
    const ChannelHandle rootChannel = ctx.arg->channels[0];

    CCU_CHK_RET(ccu::WriteVariableWithNotify(
        rootChannel,
        ctx.localBuffer,
        BUFFER_XN_ID,
        CKE_INDEX,
        1U << BUFFER_XN_ID));

    CCU_CHK_RET(ccu::WriteVariableWithNotify(
        rootChannel,
        ctx.localToken,
        TOKEN_XN_ID,
        CKE_INDEX,
        1U << TOKEN_XN_ID));

    return CcuResult::CCU_SUCCESS;
}

CcuResult RunReceiver(Broadcast8p4SmallSpecializedContext &ctx)
{
    // The root-specialized receiver kernel owns exactly one channel: root.
    CCU_CHK_RET(PublishLocalAddressToRoot(ctx));
    CCU_CHK_RET(ccu::NotifyWait(
        ctx.arg->channels[0],
        CKE_INDEX,
        1U << DONE_SYNC_ID));
    return CcuResult::CCU_SUCCESS;
}

CcuResult WaitAllReceiverAddresses(
    Broadcast8p4SmallSpecializedContext &ctx)
{
    constexpr uint32_t addressReadyMask =
        (1U << BUFFER_XN_ID) | (1U << TOKEN_XN_ID);

    for (uint32_t channelIndex = 0;
         channelIndex < ctx.arg->channelCount;
         ++channelIndex) {
        CCU_CHK_RET(ccu::NotifyWait(
            ctx.arg->channels[channelIndex],
            CKE_INDEX,
            addressReadyMask));
    }
    return CcuResult::CCU_SUCCESS;
}

CcuResult IssueAllWrites(Broadcast8p4SmallSpecializedContext &ctx)
{
    for (uint32_t channelIndex = 0;
         channelIndex < ctx.arg->channelCount;
         ++channelIndex) {
        const uint32_t peer = ctx.arg->peerRanks[channelIndex];

        ccu::LocalAddr src;
        src.addr = ctx.localBuffer;
        src.token = ctx.localToken;

        ccu::RemoteAddr dst;
        dst.addr = ctx.remoteBuffer[peer];
        dst.token = ctx.remoteToken[peer];

        CCU_CHK_RET(ccu::Write(
            ctx.arg->channels[channelIndex],
            dst,
            src,
            ctx.dataSize,
            ctx.transferEvent,
            RankBit(peer)));
    }

    CCU_CHK_RET(ccu::EventWait(
        ctx.transferEvent,
        BuildPeerMask(*ctx.arg)));
    return CcuResult::CCU_SUCCESS;
}

CcuResult RecordAllDone(Broadcast8p4SmallSpecializedContext &ctx)
{
    for (uint32_t channelIndex = 0;
         channelIndex < ctx.arg->channelCount;
         ++channelIndex) {
        CCU_CHK_RET(ccu::NotifyRecord(
            ctx.arg->channels[channelIndex],
            CKE_INDEX,
            1U << DONE_SYNC_ID));
    }
    return CcuResult::CCU_SUCCESS;
}

CcuResult RunSender(Broadcast8p4SmallSpecializedContext &ctx)
{
    CCU_CHK_RET(WaitAllReceiverAddresses(ctx));

    // All peer Writes are issued first and share one Event. There is no
    // per-peer EventWait, so transfers on distinct channels may overlap.
    CCU_CHK_RET(IssueAllWrites(ctx));

    CCU_CHK_RET(RecordAllDone(ctx));
    return CcuResult::CCU_SUCCESS;
}

CcuResult RunSpecializedKernel(CcuKernelArg kernelArg)
{
    auto *arg = static_cast<Broadcast8p4SmallSpecializedKernelArg *>(kernelArg);
    CCU_CHK_RET(ValidateKernelArg(arg));

    // Idle kernels exist only to give the translator an unambiguous IO-Die
    // channel binding. The host never launches them on the data path.
    if (arg->role == BCAST_8P4_ROLE_IDLE) {
        return CcuResult::CCU_SUCCESS;
    }

    Broadcast8p4SmallSpecializedContext ctx{};
    ctx.arg = arg;

    CCU_CHK_RET(InitPeerResources(ctx));
    CCU_CHK_RET(LoadTaskArgs(ctx));

    if (arg->role == BCAST_8P4_ROLE_RECEIVER) {
        return RunReceiver(ctx);
    }
    return RunSender(ctx);
}

} // namespace impl_8p4_small

CcuResult Ccu8p4SmallRootSpecializedIntraKernel(CcuKernelArg arg)
{
    using namespace impl_8p4_small;
    return RunSpecializedKernel(arg);
}

CcuResult Ccu8p4SmallRootSpecializedInterKernel(CcuKernelArg arg)
{
    using namespace impl_8p4_small;
    return RunSpecializedKernel(arg);
}

} // namespace ops_hccl


namespace ops_hccl {
namespace impl_8p4_large {

// Five resources only. A bit is reused by a later stage only after the earlier
// stage has completed its full Record -> Wait lifecycle on that same channel.
constexpr uint32_t BUFFER_XN_ID = 1;
constexpr uint32_t TOKEN_XN_ID = 2;
constexpr uint32_t SCATTER_DONE_ID = 3;
constexpr uint32_t FANOUT_DONE_ID = 4;
constexpr uint32_t CKE_INDEX = 0;

constexpr uint64_t PIPE_STAGE_SCATTER_R0 = 0;
constexpr uint64_t PIPE_STAGE_SCATTER_R1_FANOUT_R0 = 1;
constexpr uint64_t PIPE_STAGE_FANOUT_R1 = 2;
constexpr uint32_t ROUND0 = 0;
constexpr uint32_t ROUND1 = 1;

struct PipelineTaskLayout {
    ccu::Variable localBuffer;
    ccu::Variable localToken;
    ccu::Variable baseSliceBytes;
    ccu::Variable baseRound0Bytes;
    ccu::Variable baseRound1Bytes;
    ccu::Variable lastRound0Bytes;
    ccu::Variable lastRound1Bytes;
    ccu::Variable stageId;
};

struct Broadcast8p4PipelineContext {
    Broadcast8p4OwnerPipelineKernelArg *arg = nullptr;
    PipelineTaskLayout task;
    std::vector<ccu::Variable> peerBuffer;
    std::vector<ccu::Variable> peerToken;
    ccu::Event transferEvent;
};

uint16_t RankBit(uint32_t rank)
{
    return static_cast<uint16_t>(1U << rank);
}

uint32_t OwnerSliceIndex(uint32_t ownerRank, uint32_t rootRank)
{
    return ownerRank < rootRank ? ownerRank : ownerRank - 1U;
}

CcuResult ValidateKernelArg(const Broadcast8p4OwnerPipelineKernelArg *arg)
{
    if (arg == nullptr || arg->rankSize != BCAST_8P4_RANK_SIZE ||
        arg->rankId >= arg->rankSize || arg->rootRank >= arg->rankSize) {
        return CcuResult::CCU_E_PARA;
    }
    if (arg->channelCount == 0 || arg->channelCount >= arg->rankSize) {
        return CcuResult::CCU_E_INTERNAL;
    }
    for (uint32_t channelIndex = 0; channelIndex < arg->channelCount; ++channelIndex) {
        const uint32_t peer = arg->peerRanks[channelIndex];
        if (peer >= arg->rankSize || peer == arg->rankId) {
            return CcuResult::CCU_E_INTERNAL;
        }
    }
    return CcuResult::CCU_SUCCESS;
}

CcuResult InitPeerResources(Broadcast8p4PipelineContext &ctx)
{
    ctx.peerBuffer.resize(ctx.arg->rankSize);
    ctx.peerToken.resize(ctx.arg->rankSize);
    for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
        const uint32_t peer = ctx.arg->peerRanks[channelIndex];
        ChannelHandle channel = ctx.arg->channels[channelIndex];
        ctx.peerBuffer[peer] = ccu::GetResByChannel<ccu::Variable>(channel, BUFFER_XN_ID);
        ctx.peerToken[peer] = ccu::GetResByChannel<ccu::Variable>(channel, TOKEN_XN_ID);
    }
    return CcuResult::CCU_SUCCESS;
}

CcuResult LoadTaskArgs(Broadcast8p4PipelineContext &ctx)
{
    uint32_t argId = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.task.localBuffer, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.task.localToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.task.baseSliceBytes, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.task.baseRound0Bytes, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.task.baseRound1Bytes, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.task.lastRound0Bytes, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.task.lastRound1Bytes, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.task.stageId, argId++));
    return CcuResult::CCU_SUCCESS;
}

CcuResult PublishAddrToPeer(Broadcast8p4PipelineContext &ctx, uint32_t channelIndex)
{
    ChannelHandle channel = ctx.arg->channels[channelIndex];
    CCU_CHK_RET(ccu::WriteVariableWithNotify(
        channel, ctx.task.localBuffer, BUFFER_XN_ID,
        CKE_INDEX, 1U << BUFFER_XN_ID));
    CCU_CHK_RET(ccu::WriteVariableWithNotify(
        channel, ctx.task.localToken, TOKEN_XN_ID,
        CKE_INDEX, 1U << TOKEN_XN_ID));
    return CcuResult::CCU_SUCCESS;
}

CcuResult WaitPeerAddr(Broadcast8p4PipelineContext &ctx, uint32_t channelIndex)
{
    constexpr uint32_t readyMask =
        (1U << BUFFER_XN_ID) | (1U << TOKEN_XN_ID);
    CCU_CHK_RET(ccu::NotifyWait(
        ctx.arg->channels[channelIndex], CKE_INDEX, readyMask));
    return CcuResult::CCU_SUCCESS;
}

CcuResult RecordSync(Broadcast8p4PipelineContext &ctx,
    uint32_t channelIndex, uint32_t syncId)
{
    CCU_CHK_RET(ccu::NotifyRecord(
        ctx.arg->channels[channelIndex], CKE_INDEX, 1U << syncId));
    return CcuResult::CCU_SUCCESS;
}

CcuResult WaitSync(Broadcast8p4PipelineContext &ctx,
    uint32_t channelIndex, uint32_t syncId)
{
    CCU_CHK_RET(ccu::NotifyWait(
        ctx.arg->channels[channelIndex], CKE_INDEX, 1U << syncId));
    return CcuResult::CCU_SUCCESS;
}

ccu::Variable MakeSliceOffsetBytes(uint32_t sliceIndex,
    ccu::Variable &baseSliceBytes)
{
    ccu::Variable offset;
    offset = uint64_t{0};
    for (uint32_t index = 0; index < sliceIndex; ++index) {
        offset += baseSliceBytes;
    }
    return offset;
}

ccu::Variable MakeRound0Bytes(uint32_t sliceIndex, PipelineTaskLayout &task)
{
    ccu::Variable bytes;
    bytes = task.baseRound0Bytes;
    if (sliceIndex == BCAST_8P4_OWNER_NUM - 1U) {
        bytes = task.lastRound0Bytes;
    }
    return bytes;
}

ccu::Variable MakeRound1Bytes(uint32_t sliceIndex, PipelineTaskLayout &task)
{
    ccu::Variable bytes;
    bytes = task.baseRound1Bytes;
    if (sliceIndex == BCAST_8P4_OWNER_NUM - 1U) {
        bytes = task.lastRound1Bytes;
    }
    return bytes;
}

ccu::Variable MakeRoundOffsetBytes(uint32_t sliceIndex, uint32_t round,
    PipelineTaskLayout &task)
{
    ccu::Variable offset = MakeSliceOffsetBytes(sliceIndex, task.baseSliceBytes);
    if (round == ROUND1) {
        ccu::Variable round0Bytes = MakeRound0Bytes(sliceIndex, task);
        offset += round0Bytes;
    }
    return offset;
}

ccu::Variable MakeRoundBytes(uint32_t sliceIndex, uint32_t round,
    PipelineTaskLayout &task)
{
    if (round == ROUND0) {
        return MakeRound0Bytes(sliceIndex, task);
    }
    return MakeRound1Bytes(sliceIndex, task);
}

CcuResult WriteRoundToPeer(Broadcast8p4PipelineContext &ctx,
    uint32_t channelIndex, uint32_t sliceIndex, uint32_t round,
    uint16_t eventMask)
{
    const uint32_t peer = ctx.arg->peerRanks[channelIndex];
    ccu::Variable offset = MakeRoundOffsetBytes(sliceIndex, round, ctx.task);
    ccu::Variable bytes = MakeRoundBytes(sliceIndex, round, ctx.task);

    ccu::LocalAddr src;
    src.addr = ctx.task.localBuffer;
    src.addr += offset;
    src.token = ctx.task.localToken;

    ccu::RemoteAddr dst;
    dst.addr = ctx.peerBuffer[peer];
    dst.addr += offset;
    dst.token = ctx.peerToken[peer];

    CCU_CHK_RET(ccu::Write(ctx.arg->channels[channelIndex], dst, src, bytes,
        ctx.transferEvent, eventMask));
    return CcuResult::CCU_SUCCESS;
}

uint16_t AllPeerMask(const Broadcast8p4OwnerPipelineKernelArg &arg)
{
    uint16_t mask = 0;
    for (uint32_t channelIndex = 0; channelIndex < arg.channelCount; ++channelIndex) {
        mask = static_cast<uint16_t>(mask | RankBit(arg.peerRanks[channelIndex]));
    }
    return mask;
}

uint16_t FanoutPeerMask(const Broadcast8p4OwnerPipelineKernelArg &arg)
{
    uint16_t mask = 0;
    for (uint32_t channelIndex = 0; channelIndex < arg.channelCount; ++channelIndex) {
        const uint32_t peer = arg.peerRanks[channelIndex];
        if (peer != arg.rootRank) {
            mask = static_cast<uint16_t>(mask | RankBit(peer));
        }
    }
    return mask;
}

CcuResult RootScatterRound(Broadcast8p4PipelineContext &ctx, uint32_t round)
{
    for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
        CCU_CHK_RET(WaitPeerAddr(ctx, channelIndex));
    }

    // All writes in one layer are issued before one shared EventWait.
    for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
        const uint32_t ownerRank = ctx.arg->peerRanks[channelIndex];
        const uint32_t sliceIndex = OwnerSliceIndex(ownerRank, ctx.arg->rootRank);
        CCU_CHK_RET(WriteRoundToPeer(
            ctx, channelIndex, sliceIndex, round, RankBit(ownerRank)));
    }

    CCU_CHK_RET(ccu::EventWait(ctx.transferEvent, AllPeerMask(*ctx.arg)));
    for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
        CCU_CHK_RET(RecordSync(ctx, channelIndex, SCATTER_DONE_ID));
    }
    return CcuResult::CCU_SUCCESS;
}

CcuResult OwnerScatterReceive(Broadcast8p4PipelineContext &ctx)
{
    // Only one of the two layer kernels contains the root channel.
    const uint32_t rootChannelIndex =
        ctx.arg->channelIndexByRank[ctx.arg->rootRank];
    if (rootChannelIndex != INVALID_VALUE_RANKID &&
        rootChannelIndex < ctx.arg->channelCount) {
        CCU_CHK_RET(PublishAddrToPeer(ctx, rootChannelIndex));
        CCU_CHK_RET(WaitSync(ctx, rootChannelIndex, SCATTER_DONE_ID));
    }
    return CcuResult::CCU_SUCCESS;
}

CcuResult OwnerFanoutRound(Broadcast8p4PipelineContext &ctx, uint32_t round)
{
    // Root is excluded because it already owns the complete source buffer.
    for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
        if (ctx.arg->peerRanks[channelIndex] != ctx.arg->rootRank) {
            CCU_CHK_RET(PublishAddrToPeer(ctx, channelIndex));
        }
    }
    for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
        if (ctx.arg->peerRanks[channelIndex] != ctx.arg->rootRank) {
            CCU_CHK_RET(WaitPeerAddr(ctx, channelIndex));
        }
    }

    const uint32_t ownSliceIndex =
        OwnerSliceIndex(ctx.arg->rankId, ctx.arg->rootRank);
    for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
        const uint32_t peer = ctx.arg->peerRanks[channelIndex];
        if (peer != ctx.arg->rootRank) {
            CCU_CHK_RET(WriteRoundToPeer(
                ctx, channelIndex, ownSliceIndex, round, RankBit(peer)));
        }
    }

    const uint16_t activeMask = FanoutPeerMask(*ctx.arg);
    if (activeMask != 0) {
        CCU_CHK_RET(ccu::EventWait(ctx.transferEvent, activeMask));
    }

    // The symmetric handshake guarantees that all pieces of this round have
    // arrived before the notify resource is reused by a later stage.
    for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
        if (ctx.arg->peerRanks[channelIndex] != ctx.arg->rootRank) {
            CCU_CHK_RET(RecordSync(ctx, channelIndex, FANOUT_DONE_ID));
        }
    }
    for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
        if (ctx.arg->peerRanks[channelIndex] != ctx.arg->rootRank) {
            CCU_CHK_RET(WaitSync(ctx, channelIndex, FANOUT_DONE_ID));
        }
    }
    return CcuResult::CCU_SUCCESS;
}

CcuResult RunStage0(Broadcast8p4PipelineContext &ctx)
{
    if (ctx.arg->rankId == ctx.arg->rootRank) {
        return RootScatterRound(ctx, ROUND0);
    }
    return OwnerScatterReceive(ctx);
}

CcuResult RunStage1(Broadcast8p4PipelineContext &ctx)
{
    if (ctx.arg->rankId == ctx.arg->rootRank) {
        return RootScatterRound(ctx, ROUND1);
    }

    // In the one layer that contains root, publish the R1 destination first.
    // Do not wait for R1 yet: fanout R0 runs while root writes R1 into the
    // disjoint second half of this owner's slice.
    const uint32_t rootChannelIndex =
        ctx.arg->channelIndexByRank[ctx.arg->rootRank];
    if (rootChannelIndex != INVALID_VALUE_RANKID &&
        rootChannelIndex < ctx.arg->channelCount) {
        CCU_CHK_RET(PublishAddrToPeer(ctx, rootChannelIndex));
    }

    CCU_CHK_RET(OwnerFanoutRound(ctx, ROUND0));

    if (rootChannelIndex != INVALID_VALUE_RANKID &&
        rootChannelIndex < ctx.arg->channelCount) {
        CCU_CHK_RET(WaitSync(ctx, rootChannelIndex, SCATTER_DONE_ID));
    }
    return CcuResult::CCU_SUCCESS;
}

CcuResult RunStage2(Broadcast8p4PipelineContext &ctx)
{
    if (ctx.arg->rankId == ctx.arg->rootRank) {
        return CcuResult::CCU_SUCCESS;
    }
    return OwnerFanoutRound(ctx, ROUND1);
}

CcuResult RunCompactPipelineKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<Broadcast8p4OwnerPipelineKernelArg *>(arg);
    CCU_CHK_RET(ValidateKernelArg(kernelArg));

    Broadcast8p4PipelineContext ctx{};
    ctx.arg = kernelArg;
    CCU_CHK_RET(InitPeerResources(ctx));
    CCU_CHK_RET(LoadTaskArgs(ctx));

    CCU_IF(ctx.task.stageId == PIPE_STAGE_SCATTER_R0)
    {
        CCU_CHK_RET(RunStage0(ctx));
    }

    CCU_IF(ctx.task.stageId == PIPE_STAGE_SCATTER_R1_FANOUT_R0)
    {
        CCU_CHK_RET(RunStage1(ctx));
    }

    CCU_IF(ctx.task.stageId == PIPE_STAGE_FANOUT_R1)
    {
        CCU_CHK_RET(RunStage2(ctx));
    }

    return CcuResult::CCU_SUCCESS;
}

} // namespace impl_8p4_large

CcuResult Ccu8p4OwnerPipeline2CompactIntraKernel(CcuKernelArg arg)
{
    using namespace impl_8p4_large;
    return RunCompactPipelineKernel(arg);
}

CcuResult Ccu8p4OwnerPipeline2CompactInterKernel(CcuKernelArg arg)
{
    using namespace impl_8p4_large;
    return RunCompactPipelineKernel(arg);
}

} // namespace ops_hccl
