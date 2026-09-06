/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <hcomm/hcomm_primitives.h>

#include "ccu_kernel.h"
#include "log.h"

#ifndef CCU_CHK_RET
#define CCU_CHK_RET(call) \
    do { \
        CcuResult ccuResult = (call); \
        if (ccuResult != CCU_SUCCESS) { \
            return ccuResult; \
        } \
    } while (0)
#endif

namespace ops_hccl {
namespace {

constexpr uint16_t OUTPUT_XN_ID = 1;
constexpr uint16_t TOKEN_XN_ID = 2;
constexpr uint16_t POST_SYNC_ID = 3;
constexpr uint16_t CKE_IDX_0 = 0;
constexpr uint16_t EIGHT_P_FOUR_INPUT_XN_ID = 1;
constexpr uint16_t EIGHT_P_FOUR_OUTPUT_XN_ID = 2;
constexpr uint16_t EIGHT_P_FOUR_TOKEN_XN_ID = 3;
constexpr uint16_t EIGHT_P_FOUR_SEED_DATA_ID = 2;
constexpr uint16_t EIGHT_P_FOUR_DONE_ID = 3;
constexpr uint32_t EIGHT_P_FOUR_FULL_CHANNEL_COUNT = 4;
constexpr uint32_t EIGHT_P_FOUR_PARTIAL_CHANNEL_COUNT = 8;

constexpr uint64_t SetBits(uint16_t end)
{
    return (uint64_t(1) << (end + 1)) - uint64_t(1);
}

uint64_t GetLoopParam(uint64_t loopContextId, uint64_t addressOffset, uint64_t loopIteration)
{
    constexpr uint16_t contextIdBitNum = 8;
    constexpr uint16_t contextIdShift = 45;
    constexpr uint16_t addressBitNum = 32;
    constexpr uint16_t addressShift = 13;
    constexpr uint16_t loopBitNum = 13;
    return ((loopContextId & SetBits(contextIdBitNum)) << contextIdShift) |
        ((addressOffset & SetBits(addressBitNum)) << addressShift) |
        (loopIteration & SetBits(loopBitNum));
}

uint64_t GetParallelParam(
    uint64_t repeatNum, uint64_t repeatLoopIndex, uint64_t totalLoopNum)
{
    constexpr uint16_t bitNum = 7;
    constexpr uint16_t repeatNumShift = 55;
    constexpr uint16_t repeatLoopShift = 48;
    constexpr uint16_t totalLoopShift = 41;
    return ((repeatNum & SetBits(bitNum)) << repeatNumShift) |
        ((repeatLoopIndex & SetBits(bitNum)) << repeatLoopShift) |
        ((totalLoopNum & SetBits(bitNum)) << totalLoopShift);
}

uint64_t GetOffsetParam(
    uint64_t addressOffset, uint64_t memorySliceOffset, uint64_t eventOffset)
{
    constexpr uint16_t addressBitNum = 32;
    constexpr uint16_t addressShift = 21;
    constexpr uint16_t memorySliceBitNum = 11;
    constexpr uint16_t memorySliceShift = 10;
    constexpr uint16_t eventBitNum = 10;
    return ((addressOffset & SetBits(addressBitNum)) << addressShift) |
        ((memorySliceOffset & SetBits(memorySliceBitNum)) << memorySliceShift) |
        (eventOffset & SetBits(eventBitNum));
}

CcuResult InitDirectResources(CcuDirectContext &ctx)
{
    if (ctx.arg->channelCount == 0 || ctx.arg->channelCount >= MAX_RANK_SIZE) {
        HCCL_ERROR("[CcuDirectKernel] invalid channel count %u", ctx.arg->channelCount);
        return CCU_E_PARA;
    }
    const uint32_t channelMask = (1U << ctx.arg->channelCount) - 1U;
    if (ctx.arg->writeMask == 0 ||
        (ctx.arg->writeMask & ~channelMask) != 0 ||
        (ctx.arg->publishMask & ~channelMask) != 0 ||
        (ctx.arg->readyWaitMask & ~channelMask) != 0 ||
        ctx.arg->syncMask == 0 ||
        (ctx.arg->syncMask & ~channelMask) != 0 ||
        (ctx.arg->syncWaitMask & ~channelMask) != 0) {
        HCCL_ERROR("[CcuDirectKernel] invalid masks %x/%x/%x/%x/%x for %u channels",
            ctx.arg->writeMask, ctx.arg->publishMask,
            ctx.arg->readyWaitMask, ctx.arg->syncMask,
            ctx.arg->syncWaitMask,
            ctx.arg->channelCount);
        return CCU_E_PARA;
    }

    ctx.remoteOutputs.resize(ctx.arg->channelCount);
    ctx.remoteTokens.resize(ctx.arg->channelCount);
    for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
        ctx.remoteOutputs[index] =
            ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[index], OUTPUT_XN_ID);
        ctx.remoteTokens[index] =
            ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[index], TOKEN_XN_ID);
    }
    return CCU_SUCCESS;
}

CcuResult LoadDirectArgs(CcuDirectContext &ctx)
{
    uint32_t argIndex = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.input, argIndex++));
    CCU_CHK_RET(ccu::LoadArg(ctx.output, argIndex++));
    CCU_CHK_RET(ccu::LoadArg(ctx.token, argIndex++));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputOffset, argIndex++));
    CCU_CHK_RET(ccu::LoadArg(ctx.firstSize, argIndex++));
    CCU_CHK_RET(ccu::LoadArg(ctx.secondSize, argIndex++));
    CCU_CHK_RET(ccu::LoadArg(ctx.copyFlag, argIndex++));
    CCU_CHK_RET(ccu::LoadArg(ctx.skipDone, argIndex++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.addrOffset, argIndex++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.loopParam, argIndex++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.parallelParam, argIndex++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.residual, argIndex++));
    return CCU_SUCCESS;
}

CcuResult LoadDirectAddressArgs(CcuDirectContext &ctx)
{
    CCU_CHK_RET(ccu::LoadArg(ctx.output, 1));
    CCU_CHK_RET(ccu::LoadArg(ctx.token, 2));
    return CCU_SUCCESS;
}

CcuResult LoadDirectTransferArgs(CcuDirectContext &ctx)
{
    CCU_CHK_RET(ccu::LoadArg(ctx.input, 0));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputOffset, 3));
    CCU_CHK_RET(ccu::LoadArg(ctx.firstSize, 4));
    CCU_CHK_RET(ccu::LoadArg(ctx.secondSize, 5));
    CCU_CHK_RET(ccu::LoadArg(ctx.copyFlag, 6));
    CCU_CHK_RET(ccu::LoadArg(ctx.skipDone, 7));
    if (ctx.arg->doLocalCopy != 0) {
        CCU_CHK_RET(ccu::LoadArg(ctx.goSize.addrOffset, 8));
        CCU_CHK_RET(ccu::LoadArg(ctx.goSize.loopParam, 9));
        CCU_CHK_RET(ccu::LoadArg(ctx.goSize.parallelParam, 10));
        CCU_CHK_RET(ccu::LoadArg(ctx.goSize.residual, 11));
    }
    return CCU_SUCCESS;
}

template <typename Context>
CcuResult PublishDirectAddresses(Context &ctx)
{
    for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
        CCU_CHK_RET(ccu::WriteVariableWithNotify(
            ctx.arg->channels[index], ctx.output, OUTPUT_XN_ID, CKE_IDX_0, 1U << OUTPUT_XN_ID));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(
            ctx.arg->channels[index], ctx.token, TOKEN_XN_ID, CKE_IDX_0, 1U << TOKEN_XN_ID));
    }
    return CCU_SUCCESS;
}

CcuResult PublishDirectAddresses(CcuDirectContext &ctx)
{
    for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
        if ((ctx.arg->publishMask & (1U << index)) == 0) {
            continue;
        }
        CCU_CHK_RET(ccu::WriteVariableWithNotify(
            ctx.arg->channels[index], ctx.output, OUTPUT_XN_ID,
            CKE_IDX_0, 1U << OUTPUT_XN_ID));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(
            ctx.arg->channels[index], ctx.token, TOKEN_XN_ID,
            CKE_IDX_0, 1U << TOKEN_XN_ID));
    }
    return CCU_SUCCESS;
}

template <typename Context>
CcuResult WaitDirectAddresses(Context &ctx)
{
    constexpr uint16_t readyMask = (1U << OUTPUT_XN_ID) | (1U << TOKEN_XN_ID);
    for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
        CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[index], CKE_IDX_0, readyMask));
    }
    return CCU_SUCCESS;
}

CcuResult WaitDirectAddresses(CcuDirectContext &ctx)
{
    constexpr uint16_t readyMask =
        (1U << OUTPUT_XN_ID) | (1U << TOKEN_XN_ID);
    for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
        if ((ctx.arg->readyWaitMask & (1U << index)) == 0) {
            continue;
        }
        CCU_CHK_RET(ccu::NotifyWait(
            ctx.arg->channels[index], CKE_IDX_0, readyMask));
    }
    return CCU_SUCCESS;
}

CcuResult PostSyncDirect(CcuDirectContext &ctx)
{
    constexpr uint16_t doneMask = 1U << POST_SYNC_ID;
    for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
        if ((ctx.arg->syncMask & (1U << index)) == 0) {
            continue;
        }
        CCU_CHK_RET(ccu::NotifyRecord(ctx.arg->channels[index], CKE_IDX_0, doneMask));
    }
    for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
        if ((ctx.arg->syncWaitMask & (1U << index)) == 0) {
            continue;
        }
        CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[index], CKE_IDX_0, doneMask));
    }
    return CCU_SUCCESS;
}

template <typename Context>
void InitGroupCopyResources(Context &ctx, ccu::LocalAddr *loopSrc,
    ccu::LocalAddr *loopDst, ccu::Variable *loopLen)
{
    if (!ctx.copyResourceAllocated) {
        ctx.copyConfig.msInterleave = CCU_MS_INTERLEAVE;
        ctx.copyConfig.loopCount = CCU_MS_LOCAL_COPY_LOOP_COUNT;
        ctx.copyConfig.memSlice = CCU_LOCAL_COPY_MS_PER_LOOP * CCU_MS_SIZE;
        ctx.copyResource.eventCount = ctx.copyConfig.loopCount;
        ctx.copyResource.completedEvent = ccu::Array<ccu::Event>(ctx.copyResource.eventCount);
        ctx.copyResource.bufCount = ctx.copyConfig.loopCount * ctx.copyConfig.msInterleave;
        ctx.copyResource.ccuBuf = ccu::Array<ccu::CcuBuffer>(ctx.copyResource.bufCount);
        ctx.copyResourceAllocated = true;
    }

    const std::string loopName = "allgather_local_copy";
    if (!ctx.IsLoopEntityRegistered(loopName)) {
        ctx.CreateLoopEntity(loopName);
        auto &entity = ctx.loopMap[loopName];
        for (uint32_t index = 0; index < 2; ++index) {
            const uint32_t bufferBase = index * ctx.copyConfig.msInterleave;
            const ccu::Event loopEvent = ctx.copyResource.completedEvent[index];
            entity.body[index].reset(new ccu::Func(
                [&ctx, index, bufferBase, loopEvent, loopSrc, loopDst, loopLen]() {
                    ccu::LocalCopy(ctx.copyResource.ccuBuf[bufferBase], loopSrc[index],
                        loopLen[index], loopEvent, 1);
                    ccu::EventWait(loopEvent, 1);
                    ccu::LocalCopy(loopDst[index], ctx.copyResource.ccuBuf[bufferBase],
                        loopLen[index], loopEvent, 1);
                    ccu::EventWait(loopEvent, 1);
                }));
            entity.loops[index].reset(new ccu::Loop(entity.loopParam[index], *entity.body[index]));
        }
    }
}

template <typename Context>
CcuResult GroupCopy(Context &ctx, ccu::LocalAddr destination, ccu::LocalAddr source)
{
    ccu::LocalAddr loopSource[2];
    ccu::LocalAddr loopDestination[2];
    ccu::Variable loopLength[2];
    InitGroupCopyResources(ctx, loopSource, loopDestination, loopLength);
    auto &loops = ctx.loopMap["allgather_local_copy"];

    CCU_IF(ctx.goSize.addrOffset != 0)
    {
        ccu::Variable loopParam;
        loopParam = GetLoopParam(0, ctx.copyConfig.memSlice * ctx.copyConfig.loopCount, 0);
        loopParam += ctx.goSize.loopParam;

        ccu::Variable sliceSize;
        sliceSize = ctx.copyConfig.memSlice;
        loopSource[0].addr = source.addr;
        loopSource[0].token = source.token;
        loopDestination[0].addr = destination.addr;
        loopDestination[0].token = destination.token;
        loopLength[0] = sliceSize;
        loops.loopParam[0] = loopParam;

        ccu::Variable parallelConfig;
        parallelConfig = GetParallelParam(ctx.copyConfig.loopCount - 1, 0, 1);
        ccu::Variable offsetConfig;
        offsetConfig = GetOffsetParam(ctx.copyConfig.memSlice, ctx.copyConfig.msInterleave, 1);
        std::vector<ccu::Loop> groupLoops{*loops.loops[0]};
        ccu::LoopGroup group(parallelConfig, offsetConfig, ctx.copyConfig.loopCount, groupLoops);
    }

    CCU_IF(ctx.goSize.parallelParam != 0)
    {
        source.addr += ctx.goSize.addrOffset;
        destination.addr += ctx.goSize.addrOffset;

        loopSource[0].addr = source.addr;
        loopSource[0].token = source.token;
        loopDestination[0].addr = destination.addr;
        loopDestination[0].token = destination.token;
        loopLength[0] = ctx.goSize.residual;

        source.addr += ctx.goSize.residual;
        destination.addr += ctx.goSize.residual;

        ccu::Variable sliceSize;
        sliceSize = ctx.copyConfig.memSlice;
        loopSource[1].addr = source.addr;
        loopSource[1].token = source.token;
        loopDestination[1].addr = destination.addr;
        loopDestination[1].token = destination.token;
        loopLength[1] = sliceSize;

        ccu::Variable loopConfig0;
        loopConfig0 = GetLoopParam(0, 0, 1);
        ccu::Variable loopConfig1;
        loopConfig1 = GetLoopParam(0, 0, 1);
        ccu::Variable offsetConfig;
        offsetConfig = GetOffsetParam(ctx.copyConfig.memSlice, ctx.copyConfig.msInterleave, 1);
        loops.loopParam[0] = loopConfig0;
        loops.loopParam[1] = loopConfig1;
        std::vector<ccu::Loop> groupLoops{*loops.loops[0], *loops.loops[1]};
        ccu::LoopGroup group(ctx.goSize.parallelParam, offsetConfig,
            ctx.copyConfig.loopCount, groupLoops);
    }
    return CCU_SUCCESS;
}

CcuResult TransferDirect(CcuDirectContext &ctx)
{
    ccu::LocalAddr firstSource;
    firstSource.addr = ctx.input;
    firstSource.token = ctx.token;

    std::vector<ccu::RemoteAddr> firstDestinations(ctx.arg->channelCount);
    for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
        if ((ctx.arg->writeMask & (1U << index)) == 0) {
            continue;
        }
        firstDestinations[index].addr = ctx.remoteOutputs[index];
        firstDestinations[index].addr += ctx.outputOffset;
        firstDestinations[index].token = ctx.remoteTokens[index];
        CCU_CHK_RET(ccu::Write(ctx.arg->channels[index], firstDestinations[index],
            firstSource, ctx.firstSize, ctx.firstEvent, static_cast<uint16_t>(1U << index)));
    }

    CCU_IF(ctx.secondSize != 0)
    {
        ccu::LocalAddr secondSource;
        secondSource.addr = ctx.input;
        secondSource.addr += ctx.firstSize;
        secondSource.token = ctx.token;
        std::vector<ccu::RemoteAddr> secondDestinations(ctx.arg->channelCount);
        for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
            if ((ctx.arg->writeMask & (1U << index)) == 0) {
                continue;
            }
            secondDestinations[index].addr = ctx.remoteOutputs[index];
            secondDestinations[index].addr += ctx.outputOffset;
            secondDestinations[index].addr += ctx.firstSize;
            secondDestinations[index].token = ctx.remoteTokens[index];
            CCU_CHK_RET(ccu::Write(ctx.arg->channels[index], secondDestinations[index],
                secondSource, ctx.secondSize, ctx.secondEvent,
                static_cast<uint16_t>(1U << index)));
        }
    }

    if (ctx.arg->doLocalCopy != 0) {
        CCU_IF(ctx.copyFlag != 0)
        {
            ccu::LocalAddr localSource;
            localSource.addr = ctx.input;
            localSource.token = ctx.token;
            ccu::LocalAddr localDestination;
            localDestination.addr = ctx.output;
            localDestination.addr += ctx.outputOffset;
            localDestination.token = ctx.token;
            CCU_CHK_RET(GroupCopy(ctx, localDestination, localSource));
        }
    }

    if (ctx.arg->waitWriteCq == 0) {
        if (ctx.arg->doLocalCopy != 0) {
            // GroupCopy is already complete here.  A LoopGroup cannot be the
            // final CCU graph node, so terminate it with a local-only fence.
            // This deliberately does not consume any outbound Write CQ bit.
            const uint16_t copyFence =
                static_cast<uint16_t>(1U << ctx.arg->channelCount);
            CCU_CHK_RET(ccu::EventRecord(ctx.firstEvent, copyFence));
            CCU_CHK_RET(ccu::EventWait(ctx.firstEvent, copyFence));
        }
        return CCU_SUCCESS;
    }

    const uint16_t allEvents = static_cast<uint16_t>(ctx.arg->writeMask);
    CCU_CHK_RET(ccu::EventWait(ctx.firstEvent, allEvents));
    CCU_IF(ctx.secondSize != 0)
    {
        CCU_CHK_RET(ccu::EventWait(ctx.secondEvent, allEvents));
    }
    return CCU_SUCCESS;
}

CcuResult Transfer4x1Direct(CcuDirectContext &ctx)
{
    constexpr uint32_t channelCount = 3;
    constexpr uint16_t channelMask = (1U << channelCount) - 1U;
    constexpr uint16_t copyMask = 1U << (channelCount * 2);
    constexpr uint16_t completionMask =
        static_cast<uint16_t>(channelMask | copyMask);
    constexpr uint16_t readyMask =
        (1U << OUTPUT_XN_ID) | (1U << TOKEN_XN_ID);
    if (ctx.arg->channelCount != channelCount ||
        ctx.arg->writeMask != channelMask) {
        HCCL_ERROR("[Ccu4x1DirectKernel] invalid channel/write mask %u/0x%x",
            ctx.arg->channelCount, ctx.arg->writeMask);
        return CCU_E_PARA;
    }

    ccu::LocalAddr firstSource;
    firstSource.addr = ctx.input;
    firstSource.token = ctx.token;
    std::vector<ccu::RemoteAddr> firstDestinations(channelCount);
    for (uint32_t index = 0; index < channelCount; ++index) {
        // Every rank publishes all addresses before entering this loop.
        // Start each asynchronous Write as soon as that peer is ready instead
        // of holding all three links idle behind the slowest READY.
        CCU_CHK_RET(ccu::NotifyWait(
            ctx.arg->channels[index], CKE_IDX_0, readyMask));
        firstDestinations[index].addr = ctx.remoteOutputs[index];
        firstDestinations[index].addr += ctx.outputOffset;
        firstDestinations[index].token = ctx.remoteTokens[index];
        CCU_CHK_RET(ccu::Write(ctx.arg->channels[index],
            firstDestinations[index], firstSource, ctx.firstSize,
            ctx.firstEvent, static_cast<uint16_t>(1U << index)));
    }

    CCU_IF(ctx.secondSize != 0)
    {
        ccu::LocalAddr secondSource;
        secondSource.addr = ctx.input;
        secondSource.addr += ctx.firstSize;
        secondSource.token = ctx.token;
        std::vector<ccu::RemoteAddr> secondDestinations(channelCount);
        for (uint32_t index = 0; index < channelCount; ++index) {
            secondDestinations[index].addr = ctx.remoteOutputs[index];
            secondDestinations[index].addr += ctx.outputOffset;
            secondDestinations[index].addr += ctx.firstSize;
            secondDestinations[index].token = ctx.remoteTokens[index];
            CCU_CHK_RET(ccu::Write(ctx.arg->channels[index],
                secondDestinations[index], secondSource, ctx.secondSize,
                ctx.firstEvent,
                static_cast<uint16_t>(1U << (index + channelCount))));
        }
    }
    if (ctx.arg->doLocalCopy != 0) {
        CCU_IF(ctx.copyFlag != 0)
        {
            ccu::LocalAddr localDestination;
            localDestination.addr = ctx.output;
            localDestination.addr += ctx.outputOffset;
            localDestination.token = ctx.token;
            CCU_CHK_RET(GroupCopy(ctx, localDestination, firstSource));
        }
    }

    // Conservative early return: wait until every peer's first <=256 MiB WQE
    // and the independent self-copy have completed.  The three second WQEs
    // were already submitted and may remain in flight after this kernel exits.
    // The local record also gives GroupCopy a concrete graph successor.
    CCU_CHK_RET(ccu::EventRecord(ctx.firstEvent, copyMask));
    CCU_CHK_RET(ccu::EventWait(ctx.firstEvent, completionMask));
    return CCU_SUCCESS;
}

CcuResult InitSmallDirectResources(CcuSmallDirectContext &ctx)
{
    if (ctx.arg->channelCount == 0 || ctx.arg->channelCount >= MAX_RANK_SIZE) {
        HCCL_ERROR("[CcuSmallDirectKernel] invalid channel count %u",
            ctx.arg->channelCount);
        return CCU_E_PARA;
    }

    ctx.remoteOutputs.resize(ctx.arg->channelCount);
    ctx.remoteTokens.resize(ctx.arg->channelCount);
    for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
        ctx.remoteOutputs[index] =
            ccu::GetResByChannel<ccu::Variable>(
                ctx.arg->channels[index], OUTPUT_XN_ID);
        ctx.remoteTokens[index] =
            ccu::GetResByChannel<ccu::Variable>(
                ctx.arg->channels[index], TOKEN_XN_ID);
    }
    return CCU_SUCCESS;
}

CcuResult LoadSmallDirectArgs(CcuSmallDirectContext &ctx)
{
    uint32_t argIndex = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.input, argIndex++));
    CCU_CHK_RET(ccu::LoadArg(ctx.output, argIndex++));
    CCU_CHK_RET(ccu::LoadArg(ctx.token, argIndex++));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputOffset, argIndex++));
    CCU_CHK_RET(ccu::LoadArg(ctx.size, argIndex++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.addrOffset, argIndex++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.loopParam, argIndex++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.parallelParam, argIndex++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.residual, argIndex++));
    return CCU_SUCCESS;
}

CcuResult Load4x1SmallAddressArgs(CcuSmallDirectContext &ctx)
{
    CCU_CHK_RET(ccu::LoadArg(ctx.output, 0));
    CCU_CHK_RET(ccu::LoadArg(ctx.token, 1));
    return CCU_SUCCESS;
}

CcuResult Load4x1SmallTransferArgs(CcuSmallDirectContext &ctx)
{
    CCU_CHK_RET(ccu::LoadArg(ctx.input, 2));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputOffset, 3));
    CCU_CHK_RET(ccu::LoadArg(ctx.size, 4));
    return CCU_SUCCESS;
}

CcuResult TransferSmallDirect(CcuSmallDirectContext &ctx)
{
    ccu::LocalAddr source;
    source.addr = ctx.input;
    source.token = ctx.token;

    std::vector<ccu::RemoteAddr> destinations(ctx.arg->channelCount);
    for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
        destinations[index].addr = ctx.remoteOutputs[index];
        destinations[index].addr += ctx.outputOffset;
        destinations[index].token = ctx.remoteTokens[index];
        CCU_CHK_RET(ccu::Write(ctx.arg->channels[index], destinations[index],
            source, ctx.size, ctx.event, static_cast<uint16_t>(1U << index)));
    }

    if (ctx.arg->doLocalCopy != 0) {
        ccu::LocalAddr localDestination;
        localDestination.addr = ctx.output;
        localDestination.addr += ctx.outputOffset;
        localDestination.token = ctx.token;
        CCU_CHK_RET(GroupCopy(ctx, localDestination, source));
    }

    if (ctx.arg->waitWriteCq == 0) {
        if (ctx.arg->doLocalCopy != 0) {
            // Preserve the mandatory concrete successor of GroupCopy while
            // allowing all communication Writes to remain in flight.
            const uint16_t copyFence =
                static_cast<uint16_t>(1U << ctx.arg->channelCount);
            CCU_CHK_RET(ccu::EventRecord(ctx.event, copyFence));
            CCU_CHK_RET(ccu::EventWait(ctx.event, copyFence));
        }
        return CCU_SUCCESS;
    }

    const uint16_t allEvents =
        static_cast<uint16_t>((1U << ctx.arg->channelCount) - 1U);
    CCU_CHK_RET(ccu::EventWait(ctx.event, allEvents));
    return CCU_SUCCESS;
}

CcuResult Submit4x1SmallSelfCopy(CcuSmallDirectContext &ctx)
{
    constexpr uint16_t copyMask = 1U << 3;
    ccu::LocalAddr source;
    source.addr = ctx.input;
    source.token = ctx.token;
    ccu::LocalAddr destination;
    destination.addr = ctx.output;
    destination.addr += ctx.outputOffset;
    destination.token = ctx.token;
    return ccu::LocalCopy(
        destination, source, ctx.size, ctx.event, copyMask);
}

CcuResult Transfer4x1SmallDirect(CcuSmallDirectContext &ctx)
{
    constexpr uint32_t channelCount = 3;
    constexpr uint16_t copyMask = 1U << channelCount;
    constexpr uint16_t readyMask =
        (1U << OUTPUT_XN_ID) | (1U << TOKEN_XN_ID);
    if (ctx.arg->channelCount != channelCount ||
        ctx.arg->doLocalCopy == 0) {
        HCCL_ERROR("[Ccu4x1SmallDirectKernel] invalid channel/copy %u/%u",
            ctx.arg->channelCount, ctx.arg->doLocalCopy);
        return CCU_E_PARA;
    }

    ccu::LocalAddr source;
    source.addr = ctx.input;
    source.token = ctx.token;
    std::vector<ccu::RemoteAddr> destinations(channelCount);
    for (uint32_t index = 0; index < channelCount; ++index) {
        CCU_CHK_RET(ccu::NotifyWait(
            ctx.arg->channels[index], CKE_IDX_0, readyMask));
        destinations[index].addr = ctx.remoteOutputs[index];
        destinations[index].addr += ctx.outputOffset;
        destinations[index].token = ctx.remoteTokens[index];
        CCU_CHK_RET(ccu::Write(
            ctx.arg->channels[index], destinations[index], source, ctx.size,
            ctx.event, static_cast<uint16_t>(1U << index)));
    }

    // Aggressive score path: the three network Writes remain in flight after
    // submission.  Return as soon as the independent self-copy completes,
    // without waiting for the three outbound Write CQ bits.
    CCU_CHK_RET(ccu::EventWait(ctx.event, copyMask));
    return CCU_SUCCESS;
}

template <typename Context>
CcuResult Init8p4NetworkResources(Context &ctx)
{
    const uint32_t expectedChannels = ctx.arg->isPartialSide != 0 ?
        EIGHT_P_FOUR_PARTIAL_CHANNEL_COUNT :
        EIGHT_P_FOUR_FULL_CHANNEL_COUNT;
    const uint32_t expectedSeedPeers = ctx.arg->isPartialSide != 0 ? 2U : 1U;
    const uint32_t localGroupSize = ctx.arg->isPartialSide != 0 ?
        EIGHT_P_FOUR_FULL_CHANNEL_COUNT :
        EIGHT_P_FOUR_PARTIAL_CHANNEL_COUNT;
    if (ctx.arg->channelCount != expectedChannels ||
        ctx.arg->seedPeerCount != expectedSeedPeers ||
        ctx.arg->localIndex >= localGroupSize) {
        HCCL_ERROR("[Ccu8p4Network] invalid role/channel/index %u/%u/%u",
            ctx.arg->isPartialSide, ctx.arg->channelCount,
            ctx.arg->localIndex);
        return CCU_E_PARA;
    }
    for (uint32_t seed = 0; seed < ctx.arg->seedPeerCount; ++seed) {
        if (ctx.arg->seedPeerIndices[seed] >= ctx.arg->channelCount) {
            HCCL_ERROR("[Ccu8p4Network] invalid seed peer index %u",
                ctx.arg->seedPeerIndices[seed]);
            return CCU_E_PARA;
        }
    }

    ctx.remoteInputs.resize(ctx.arg->channelCount);
    ctx.remoteOutputs.resize(ctx.arg->channelCount);
    ctx.remoteTokens.resize(ctx.arg->channelCount);
    for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
        ctx.remoteInputs[index] =
            ccu::GetResByChannel<ccu::Variable>(
                ctx.arg->channels[index], EIGHT_P_FOUR_INPUT_XN_ID);
        ctx.remoteOutputs[index] =
            ccu::GetResByChannel<ccu::Variable>(
                ctx.arg->channels[index], EIGHT_P_FOUR_OUTPUT_XN_ID);
        ctx.remoteTokens[index] =
            ccu::GetResByChannel<ccu::Variable>(
                ctx.arg->channels[index], EIGHT_P_FOUR_TOKEN_XN_ID);
    }
    return CCU_SUCCESS;
}

CcuResult Load8p4SeedArgs(Ccu8p4SeedContext &ctx)
{
    uint32_t argIndex = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.input, argIndex++));
    CCU_CHK_RET(ccu::LoadArg(ctx.output, argIndex++));
    CCU_CHK_RET(ccu::LoadArg(ctx.token, argIndex++));
    CCU_CHK_RET(ccu::LoadArg(ctx.fullSeedSize, argIndex++));
    CCU_CHK_RET(ccu::LoadArg(ctx.partialSeedSize, argIndex++));
    CCU_CHK_RET(ccu::LoadArg(ctx.ownOutputOffset, argIndex++));
    for (uint32_t index = 0; index < EIGHT_P_FOUR_PARTIAL_CHANNEL_COUNT; ++index) {
        CCU_CHK_RET(ccu::LoadArg(
            ctx.remoteOutputOffsets[index], argIndex++));
    }
    return CCU_SUCCESS;
}

CcuResult Publish8p4FullSideAddresses(Ccu8p4SeedContext &ctx)
{
    for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
        CCU_CHK_RET(ccu::WriteVariableWithNotify(
            ctx.arg->channels[index], ctx.input,
            EIGHT_P_FOUR_INPUT_XN_ID, CKE_IDX_0,
            1U << EIGHT_P_FOUR_INPUT_XN_ID));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(
            ctx.arg->channels[index], ctx.output,
            EIGHT_P_FOUR_OUTPUT_XN_ID, CKE_IDX_0,
            1U << EIGHT_P_FOUR_OUTPUT_XN_ID));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(
            ctx.arg->channels[index], ctx.token,
            EIGHT_P_FOUR_TOKEN_XN_ID, CKE_IDX_0,
            1U << EIGHT_P_FOUR_TOKEN_XN_ID));
    }

    return CCU_SUCCESS;
}

CcuResult Wait8p4PartialSideAddresses(Ccu8p4SeedContext &ctx)
{
    constexpr uint16_t readyMask =
        (1U << EIGHT_P_FOUR_INPUT_XN_ID) |
        (1U << EIGHT_P_FOUR_OUTPUT_XN_ID) |
        (1U << EIGHT_P_FOUR_TOKEN_XN_ID);
    for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
        CCU_CHK_RET(ccu::NotifyWait(
            ctx.arg->channels[index], CKE_IDX_0, readyMask));
    }

    return CCU_SUCCESS;
}

CcuResult Transfer8p4Seed(Ccu8p4SeedContext &ctx)
{
    if (ctx.arg->isPartialSide == 0) {
        CCU_CHK_RET(Publish8p4FullSideAddresses(ctx));
        // Only the first four full-side ranks are gateways for partial-side
        // prefixes. The other four ranks have no seed dependency.
        if (ctx.arg->waitSeedData != 0) {
            constexpr uint16_t seedMask = 1U << EIGHT_P_FOUR_SEED_DATA_ID;
            CCU_CHK_RET(ccu::NotifyWait(
                ctx.arg->channels[ctx.arg->seedPeerIndices[0]],
                CKE_IDX_0, seedMask));
        }
        return CCU_SUCCESS;
    }

    CCU_CHK_RET(Wait8p4PartialSideAddresses(ctx));
    const uint32_t firstReadIndex = ctx.arg->seedPeerIndices[0];
    const uint32_t secondReadIndex = ctx.arg->seedPeerIndices[1];
    const uint32_t writeIndex = firstReadIndex;

    ccu::LocalAddr writeSource;
    writeSource.addr = ctx.input;
    writeSource.token = ctx.token;
    ccu::RemoteAddr writeDestination;
    writeDestination.addr = ctx.remoteOutputs[writeIndex];
    writeDestination.addr += ctx.ownOutputOffset;
    writeDestination.token = ctx.remoteTokens[writeIndex];
    CCU_CHK_RET(ccu::Write(ctx.arg->channels[writeIndex],
        writeDestination, writeSource, ctx.partialSeedSize,
        ctx.writeEvent, 1));

    ccu::RemoteAddr firstReadSource;
    firstReadSource.addr = ctx.remoteInputs[firstReadIndex];
    firstReadSource.token = ctx.remoteTokens[firstReadIndex];
    ccu::LocalAddr firstReadDestination;
    firstReadDestination.addr = ctx.output;
    firstReadDestination.addr += ctx.remoteOutputOffsets[firstReadIndex];
    firstReadDestination.token = ctx.token;
    CCU_CHK_RET(ccu::Read(ctx.arg->channels[firstReadIndex],
        firstReadDestination, firstReadSource, ctx.fullSeedSize,
        ctx.readEvent, 1));

    ccu::RemoteAddr secondReadSource;
    secondReadSource.addr = ctx.remoteInputs[secondReadIndex];
    secondReadSource.token = ctx.remoteTokens[secondReadIndex];
    ccu::LocalAddr secondReadDestination;
    secondReadDestination.addr = ctx.output;
    secondReadDestination.addr += ctx.remoteOutputOffsets[secondReadIndex];
    secondReadDestination.token = ctx.token;
    CCU_CHK_RET(ccu::Read(ctx.arg->channels[secondReadIndex],
        secondReadDestination, secondReadSource, ctx.fullSeedSize,
        ctx.readEvent, 2));

    // The partial->full seed can release its relay as soon as its own Write
    // completes; both inbound Reads continue independently.
    CCU_CHK_RET(ccu::EventWait(ctx.writeEvent, 1));
    constexpr uint16_t seedMask = 1U << EIGHT_P_FOUR_SEED_DATA_ID;
    CCU_CHK_RET(ccu::NotifyRecord(
        ctx.arg->channels[writeIndex], CKE_IDX_0, seedMask));
    CCU_CHK_RET(ccu::EventWait(ctx.readEvent, 3));
    return CCU_SUCCESS;
}

CcuResult Load8p4SuffixArgs(Ccu8p4SuffixContext &ctx)
{
    uint32_t argIndex = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.input, argIndex++));
    CCU_CHK_RET(ccu::LoadArg(ctx.output, argIndex++));
    CCU_CHK_RET(ccu::LoadArg(ctx.token, argIndex++));
    CCU_CHK_RET(ccu::LoadArg(ctx.fullSeedSize, argIndex++));
    CCU_CHK_RET(ccu::LoadArg(ctx.partialSeedSize, argIndex++));
    CCU_CHK_RET(ccu::LoadArg(ctx.fullSuffixFirstSize, argIndex++));
    CCU_CHK_RET(ccu::LoadArg(ctx.fullSuffixSecondSize, argIndex++));
    CCU_CHK_RET(ccu::LoadArg(ctx.partialSuffixFirstSize, argIndex++));
    CCU_CHK_RET(ccu::LoadArg(ctx.partialSuffixSecondSize, argIndex++));
    CCU_CHK_RET(ccu::LoadArg(ctx.ownOutputOffset, argIndex++));
    for (uint32_t index = 0; index < EIGHT_P_FOUR_PARTIAL_CHANNEL_COUNT; ++index) {
        CCU_CHK_RET(ccu::LoadArg(
            ctx.remoteOutputOffsets[index], argIndex++));
    }
    return CCU_SUCCESS;
}

CcuResult Submit8p4SuffixRead(
    Ccu8p4SuffixContext &ctx, uint32_t channelIndex, uint32_t slot)
{
    const uint16_t eventMask = static_cast<uint16_t>(1U << slot);
    ccu::RemoteAddr firstSource;
    firstSource.addr = ctx.remoteInputs[channelIndex];
    firstSource.addr += ctx.fullSeedSize;
    firstSource.token = ctx.remoteTokens[channelIndex];
    ccu::LocalAddr firstDestination;
    firstDestination.addr = ctx.output;
    firstDestination.addr += ctx.remoteOutputOffsets[channelIndex];
    firstDestination.addr += ctx.fullSeedSize;
    firstDestination.token = ctx.token;
    CCU_CHK_RET(ccu::Read(ctx.arg->channels[channelIndex],
        firstDestination, firstSource, ctx.fullSuffixFirstSize,
        ctx.readFirstEvent, eventMask));

    CCU_IF(ctx.fullSuffixSecondSize != 0)
    {
        firstSource.addr += ctx.fullSuffixFirstSize;
        firstDestination.addr += ctx.fullSuffixFirstSize;
        CCU_CHK_RET(ccu::Read(ctx.arg->channels[channelIndex],
            firstDestination, firstSource, ctx.fullSuffixSecondSize,
            ctx.readSecondEvent, eventMask));
    }
    return CCU_SUCCESS;
}

CcuResult Wait8p4SuffixReads(Ccu8p4SuffixContext &ctx)
{
    const uint16_t allEvents =
        static_cast<uint16_t>((1U << ctx.arg->channelCount) - 1U);
    CCU_CHK_RET(ccu::EventWait(ctx.readFirstEvent, allEvents));
    CCU_IF(ctx.fullSuffixSecondSize != 0)
    {
        CCU_CHK_RET(ccu::EventWait(ctx.readSecondEvent, allEvents));
    }
    return CCU_SUCCESS;
}

CcuResult Submit8p4SuffixWrites(Ccu8p4SuffixContext &ctx)
{
    ccu::LocalAddr firstSource;
    firstSource.addr = ctx.input;
    firstSource.addr += ctx.partialSeedSize;
    firstSource.token = ctx.token;
    std::vector<ccu::RemoteAddr> firstDestinations(ctx.arg->channelCount);
    for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
        const uint16_t eventMask = static_cast<uint16_t>(1U << index);
        firstDestinations[index].addr = ctx.remoteOutputs[index];
        firstDestinations[index].addr += ctx.ownOutputOffset;
        firstDestinations[index].addr += ctx.partialSeedSize;
        firstDestinations[index].token = ctx.remoteTokens[index];
        CCU_CHK_RET(ccu::Write(ctx.arg->channels[index],
            firstDestinations[index], firstSource,
            ctx.partialSuffixFirstSize, ctx.writeFirstEvent, eventMask));
    }

    CCU_IF(ctx.partialSuffixSecondSize != 0)
    {
        firstSource.addr += ctx.partialSuffixFirstSize;
        std::vector<ccu::RemoteAddr> secondDestinations(ctx.arg->channelCount);
        for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
            const uint16_t eventMask = static_cast<uint16_t>(1U << index);
            secondDestinations[index].addr = ctx.remoteOutputs[index];
            secondDestinations[index].addr += ctx.ownOutputOffset;
            secondDestinations[index].addr += ctx.partialSeedSize;
            secondDestinations[index].addr += ctx.partialSuffixFirstSize;
            secondDestinations[index].token = ctx.remoteTokens[index];
            CCU_CHK_RET(ccu::Write(ctx.arg->channels[index],
                secondDestinations[index], firstSource,
                ctx.partialSuffixSecondSize, ctx.writeSecondEvent, eventMask));
        }
    }
    return CCU_SUCCESS;
}

CcuResult Wait8p4SuffixWrites(Ccu8p4SuffixContext &ctx)
{
    const uint16_t allEvents =
        static_cast<uint16_t>((1U << ctx.arg->channelCount) - 1U);
    CCU_CHK_RET(ccu::EventWait(ctx.writeFirstEvent, allEvents));
    CCU_IF(ctx.partialSuffixSecondSize != 0)
    {
        CCU_CHK_RET(ccu::EventWait(ctx.writeSecondEvent, allEvents));
    }
    return CCU_SUCCESS;
}

CcuResult Transfer8p4Suffix(Ccu8p4SuffixContext &ctx)
{
    constexpr uint16_t doneMask = 1U << EIGHT_P_FOUR_DONE_ID;
    if (ctx.arg->isPartialSide == 0) {
        for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
            CCU_CHK_RET(ccu::NotifyWait(
                ctx.arg->channels[index], CKE_IDX_0, doneMask));
        }
        for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
            CCU_CHK_RET(ccu::NotifyRecord(
                ctx.arg->channels[index], CKE_IDX_0, doneMask));
        }
        return CCU_SUCCESS;
    }

    // Submit both WQEs for all eight peers before observing any CQ. Besides
    // maximizing channel overlap, this deliberately avoids imposing an
    // artificial four-peer wave boundary on the performance task graph.
    for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
        CCU_CHK_RET(Submit8p4SuffixRead(ctx, index, index));
    }

    // The reverse direction has four senders and eight receivers, hence no
    // eight-to-four incast. Submit all eight Writes once and do not couple
    // their completions to the Read CQ.
    CCU_CHK_RET(Submit8p4SuffixWrites(ctx));

    CCU_CHK_RET(Wait8p4SuffixReads(ctx));
    CCU_CHK_RET(Wait8p4SuffixWrites(ctx));
    for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
        CCU_CHK_RET(ccu::NotifyRecord(
            ctx.arg->channels[index], CKE_IDX_0, doneMask));
    }
    for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
        CCU_CHK_RET(ccu::NotifyWait(
            ctx.arg->channels[index], CKE_IDX_0, doneMask));
    }
    return CCU_SUCCESS;
}

CcuResult InitRelayResources(CcuRelayContext &ctx)
{
    if (ctx.arg->channelCount == 0 || ctx.arg->channelCount >= MAX_RANK_SIZE ||
        ctx.arg->relayCount > 2) {
        HCCL_ERROR("[CcuRelayKernel] invalid channel/relay count %u/%u",
            ctx.arg->channelCount, ctx.arg->relayCount);
        return CCU_E_PARA;
    }
    ctx.remoteOutputs.resize(ctx.arg->channelCount);
    ctx.remoteTokens.resize(ctx.arg->channelCount);
    for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
        ctx.remoteOutputs[index] =
            ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[index], OUTPUT_XN_ID);
        ctx.remoteTokens[index] =
            ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[index], TOKEN_XN_ID);
    }
    return CCU_SUCCESS;
}

CcuResult LoadRelayArgs(CcuRelayContext &ctx)
{
    uint32_t argIndex = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.input, argIndex++));
    CCU_CHK_RET(ccu::LoadArg(ctx.output, argIndex++));
    CCU_CHK_RET(ccu::LoadArg(ctx.token, argIndex++));
    CCU_CHK_RET(ccu::LoadArg(ctx.ownOffset, argIndex++));
    CCU_CHK_RET(ccu::LoadArg(ctx.firstSize, argIndex++));
    CCU_CHK_RET(ccu::LoadArg(ctx.secondSize, argIndex++));
    CCU_CHK_RET(ccu::LoadArg(ctx.sendOwn, argIndex++));
    CCU_CHK_RET(ccu::LoadArg(ctx.relayOffsets[0], argIndex++));
    CCU_CHK_RET(ccu::LoadArg(ctx.relayOffsets[1], argIndex++));
    return CCU_SUCCESS;
}

CcuResult PreSyncRelay(CcuRelayContext &ctx)
{
    for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
        CCU_CHK_RET(ccu::WriteVariableWithNotify(
            ctx.arg->channels[index], ctx.output, OUTPUT_XN_ID,
            CKE_IDX_0, 1U << OUTPUT_XN_ID));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(
            ctx.arg->channels[index], ctx.token, TOKEN_XN_ID,
            CKE_IDX_0, 1U << TOKEN_XN_ID));
    }
    constexpr uint16_t readyMask = (1U << OUTPUT_XN_ID) | (1U << TOKEN_XN_ID);
    for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
        CCU_CHK_RET(ccu::NotifyWait(
            ctx.arg->channels[index], CKE_IDX_0, readyMask));
    }
    return CCU_SUCCESS;
}

CcuResult WriteRelayBlock(CcuRelayContext &ctx, ccu::LocalAddr source,
    ccu::Variable destinationOffset, ccu::Event &firstEvent, ccu::Event &secondEvent)
{
    std::vector<ccu::RemoteAddr> firstDestinations(ctx.arg->channelCount);
    for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
        firstDestinations[index].addr = ctx.remoteOutputs[index];
        firstDestinations[index].addr += destinationOffset;
        firstDestinations[index].token = ctx.remoteTokens[index];
        CCU_CHK_RET(ccu::Write(ctx.arg->channels[index], firstDestinations[index],
            source, ctx.firstSize, firstEvent, static_cast<uint16_t>(1U << index)));
    }

    CCU_IF(ctx.secondSize != 0)
    {
        source.addr += ctx.firstSize;
        std::vector<ccu::RemoteAddr> secondDestinations(ctx.arg->channelCount);
        for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
            secondDestinations[index].addr = ctx.remoteOutputs[index];
            secondDestinations[index].addr += destinationOffset;
            secondDestinations[index].addr += ctx.firstSize;
            secondDestinations[index].token = ctx.remoteTokens[index];
            CCU_CHK_RET(ccu::Write(ctx.arg->channels[index], secondDestinations[index],
                source, ctx.secondSize, secondEvent, static_cast<uint16_t>(1U << index)));
        }
    }

    return CCU_SUCCESS;
}

CcuResult WaitRelayBlock(
    CcuRelayContext &ctx, ccu::Event &firstEvent, ccu::Event &secondEvent)
{
    const uint16_t allEvents =
        static_cast<uint16_t>((1U << ctx.arg->channelCount) - 1U);
    CCU_CHK_RET(ccu::EventWait(firstEvent, allEvents));
    CCU_IF(ctx.secondSize != 0)
    {
        CCU_CHK_RET(ccu::EventWait(secondEvent, allEvents));
    }
    return CCU_SUCCESS;
}

CcuResult TransferRelay(CcuRelayContext &ctx)
{
    CCU_IF(ctx.sendOwn != 0)
    {
        ccu::LocalAddr ownSource;
        ownSource.addr = ctx.input;
        ownSource.token = ctx.token;
        CCU_CHK_RET(WriteRelayBlock(
            ctx, ownSource, ctx.ownOffset, ctx.ownFirstEvent, ctx.ownSecondEvent));
    }

    for (uint32_t relay = 0; relay < ctx.arg->relayCount; ++relay) {
        ccu::LocalAddr relaySource;
        relaySource.addr = ctx.output;
        relaySource.addr += ctx.relayOffsets[relay];
        relaySource.token = ctx.token;
        CCU_CHK_RET(WriteRelayBlock(ctx, relaySource, ctx.relayOffsets[relay],
            ctx.relayFirstEvents[relay], ctx.relaySecondEvents[relay]));
    }

    CCU_IF(ctx.sendOwn != 0)
    {
        CCU_CHK_RET(WaitRelayBlock(
            ctx, ctx.ownFirstEvent, ctx.ownSecondEvent));
    }
    for (uint32_t relay = 0; relay < ctx.arg->relayCount; ++relay) {
        CCU_CHK_RET(WaitRelayBlock(ctx,
            ctx.relayFirstEvents[relay], ctx.relaySecondEvents[relay]));
    }
    return CCU_SUCCESS;
}

CcuResult PostSyncRelay(CcuRelayContext &ctx)
{
    constexpr uint16_t doneMask = 1U << POST_SYNC_ID;
    for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
        CCU_CHK_RET(ccu::NotifyRecord(
            ctx.arg->channels[index], CKE_IDX_0, doneMask));
    }
    for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
        CCU_CHK_RET(ccu::NotifyWait(
            ctx.arg->channels[index], CKE_IDX_0, doneMask));
    }
    return CCU_SUCCESS;
}

} // namespace

CcuResult CcuDirectKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgDirect *>(arg);
    if (kernelArg == nullptr) {
        return CCU_E_PTR;
    }

    CcuDirectContext ctx;
    ctx.arg = kernelArg;
    CCU_CHK_RET(InitDirectResources(ctx));
    CCU_CHK_RET(LoadDirectArgs(ctx));
    CCU_CHK_RET(PublishDirectAddresses(ctx));
    CCU_CHK_RET(WaitDirectAddresses(ctx));
    CCU_CHK_RET(TransferDirect(ctx));
    CCU_IF(ctx.skipDone == 0)
    {
        CCU_CHK_RET(PostSyncDirect(ctx));
    }
    return CCU_SUCCESS;
}

CcuResult Ccu4x1DirectKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgDirect *>(arg);
    if (kernelArg == nullptr) {
        return CCU_E_PTR;
    }

    CcuDirectContext ctx;
    ctx.arg = kernelArg;
    CCU_CHK_RET(InitDirectResources(ctx));
    // OUTPUT/TOKEN are the only values peers need before they can start.
    // Publish them before loading the remaining ten transfer/copy arguments.
    CCU_CHK_RET(LoadDirectAddressArgs(ctx));
    CCU_CHK_RET(PublishDirectAddresses(ctx));
    CCU_CHK_RET(LoadDirectTransferArgs(ctx));
    CCU_CHK_RET(Transfer4x1Direct(ctx));
    CCU_IF(ctx.skipDone == 0)
    {
        CCU_CHK_RET(PostSyncDirect(ctx));
    }
    return CCU_SUCCESS;
}

CcuResult CcuLargeDirectKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgDirect *>(arg);
    if (kernelArg == nullptr) {
        return CCU_E_PTR;
    }

    CcuDirectContext ctx;
    ctx.arg = kernelArg;
    CCU_CHK_RET(InitDirectResources(ctx));
    // Overlap address propagation with the remaining argument loads.  For
    // network-only seed/suffix kernels only the compact first eight task
    // arguments are loaded; local-copy kernels retain all twelve.
    CCU_CHK_RET(LoadDirectAddressArgs(ctx));
    CCU_CHK_RET(PublishDirectAddresses(ctx));
    CCU_CHK_RET(LoadDirectTransferArgs(ctx));
    CCU_CHK_RET(WaitDirectAddresses(ctx));
    CCU_CHK_RET(TransferDirect(ctx));
    CCU_IF(ctx.skipDone == 0)
    {
        CCU_CHK_RET(PostSyncDirect(ctx));
    }
    return CCU_SUCCESS;
}

CcuResult CcuSmallDirectKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgSmallDirect *>(arg);
    if (kernelArg == nullptr) {
        return CCU_E_PTR;
    }

    CcuSmallDirectContext ctx;
    ctx.arg = kernelArg;
    CCU_CHK_RET(InitSmallDirectResources(ctx));
    CCU_CHK_RET(LoadSmallDirectArgs(ctx));
    CCU_CHK_RET(PublishDirectAddresses(ctx));
    CCU_CHK_RET(WaitDirectAddresses(ctx));
    CCU_CHK_RET(TransferSmallDirect(ctx));
    return CCU_SUCCESS;
}

CcuResult Ccu4x1SmallDirectKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgSmallDirect *>(arg);
    if (kernelArg == nullptr) {
        return CCU_E_PTR;
    }

    CcuSmallDirectContext ctx;
    ctx.arg = kernelArg;
    CCU_CHK_RET(InitSmallDirectResources(ctx));
    // Publish the two values needed by peers as soon as they are loaded.
    // The remaining argument loads and the asynchronous self copy then overlap
    // the control-plane address exchange.
    CCU_CHK_RET(Load4x1SmallAddressArgs(ctx));
    CCU_CHK_RET(PublishDirectAddresses(ctx));
    CCU_CHK_RET(Load4x1SmallTransferArgs(ctx));
    CCU_CHK_RET(Submit4x1SmallSelfCopy(ctx));
    CCU_CHK_RET(Transfer4x1SmallDirect(ctx));
    return CCU_SUCCESS;
}

CcuResult Ccu8p4SeedKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArg8p4Network *>(arg);
    if (kernelArg == nullptr) {
        return CCU_E_PTR;
    }

    Ccu8p4SeedContext ctx;
    ctx.arg = kernelArg;
    CCU_CHK_RET(Init8p4NetworkResources(ctx));
    CCU_CHK_RET(Load8p4SeedArgs(ctx));
    CCU_CHK_RET(Transfer8p4Seed(ctx));
    return CCU_SUCCESS;
}

CcuResult Ccu8p4SuffixKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArg8p4Network *>(arg);
    if (kernelArg == nullptr) {
        return CCU_E_PTR;
    }

    Ccu8p4SuffixContext ctx;
    ctx.arg = kernelArg;
    CCU_CHK_RET(Init8p4NetworkResources(ctx));
    CCU_CHK_RET(Load8p4SuffixArgs(ctx));
    CCU_CHK_RET(Transfer8p4Suffix(ctx));
    return CCU_SUCCESS;
}

CcuResult CcuRelayKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgRelay *>(arg);
    if (kernelArg == nullptr) {
        return CCU_E_PTR;
    }

    CcuRelayContext ctx;
    ctx.arg = kernelArg;
    CCU_CHK_RET(InitRelayResources(ctx));
    CCU_CHK_RET(LoadRelayArgs(ctx));
    if (kernelArg->reuseDirectAddresses == 0) {
        CCU_CHK_RET(PreSyncRelay(ctx));
    }
    CCU_CHK_RET(TransferRelay(ctx));
    if (kernelArg->postSync != 0) {
        CCU_CHK_RET(PostSyncRelay(ctx));
    }
    return CCU_SUCCESS;
}

} // namespace ops_hccl
