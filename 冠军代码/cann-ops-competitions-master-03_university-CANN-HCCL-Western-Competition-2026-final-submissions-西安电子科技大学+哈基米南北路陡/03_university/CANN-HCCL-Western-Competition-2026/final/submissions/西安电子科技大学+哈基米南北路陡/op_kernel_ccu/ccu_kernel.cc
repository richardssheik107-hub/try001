/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdint>
#include <vector>

#include <ccu/ccu_array.hpp>
#include <ccu/ccu_loop.hpp>
#include <hcomm/hcomm_primitives.h>

#include "ccu_kernel.h"
#include "log.h"

#ifndef CCU_CHK_RET
#define CCU_CHK_RET(call) \
    do { \
        const CcuResult ccuCallRet = (call); \
        if (ccuCallRet != CCU_SUCCESS) { \
            return ccuCallRet; \
        } \
    } while (0)
#endif

namespace ops_hccl {
namespace ccu = ::AscendC::ccu;

namespace {
    constexpr uint32_t INPUT_ADDR_XN_ID = 0;
    constexpr uint32_t INPUT_TOKEN_XN_ID = 1;
    constexpr uint32_t OUTPUT_ADDR_XN_ID = 2;
    constexpr uint32_t OUTPUT_TOKEN_XN_ID = 3;
    constexpr uint32_t PARAM_SYNC_NOTIFY_INDEX = 0;
    constexpr uint32_t PARTIAL_DONE_NOTIFY_INDEX = 1;
    constexpr uint32_t BROADCAST_DONE_NOTIFY_INDEX = 2;
    constexpr uint32_t PHASE_DONE_BIT_INDEX = 0;
    constexpr uint32_t SCRATCH_ADDR_XN_ID = 0;
    constexpr uint32_t SCRATCH_TOKEN_XN_ID = 1;
    constexpr uint32_t BFLY_DATA_NOTIFY_INDEX = 1;
    constexpr uint32_t SAME_DONE_NOTIFY_INDEX = 1;
    constexpr uint32_t LANE_DATA_NOTIFY_INDEX = 2;
    constexpr uint32_t LANE_NEXT_NOTIFY_INDEX = 3;
    constexpr uint32_t CROSS_BROADCAST_DONE_NOTIFY_INDEX = 4;
    constexpr uint32_t CROSS_PHASE_DONE_BIT_INDEX = 0;
    constexpr uint32_t CROSS_MAX_LANE_SIZE = 4;

    struct OwnerRsagContext {
        const CcuKernelArgOwnerRsag *arg{nullptr};

        ccu::Variable inputAddr;
        ccu::Variable outputAddr;
        ccu::Variable inputToken;
        ccu::Variable outputToken;
        ccu::Variable scratchAddr;
        ccu::Variable scratchToken;
        ccu::Variable ownerOffset;
        ccu::Variable ownerBytes;
        ccu::Variable scratchStride;
        ccu::Variable phase;

        std::vector<ccu::Variable> peerInputAddr;
        std::vector<ccu::Variable> peerInputToken;
        std::vector<ccu::Variable> peerOutputAddr;
        std::vector<ccu::Variable> peerOutputToken;
        ccu::Event dataEvent;
    };

    CcuResult InitResources(OwnerRsagContext &ctx)
    {
        const auto *arg = ctx.arg;
        ctx.peerInputAddr.resize(arg->channelCount);
        ctx.peerInputToken.resize(arg->channelCount);
        ctx.peerOutputAddr.resize(arg->channelCount);
        ctx.peerOutputToken.resize(arg->channelCount);

        for (uint32_t i = 0; i < arg->channelCount; ++i) {
            ctx.peerInputAddr[i] = ccu::GetResByChannel<ccu::Variable>(arg->channels[i], INPUT_ADDR_XN_ID);
            ctx.peerInputToken[i] = ccu::GetResByChannel<ccu::Variable>(arg->channels[i], INPUT_TOKEN_XN_ID);
            ctx.peerOutputAddr[i] = ccu::GetResByChannel<ccu::Variable>(arg->channels[i], OUTPUT_ADDR_XN_ID);
            ctx.peerOutputToken[i] = ccu::GetResByChannel<ccu::Variable>(arg->channels[i], OUTPUT_TOKEN_XN_ID);
        }
        return CCU_SUCCESS;
    }

    CcuResult LoadArgs(OwnerRsagContext &ctx)
    {
        uint32_t argId = 0;
        CCU_CHK_RET(ccu::LoadArg(ctx.inputAddr, argId++));
        CCU_CHK_RET(ccu::LoadArg(ctx.outputAddr, argId++));
        CCU_CHK_RET(ccu::LoadArg(ctx.inputToken, argId++));
        CCU_CHK_RET(ccu::LoadArg(ctx.outputToken, argId++));
        CCU_CHK_RET(ccu::LoadArg(ctx.scratchAddr, argId++));
        CCU_CHK_RET(ccu::LoadArg(ctx.scratchToken, argId++));
        CCU_CHK_RET(ccu::LoadArg(ctx.ownerOffset, argId++));
        CCU_CHK_RET(ccu::LoadArg(ctx.ownerBytes, argId++));
        CCU_CHK_RET(ccu::LoadArg(ctx.scratchStride, argId++));
        CCU_CHK_RET(ccu::LoadArg(ctx.phase, argId++));
        return CCU_SUCCESS;
    }

    CcuResult PreSync(OwnerRsagContext &ctx)
    {
        const auto *arg = ctx.arg;
        for (uint32_t i = 0; i < arg->channelCount; ++i) {
            CCU_CHK_RET(ccu::WriteVariableWithNotify(
                arg->channels[i], ctx.inputAddr, INPUT_ADDR_XN_ID, PARAM_SYNC_NOTIFY_INDEX, 1U << INPUT_ADDR_XN_ID));
            CCU_CHK_RET(ccu::WriteVariableWithNotify(
                arg->channels[i], ctx.inputToken, INPUT_TOKEN_XN_ID, PARAM_SYNC_NOTIFY_INDEX, 1U << INPUT_TOKEN_XN_ID));
            CCU_CHK_RET(ccu::WriteVariableWithNotify(
                arg->channels[i], ctx.outputAddr, OUTPUT_ADDR_XN_ID, PARAM_SYNC_NOTIFY_INDEX, 1U << OUTPUT_ADDR_XN_ID));
            CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[i], ctx.outputToken, OUTPUT_TOKEN_XN_ID,
                PARAM_SYNC_NOTIFY_INDEX, 1U << OUTPUT_TOKEN_XN_ID));
        }

        constexpr uint16_t allBits = (1U << INPUT_ADDR_XN_ID) | (1U << INPUT_TOKEN_XN_ID) | (1U << OUTPUT_ADDR_XN_ID)
                                     | (1U << OUTPUT_TOKEN_XN_ID);
        for (uint32_t i = 0; i < arg->channelCount; ++i) {
            CCU_CHK_RET(ccu::NotifyWait(arg->channels[i], PARAM_SYNC_NOTIFY_INDEX, allBits));
        }
        return CCU_SUCCESS;
    }

    CcuResult PostSync(OwnerRsagContext &ctx, uint32_t notifyIndex)
    {
        const auto *arg = ctx.arg;
        constexpr uint16_t phaseDoneBit = 1U << PHASE_DONE_BIT_INDEX;
        for (uint32_t i = 0; i < arg->channelCount; ++i) {
            CCU_CHK_RET(ccu::NotifyRecord(arg->channels[i], notifyIndex, phaseDoneBit));
        }
        for (uint32_t i = 0; i < arg->channelCount; ++i) {
            CCU_CHK_RET(ccu::NotifyWait(arg->channels[i], notifyIndex, phaseDoneBit));
        }
        return CCU_SUCCESS;
    }

    void SetPartialAddr(const OwnerRsagContext &ctx, ccu::LocalAddr &partial)
    {
        partial.addr = ctx.scratchAddr;
        if (ctx.arg->isPrimary == 0) {
            partial.addr += ctx.scratchStride;
        }
        partial.token = ctx.scratchToken;
    }

    CcuResult DoPartialReduce(OwnerRsagContext &ctx)
    {
        const auto *arg = ctx.arg;
        ccu::LocalAddr partial;
        SetPartialAddr(ctx, partial);

        uint32_t firstPeer = 0;
        if (arg->isPrimary != 0) {
            ccu::LocalAddr localInput;
            localInput.addr = ctx.inputAddr;
            localInput.addr += ctx.ownerOffset;
            localInput.token = ctx.inputToken;
            CCU_CHK_RET(ccu::LocalCopy(partial, localInput, ctx.ownerBytes, ctx.dataEvent, 1));
            CCU_CHK_RET(ccu::EventWait(ctx.dataEvent, 1));
        } else {
            ccu::RemoteAddr firstInput;
            firstInput.addr = ctx.peerInputAddr[0];
            firstInput.addr += ctx.ownerOffset;
            firstInput.token = ctx.peerInputToken[0];
            CCU_CHK_RET(ccu::Read(arg->channels[0], partial, firstInput, ctx.ownerBytes, ctx.dataEvent, 1));
            CCU_CHK_RET(ccu::EventWait(ctx.dataEvent, 1));
            firstPeer = 1;
        }

        for (uint32_t i = firstPeer; i < arg->channelCount; ++i) {
            ccu::RemoteAddr peerInput;
            peerInput.addr = ctx.peerInputAddr[i];
            peerInput.addr += ctx.ownerOffset;
            peerInput.token = ctx.peerInputToken[i];
            CCU_CHK_RET(ccu::ReadReduce(arg->channels[i], partial, peerInput, ctx.ownerBytes, arg->dataType,
                arg->reduceType, ctx.dataEvent, 1));
            CCU_CHK_RET(ccu::EventWait(ctx.dataEvent, 1));
        }
        return CCU_SUCCESS;
    }

    CcuResult FinalizeOwnerSlice(OwnerRsagContext &ctx)
    {
        ccu::LocalAddr partial0;
        partial0.addr = ctx.scratchAddr;
        partial0.token = ctx.scratchToken;

        ccu::LocalAddr output;
        output.addr = ctx.outputAddr;
        output.addr += ctx.ownerOffset;
        output.token = ctx.outputToken;

        CCU_CHK_RET(ccu::LocalCopy(output, partial0, ctx.ownerBytes, ctx.dataEvent, 1));
        CCU_CHK_RET(ccu::EventWait(ctx.dataEvent, 1));

        if (ctx.arg->kernelCount == 2) {
            ccu::LocalAddr partial1;
            partial1.addr = ctx.scratchAddr;
            partial1.addr += ctx.scratchStride;
            partial1.token = ctx.scratchToken;
            CCU_CHK_RET(ccu::LocalReduce(
                output, partial1, ctx.ownerBytes, ctx.arg->dataType, ctx.arg->reduceType, ctx.dataEvent, 1));
            CCU_CHK_RET(ccu::EventWait(ctx.dataEvent, 1));
        }
        return CCU_SUCCESS;
    }

    CcuResult BroadcastOwnerSlice(OwnerRsagContext &ctx)
    {
        const auto *arg = ctx.arg;
        ccu::LocalAddr source;
        source.addr = ctx.outputAddr;
        source.addr += ctx.ownerOffset;
        source.token = ctx.outputToken;

        uint16_t allBits = 0;
        for (uint32_t i = 0; i < arg->channelCount; ++i) {
            const uint16_t bit = static_cast<uint16_t>(1U << i);
            ccu::RemoteAddr peerOutput;
            peerOutput.addr = ctx.peerOutputAddr[i];
            peerOutput.addr += ctx.ownerOffset;
            peerOutput.token = ctx.peerOutputToken[i];
            CCU_CHK_RET(ccu::Write(arg->channels[i], peerOutput, source, ctx.ownerBytes, ctx.dataEvent, bit));
            allBits = static_cast<uint16_t>(allBits | bit);
        }
        CCU_CHK_RET(ccu::EventWait(ctx.dataEvent, allBits));
        return CCU_SUCCESS;
    }

    struct ButterflyContext {
        const CcuKernelArgButterfly2x8 *arg{nullptr};
        ccu::Variable inputAddr;
        ccu::Variable outputAddr;
        ccu::Variable inputToken;
        ccu::Variable outputToken;
        ccu::Variable scratchAddr;
        ccu::Variable scratchToken;
        std::vector<ccu::Variable> peerScratchAddr;
        std::vector<ccu::Variable> peerScratchToken;
        ccu::Variable transferBytes;
        ccu::Variable addressOffset;
        ccu::LocalAddr localSource;
        ccu::LocalAddr localDestination;
        ccu::RemoteAddr remoteScratch;
        ccu::Event dataEvent;
    };

    CcuResult InitButterfly(ButterflyContext &ctx)
    {
        ctx.peerScratchAddr.resize(ctx.arg->channelCount);
        ctx.peerScratchToken.resize(ctx.arg->channelCount);
        for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
            ctx.peerScratchAddr[i] = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[i], SCRATCH_ADDR_XN_ID);
            ctx.peerScratchToken[i] = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[i], SCRATCH_TOKEN_XN_ID);
        }
        uint32_t argId = 0;
        CCU_CHK_RET(ccu::LoadArg(ctx.inputAddr, argId++));
        CCU_CHK_RET(ccu::LoadArg(ctx.outputAddr, argId++));
        CCU_CHK_RET(ccu::LoadArg(ctx.inputToken, argId++));
        CCU_CHK_RET(ccu::LoadArg(ctx.outputToken, argId++));
        ctx.scratchAddr = ctx.arg->scratchAddr;
        ctx.scratchToken = ctx.arg->scratchToken;
        ctx.transferBytes = ctx.arg->totalBytes;
        return CCU_SUCCESS;
    }

    CcuResult ButterflyPreSync(ButterflyContext &ctx)
    {
        for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
            const ChannelHandle channel = ctx.arg->channels[channelIndex];
            CCU_CHK_RET(ccu::WriteVariableWithNotify(
                channel, ctx.scratchAddr, SCRATCH_ADDR_XN_ID, PARAM_SYNC_NOTIFY_INDEX, 1U << SCRATCH_ADDR_XN_ID));
            CCU_CHK_RET(ccu::WriteVariableWithNotify(
                channel, ctx.scratchToken, SCRATCH_TOKEN_XN_ID, PARAM_SYNC_NOTIFY_INDEX, 1U << SCRATCH_TOKEN_XN_ID));
        }
        constexpr uint16_t allBits = (1U << SCRATCH_ADDR_XN_ID) | (1U << SCRATCH_TOKEN_XN_ID);
        for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
            CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[channelIndex], PARAM_SYNC_NOTIFY_INDEX, allBits));
        }
        return CCU_SUCCESS;
    }

    CcuResult ButterflyDataHandshake(ButterflyContext &ctx, uint32_t channelIndex)
    {
        const ChannelHandle channel = ctx.arg->channels[channelIndex];
        constexpr uint16_t dataBit = 1U << PHASE_DONE_BIT_INDEX;
        CCU_CHK_RET(ccu::NotifyRecord(channel, BFLY_DATA_NOTIFY_INDEX, dataBit));
        CCU_CHK_RET(ccu::NotifyWait(channel, BFLY_DATA_NOTIFY_INDEX, dataBit));
        return CCU_SUCCESS;
    }

    CcuResult DoButterflyPhase(ButterflyContext &ctx, uint32_t channelIndex, uint32_t phase)
    {
        const ChannelHandle channel = ctx.arg->channels[channelIndex];
        if (phase < 3) {
            ctx.localSource.addr = phase == 0 ? ctx.inputAddr : ctx.scratchAddr;
            if (phase != 0) {
                ctx.addressOffset = static_cast<uint64_t>(phase - 1U) * ctx.arg->scratchStride;
                ctx.localSource.addr += ctx.addressOffset;
            }
            ctx.localSource.token = phase == 0 ? ctx.inputToken : ctx.scratchToken;
            ctx.remoteScratch.addr = ctx.peerScratchAddr[channelIndex];
            ctx.addressOffset = static_cast<uint64_t>(phase) * ctx.arg->scratchStride;
            ctx.remoteScratch.addr += ctx.addressOffset;
            ctx.remoteScratch.token = ctx.peerScratchToken[channelIndex];
            CCU_CHK_RET(ccu::Write(channel, ctx.remoteScratch, ctx.localSource, ctx.transferBytes, ctx.dataEvent, 1));
            CCU_CHK_RET(ccu::EventWait(ctx.dataEvent, 1));
            CCU_CHK_RET(ButterflyDataHandshake(ctx, channelIndex));

            ctx.localDestination.addr = ctx.scratchAddr;
            ctx.addressOffset = static_cast<uint64_t>(phase) * ctx.arg->scratchStride;
            ctx.localDestination.addr += ctx.addressOffset;
            ctx.localDestination.token = ctx.scratchToken;
            CCU_CHK_RET(ccu::LocalReduce(ctx.localDestination, ctx.localSource, ctx.transferBytes, ctx.arg->dataType,
                ctx.arg->reduceType, ctx.dataEvent, 1));
            CCU_CHK_RET(ccu::EventWait(ctx.dataEvent, 1));
            return CCU_SUCCESS;
        }

        CCU_CHK_RET(ButterflyDataHandshake(ctx, channelIndex));
        ctx.remoteScratch.addr = ctx.peerScratchAddr[channelIndex];
        ctx.addressOffset = 2ULL * ctx.arg->scratchStride;
        ctx.remoteScratch.addr += ctx.addressOffset;
        ctx.remoteScratch.token = ctx.peerScratchToken[channelIndex];
        ctx.localDestination.addr = ctx.outputAddr;
        ctx.localDestination.token = ctx.outputToken;
        CCU_CHK_RET(ccu::Read(channel, ctx.localDestination, ctx.remoteScratch, ctx.transferBytes, ctx.dataEvent, 1));
        CCU_CHK_RET(ccu::EventWait(ctx.dataEvent, 1));
        ctx.localSource.addr = ctx.scratchAddr;
        ctx.addressOffset = 2ULL * ctx.arg->scratchStride;
        ctx.localSource.addr += ctx.addressOffset;
        ctx.localSource.token = ctx.scratchToken;
        CCU_CHK_RET(ccu::LocalReduce(ctx.localDestination, ctx.localSource, ctx.transferBytes, ctx.arg->dataType,
            ctx.arg->reduceType, ctx.dataEvent, 1));
        CCU_CHK_RET(ccu::EventWait(ctx.dataEvent, 1));
        return CCU_SUCCESS;
    }

    const char *ButterflyKernelEventTag(uint32_t kernelIndex)
    {
        return kernelIndex == 0 ? "bfly_kernel0_sync" : "bfly_kernel1_sync";
    }

    CcuResult ButterflyWaitPreviousKernel(const CcuKernelArgButterfly2x8 &arg, uint32_t phase)
    {
        if (phase == 0 || arg.phaseOwnerKernel[phase - 1] == arg.kernelIndex) {
            return CCU_SUCCESS;
        }
        return ccu::EventWait(ButterflyKernelEventTag(arg.kernelIndex), static_cast<uint16_t>(1U << (phase - 1U)));
    }

    CcuResult ButterflyRecordNextKernel(const CcuKernelArgButterfly2x8 &arg, uint32_t phase)
    {
        if (phase + 1U >= 4U || arg.phaseOwnerKernel[phase + 1U] == arg.kernelIndex) {
            return CCU_SUCCESS;
        }
        return ccu::EventRecord(
            ButterflyKernelEventTag(arg.phaseOwnerKernel[phase + 1U]), static_cast<uint16_t>(1U << phase));
    }

    struct CrossLaneContext {
        const CcuKernelArgCrossLane2x8 *arg{nullptr};
        ccu::Variable inputAddr;
        ccu::Variable outputAddr;
        ccu::Variable inputToken;
        ccu::Variable outputToken;
        ccu::Variable scratchAddr;
        ccu::Variable scratchToken;
        ccu::Variable phase;
        std::vector<ccu::Variable> peerScratchAddr;
        std::vector<ccu::Variable> peerScratchToken;
        std::vector<ccu::Variable> peerOutputAddr;
        std::vector<ccu::Variable> peerOutputToken;
        ccu::Variable transferBytes;
        ccu::Variable addressOffset;
        ccu::LocalAddr localSource;
        ccu::LocalAddr localDestination;
        ccu::RemoteAddr remoteDestination;
        ccu::Event dataEvent;
    };

    CcuResult InitCrossLane(CrossLaneContext &ctx)
    {
        const auto *arg = ctx.arg;
        ctx.peerScratchAddr.resize(arg->channelCount);
        ctx.peerScratchToken.resize(arg->channelCount);
        ctx.peerOutputAddr.resize(arg->channelCount);
        ctx.peerOutputToken.resize(arg->channelCount);
        for (uint32_t i = 0; i < arg->channelCount; ++i) {
            ctx.peerScratchAddr[i] = ccu::GetResByChannel<ccu::Variable>(arg->channels[i], SCRATCH_ADDR_XN_ID);
            ctx.peerScratchToken[i] = ccu::GetResByChannel<ccu::Variable>(arg->channels[i], SCRATCH_TOKEN_XN_ID);
            ctx.peerOutputAddr[i] = ccu::GetResByChannel<ccu::Variable>(arg->channels[i], OUTPUT_ADDR_XN_ID);
            ctx.peerOutputToken[i] = ccu::GetResByChannel<ccu::Variable>(arg->channels[i], OUTPUT_TOKEN_XN_ID);
        }
        uint32_t argId = 0;
        CCU_CHK_RET(ccu::LoadArg(ctx.inputAddr, argId++));
        CCU_CHK_RET(ccu::LoadArg(ctx.outputAddr, argId++));
        CCU_CHK_RET(ccu::LoadArg(ctx.inputToken, argId++));
        CCU_CHK_RET(ccu::LoadArg(ctx.outputToken, argId++));
        CCU_CHK_RET(ccu::LoadArg(ctx.scratchAddr, argId++));
        CCU_CHK_RET(ccu::LoadArg(ctx.scratchToken, argId++));
        CCU_CHK_RET(ccu::LoadArg(ctx.phase, argId++));
        return CCU_SUCCESS;
    }

    CcuResult CrossLanePreSync(CrossLaneContext &ctx)
    {
        const auto *arg = ctx.arg;
        for (uint32_t i = 0; i < arg->channelCount; ++i) {
            CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[i], ctx.scratchAddr, SCRATCH_ADDR_XN_ID,
                PARAM_SYNC_NOTIFY_INDEX, 1U << SCRATCH_ADDR_XN_ID));
            CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[i], ctx.scratchToken, SCRATCH_TOKEN_XN_ID,
                PARAM_SYNC_NOTIFY_INDEX, 1U << SCRATCH_TOKEN_XN_ID));
            CCU_CHK_RET(ccu::WriteVariableWithNotify(
                arg->channels[i], ctx.outputAddr, OUTPUT_ADDR_XN_ID, PARAM_SYNC_NOTIFY_INDEX, 1U << OUTPUT_ADDR_XN_ID));
            CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[i], ctx.outputToken, OUTPUT_TOKEN_XN_ID,
                PARAM_SYNC_NOTIFY_INDEX, 1U << OUTPUT_TOKEN_XN_ID));
        }
        constexpr uint16_t allBits = (1U << SCRATCH_ADDR_XN_ID) | (1U << SCRATCH_TOKEN_XN_ID)
                                     | (1U << OUTPUT_ADDR_XN_ID) | (1U << OUTPUT_TOKEN_XN_ID);
        for (uint32_t i = 0; i < arg->channelCount; ++i) {
            CCU_CHK_RET(ccu::NotifyWait(arg->channels[i], PARAM_SYNC_NOTIFY_INDEX, allBits));
        }
        return CCU_SUCCESS;
    }

    CcuResult DoSameServerPush(CrossLaneContext &ctx)
    {
        const auto *arg = ctx.arg;
        for (uint32_t shard = 0; shard < CROSS_LANE_SHARD_COUNT; ++shard) {
            uint16_t eventMask = 0;
            for (uint32_t i = 0; i < arg->channelCount; ++i) {
                if (arg->peerIsLocal[i] == 0) {
                    continue;
                }
                const uint16_t bit = static_cast<uint16_t>(1U << i);
                ctx.localSource.addr = ctx.inputAddr;
                ctx.addressOffset = arg->targetOwnerOffset[i] + arg->targetShardOffset[i][shard];
                ctx.localSource.addr += ctx.addressOffset;
                ctx.localSource.token = ctx.inputToken;
                if (arg->directOutputAccumulator != 0 && arg->targetSlot[i] == 0) {
                    ctx.remoteDestination.addr = ctx.peerOutputAddr[i];
                    ctx.addressOffset = arg->targetOwnerOffset[i] + arg->targetShardOffset[i][shard];
                    ctx.remoteDestination.addr += ctx.addressOffset;
                    ctx.remoteDestination.token = ctx.peerOutputToken[i];
                } else {
                    ctx.remoteDestination.addr = ctx.peerScratchAddr[i];
                    ctx.addressOffset = static_cast<uint64_t>(arg->targetSlot[i]) * arg->scratchStride
                                        + arg->targetShardOffset[i][shard];
                    ctx.remoteDestination.addr += ctx.addressOffset;
                    ctx.remoteDestination.token = ctx.peerScratchToken[i];
                }
                ctx.transferBytes = arg->targetShardBytes[i][shard];
                CCU_CHK_RET(ccu::Write(
                    arg->channels[i], ctx.remoteDestination, ctx.localSource, ctx.transferBytes, ctx.dataEvent, bit));
                eventMask = static_cast<uint16_t>(eventMask | bit);
            }
            if (eventMask != 0) {
                CCU_CHK_RET(ccu::EventWait(ctx.dataEvent, eventMask));
            }
        }

        constexpr uint16_t doneBit = 1U << CROSS_PHASE_DONE_BIT_INDEX;
        for (uint32_t i = 0; i < arg->channelCount; ++i) {
            if (arg->peerIsLocal[i] != 0) {
                CCU_CHK_RET(ccu::NotifyRecord(arg->channels[i], SAME_DONE_NOTIFY_INDEX, doneBit));
            }
        }
        for (uint32_t i = 0; i < arg->channelCount; ++i) {
            if (arg->peerIsLocal[i] != 0) {
                CCU_CHK_RET(ccu::NotifyWait(arg->channels[i], SAME_DONE_NOTIFY_INDEX, doneBit));
            }
        }
        return CCU_SUCCESS;
    }

    void GetEqualLaneShard(
        uint64_t ownerBytes, uint32_t laneSize, uint32_t shard, uint64_t &shardOffset, uint64_t &shardBytes)
    {
        const uint64_t elements = ownerBytes / sizeof(float);
        const uint64_t base = elements / laneSize;
        const uint64_t remainder = elements % laneSize;
        const uint64_t shardElements = base + (shard < remainder ? 1U : 0U);
        const uint64_t offsetElements = static_cast<uint64_t>(shard) * base + std::min<uint64_t>(shard, remainder);
        shardOffset = offsetElements * sizeof(float);
        shardBytes = shardElements * sizeof(float);
    }

    CcuResult DoCrossLanePush(CrossLaneContext &ctx)
    {
        const auto *arg = ctx.arg;
        constexpr uint16_t doneBit = 1U << CROSS_PHASE_DONE_BIT_INDEX;
        for (uint32_t lanePhase = 0; lanePhase < CROSS_MAX_LANE_SIZE; ++lanePhase) {
            uint16_t eventMask = 0;
            for (uint32_t i = 0; i < arg->channelCount; ++i) {
                if (arg->peerIsLocal[i] != 0 || lanePhase >= arg->targetLaneSize[i]) {
                    continue;
                }
                const uint32_t shard = (arg->targetLanePosition[i] + lanePhase) % arg->targetLaneSize[i];
                uint64_t shardOffset = 0;
                uint64_t shardBytes = 0;
                GetEqualLaneShard(arg->targetOwnerBytes[i], arg->targetLaneSize[i], shard, shardOffset, shardBytes);
                const uint16_t bit = static_cast<uint16_t>(1U << i);
                ctx.localSource.addr = ctx.inputAddr;
                ctx.addressOffset = arg->targetOwnerOffset[i] + shardOffset;
                ctx.localSource.addr += ctx.addressOffset;
                ctx.localSource.token = ctx.inputToken;
                ctx.remoteDestination.addr = ctx.peerScratchAddr[i];
                ctx.addressOffset
                    = static_cast<uint64_t>(CROSS_LANE_SAME_SLOT_COUNT + arg->targetLane[i]) * arg->scratchStride
                      + shardOffset;
                ctx.remoteDestination.addr += ctx.addressOffset;
                ctx.remoteDestination.token = ctx.peerScratchToken[i];
                ctx.transferBytes = shardBytes;
                if (lanePhase == 0) {
                    CCU_CHK_RET(ccu::Write(arg->channels[i], ctx.remoteDestination, ctx.localSource, ctx.transferBytes,
                        ctx.dataEvent, bit));
                } else {
                    CCU_CHK_RET(ccu::WriteReduce(arg->channels[i], ctx.remoteDestination, ctx.localSource,
                        ctx.transferBytes, arg->dataType, arg->reduceType, ctx.dataEvent, bit));
                }
                eventMask = static_cast<uint16_t>(eventMask | bit);
            }
            if (eventMask != 0) {
                CCU_CHK_RET(ccu::EventWait(ctx.dataEvent, eventMask));
            }

            for (uint32_t i = 0; i < arg->channelCount; ++i) {
                if (arg->peerIsLocal[i] == 0 && lanePhase < arg->targetLaneSize[i]) {
                    CCU_CHK_RET(ccu::NotifyRecord(arg->channels[i], LANE_DATA_NOTIFY_INDEX, doneBit));
                }
            }
            for (uint32_t i = 0; i < arg->channelCount; ++i) {
                if (arg->peerIsLocal[i] == 0 && lanePhase < arg->sourceLaneSize[i]) {
                    CCU_CHK_RET(ccu::NotifyWait(arg->channels[i], LANE_DATA_NOTIFY_INDEX, doneBit));
                }
            }
            for (uint32_t i = 0; i < arg->channelCount; ++i) {
                if (arg->peerIsLocal[i] == 0 && lanePhase < arg->sourceLaneSize[i]) {
                    CCU_CHK_RET(ccu::NotifyRecord(arg->channels[i], LANE_NEXT_NOTIFY_INDEX, doneBit));
                }
            }
            for (uint32_t i = 0; i < arg->channelCount; ++i) {
                if (arg->peerIsLocal[i] == 0 && lanePhase < arg->targetLaneSize[i]) {
                    CCU_CHK_RET(ccu::NotifyWait(arg->channels[i], LANE_NEXT_NOTIFY_INDEX, doneBit));
                }
            }
        }
        return CCU_SUCCESS;
    }

    CcuResult DoMeshLanePhase(CrossLaneContext &ctx, uint32_t lanePhase)
    {
        const auto *arg = ctx.arg;
        if (lanePhase == 0) {
            CCU_CHK_RET(CrossLanePreSync(ctx));
        }

        uint16_t eventMask = 0;
        for (uint32_t i = 0; i < arg->channelCount; ++i) {
            if (lanePhase >= arg->targetLaneSize[i]) {
                continue;
            }
            const uint32_t shard = (arg->targetLanePosition[i] + lanePhase) % arg->targetLaneSize[i];
            uint64_t shardOffset = 0;
            uint64_t shardBytes = 0;
            GetEqualLaneShard(arg->targetOwnerBytes[i], arg->targetLaneSize[i], shard, shardOffset, shardBytes);
            const uint16_t bit = static_cast<uint16_t>(1U << i);
            ctx.localSource.addr = ctx.inputAddr;
            ctx.addressOffset = arg->targetOwnerOffset[i] + shardOffset;
            ctx.localSource.addr += ctx.addressOffset;
            ctx.localSource.token = ctx.inputToken;
            ctx.remoteDestination.addr = ctx.peerScratchAddr[i];
            ctx.addressOffset = static_cast<uint64_t>(arg->targetLane[i]) * arg->scratchStride + shardOffset;
            ctx.remoteDestination.addr += ctx.addressOffset;
            ctx.remoteDestination.token = ctx.peerScratchToken[i];
            ctx.transferBytes = shardBytes;
            if (lanePhase == 0) {
                CCU_CHK_RET(ccu::Write(
                    arg->channels[i], ctx.remoteDestination, ctx.localSource, ctx.transferBytes, ctx.dataEvent, bit));
            } else {
                CCU_CHK_RET(ccu::WriteReduce(arg->channels[i], ctx.remoteDestination, ctx.localSource,
                    ctx.transferBytes, arg->dataType, arg->reduceType, ctx.dataEvent, bit));
            }
            eventMask = static_cast<uint16_t>(eventMask | bit);
        }
        if (eventMask != 0) {
            CCU_CHK_RET(ccu::EventWait(ctx.dataEvent, eventMask));
        }

        constexpr uint16_t doneBit = 1U << CROSS_PHASE_DONE_BIT_INDEX;
        for (uint32_t i = 0; i < arg->channelCount; ++i) {
            if (lanePhase < arg->targetLaneSize[i]) {
                CCU_CHK_RET(ccu::NotifyRecord(arg->channels[i], LANE_DATA_NOTIFY_INDEX, doneBit));
            }
        }
        for (uint32_t i = 0; i < arg->channelCount; ++i) {
            if (lanePhase < arg->sourceLaneSize[i]) {
                CCU_CHK_RET(ccu::NotifyWait(arg->channels[i], LANE_DATA_NOTIFY_INDEX, doneBit));
            }
        }
        if (arg->kernelCount > 1) {
            const uint16_t phaseBit = static_cast<uint16_t>(1U << lanePhase);
            const char *myReady = arg->kernelIndex == 0 ? "mesh_lane_phase_ready0" : "mesh_lane_phase_ready1";
            const char *peerReady = arg->kernelIndex == 0 ? "mesh_lane_phase_ready1" : "mesh_lane_phase_ready0";
            CCU_CHK_RET(ccu::EventRecord(peerReady, phaseBit));
            CCU_CHK_RET(ccu::EventWait(myReady, phaseBit));
        }
        for (uint32_t i = 0; i < arg->channelCount; ++i) {
            if (lanePhase < arg->sourceLaneSize[i]) {
                CCU_CHK_RET(ccu::NotifyRecord(arg->channels[i], LANE_NEXT_NOTIFY_INDEX, doneBit));
            }
        }
        for (uint32_t i = 0; i < arg->channelCount; ++i) {
            if (lanePhase < arg->targetLaneSize[i]) {
                CCU_CHK_RET(ccu::NotifyWait(arg->channels[i], LANE_NEXT_NOTIFY_INDEX, doneBit));
            }
        }
        return CCU_SUCCESS;
    }

    CcuResult DoCrossLaneReduceScatter(CrossLaneContext &ctx)
    {
        CCU_CHK_RET(CrossLanePreSync(ctx));
        CCU_CHK_RET(DoSameServerPush(ctx));
        CCU_CHK_RET(DoCrossLanePush(ctx));
        return CCU_SUCCESS;
    }

    CcuResult DoCrossLaneLocalReduceSerial(CrossLaneContext &ctx)
    {
        const auto *arg = ctx.arg;
        for (uint32_t shard = 0; shard < CROSS_LANE_SHARD_COUNT; ++shard) {
            if ((arg->localShardMask & (1U << shard)) == 0) {
                continue;
            }
            const uint64_t shardOffset = arg->ownerShardOffset[shard];
            ctx.transferBytes = arg->ownerShardBytes[shard];
            if (arg->directOutputAccumulator != 0) {
                ctx.localDestination.addr = ctx.outputAddr;
                ctx.addressOffset = arg->ownerOffset + shardOffset;
                ctx.localDestination.addr += ctx.addressOffset;
                ctx.localDestination.token = ctx.outputToken;
            } else {
                ctx.localDestination.addr = ctx.scratchAddr;
                ctx.addressOffset = shardOffset;
                ctx.localDestination.addr += ctx.addressOffset;
                ctx.localDestination.token = ctx.scratchToken;
            }
            const uint32_t firstExtraSlot = 1;
            const uint32_t extraSlotCount
                = arg->meshLaneMode != 0 ? arg->localLaneCount - 1U : CROSS_LANE_SAME_SLOT_COUNT - 1U;
            for (uint32_t slot = firstExtraSlot; slot < firstExtraSlot + extraSlotCount; ++slot) {
                ctx.localSource.addr = ctx.scratchAddr;
                ctx.addressOffset = static_cast<uint64_t>(slot) * arg->scratchStride + shardOffset;
                ctx.localSource.addr += ctx.addressOffset;
                ctx.localSource.token = ctx.scratchToken;
                CCU_CHK_RET(ccu::LocalReduce(ctx.localDestination, ctx.localSource, ctx.transferBytes, arg->dataType,
                    arg->reduceType, ctx.dataEvent, 1));
                CCU_CHK_RET(ccu::EventWait(ctx.dataEvent, 1));
            }
            ctx.localSource.addr = ctx.inputAddr;
            ctx.addressOffset = arg->ownerOffset + shardOffset;
            ctx.localSource.addr += ctx.addressOffset;
            ctx.localSource.token = ctx.inputToken;
            CCU_CHK_RET(ccu::LocalReduce(ctx.localDestination, ctx.localSource, ctx.transferBytes, arg->dataType,
                arg->reduceType, ctx.dataEvent, 1));
            CCU_CHK_RET(ccu::EventWait(ctx.dataEvent, 1));
            if (arg->meshLaneMode == 0) {
                for (uint32_t lane = 0; lane < arg->localLaneCount; ++lane) {
                    ctx.localSource.addr = ctx.scratchAddr;
                    ctx.addressOffset
                        = static_cast<uint64_t>(CROSS_LANE_SAME_SLOT_COUNT + lane) * arg->scratchStride + shardOffset;
                    ctx.localSource.addr += ctx.addressOffset;
                    ctx.localSource.token = ctx.scratchToken;
                    CCU_CHK_RET(ccu::LocalReduce(ctx.localDestination, ctx.localSource, ctx.transferBytes,
                        arg->dataType, arg->reduceType, ctx.dataEvent, 1));
                    CCU_CHK_RET(ccu::EventWait(ctx.dataEvent, 1));
                }
            }
            if (arg->directOutputAccumulator == 0) {
                ctx.localSource.addr = ctx.scratchAddr;
                ctx.addressOffset = shardOffset;
                ctx.localSource.addr += ctx.addressOffset;
                ctx.localSource.token = ctx.scratchToken;
                ctx.localDestination.addr = ctx.outputAddr;
                ctx.addressOffset = arg->ownerOffset + shardOffset;
                ctx.localDestination.addr += ctx.addressOffset;
                ctx.localDestination.token = ctx.outputToken;
                CCU_CHK_RET(
                    ccu::LocalCopy(ctx.localDestination, ctx.localSource, ctx.transferBytes, ctx.dataEvent, 1));
                CCU_CHK_RET(ccu::EventWait(ctx.dataEvent, 1));
            }
        }
        return CCU_SUCCESS;
    }

    CcuResult DoCrossLaneLocalReduceParallel(CrossLaneContext &ctx)
    {
        const auto *arg = ctx.arg;
        for (uint32_t slot = 1; slot < CROSS_LANE_SAME_SLOT_COUNT; ++slot) {
            uint16_t eventMask = 0;
            for (uint32_t shard = 0; shard < CROSS_LANE_SHARD_COUNT; ++shard) {
                if ((arg->localShardMask & (1U << shard)) == 0) {
                    continue;
                }
                const uint16_t bit = static_cast<uint16_t>(1U << shard);
                const uint64_t shardOffset = arg->ownerShardOffset[shard];
                ctx.localDestination.addr = ctx.scratchAddr;
                ctx.addressOffset = shardOffset;
                ctx.localDestination.addr += ctx.addressOffset;
                ctx.localDestination.token = ctx.scratchToken;
                ctx.localSource.addr = ctx.scratchAddr;
                ctx.addressOffset = static_cast<uint64_t>(slot) * arg->scratchStride + shardOffset;
                ctx.localSource.addr += ctx.addressOffset;
                ctx.localSource.token = ctx.scratchToken;
                ctx.transferBytes = arg->ownerShardBytes[shard];
                CCU_CHK_RET(ccu::LocalReduce(ctx.localDestination, ctx.localSource, ctx.transferBytes, arg->dataType,
                    arg->reduceType, ctx.dataEvent, bit));
                eventMask = static_cast<uint16_t>(eventMask | bit);
            }
            CCU_CHK_RET(ccu::EventWait(ctx.dataEvent, eventMask));
        }

        uint16_t eventMask = 0;
        for (uint32_t shard = 0; shard < CROSS_LANE_SHARD_COUNT; ++shard) {
            if ((arg->localShardMask & (1U << shard)) == 0) {
                continue;
            }
            const uint16_t bit = static_cast<uint16_t>(1U << shard);
            const uint64_t shardOffset = arg->ownerShardOffset[shard];
            ctx.localDestination.addr = ctx.scratchAddr;
            ctx.addressOffset = shardOffset;
            ctx.localDestination.addr += ctx.addressOffset;
            ctx.localDestination.token = ctx.scratchToken;
            ctx.localSource.addr = ctx.inputAddr;
            ctx.addressOffset = arg->ownerOffset + shardOffset;
            ctx.localSource.addr += ctx.addressOffset;
            ctx.localSource.token = ctx.inputToken;
            ctx.transferBytes = arg->ownerShardBytes[shard];
            CCU_CHK_RET(ccu::LocalReduce(ctx.localDestination, ctx.localSource, ctx.transferBytes, arg->dataType,
                arg->reduceType, ctx.dataEvent, bit));
            eventMask = static_cast<uint16_t>(eventMask | bit);
        }
        CCU_CHK_RET(ccu::EventWait(ctx.dataEvent, eventMask));

        for (uint32_t lane = 0; lane < arg->localLaneCount; ++lane) {
            eventMask = 0;
            for (uint32_t shard = 0; shard < CROSS_LANE_SHARD_COUNT; ++shard) {
                if ((arg->localShardMask & (1U << shard)) == 0) {
                    continue;
                }
                const uint16_t bit = static_cast<uint16_t>(1U << shard);
                const uint64_t shardOffset = arg->ownerShardOffset[shard];
                ctx.localDestination.addr = ctx.scratchAddr;
                ctx.addressOffset = shardOffset;
                ctx.localDestination.addr += ctx.addressOffset;
                ctx.localDestination.token = ctx.scratchToken;
                ctx.localSource.addr = ctx.scratchAddr;
                ctx.addressOffset
                    = static_cast<uint64_t>(CROSS_LANE_SAME_SLOT_COUNT + lane) * arg->scratchStride + shardOffset;
                ctx.localSource.addr += ctx.addressOffset;
                ctx.localSource.token = ctx.scratchToken;
                ctx.transferBytes = arg->ownerShardBytes[shard];
                CCU_CHK_RET(ccu::LocalReduce(ctx.localDestination, ctx.localSource, ctx.transferBytes, arg->dataType,
                    arg->reduceType, ctx.dataEvent, bit));
                eventMask = static_cast<uint16_t>(eventMask | bit);
            }
            CCU_CHK_RET(ccu::EventWait(ctx.dataEvent, eventMask));
        }

        eventMask = 0;
        for (uint32_t shard = 0; shard < CROSS_LANE_SHARD_COUNT; ++shard) {
            if ((arg->localShardMask & (1U << shard)) == 0) {
                continue;
            }
            const uint16_t bit = static_cast<uint16_t>(1U << shard);
            const uint64_t shardOffset = arg->ownerShardOffset[shard];
            ctx.localSource.addr = ctx.scratchAddr;
            ctx.addressOffset = shardOffset;
            ctx.localSource.addr += ctx.addressOffset;
            ctx.localSource.token = ctx.scratchToken;
            ctx.localDestination.addr = ctx.outputAddr;
            ctx.addressOffset = arg->ownerOffset + shardOffset;
            ctx.localDestination.addr += ctx.addressOffset;
            ctx.localDestination.token = ctx.outputToken;
            ctx.transferBytes = arg->ownerShardBytes[shard];
            CCU_CHK_RET(ccu::LocalCopy(ctx.localDestination, ctx.localSource, ctx.transferBytes, ctx.dataEvent, bit));
            eventMask = static_cast<uint16_t>(eventMask | bit);
        }
        return ccu::EventWait(ctx.dataEvent, eventMask);
    }

    CcuResult DoCrossLaneBroadcast(CrossLaneContext &ctx)
    {
        const auto *arg = ctx.arg;
        CCU_CHK_RET(CrossLanePreSync(ctx));
        ctx.localSource.addr = ctx.outputAddr;
        ctx.addressOffset = arg->ownerOffset;
        ctx.localSource.addr += ctx.addressOffset;
        ctx.localSource.token = ctx.outputToken;
        ctx.transferBytes = arg->ownerBytes;
        uint16_t eventMask = 0;
        for (uint32_t i = 0; i < arg->channelCount; ++i) {
            const uint16_t bit = static_cast<uint16_t>(1U << i);
            ctx.remoteDestination.addr = ctx.peerOutputAddr[i];
            ctx.addressOffset = arg->ownerOffset;
            ctx.remoteDestination.addr += ctx.addressOffset;
            ctx.remoteDestination.token = ctx.peerOutputToken[i];
            CCU_CHK_RET(ccu::Write(
                arg->channels[i], ctx.remoteDestination, ctx.localSource, ctx.transferBytes, ctx.dataEvent, bit));
            eventMask = static_cast<uint16_t>(eventMask | bit);
        }
        CCU_CHK_RET(ccu::EventWait(ctx.dataEvent, eventMask));
        constexpr uint16_t doneBit = 1U << CROSS_PHASE_DONE_BIT_INDEX;
        for (uint32_t i = 0; i < arg->channelCount; ++i) {
            CCU_CHK_RET(ccu::NotifyRecord(arg->channels[i], CROSS_BROADCAST_DONE_NOTIFY_INDEX, doneBit));
        }
        for (uint32_t i = 0; i < arg->channelCount; ++i) {
            CCU_CHK_RET(ccu::NotifyWait(arg->channels[i], CROSS_BROADCAST_DONE_NOTIFY_INDEX, doneBit));
        }
        return CCU_SUCCESS;
    }

    constexpr uint32_t FUSED_INPLACE_ARG_ID = 4;
    constexpr uint32_t FUSED_PHASE_ARG_ID = 5;
    constexpr uint32_t P12_ALLPAIRS_SEGMENT_ARG_ID = 6;
    constexpr uint32_t ROTATE_DATA_NOTIFY_INDEX = 1;
    constexpr uint32_t ROTATE_NEXT_NOTIFY_INDEX = 2;
    constexpr uint32_t ROTATE_BROADCAST_NOTIFY_INDEX = 3;
    constexpr uint32_t ROTATE_UNIQUE_PHASE_NOTIFY_BASE_INDEX = 1;
    constexpr uint32_t ROTATE_UNIQUE_BROADCAST_NOTIFY_INDEX = 4;
    constexpr uint32_t ROTATE_WAVE2_CHUNK_COUNT = 2;
    constexpr uint32_t HIER_LOCAL_DATA_NOTIFY_INDEX = 1;
    constexpr uint32_t HIER_LOCAL_NEXT_NOTIFY_INDEX = 2;
    constexpr uint32_t HIER_CROSS_READY_NOTIFY_INDEX = 3;
    constexpr uint32_t HIER_CROSS_DATA_NOTIFY_INDEX = 4;
    constexpr uint32_t HIER_CROSS_RESULT_NOTIFY_INDEX = 5;
    constexpr uint32_t P12_CLOS_PHASE_NOTIFY_INDEX = 1;
    constexpr uint32_t P12_MESH_DONE_NOTIFY_INDEX = 2;
    constexpr uint32_t P12_CLOS_BROADCAST_NOTIFY_INDEX = 3;

    struct FusedContext {
        ccu::Variable inputAddr;
        ccu::Variable outputAddr;
        ccu::Variable inputToken;
        ccu::Variable outputToken;
        ccu::Variable isInplace;
        ccu::Variable phase;
        ccu::Variable segment;
        ccu::Variable scratchAddr;
        ccu::Variable scratchToken;
        ccu::Variable transferBytes;
        ccu::Variable addressOffset;
        std::vector<ccu::Variable> peerInputAddr;
        std::vector<ccu::Variable> peerInputToken;
        std::vector<ccu::Variable> peerScratchAddr;
        std::vector<ccu::Variable> peerScratchToken;
        std::vector<ccu::Variable> peerOutputAddr;
        std::vector<ccu::Variable> peerOutputToken;
        ccu::LocalAddr localSource;
        ccu::LocalAddr localDestination;
        ccu::RemoteAddr remoteDestination;
        ccu::Event dataEvent;
    };

    CcuResult InitFusedContext(
        const CcuKernelArgBase &arg, uint64_t scratchAddr, uint64_t scratchToken, FusedContext &ctx)
    {
        ctx.peerScratchAddr.resize(arg.channelCount);
        ctx.peerScratchToken.resize(arg.channelCount);
        ctx.peerOutputAddr.resize(arg.channelCount);
        ctx.peerOutputToken.resize(arg.channelCount);
        for (uint32_t i = 0; i < arg.channelCount; ++i) {
            ctx.peerScratchAddr[i] = ccu::GetResByChannel<ccu::Variable>(arg.channels[i], SCRATCH_ADDR_XN_ID);
            ctx.peerScratchToken[i] = ccu::GetResByChannel<ccu::Variable>(arg.channels[i], SCRATCH_TOKEN_XN_ID);
            ctx.peerOutputAddr[i] = ccu::GetResByChannel<ccu::Variable>(arg.channels[i], OUTPUT_ADDR_XN_ID);
            ctx.peerOutputToken[i] = ccu::GetResByChannel<ccu::Variable>(arg.channels[i], OUTPUT_TOKEN_XN_ID);
        }
        uint32_t argId = 0;
        CCU_CHK_RET(ccu::LoadArg(ctx.inputAddr, argId++));
        CCU_CHK_RET(ccu::LoadArg(ctx.outputAddr, argId++));
        CCU_CHK_RET(ccu::LoadArg(ctx.inputToken, argId++));
        CCU_CHK_RET(ccu::LoadArg(ctx.outputToken, argId++));
        CCU_CHK_RET(ccu::LoadArg(ctx.isInplace, FUSED_INPLACE_ARG_ID));
        CCU_CHK_RET(ccu::LoadArg(ctx.phase, FUSED_PHASE_ARG_ID));
        ctx.scratchAddr = scratchAddr;
        ctx.scratchToken = scratchToken;
        return CCU_SUCCESS;
    }

    CcuResult FusedPreSync(const CcuKernelArgBase &arg, FusedContext &ctx)
    {
        for (uint32_t i = 0; i < arg.channelCount; ++i) {
            CCU_CHK_RET(ccu::WriteVariableWithNotify(arg.channels[i], ctx.scratchAddr, SCRATCH_ADDR_XN_ID,
                PARAM_SYNC_NOTIFY_INDEX, 1U << SCRATCH_ADDR_XN_ID));
            CCU_CHK_RET(ccu::WriteVariableWithNotify(arg.channels[i], ctx.scratchToken, SCRATCH_TOKEN_XN_ID,
                PARAM_SYNC_NOTIFY_INDEX, 1U << SCRATCH_TOKEN_XN_ID));
            CCU_CHK_RET(ccu::WriteVariableWithNotify(
                arg.channels[i], ctx.outputAddr, OUTPUT_ADDR_XN_ID, PARAM_SYNC_NOTIFY_INDEX, 1U << OUTPUT_ADDR_XN_ID));
            CCU_CHK_RET(ccu::WriteVariableWithNotify(arg.channels[i], ctx.outputToken, OUTPUT_TOKEN_XN_ID,
                PARAM_SYNC_NOTIFY_INDEX, 1U << OUTPUT_TOKEN_XN_ID));
        }
        constexpr uint16_t allBits = (1U << SCRATCH_ADDR_XN_ID) | (1U << SCRATCH_TOKEN_XN_ID)
                                     | (1U << OUTPUT_ADDR_XN_ID) | (1U << OUTPUT_TOKEN_XN_ID);
        for (uint32_t i = 0; i < arg.channelCount; ++i) {
            CCU_CHK_RET(ccu::NotifyWait(arg.channels[i], PARAM_SYNC_NOTIFY_INDEX, allBits));
        }
        return CCU_SUCCESS;
    }

    CcuResult FusedPullPreSync(const CcuKernelArgBase &arg, FusedContext &ctx)
    {
        for (uint32_t i = 0; i < arg.channelCount; ++i) {
            CCU_CHK_RET(ccu::WriteVariableWithNotify(arg.channels[i], ctx.inputAddr, INPUT_ADDR_XN_ID,
                PARAM_SYNC_NOTIFY_INDEX, 1U << INPUT_ADDR_XN_ID));
            CCU_CHK_RET(ccu::WriteVariableWithNotify(arg.channels[i], ctx.inputToken, INPUT_TOKEN_XN_ID,
                PARAM_SYNC_NOTIFY_INDEX, 1U << INPUT_TOKEN_XN_ID));
            CCU_CHK_RET(ccu::WriteVariableWithNotify(
                arg.channels[i], ctx.outputAddr, OUTPUT_ADDR_XN_ID, PARAM_SYNC_NOTIFY_INDEX, 1U << OUTPUT_ADDR_XN_ID));
            CCU_CHK_RET(ccu::WriteVariableWithNotify(arg.channels[i], ctx.outputToken, OUTPUT_TOKEN_XN_ID,
                PARAM_SYNC_NOTIFY_INDEX, 1U << OUTPUT_TOKEN_XN_ID));
        }
        constexpr uint16_t allBits = (1U << INPUT_ADDR_XN_ID) | (1U << INPUT_TOKEN_XN_ID)
                                     | (1U << OUTPUT_ADDR_XN_ID) | (1U << OUTPUT_TOKEN_XN_ID);
        for (uint32_t i = 0; i < arg.channelCount; ++i) {
            CCU_CHK_RET(ccu::NotifyWait(arg.channels[i], PARAM_SYNC_NOTIFY_INDEX, allBits));
        }
        return CCU_SUCCESS;
    }

    CcuResult SendOutputAddress(const CcuKernelArgBase &arg, FusedContext &ctx)
    {
        for (uint32_t i = 0; i < arg.channelCount; ++i) {
            CCU_CHK_RET(ccu::WriteVariableWithNotify(
                arg.channels[i], ctx.outputAddr, OUTPUT_ADDR_XN_ID, PARAM_SYNC_NOTIFY_INDEX, 1U << OUTPUT_ADDR_XN_ID));
        }
        return CCU_SUCCESS;
    }

    CcuResult KernelBarrier(uint32_t kernelIndex, uint32_t kernelCount, uint32_t isPrimary, uint16_t mask,
        const char *ready0, const char *ready1, const char *release0, const char *release1)
    {
        if (kernelCount == 1) {
            return CCU_SUCCESS;
        }
        (void)isPrimary;
        (void)release0;
        (void)release1;
        const char *myReady = kernelIndex == 0 ? ready0 : ready1;
        const char *peerReady = kernelIndex == 0 ? ready1 : ready0;
        CCU_CHK_RET(ccu::EventRecord(peerReady, mask));
        CCU_CHK_RET(ccu::EventWait(myReady, mask));
        return CCU_SUCCESS;
    }

    CcuResult NotifyAll(const CcuKernelArgBase &arg, uint32_t notifyIndex)
    {
        constexpr uint16_t doneBit = 1;
        for (uint32_t i = 0; i < arg.channelCount; ++i) {
            CCU_CHK_RET(ccu::NotifyRecord(arg.channels[i], notifyIndex, doneBit));
        }
        for (uint32_t i = 0; i < arg.channelCount; ++i) {
            CCU_CHK_RET(ccu::NotifyWait(arg.channels[i], notifyIndex, doneBit));
        }
        return CCU_SUCCESS;
    }

    CcuResult BroadcastOwner(const CcuKernelArgBase &baseArg, uint64_t ownerOffset, uint64_t ownerBytes,
        uint32_t notifyIndex, FusedContext &ctx)
    {
        ctx.localSource.addr = ctx.outputAddr;
        ctx.addressOffset = ownerOffset;
        ctx.localSource.addr += ctx.addressOffset;
        ctx.localSource.token = ctx.outputToken;
        ctx.transferBytes = ownerBytes;
        uint16_t eventMask = 0;
        for (uint32_t i = 0; i < baseArg.channelCount; ++i) {
            const uint16_t bit = static_cast<uint16_t>(1U << i);
            ctx.remoteDestination.addr = ctx.peerOutputAddr[i];
            ctx.addressOffset = ownerOffset;
            ctx.remoteDestination.addr += ctx.addressOffset;
            ctx.remoteDestination.token = ctx.peerOutputToken[i];
            CCU_CHK_RET(ccu::Write(
                baseArg.channels[i], ctx.remoteDestination, ctx.localSource, ctx.transferBytes, ctx.dataEvent, bit));
            eventMask = static_cast<uint16_t>(eventMask | bit);
        }
        if (eventMask != 0) {
            CCU_CHK_RET(ccu::EventWait(ctx.dataEvent, eventMask));
        }
        return NotifyAll(baseArg, notifyIndex);
    }

    uint32_t DirectSourceSlot(uint32_t sourceRank, uint32_t ownerRank)
    {
        return sourceRank < ownerRank ? sourceRank : sourceRank - 1U;
    }

    CcuResult DoDirectPush(const CcuKernelArgDirectRsag &arg, FusedContext &ctx)
    {
        const uint16_t initBit = static_cast<uint16_t>(1U << arg.channelCount);
        if (arg.earlyInit != 0 && arg.isPrimary != 0) {
            CCU_IF(ctx.isInplace == 0)
            {
                ctx.localDestination.addr = ctx.outputAddr;
                ctx.addressOffset = arg.ownerOffset;
                ctx.localDestination.addr += ctx.addressOffset;
                ctx.localDestination.token = ctx.outputToken;
                ctx.localSource.addr = ctx.inputAddr;
                ctx.addressOffset = arg.ownerOffset;
                ctx.localSource.addr += ctx.addressOffset;
                ctx.localSource.token = ctx.inputToken;
                ctx.transferBytes = arg.ownerBytes;
                CCU_CHK_RET(
                    ccu::LocalCopy(ctx.localDestination, ctx.localSource, ctx.transferBytes, ctx.dataEvent, initBit));
            }
        }
        uint16_t eventMask = 0;
        for (uint32_t i = 0; i < arg.channelCount; ++i) {
            const uint16_t bit = static_cast<uint16_t>(1U << i);
            ctx.localSource.addr = ctx.inputAddr;
            ctx.addressOffset = arg.targetOwnerOffset[i];
            ctx.localSource.addr += ctx.addressOffset;
            ctx.localSource.token = ctx.inputToken;
            ctx.remoteDestination.addr = ctx.peerScratchAddr[i];
            ctx.addressOffset = static_cast<uint64_t>(arg.targetSlot[i]) * arg.scratchStride;
            ctx.remoteDestination.addr += ctx.addressOffset;
            ctx.remoteDestination.token = ctx.peerScratchToken[i];
            ctx.transferBytes = arg.targetOwnerBytes[i];
            CCU_CHK_RET(ccu::Write(
                arg.channels[i], ctx.remoteDestination, ctx.localSource, ctx.transferBytes, ctx.dataEvent, bit));
            eventMask = static_cast<uint16_t>(eventMask | bit);
        }
        if (eventMask != 0) {
            CCU_CHK_RET(ccu::EventWait(ctx.dataEvent, eventMask));
        }
        if (arg.earlyInit != 0 && arg.isPrimary != 0) {
            CCU_IF(ctx.isInplace == 0)
            {
                CCU_CHK_RET(ccu::EventWait(ctx.dataEvent, initBit));
            }
        }
        return NotifyAll(arg, PARTIAL_DONE_NOTIFY_INDEX);
    }

    CcuResult DoDirectLocalReduce(const CcuKernelArgDirectRsag &arg, FusedContext &ctx)
    {
        ctx.localDestination.addr = ctx.outputAddr;
        ctx.addressOffset = arg.ownerOffset;
        ctx.localDestination.addr += ctx.addressOffset;
        ctx.localDestination.token = ctx.outputToken;
        ctx.transferBytes = arg.ownerBytes;

        if (arg.earlyInit == 0) {
            CCU_IF(ctx.isInplace == 0)
            {
                ctx.localSource.addr = ctx.inputAddr;
                ctx.addressOffset = arg.ownerOffset;
                ctx.localSource.addr += ctx.addressOffset;
                ctx.localSource.token = ctx.inputToken;
                CCU_CHK_RET(
                    ccu::LocalCopy(ctx.localDestination, ctx.localSource, ctx.transferBytes, ctx.dataEvent, 1));
                CCU_CHK_RET(ccu::EventWait(ctx.dataEvent, 1));
            }
        }

        for (uint32_t sourceRank = 0; sourceRank < arg.rankSize; ++sourceRank) {
            if (sourceRank == arg.rankId) {
                continue;
            }
            ctx.localSource.addr = ctx.scratchAddr;
            ctx.addressOffset = static_cast<uint64_t>(DirectSourceSlot(sourceRank, arg.rankId)) * arg.scratchStride;
            ctx.localSource.addr += ctx.addressOffset;
            ctx.localSource.token = ctx.scratchToken;
            CCU_CHK_RET(ccu::LocalReduce(ctx.localDestination, ctx.localSource, ctx.transferBytes, arg.dataType,
                arg.reduceType, ctx.dataEvent, 1));
            CCU_CHK_RET(ccu::EventWait(ctx.dataEvent, 1));
        }
        return CCU_SUCCESS;
    }

    CcuResult RunDirectRsag(const CcuKernelArgDirectRsag &arg, FusedContext &ctx)
    {
        if (arg.kernelCount == 1) {
            CCU_CHK_RET(FusedPreSync(arg, ctx));
            CCU_CHK_RET(DoDirectPush(arg, ctx));
            CCU_CHK_RET(DoDirectLocalReduce(arg, ctx));
            return BroadcastOwner(arg, arg.ownerOffset, arg.ownerBytes, BROADCAST_DONE_NOTIFY_INDEX, ctx);
        }
        CCU_IF(ctx.phase == static_cast<uint64_t>(DirectRsagPhase::SOURCE_PUSH))
        {
            CCU_CHK_RET(FusedPreSync(arg, ctx));
            CCU_CHK_RET(DoDirectPush(arg, ctx));
        }
        CCU_IF(ctx.phase == static_cast<uint64_t>(DirectRsagPhase::LOCAL_REDUCE))
        {
            if (arg.isPrimary != 0) {
                CCU_CHK_RET(DoDirectLocalReduce(arg, ctx));
            }
        }
        CCU_IF(ctx.phase == static_cast<uint64_t>(DirectRsagPhase::BROADCAST))
        {
            if (arg.preSyncOnce == 0) {
                CCU_CHK_RET(FusedPreSync(arg, ctx));
            }
            CCU_CHK_RET(BroadcastOwner(arg, arg.ownerOffset, arg.ownerBytes, BROADCAST_DONE_NOTIFY_INDEX, ctx));
        }
        return CCU_SUCCESS;
    }

    CcuResult DoP12AllPairsPush(const CcuKernelArgP12AllPairs &arg, uint32_t segment, FusedContext &ctx)
    {
        const uint16_t initBit = static_cast<uint16_t>(1U << arg.channelCount);
        if (arg.earlyInit != 0 && arg.isPrimary != 0) {
            CCU_IF(ctx.isInplace == 0)
            {
                ctx.localDestination.addr = ctx.outputAddr;
                ctx.addressOffset = arg.ownerOffset[segment];
                ctx.localDestination.addr += ctx.addressOffset;
                ctx.localDestination.token = ctx.outputToken;
                ctx.localSource.addr = ctx.inputAddr;
                ctx.addressOffset = arg.ownerOffset[segment];
                ctx.localSource.addr += ctx.addressOffset;
                ctx.localSource.token = ctx.inputToken;
                ctx.transferBytes = arg.ownerBytes[segment];
                CCU_CHK_RET(
                    ccu::LocalCopy(ctx.localDestination, ctx.localSource, ctx.transferBytes, ctx.dataEvent, initBit));
            }
        }
        uint16_t eventMask = 0;
        for (uint32_t i = 0; i < arg.channelCount; ++i) {
            const uint16_t bit = static_cast<uint16_t>(1U << i);
            ctx.localSource.addr = ctx.inputAddr;
            ctx.addressOffset = arg.targetOwnerOffset[segment][i];
            ctx.localSource.addr += ctx.addressOffset;
            ctx.localSource.token = ctx.inputToken;
            ctx.remoteDestination.addr = ctx.peerScratchAddr[i];
            ctx.addressOffset = static_cast<uint64_t>(arg.targetSlot[i]) * arg.scratchStride;
            ctx.remoteDestination.addr += ctx.addressOffset;
            ctx.remoteDestination.token = ctx.peerScratchToken[i];
            ctx.transferBytes = arg.targetOwnerBytes[segment][i];
            CCU_CHK_RET(ccu::Write(
                arg.channels[i], ctx.remoteDestination, ctx.localSource, ctx.transferBytes, ctx.dataEvent, bit));
            eventMask = static_cast<uint16_t>(eventMask | bit);
        }
        if (eventMask != 0) {
            CCU_CHK_RET(ccu::EventWait(ctx.dataEvent, eventMask));
        }
        if (arg.earlyInit != 0 && arg.isPrimary != 0) {
            CCU_IF(ctx.isInplace == 0)
            {
                CCU_CHK_RET(ccu::EventWait(ctx.dataEvent, initBit));
            }
        }
        return NotifyAll(arg, PARTIAL_DONE_NOTIFY_INDEX);
    }

    CcuResult P12ClosPreSync(const CcuKernelArgP12AllPairs &arg, uint32_t segment, FusedContext &ctx)
    {
        if (segment != 0) {
            if (arg.isClosOwner != 0) {
                ctx.localDestination.addr = ctx.outputAddr;
                ctx.addressOffset = arg.ownerOffset[segment];
                ctx.localDestination.addr += ctx.addressOffset;
                ctx.localDestination.token = ctx.outputToken;
                ctx.transferBytes = arg.ownerBytes[segment];
                CCU_IF(ctx.isInplace == 0)
                {
                    ctx.localSource.addr = ctx.inputAddr;
                    ctx.addressOffset = arg.ownerOffset[segment];
                    ctx.localSource.addr += ctx.addressOffset;
                    ctx.localSource.token = ctx.inputToken;
                    CCU_CHK_RET(
                        ccu::LocalCopy(ctx.localDestination, ctx.localSource, ctx.transferBytes, ctx.dataEvent, 1));
                    CCU_CHK_RET(ccu::EventWait(ctx.dataEvent, 1));
                }
            }
            return CCU_SUCCESS;
        }

        if (arg.isClosOwner == 0) {
            return FusedPreSync(arg, ctx);
        }

        ctx.localDestination.addr = ctx.outputAddr;
        ctx.addressOffset = arg.ownerOffset[segment];
        ctx.localDestination.addr += ctx.addressOffset;
        ctx.localDestination.token = ctx.outputToken;
        ctx.transferBytes = arg.ownerBytes[segment];
        CCU_IF(ctx.isInplace == 0)
        {
            ctx.localSource.addr = ctx.inputAddr;
            ctx.addressOffset = arg.ownerOffset[segment];
            ctx.localSource.addr += ctx.addressOffset;
            ctx.localSource.token = ctx.inputToken;
            CCU_CHK_RET(ccu::LocalCopy(ctx.localDestination, ctx.localSource, ctx.transferBytes, ctx.dataEvent, 1));
        }

        for (uint32_t i = 0; i < arg.channelCount; ++i) {
            CCU_CHK_RET(ccu::WriteVariableWithNotify(arg.channels[i], ctx.scratchAddr, SCRATCH_ADDR_XN_ID,
                PARAM_SYNC_NOTIFY_INDEX, 1U << SCRATCH_ADDR_XN_ID));
            CCU_CHK_RET(ccu::WriteVariableWithNotify(arg.channels[i], ctx.scratchToken, SCRATCH_TOKEN_XN_ID,
                PARAM_SYNC_NOTIFY_INDEX, 1U << SCRATCH_TOKEN_XN_ID));
            CCU_CHK_RET(ccu::WriteVariableWithNotify(
                arg.channels[i], ctx.outputAddr, OUTPUT_ADDR_XN_ID, PARAM_SYNC_NOTIFY_INDEX, 1U << OUTPUT_ADDR_XN_ID));
        }
        CCU_IF(ctx.isInplace == 0)
        {
            CCU_CHK_RET(ccu::EventWait(ctx.dataEvent, 1));
        }
        for (uint32_t i = 0; i < arg.channelCount; ++i) {
            CCU_CHK_RET(ccu::WriteVariableWithNotify(arg.channels[i], ctx.outputToken, OUTPUT_TOKEN_XN_ID,
                PARAM_SYNC_NOTIFY_INDEX, 1U << OUTPUT_TOKEN_XN_ID));
        }
        constexpr uint16_t allBits = (1U << SCRATCH_ADDR_XN_ID) | (1U << SCRATCH_TOKEN_XN_ID)
                                     | (1U << OUTPUT_ADDR_XN_ID) | (1U << OUTPUT_TOKEN_XN_ID);
        for (uint32_t i = 0; i < arg.channelCount; ++i) {
            CCU_CHK_RET(ccu::NotifyWait(arg.channels[i], PARAM_SYNC_NOTIFY_INDEX, allBits));
        }
        return CCU_SUCCESS;
    }

    CcuResult RunP12ClosPhase(const CcuKernelArgP12AllPairs &arg, uint32_t segment, uint32_t phase, FusedContext &ctx)
    {
        uint32_t activeChannel = arg.channelCount;
        for (uint32_t i = 0; i < arg.channelCount; ++i) {
            if (arg.peerIsLocal[i] == 0 && arg.peerClosPhase[i] == phase) {
                activeChannel = i;
                break;
            }
        }
        if (activeChannel != arg.channelCount) {
            ctx.localSource.addr = ctx.inputAddr;
            ctx.addressOffset = arg.targetOwnerOffset[segment][activeChannel];
            ctx.localSource.addr += ctx.addressOffset;
            ctx.localSource.token = ctx.inputToken;
            ctx.remoteDestination.addr = ctx.peerOutputAddr[activeChannel];
            ctx.remoteDestination.addr += ctx.addressOffset;
            ctx.remoteDestination.token = ctx.peerOutputToken[activeChannel];
            ctx.transferBytes = arg.targetOwnerBytes[segment][activeChannel];
            CCU_CHK_RET(ccu::WriteReduce(arg.channels[activeChannel], ctx.remoteDestination, ctx.localSource,
                ctx.transferBytes, arg.dataType, arg.reduceType, ctx.dataEvent, 1));
            CCU_CHK_RET(ccu::EventWait(ctx.dataEvent, 1));
        }

        // A 侧每个 rank 只有 4 个活跃 phase，B 侧有 8 个。若只在活跃边
        // 上 ACK，两侧推进速度不同会在下一轮形成环。每轮在完整 K8,4
        // CLOS 子图上做 barrier，保证同一 output owner 的 WriteReduce 严格
        // 按 phase 串行；不同 phase 使用不同 bit，避免复用通知值。
        const uint16_t phaseBit = static_cast<uint16_t>(1U << phase);
        for (uint32_t i = 0; i < arg.channelCount; ++i) {
            if (arg.peerIsLocal[i] == 0) {
                CCU_CHK_RET(ccu::NotifyRecord(arg.channels[i], P12_CLOS_PHASE_NOTIFY_INDEX, phaseBit));
            }
        }
        for (uint32_t i = 0; i < arg.channelCount; ++i) {
            if (arg.peerIsLocal[i] == 0) {
                CCU_CHK_RET(ccu::NotifyWait(arg.channels[i], P12_CLOS_PHASE_NOTIFY_INDEX, phaseBit));
            }
        }
        return CCU_SUCCESS;
    }

    CcuResult DoP12ClosAccumulatePush(const CcuKernelArgP12AllPairs &arg, uint32_t segment, FusedContext &ctx)
    {
        CCU_CHK_RET(P12ClosPreSync(arg, segment, ctx));

        uint16_t meshEventMask = 0;
        for (uint32_t i = 0; i < arg.channelCount; ++i) {
            if (arg.peerIsLocal[i] == 0) {
                continue;
            }
            const uint16_t bit = static_cast<uint16_t>(1U << i);
            ctx.localSource.addr = ctx.inputAddr;
            ctx.addressOffset = arg.targetOwnerOffset[segment][i];
            ctx.localSource.addr += ctx.addressOffset;
            ctx.localSource.token = ctx.inputToken;
            ctx.remoteDestination.addr = ctx.peerScratchAddr[i];
            ctx.addressOffset = static_cast<uint64_t>(arg.targetSlot[i]) * arg.scratchStride;
            ctx.remoteDestination.addr += ctx.addressOffset;
            ctx.remoteDestination.token = ctx.peerScratchToken[i];
            ctx.transferBytes = arg.targetOwnerBytes[segment][i];
            CCU_CHK_RET(ccu::Write(
                arg.channels[i], ctx.remoteDestination, ctx.localSource, ctx.transferBytes, ctx.dataEvent, bit));
            meshEventMask = static_cast<uint16_t>(meshEventMask | bit);
        }

        if (arg.isClosOwner != 0) {
            for (uint32_t phase = 0; phase < P12_ALLPAIRS_A_RANK_COUNT; ++phase) {
                CCU_CHK_RET(RunP12ClosPhase(arg, segment, phase, ctx));
            }
        }
        if (meshEventMask != 0) {
            CCU_CHK_RET(ccu::EventWait(ctx.dataEvent, meshEventMask));
        }

        constexpr uint16_t doneBit = 1;
        for (uint32_t i = 0; i < arg.channelCount; ++i) {
            if (arg.peerIsLocal[i] != 0) {
                CCU_CHK_RET(ccu::NotifyRecord(arg.channels[i], P12_MESH_DONE_NOTIFY_INDEX, doneBit));
            }
        }
        for (uint32_t i = 0; i < arg.channelCount; ++i) {
            if (arg.peerIsLocal[i] != 0) {
                CCU_CHK_RET(ccu::NotifyWait(arg.channels[i], P12_MESH_DONE_NOTIFY_INDEX, doneBit));
            }
        }
        return CCU_SUCCESS;
    }

    CcuResult DoP12ClosLocalReduce(const CcuKernelArgP12AllPairs &arg, uint32_t segment, FusedContext &ctx)
    {
        const bool ownerInA = std::find(arg.treeRanks, arg.treeRanks + P12_ALLPAIRS_A_RANK_COUNT, arg.rankId)
                              != arg.treeRanks + P12_ALLPAIRS_A_RANK_COUNT;
        const uint32_t *localRanks = ownerInA ? arg.treeRanks : arg.treeRanks + P12_ALLPAIRS_A_RANK_COUNT;
        const uint32_t localRankCount = ownerInA ? P12_ALLPAIRS_A_RANK_COUNT : P12_ALLPAIRS_B_RANK_COUNT;

        ctx.localDestination.addr = ctx.outputAddr;
        ctx.addressOffset = arg.ownerOffset[segment];
        ctx.localDestination.addr += ctx.addressOffset;
        ctx.localDestination.token = ctx.outputToken;
        ctx.transferBytes = arg.ownerBytes[segment];
        uint32_t slot = 0;
        for (uint32_t i = 0; i < localRankCount; ++i) {
            if (localRanks[i] == arg.rankId) {
                continue;
            }
            ctx.localSource.addr = ctx.scratchAddr;
            ctx.addressOffset = static_cast<uint64_t>(slot++) * arg.scratchStride;
            ctx.localSource.addr += ctx.addressOffset;
            ctx.localSource.token = ctx.scratchToken;
            CCU_CHK_RET(ccu::LocalReduce(ctx.localDestination, ctx.localSource, ctx.transferBytes, arg.dataType,
                arg.reduceType, ctx.dataEvent, 1));
            CCU_CHK_RET(ccu::EventWait(ctx.dataEvent, 1));
        }
        return CCU_SUCCESS;
    }

    void SetP12AllPairsLeafDestination(
        const CcuKernelArgP12AllPairs &arg, uint32_t segment, uint32_t leafRank, FusedContext &ctx)
    {
        if (leafRank == arg.rankId) {
            ctx.localDestination.addr = ctx.outputAddr;
            ctx.addressOffset = arg.ownerOffset[segment];
            ctx.localDestination.addr += ctx.addressOffset;
            ctx.localDestination.token = ctx.outputToken;
            return;
        }
        ctx.localDestination.addr = ctx.scratchAddr;
        ctx.addressOffset = static_cast<uint64_t>(DirectSourceSlot(leafRank, arg.rankId)) * arg.scratchStride;
        ctx.localDestination.addr += ctx.addressOffset;
        ctx.localDestination.token = ctx.scratchToken;
    }

    void SetP12AllPairsLeafSource(
        const CcuKernelArgP12AllPairs &arg, uint32_t segment, uint32_t leafRank, FusedContext &ctx)
    {
        if (leafRank == arg.rankId) {
            ctx.localSource.addr = ctx.outputAddr;
            ctx.addressOffset = arg.ownerOffset[segment];
            ctx.localSource.addr += ctx.addressOffset;
            ctx.localSource.token = ctx.outputToken;
            return;
        }
        ctx.localSource.addr = ctx.scratchAddr;
        ctx.addressOffset = static_cast<uint64_t>(DirectSourceSlot(leafRank, arg.rankId)) * arg.scratchStride;
        ctx.localSource.addr += ctx.addressOffset;
        ctx.localSource.token = ctx.scratchToken;
    }

    CcuResult IssueP12AllPairsTreeReduce(const CcuKernelArgP12AllPairs &arg, uint32_t segment,
        uint32_t destinationRank, uint32_t sourceRank, uint16_t eventBit, FusedContext &ctx)
    {
        SetP12AllPairsLeafDestination(arg, segment, destinationRank, ctx);
        SetP12AllPairsLeafSource(arg, segment, sourceRank, ctx);
        return ccu::LocalReduce(ctx.localDestination, ctx.localSource, ctx.transferBytes, arg.dataType, arg.reduceType,
            ctx.dataEvent, eventBit);
    }

    void BuildP12AllPairsRootedGroup(
        const uint32_t *canonicalRanks, uint32_t rankCount, uint32_t ownerRank, uint32_t *rootedRanks)
    {
        uint32_t next = 0;
        for (uint32_t i = 0; i < rankCount; ++i) {
            if (canonicalRanks[i] == ownerRank) {
                rootedRanks[next++] = ownerRank;
                break;
            }
        }
        for (uint32_t i = 0; i < rankCount; ++i) {
            if (canonicalRanks[i] != ownerRank) {
                rootedRanks[next++] = canonicalRanks[i];
            }
        }
    }

    CcuResult DoP12AllPairsBalancedLocalReduce(
        const CcuKernelArgP12AllPairs &arg, uint32_t segment, FusedContext &ctx)
    {
        uint32_t aRanks[P12_ALLPAIRS_A_RANK_COUNT]{};
        uint32_t bRanks[P12_ALLPAIRS_B_RANK_COUNT]{};
        const bool ownerInA = std::find(arg.treeRanks, arg.treeRanks + P12_ALLPAIRS_A_RANK_COUNT, arg.rankId)
                              != arg.treeRanks + P12_ALLPAIRS_A_RANK_COUNT;
        BuildP12AllPairsRootedGroup(
            arg.treeRanks, P12_ALLPAIRS_A_RANK_COUNT, ownerInA ? arg.rankId : MAX_RANK_SIZE, aRanks);
        BuildP12AllPairsRootedGroup(arg.treeRanks + P12_ALLPAIRS_A_RANK_COUNT, P12_ALLPAIRS_B_RANK_COUNT,
            ownerInA ? MAX_RANK_SIZE : arg.rankId, bRanks);

        // Level 0: A8 has four independent pairs and B4 has two.  Each LocalReduce
        // uses a distinct Event bit, so the six HBM operations can overlap.
        uint16_t eventMask = 0;
        uint32_t eventIndex = 0;
        for (uint32_t i = 0; i < P12_ALLPAIRS_A_RANK_COUNT; i += 2) {
            const uint16_t bit = static_cast<uint16_t>(1U << eventIndex++);
            CCU_CHK_RET(IssueP12AllPairsTreeReduce(arg, segment, aRanks[i], aRanks[i + 1], bit, ctx));
            eventMask = static_cast<uint16_t>(eventMask | bit);
        }
        for (uint32_t i = 0; i < P12_ALLPAIRS_B_RANK_COUNT; i += 2) {
            const uint16_t bit = static_cast<uint16_t>(1U << eventIndex++);
            CCU_CHK_RET(IssueP12AllPairsTreeReduce(arg, segment, bRanks[i], bRanks[i + 1], bit, ctx));
            eventMask = static_cast<uint16_t>(eventMask | bit);
        }
        CCU_CHK_RET(ccu::EventWait(ctx.dataEvent, eventMask));

        // Level 1: two A reductions and one B reduction remain independent.
        eventMask = 0;
        CCU_CHK_RET(IssueP12AllPairsTreeReduce(arg, segment, aRanks[0], aRanks[2], 1U, ctx));
        eventMask = static_cast<uint16_t>(eventMask | 1U);
        CCU_CHK_RET(IssueP12AllPairsTreeReduce(arg, segment, aRanks[4], aRanks[6], 2U, ctx));
        eventMask = static_cast<uint16_t>(eventMask | 2U);
        CCU_CHK_RET(IssueP12AllPairsTreeReduce(arg, segment, bRanks[0], bRanks[2], 4U, ctx));
        eventMask = static_cast<uint16_t>(eventMask | 4U);
        CCU_CHK_RET(ccu::EventWait(ctx.dataEvent, eventMask));

        // Level 2 finishes A8; level 3 combines A and B into this owner's output root.
        CCU_CHK_RET(IssueP12AllPairsTreeReduce(arg, segment, aRanks[0], aRanks[4], 1U, ctx));
        CCU_CHK_RET(ccu::EventWait(ctx.dataEvent, 1U));
        const uint32_t destinationRoot = ownerInA ? aRanks[0] : bRanks[0];
        const uint32_t sourceRoot = ownerInA ? bRanks[0] : aRanks[0];
        CCU_CHK_RET(IssueP12AllPairsTreeReduce(arg, segment, destinationRoot, sourceRoot, 1U, ctx));
        return ccu::EventWait(ctx.dataEvent, 1U);
    }

    CcuResult DoP12AllPairsLocalReduce(const CcuKernelArgP12AllPairs &arg, uint32_t segment, FusedContext &ctx)
    {
        ctx.localDestination.addr = ctx.outputAddr;
        ctx.addressOffset = arg.ownerOffset[segment];
        ctx.localDestination.addr += ctx.addressOffset;
        ctx.localDestination.token = ctx.outputToken;
        ctx.transferBytes = arg.ownerBytes[segment];

        if (arg.earlyInit == 0) {
            CCU_IF(ctx.isInplace == 0)
            {
                ctx.localSource.addr = ctx.inputAddr;
                ctx.addressOffset = arg.ownerOffset[segment];
                ctx.localSource.addr += ctx.addressOffset;
                ctx.localSource.token = ctx.inputToken;
                CCU_CHK_RET(
                    ccu::LocalCopy(ctx.localDestination, ctx.localSource, ctx.transferBytes, ctx.dataEvent, 1));
                CCU_CHK_RET(ccu::EventWait(ctx.dataEvent, 1));
            }
        }

        if (arg.balancedReduce != 0) {
            return DoP12AllPairsBalancedLocalReduce(arg, segment, ctx);
        }

        // 固定 source-rank 顺序，避免网络到达顺序改变 FP32 加法括号。
        for (uint32_t sourceRank = 0; sourceRank < arg.rankSize; ++sourceRank) {
            if (sourceRank == arg.rankId) {
                continue;
            }
            ctx.localSource.addr = ctx.scratchAddr;
            ctx.addressOffset = static_cast<uint64_t>(DirectSourceSlot(sourceRank, arg.rankId)) * arg.scratchStride;
            ctx.localSource.addr += ctx.addressOffset;
            ctx.localSource.token = ctx.scratchToken;
            CCU_CHK_RET(ccu::LocalReduce(ctx.localDestination, ctx.localSource, ctx.transferBytes, arg.dataType,
                arg.reduceType, ctx.dataEvent, 1));
            CCU_CHK_RET(ccu::EventWait(ctx.dataEvent, 1));
        }
        return CCU_SUCCESS;
    }

    CcuResult RunP12AllPairsSegment(const CcuKernelArgP12AllPairs &arg, uint32_t segment, FusedContext &ctx)
    {
        if (arg.kernelCount == 1) {
            if (arg.closAccumulate != 0) {
                CCU_CHK_RET(DoP12ClosAccumulatePush(arg, segment, ctx));
                CCU_CHK_RET(DoP12ClosLocalReduce(arg, segment, ctx));
                return BroadcastOwner(
                    arg, arg.ownerOffset[segment], arg.ownerBytes[segment], P12_CLOS_BROADCAST_NOTIFY_INDEX, ctx);
            }
            if (arg.preSyncOnce == 0 || segment == 0) {
                CCU_CHK_RET(FusedPreSync(arg, ctx));
            }
            CCU_CHK_RET(DoP12AllPairsPush(arg, segment, ctx));
            CCU_CHK_RET(DoP12AllPairsLocalReduce(arg, segment, ctx));
            return BroadcastOwner(
                arg, arg.ownerOffset[segment], arg.ownerBytes[segment], BROADCAST_DONE_NOTIFY_INDEX, ctx);
        }
        CCU_IF(ctx.phase == static_cast<uint64_t>(DirectRsagPhase::SOURCE_PUSH))
        {
            if (arg.closAccumulate != 0) {
                CCU_CHK_RET(DoP12ClosAccumulatePush(arg, segment, ctx));
            } else if (arg.preSyncOnce == 0 || segment == 0) {
                CCU_CHK_RET(FusedPreSync(arg, ctx));
                CCU_CHK_RET(DoP12AllPairsPush(arg, segment, ctx));
            } else {
                CCU_CHK_RET(DoP12AllPairsPush(arg, segment, ctx));
            }
        }
        CCU_IF(ctx.phase == static_cast<uint64_t>(DirectRsagPhase::LOCAL_REDUCE))
        {
            if (arg.isPrimary != 0) {
                if (arg.closAccumulate != 0) {
                    CCU_CHK_RET(DoP12ClosLocalReduce(arg, segment, ctx));
                } else {
                    CCU_CHK_RET(DoP12AllPairsLocalReduce(arg, segment, ctx));
                }
            }
        }
        CCU_IF(ctx.phase == static_cast<uint64_t>(DirectRsagPhase::BROADCAST))
        {
            if (arg.closAccumulate == 0 && arg.preSyncOnce == 0) {
                CCU_CHK_RET(FusedPreSync(arg, ctx));
            }
            CCU_CHK_RET(BroadcastOwner(arg, arg.ownerOffset[segment], arg.ownerBytes[segment],
                arg.closAccumulate != 0 ? P12_CLOS_BROADCAST_NOTIFY_INDEX : BROADCAST_DONE_NOTIFY_INDEX, ctx));
        }
        return CCU_SUCCESS;
    }

    CcuResult RunP12AllPairs(const CcuKernelArgP12AllPairs &arg, FusedContext &ctx)
    {
        CCU_IF(ctx.segment == 0)
        {
            CCU_CHK_RET(RunP12AllPairsSegment(arg, 0, ctx));
        }
        if (arg.segmentCount > 1) {
            CCU_IF(ctx.segment == 1)
            {
                CCU_CHK_RET(RunP12AllPairsSegment(arg, 1, ctx));
            }
        }
        return CCU_SUCCESS;
    }

    namespace v112_p12_detail {

    CcuResult InitContext(const CcuKernelArgP12OutputTree &arg, FusedContext &ctx)
    {
        CCU_CHK_RET(InitFusedContext(arg, arg.scratchAddr, arg.scratchToken, ctx));
        ctx.peerInputAddr.resize(arg.channelCount);
        ctx.peerInputToken.resize(arg.channelCount);
        for (uint32_t i = 0; i < arg.channelCount; ++i) {
            ctx.peerInputAddr[i] = ccu::GetResByChannel<ccu::Variable>(arg.channels[i], INPUT_ADDR_XN_ID);
            ctx.peerInputToken[i] = ccu::GetResByChannel<ccu::Variable>(arg.channels[i], INPUT_TOKEN_XN_ID);
        }
        return CCU_SUCCESS;
    }

    void GetPullShard(uint64_t bytes, uint32_t shardCount, uint32_t shard, uint64_t &offsetBytes,
        uint64_t &shardBytes)
    {
        const uint64_t elements = bytes / sizeof(float);
        const uint64_t base = elements / shardCount;
        const uint64_t remainder = elements % shardCount;
        const uint64_t offsetElements = static_cast<uint64_t>(shard) * base + std::min<uint64_t>(shard, remainder);
        offsetBytes = offsetElements * sizeof(float);
        shardBytes = (base + (shard < remainder ? 1U : 0U)) * sizeof(float);
    }

    void SetPullPartial(
        const CcuKernelArgP12OutputTree &arg, uint32_t segment, FusedContext &ctx, ccu::LocalAddr &partial)
    {
        partial.addr = arg.isPrimary != 0 ? ctx.outputAddr : ctx.scratchAddr;
        if (arg.isPrimary != 0) {
            ctx.addressOffset = arg.ownerOffset[segment];
            partial.addr += ctx.addressOffset;
            partial.token = ctx.outputToken;
        } else {
            partial.token = ctx.scratchToken;
        }
    }

    void PrepareRotatingRead(const CcuKernelArgP12OutputTree &arg, uint32_t segment,
        const ccu::LocalAddr &partial, uint32_t round, uint32_t channel, FusedContext &ctx,
        ccu::RemoteAddr &remoteSource)
    {
        const uint32_t shard = (round + channel) % arg.channelCount;
        uint64_t shardOffset = 0;
        uint64_t shardBytes = 0;
        GetPullShard(arg.ownerBytes[segment], arg.channelCount, shard, shardOffset, shardBytes);

        ctx.localDestination = partial;
        ctx.addressOffset = shardOffset;
        ctx.localDestination.addr += ctx.addressOffset;
        remoteSource.addr = ctx.peerInputAddr[channel];
        ctx.addressOffset = arg.ownerOffset[segment] + shardOffset;
        remoteSource.addr += ctx.addressOffset;
        remoteSource.token = ctx.peerInputToken[channel];
        ctx.transferBytes = shardBytes;
    }

    CcuResult DoRotatingReadReduce(const CcuKernelArgP12OutputTree &arg, uint32_t segment, FusedContext &ctx)
    {
        ccu::LocalAddr partial;
        ccu::RemoteAddr remoteSource;
        SetPullPartial(arg, segment, ctx, partial);
        if (arg.isPrimary != 0) {
            ctx.localSource.addr = ctx.inputAddr;
            ctx.addressOffset = arg.ownerOffset[segment];
            ctx.localSource.addr += ctx.addressOffset;
            ctx.localSource.token = ctx.inputToken;
            ctx.transferBytes = arg.ownerBytes[segment];
            CCU_CHK_RET(ccu::LocalCopy(partial, ctx.localSource, ctx.transferBytes, ctx.dataEvent, 1U));
            PrepareRotatingRead(arg, segment, partial, 0, 0, ctx, remoteSource);
            CCU_CHK_RET(ccu::EventWait(ctx.dataEvent, 1U));
        } else {
            PrepareRotatingRead(arg, segment, partial, 0, 0, ctx, remoteSource);
        }

        const uint16_t eventMask = static_cast<uint16_t>((1U << arg.channelCount) - 1U);
        for (uint32_t round = 0; round < arg.channelCount; ++round) {
            for (uint32_t channel = 0; channel < arg.channelCount; ++channel) {
                if (channel != 0) {
                    PrepareRotatingRead(arg, segment, partial, round, channel, ctx, remoteSource);
                }
                const uint16_t bit = static_cast<uint16_t>(1U << channel);
                if (arg.isPrimary == 0 && round == 0) {
                    CCU_CHK_RET(ccu::Read(arg.channels[channel], ctx.localDestination, remoteSource, ctx.transferBytes,
                        ctx.dataEvent, bit));
                } else {
                    CCU_CHK_RET(ccu::ReadReduce(arg.channels[channel], ctx.localDestination, remoteSource,
                        ctx.transferBytes, arg.dataType, arg.reduceType, ctx.dataEvent, bit));
                }
            }
            if (round + 1 < arg.channelCount) {
                PrepareRotatingRead(arg, segment, partial, round + 1, 0, ctx, remoteSource);
            }
            CCU_CHK_RET(ccu::EventWait(ctx.dataEvent, eventMask));
        }
        return CCU_SUCCESS;
    }

    CcuResult DoRotatingReadReduceMerge(
        const CcuKernelArgP12OutputTree &arg, uint32_t segment, FusedContext &ctx)
    {
        ctx.localDestination.addr = ctx.outputAddr;
        ctx.addressOffset = arg.ownerOffset[segment];
        ctx.localDestination.addr += ctx.addressOffset;
        ctx.localDestination.token = ctx.outputToken;
        ctx.localSource.addr = ctx.scratchAddr;
        ctx.localSource.token = ctx.scratchToken;
        ctx.transferBytes = arg.ownerBytes[segment];
        CCU_CHK_RET(ccu::LocalReduce(ctx.localDestination, ctx.localSource, ctx.transferBytes, arg.dataType,
            arg.reduceType, ctx.dataEvent, 1U));
        return ccu::EventWait(ctx.dataEvent, 1U);
    }

    CcuResult RunSegment(const CcuKernelArgP12OutputTree &arg, uint32_t segment, FusedContext &ctx)
    {
        CCU_IF(ctx.phase == static_cast<uint64_t>(DirectRsagPhase::SOURCE_PUSH))
        {
            CCU_CHK_RET(FusedPullPreSync(arg, ctx));
            CCU_CHK_RET(DoRotatingReadReduce(arg, segment, ctx));
        }
        if (arg.groupOutputRoot == 0) {
            CCU_IF(ctx.phase == static_cast<uint64_t>(DirectRsagPhase::FINAL_MERGE))
            {
                CCU_CHK_RET(DoRotatingReadReduceMerge(arg, segment, ctx));
            }
        }
        CCU_IF(ctx.phase == static_cast<uint64_t>(DirectRsagPhase::BROADCAST))
        {
            CCU_CHK_RET(NotifyAll(arg, PARTIAL_DONE_NOTIFY_INDEX));
            CCU_CHK_RET(BroadcastOwner(
                arg, arg.ownerOffset[segment], arg.ownerBytes[segment], BROADCAST_DONE_NOTIFY_INDEX, ctx));
        }
        return CCU_SUCCESS;
    }

    CcuResult Run(const CcuKernelArgP12OutputTree &arg, FusedContext &ctx)
    {
        CCU_IF(ctx.segment == 0)
        {
            CCU_CHK_RET(RunSegment(arg, 0, ctx));
        }
        return CCU_SUCCESS;
    }

    } // namespace v112_p12_detail

    void GetEqualShard(uint64_t bytes, uint32_t shardCount, uint32_t shard, uint64_t &offsetBytes, uint64_t &shardBytes)
    {
        const uint64_t elements = bytes / sizeof(float);
        const uint64_t base = elements / shardCount;
        const uint64_t remainder = elements % shardCount;
        const uint64_t offsetElements = static_cast<uint64_t>(shard) * base + std::min<uint64_t>(shard, remainder);
        offsetBytes = offsetElements * sizeof(float);
        shardBytes = (base + (shard < remainder ? 1U : 0U)) * sizeof(float);
    }

    CcuResult RunRotatePhase(const CcuKernelArgP4Rotate3 &arg, uint32_t phase, FusedContext &ctx)
    {
        uint16_t eventMask = 0;
        for (uint32_t i = 0; i < arg.channelCount; ++i) {
            const uint32_t shard = (arg.targetPosition[i] + phase) % P4_ROTATE_SHARD_COUNT;
            uint64_t shardOffset = 0;
            uint64_t shardBytes = 0;
            GetEqualShard(arg.targetOwnerBytes[i], P4_ROTATE_SHARD_COUNT, shard, shardOffset, shardBytes);
            const uint16_t bit = static_cast<uint16_t>(1U << i);
            ctx.localSource.addr = ctx.inputAddr;
            ctx.addressOffset = arg.targetOwnerOffset[i] + shardOffset;
            ctx.localSource.addr += ctx.addressOffset;
            ctx.localSource.token = ctx.inputToken;
            if (arg.directOutputAccumulator != 0) {
                ctx.remoteDestination.addr = ctx.peerOutputAddr[i];
                ctx.addressOffset = arg.targetOwnerOffset[i] + shardOffset;
                ctx.remoteDestination.addr += ctx.addressOffset;
                ctx.remoteDestination.token = ctx.peerOutputToken[i];
            } else {
                ctx.remoteDestination.addr = ctx.peerScratchAddr[i];
                ctx.addressOffset = shardOffset;
                ctx.remoteDestination.addr += ctx.addressOffset;
                ctx.remoteDestination.token = ctx.peerScratchToken[i];
            }
            ctx.transferBytes = shardBytes;
            if (arg.directOutputAccumulator != 0) {
                CCU_CHK_RET(ccu::WriteReduce(arg.channels[i], ctx.remoteDestination, ctx.localSource, ctx.transferBytes,
                    arg.dataType, arg.reduceType, ctx.dataEvent, bit));
            } else if (phase == 0) {
                CCU_CHK_RET(ccu::Write(
                    arg.channels[i], ctx.remoteDestination, ctx.localSource, ctx.transferBytes, ctx.dataEvent, bit));
            } else {
                CCU_CHK_RET(ccu::WriteReduce(arg.channels[i], ctx.remoteDestination, ctx.localSource, ctx.transferBytes,
                    arg.dataType, arg.reduceType, ctx.dataEvent, bit));
            }
            eventMask = static_cast<uint16_t>(eventMask | bit);
        }
        if (eventMask != 0) {
            CCU_CHK_RET(ccu::EventWait(ctx.dataEvent, eventMask));
        }
        constexpr uint16_t doneBit = 1;
        const uint32_t dataNotifyIndex = arg.uniquePhaseNotify != 0
                                             ? ROTATE_UNIQUE_PHASE_NOTIFY_BASE_INDEX + phase
                                             : ROTATE_DATA_NOTIFY_INDEX;
        for (uint32_t i = 0; i < arg.channelCount; ++i) {
            CCU_CHK_RET(ccu::NotifyRecord(arg.channels[i], dataNotifyIndex, doneBit));
        }
        for (uint32_t i = 0; i < arg.channelCount; ++i) {
            CCU_CHK_RET(ccu::NotifyWait(arg.channels[i], dataNotifyIndex, doneBit));
        }
        CCU_CHK_RET(KernelBarrier(arg.kernelIndex, arg.kernelCount, arg.isPrimary, static_cast<uint16_t>(1U << phase),
            "rotate_phase_ready0", "rotate_phase_ready1", "rotate_phase_release0", "rotate_phase_release1"));
        if (arg.uniquePhaseNotify == 0) {
            for (uint32_t i = 0; i < arg.channelCount; ++i) {
                CCU_CHK_RET(ccu::NotifyRecord(arg.channels[i], ROTATE_NEXT_NOTIFY_INDEX, doneBit));
            }
            for (uint32_t i = 0; i < arg.channelCount; ++i) {
                CCU_CHK_RET(ccu::NotifyWait(arg.channels[i], ROTATE_NEXT_NOTIFY_INDEX, doneBit));
            }
        }
        return CCU_SUCCESS;
    }

    CcuResult RunFullSliceAccumulatePhase(const CcuKernelArgP4Rotate3 &arg, uint32_t phase, FusedContext &ctx)
    {
        constexpr uint32_t rankCount = 4;
        const uint32_t outgoingRank = (arg.rankId + phase + 1U) % rankCount;
        uint32_t outgoingIndex = arg.channelCount;
        for (uint32_t i = 0; i < arg.channelCount; ++i) {
            if (arg.peerRanks[i] == outgoingRank) {
                outgoingIndex = i;
            }
        }
        if (outgoingIndex >= arg.channelCount) {
            return CCU_E_PARA;
        }

        if (arg.phaseChain != 0 && phase > 0) {
            const uint32_t previousWriterRank = (arg.rankId + 1U) % rankCount;
            uint32_t previousWriterIndex = arg.channelCount;
            for (uint32_t i = 0; i < arg.channelCount; ++i) {
                if (arg.peerRanks[i] == previousWriterRank) {
                    previousWriterIndex = i;
                }
            }
            if (previousWriterIndex >= arg.channelCount) {
                return CCU_E_PARA;
            }
            CCU_CHK_RET(ccu::NotifyWait(
                arg.channels[previousWriterIndex], ROTATE_UNIQUE_PHASE_NOTIFY_BASE_INDEX + phase - 1U, 1));
        }

        ctx.localSource.addr = ctx.inputAddr;
        ctx.addressOffset = arg.targetOwnerOffset[outgoingIndex];
        ctx.localSource.addr += ctx.addressOffset;
        ctx.localSource.token = ctx.inputToken;
        ctx.remoteDestination.addr = ctx.peerOutputAddr[outgoingIndex];
        ctx.remoteDestination.addr += ctx.addressOffset;
        ctx.remoteDestination.token = ctx.peerOutputToken[outgoingIndex];
        ctx.transferBytes = arg.targetOwnerBytes[outgoingIndex];
        CCU_CHK_RET(ccu::WriteReduce(arg.channels[outgoingIndex], ctx.remoteDestination, ctx.localSource,
            ctx.transferBytes, arg.dataType, arg.reduceType, ctx.dataEvent, 1));
        CCU_CHK_RET(ccu::EventWait(ctx.dataEvent, 1));

        constexpr uint16_t doneBit = 1;
        const uint32_t notifyIndex = ROTATE_UNIQUE_PHASE_NOTIFY_BASE_INDEX + phase;
        if (arg.phaseChain != 0) {
            const uint32_t nextWriterRank = (arg.rankId + rankCount - 1U) % rankCount;
            for (uint32_t i = 0; i < arg.channelCount; ++i) {
                if (arg.peerRanks[i] == nextWriterRank) {
                    CCU_CHK_RET(ccu::NotifyRecord(arg.channels[i], notifyIndex, doneBit));
                    return CCU_SUCCESS;
                }
            }
            return CCU_E_PARA;
        }
        for (uint32_t i = 0; i < arg.channelCount; ++i) {
            CCU_CHK_RET(ccu::NotifyRecord(arg.channels[i], notifyIndex, doneBit));
        }
        for (uint32_t i = 0; i < arg.channelCount; ++i) {
            CCU_CHK_RET(ccu::NotifyWait(arg.channels[i], notifyIndex, doneBit));
        }
        return CCU_SUCCESS;
    }

    CcuResult RunWave2FullSliceAccumulatePhase(
        const CcuKernelArgP4Rotate3 &arg, uint32_t chunk, uint32_t phase, FusedContext &ctx)
    {
        constexpr uint32_t rankCount = 4;
        if (chunk >= ROTATE_WAVE2_CHUNK_COUNT) {
            return CCU_E_PARA;
        }
        const uint16_t doneBit = static_cast<uint16_t>(1U << chunk);
        const uint32_t outgoingRank = (arg.rankId + phase + 1U) % rankCount;
        uint32_t outgoingIndex = arg.channelCount;
        for (uint32_t i = 0; i < arg.channelCount; ++i) {
            if (arg.peerRanks[i] == outgoingRank) {
                outgoingIndex = i;
            }
        }
        if (outgoingIndex >= arg.channelCount) {
            return CCU_E_PARA;
        }

        if (phase > 0) {
            const uint32_t previousWriterRank = (arg.rankId + 1U) % rankCount;
            uint32_t previousWriterIndex = arg.channelCount;
            for (uint32_t i = 0; i < arg.channelCount; ++i) {
                if (arg.peerRanks[i] == previousWriterRank) {
                    previousWriterIndex = i;
                }
            }
            if (previousWriterIndex >= arg.channelCount) {
                return CCU_E_PARA;
            }
            CCU_CHK_RET(ccu::NotifyWait(arg.channels[previousWriterIndex],
                ROTATE_UNIQUE_PHASE_NOTIFY_BASE_INDEX + phase - 1U, doneBit));
        }

        uint64_t chunkOffset = 0;
        uint64_t chunkBytes = 0;
        GetEqualShard(arg.targetOwnerBytes[outgoingIndex], ROTATE_WAVE2_CHUNK_COUNT, chunk, chunkOffset, chunkBytes);
        ctx.localSource.addr = ctx.inputAddr;
        ctx.addressOffset = arg.targetOwnerOffset[outgoingIndex] + chunkOffset;
        ctx.localSource.addr += ctx.addressOffset;
        ctx.localSource.token = ctx.inputToken;
        ctx.remoteDestination.addr = ctx.peerOutputAddr[outgoingIndex];
        ctx.remoteDestination.addr += ctx.addressOffset;
        ctx.remoteDestination.token = ctx.peerOutputToken[outgoingIndex];
        ctx.transferBytes = chunkBytes;
        CCU_CHK_RET(ccu::WriteReduce(arg.channels[outgoingIndex], ctx.remoteDestination, ctx.localSource,
            ctx.transferBytes, arg.dataType, arg.reduceType, ctx.dataEvent, 1));
        CCU_CHK_RET(ccu::EventWait(ctx.dataEvent, 1));

        const uint32_t notifyIndex = ROTATE_UNIQUE_PHASE_NOTIFY_BASE_INDEX + phase;
        const uint32_t nextWriterRank = (arg.rankId + rankCount - 1U) % rankCount;
        for (uint32_t i = 0; i < arg.channelCount; ++i) {
            if (arg.peerRanks[i] == nextWriterRank) {
                return ccu::NotifyRecord(arg.channels[i], notifyIndex, doneBit);
            }
        }
        return CCU_E_PARA;
    }

    CcuResult WaitWave2FinalWriter(const CcuKernelArgP4Rotate3 &arg, uint32_t chunk)
    {
        if (chunk >= ROTATE_WAVE2_CHUNK_COUNT) {
            return CCU_E_PARA;
        }
        const uint32_t finalWriterRank = (arg.rankId + 1U) % 4U;
        for (uint32_t i = 0; i < arg.channelCount; ++i) {
            if (arg.peerRanks[i] == finalWriterRank) {
                return ccu::NotifyWait(arg.channels[i],
                    ROTATE_UNIQUE_PHASE_NOTIFY_BASE_INDEX + P4_ROTATE_SHARD_COUNT - 1U,
                    static_cast<uint16_t>(1U << chunk));
            }
        }
        return CCU_E_PARA;
    }

    CcuResult RunPullRotatePhase(const CcuKernelArgP4Rotate3 &arg, uint32_t phase, FusedContext &ctx)
    {
        uint16_t eventMask = 0;
        for (uint32_t i = 0; i < arg.channelCount; ++i) {
            const uint32_t shard = (arg.targetPosition[i] + phase) % P4_ROTATE_SHARD_COUNT;
            uint64_t shardOffset = 0;
            uint64_t shardBytes = 0;
            GetEqualShard(arg.ownerBytes, P4_ROTATE_SHARD_COUNT, shard, shardOffset, shardBytes);
            const uint16_t bit = static_cast<uint16_t>(1U << i);
            ctx.localDestination.addr = ctx.outputAddr;
            ctx.addressOffset = arg.ownerOffset + shardOffset;
            ctx.localDestination.addr += ctx.addressOffset;
            ctx.localDestination.token = ctx.outputToken;
            ccu::RemoteAddr remoteSource;
            remoteSource.addr = ctx.peerScratchAddr[i];
            ctx.addressOffset = arg.ownerOffset + shardOffset;
            remoteSource.addr += ctx.addressOffset;
            remoteSource.token = ctx.peerScratchToken[i];
            ctx.transferBytes = shardBytes;
            CCU_CHK_RET(ccu::ReadReduce(arg.channels[i], ctx.localDestination, remoteSource, ctx.transferBytes,
                arg.dataType, arg.reduceType, ctx.dataEvent, bit));
            eventMask = static_cast<uint16_t>(eventMask | bit);
        }
        if (eventMask != 0) {
            CCU_CHK_RET(ccu::EventWait(ctx.dataEvent, eventMask));
        }
        return CCU_SUCCESS;
    }

    CcuResult StartRotateOutputInit(const CcuKernelArgP4Rotate3 &arg, FusedContext &ctx)
    {
        if (arg.isPrimary != 0) {
            ctx.localDestination.addr = ctx.outputAddr;
            ctx.addressOffset = arg.ownerOffset;
            ctx.localDestination.addr += ctx.addressOffset;
            ctx.localDestination.token = ctx.outputToken;
            ctx.transferBytes = arg.ownerBytes;
            CCU_IF(ctx.isInplace == 0)
            {
                ctx.localSource.addr = ctx.inputAddr;
                ctx.addressOffset = arg.ownerOffset;
                ctx.localSource.addr += ctx.addressOffset;
                ctx.localSource.token = ctx.inputToken;
                CCU_CHK_RET(ccu::LocalCopy(
                    ctx.localDestination, ctx.localSource, ctx.transferBytes, ctx.dataEvent, 1));
            }
        }
        return CCU_SUCCESS;
    }

    CcuResult FinishRotateOutputInit(const CcuKernelArgP4Rotate3 &arg, FusedContext &ctx)
    {
        if (arg.isPrimary != 0) {
            CCU_IF(ctx.isInplace == 0)
            {
                CCU_CHK_RET(ccu::EventWait(ctx.dataEvent, 1));
            }
        }
        CCU_CHK_RET(KernelBarrier(arg.kernelIndex, arg.kernelCount, arg.isPrimary, 1, "rotate_output_init_ready0",
            "rotate_output_init_ready1", "rotate_output_init_release0", "rotate_output_init_release1"));
        for (uint32_t i = 0; i < arg.channelCount; ++i) {
            CCU_CHK_RET(ccu::WriteVariableWithNotify(arg.channels[i], ctx.outputToken, OUTPUT_TOKEN_XN_ID,
                PARAM_SYNC_NOTIFY_INDEX, 1U << OUTPUT_TOKEN_XN_ID));
        }
        constexpr uint16_t outputBits = (1U << OUTPUT_ADDR_XN_ID) | (1U << OUTPUT_TOKEN_XN_ID);
        for (uint32_t i = 0; i < arg.channelCount; ++i) {
            CCU_CHK_RET(ccu::NotifyWait(arg.channels[i], PARAM_SYNC_NOTIFY_INDEX, outputBits));
        }
        return CCU_SUCCESS;
    }

    CcuResult StartWave2OutputInit(const CcuKernelArgP4Rotate3 &arg, FusedContext &ctx)
    {
        if (arg.isPrimary == 0) {
            return CCU_E_PARA;
        }
        for (uint32_t chunk = 0; chunk < ROTATE_WAVE2_CHUNK_COUNT; ++chunk) {
            uint64_t chunkOffset = 0;
            uint64_t chunkBytes = 0;
            GetEqualShard(arg.ownerBytes, ROTATE_WAVE2_CHUNK_COUNT, chunk, chunkOffset, chunkBytes);
            ctx.localDestination.addr = ctx.outputAddr;
            ctx.addressOffset = arg.ownerOffset + chunkOffset;
            ctx.localDestination.addr += ctx.addressOffset;
            ctx.localDestination.token = ctx.outputToken;
            ctx.localSource.addr = ctx.inputAddr;
            ctx.localSource.addr += ctx.addressOffset;
            ctx.localSource.token = ctx.inputToken;
            ctx.transferBytes = chunkBytes;
            CCU_CHK_RET(ccu::LocalCopy(ctx.localDestination, ctx.localSource, ctx.transferBytes, ctx.dataEvent,
                static_cast<uint16_t>(1U << chunk)));
        }
        return CCU_SUCCESS;
    }

    CcuResult FinishWave2ChunkAInit(const CcuKernelArgP4Rotate3 &arg, FusedContext &ctx)
    {
        CCU_CHK_RET(ccu::EventWait(ctx.dataEvent, 1U));
        CCU_CHK_RET(KernelBarrier(arg.kernelIndex, arg.kernelCount, arg.isPrimary, 1U,
            "rotate_wave2_chunk_a_ready0", "rotate_wave2_chunk_a_ready1", "rotate_wave2_chunk_a_release0",
            "rotate_wave2_chunk_a_release1"));
        for (uint32_t i = 0; i < arg.channelCount; ++i) {
            CCU_CHK_RET(ccu::WriteVariableWithNotify(arg.channels[i], ctx.outputToken, OUTPUT_TOKEN_XN_ID,
                PARAM_SYNC_NOTIFY_INDEX, 1U << OUTPUT_TOKEN_XN_ID));
        }
        constexpr uint16_t outputBits = (1U << OUTPUT_ADDR_XN_ID) | (1U << OUTPUT_TOKEN_XN_ID);
        for (uint32_t i = 0; i < arg.channelCount; ++i) {
            CCU_CHK_RET(ccu::NotifyWait(arg.channels[i], PARAM_SYNC_NOTIFY_INDEX, outputBits));
        }
        return CCU_SUCCESS;
    }

    CcuResult FinishWave2ChunkBInit(const CcuKernelArgP4Rotate3 &arg, FusedContext &ctx)
    {
        constexpr uint16_t chunkBBit = 1U << 1U;
        CCU_CHK_RET(ccu::EventWait(ctx.dataEvent, chunkBBit));
        CCU_CHK_RET(KernelBarrier(arg.kernelIndex, arg.kernelCount, arg.isPrimary, chunkBBit,
            "rotate_wave2_chunk_b_ready0", "rotate_wave2_chunk_b_ready1", "rotate_wave2_chunk_b_release0",
            "rotate_wave2_chunk_b_release1"));
        for (uint32_t i = 0; i < arg.channelCount; ++i) {
            CCU_CHK_RET(ccu::NotifyRecord(arg.channels[i], ROTATE_UNIQUE_BROADCAST_NOTIFY_INDEX, chunkBBit));
        }
        for (uint32_t i = 0; i < arg.channelCount; ++i) {
            CCU_CHK_RET(ccu::NotifyWait(arg.channels[i], ROTATE_UNIQUE_BROADCAST_NOTIFY_INDEX, chunkBBit));
        }
        return CCU_SUCCESS;
    }

    CcuResult PullRotatePreSync(const CcuKernelArgP4Rotate3 &arg, FusedContext &ctx)
    {
        for (uint32_t i = 0; i < arg.channelCount; ++i) {
            CCU_CHK_RET(ccu::WriteVariableWithNotify(arg.channels[i], ctx.inputAddr, SCRATCH_ADDR_XN_ID,
                PARAM_SYNC_NOTIFY_INDEX, 1U << SCRATCH_ADDR_XN_ID));
            CCU_CHK_RET(ccu::WriteVariableWithNotify(arg.channels[i], ctx.inputToken, SCRATCH_TOKEN_XN_ID,
                PARAM_SYNC_NOTIFY_INDEX, 1U << SCRATCH_TOKEN_XN_ID));
            CCU_CHK_RET(ccu::WriteVariableWithNotify(
                arg.channels[i], ctx.outputAddr, OUTPUT_ADDR_XN_ID, PARAM_SYNC_NOTIFY_INDEX, 1U << OUTPUT_ADDR_XN_ID));
            CCU_CHK_RET(ccu::WriteVariableWithNotify(arg.channels[i], ctx.outputToken, OUTPUT_TOKEN_XN_ID,
                PARAM_SYNC_NOTIFY_INDEX, 1U << OUTPUT_TOKEN_XN_ID));
        }
        constexpr uint16_t allBits = (1U << SCRATCH_ADDR_XN_ID) | (1U << SCRATCH_TOKEN_XN_ID)
                                     | (1U << OUTPUT_ADDR_XN_ID) | (1U << OUTPUT_TOKEN_XN_ID);
        for (uint32_t i = 0; i < arg.channelCount; ++i) {
            CCU_CHK_RET(ccu::NotifyWait(arg.channels[i], PARAM_SYNC_NOTIFY_INDEX, allBits));
        }
        if (arg.isPrimary != 0) {
            CCU_IF(ctx.isInplace == 0)
            {
                CCU_CHK_RET(ccu::EventWait(ctx.dataEvent, 1));
            }
        }
        return KernelBarrier(arg.kernelIndex, arg.kernelCount, arg.isPrimary, 1, "rotate_pull_init_ready0",
            "rotate_pull_init_ready1", "rotate_pull_init_release0", "rotate_pull_init_release1");
    }

    CcuResult DoRotateLocalReduce(const CcuKernelArgP4Rotate3 &arg, FusedContext &ctx)
    {
        for (uint32_t shard = 0; shard < P4_ROTATE_SHARD_COUNT; ++shard) {
            uint64_t shardOffset = 0;
            uint64_t shardBytes = 0;
            GetEqualShard(arg.ownerBytes, P4_ROTATE_SHARD_COUNT, shard, shardOffset, shardBytes);
            ctx.localDestination.addr = ctx.scratchAddr;
            ctx.addressOffset = shardOffset;
            ctx.localDestination.addr += ctx.addressOffset;
            ctx.localDestination.token = ctx.scratchToken;
            ctx.localSource.addr = ctx.inputAddr;
            ctx.addressOffset = arg.ownerOffset + shardOffset;
            ctx.localSource.addr += ctx.addressOffset;
            ctx.localSource.token = ctx.inputToken;
            ctx.transferBytes = shardBytes;
            CCU_CHK_RET(ccu::LocalReduce(ctx.localDestination, ctx.localSource, ctx.transferBytes, arg.dataType,
                arg.reduceType, ctx.dataEvent, 1));
            CCU_CHK_RET(ccu::EventWait(ctx.dataEvent, 1));

            ctx.localSource.addr = ctx.scratchAddr;
            ctx.addressOffset = shardOffset;
            ctx.localSource.addr += ctx.addressOffset;
            ctx.localSource.token = ctx.scratchToken;
            ctx.localDestination.addr = ctx.outputAddr;
            ctx.addressOffset = arg.ownerOffset + shardOffset;
            ctx.localDestination.addr += ctx.addressOffset;
            ctx.localDestination.token = ctx.outputToken;
            CCU_CHK_RET(ccu::LocalCopy(ctx.localDestination, ctx.localSource, ctx.transferBytes, ctx.dataEvent, 1));
            CCU_CHK_RET(ccu::EventWait(ctx.dataEvent, 1));
        }
        return CCU_SUCCESS;
    }

    CcuResult RunP4OutputWave2(const CcuKernelArgP4Rotate3 &arg, FusedContext &ctx)
    {
        CCU_CHK_RET(StartWave2OutputInit(arg, ctx));
        CCU_CHK_RET(SendOutputAddress(arg, ctx));
        CCU_CHK_RET(FinishWave2ChunkAInit(arg, ctx));
        for (uint32_t phase = 0; phase < P4_ROTATE_SHARD_COUNT; ++phase) {
            CCU_CHK_RET(RunWave2FullSliceAccumulatePhase(arg, 0U, phase, ctx));
        }
        CCU_CHK_RET(WaitWave2FinalWriter(arg, 0U));

        CCU_CHK_RET(FinishWave2ChunkBInit(arg, ctx));
        for (uint32_t phase = 0; phase < P4_ROTATE_SHARD_COUNT; ++phase) {
            CCU_CHK_RET(RunWave2FullSliceAccumulatePhase(arg, 1U, phase, ctx));
        }
        CCU_CHK_RET(WaitWave2FinalWriter(arg, 1U));

        CCU_CHK_RET(KernelBarrier(arg.kernelIndex, arg.kernelCount, arg.isPrimary, 1U, "rotate_result_ready0",
            "rotate_result_ready1", "rotate_result_release0", "rotate_result_release1"));
        return BroadcastOwner(
            arg, arg.ownerOffset, arg.ownerBytes, ROTATE_UNIQUE_BROADCAST_NOTIFY_INDEX, ctx);
    }

    CcuResult RunP4Rotate3(const CcuKernelArgP4Rotate3 &arg, FusedContext &ctx)
    {
        if (arg.wave2Accumulate != 0) {
            return RunP4OutputWave2(arg, ctx);
        }
        if (arg.ownerPull != 0) {
            CCU_CHK_RET(StartRotateOutputInit(arg, ctx));
            CCU_CHK_RET(PullRotatePreSync(arg, ctx));
        } else if (arg.directOutputAccumulator != 0) {
            CCU_CHK_RET(StartRotateOutputInit(arg, ctx));
            CCU_CHK_RET(SendOutputAddress(arg, ctx));
            CCU_CHK_RET(FinishRotateOutputInit(arg, ctx));
        } else {
            CCU_CHK_RET(FusedPreSync(arg, ctx));
        }
        for (uint32_t phase = 0; phase < P4_ROTATE_SHARD_COUNT; ++phase) {
            if (arg.ownerPull != 0) {
                CCU_CHK_RET(RunPullRotatePhase(arg, phase, ctx));
            } else if (arg.fullSliceAccumulate != 0) {
                CCU_CHK_RET(RunFullSliceAccumulatePhase(arg, phase, ctx));
            } else {
                CCU_CHK_RET(RunRotatePhase(arg, phase, ctx));
            }
        }
        if (arg.fullSliceAccumulate != 0 && arg.phaseChain != 0) {
            const uint32_t finalWriterRank = (arg.rankId + 1U) % 4U;
            uint32_t finalWriterIndex = arg.channelCount;
            for (uint32_t i = 0; i < arg.channelCount; ++i) {
                if (arg.peerRanks[i] == finalWriterRank) {
                    finalWriterIndex = i;
                }
            }
            if (finalWriterIndex >= arg.channelCount) {
                return CCU_E_PARA;
            }
            CCU_CHK_RET(ccu::NotifyWait(
                arg.channels[finalWriterIndex], ROTATE_UNIQUE_PHASE_NOTIFY_BASE_INDEX + P4_ROTATE_SHARD_COUNT - 1U, 1));
        }
        if (arg.directOutputAccumulator == 0 && arg.isPrimary != 0) {
            CCU_CHK_RET(DoRotateLocalReduce(arg, ctx));
        }
        CCU_CHK_RET(KernelBarrier(arg.kernelIndex, arg.kernelCount, arg.isPrimary, 1, "rotate_result_ready0",
            "rotate_result_ready1", "rotate_result_release0", "rotate_result_release1"));
        const uint32_t broadcastNotifyIndex = arg.uniquePhaseNotify != 0 ? ROTATE_UNIQUE_BROADCAST_NOTIFY_INDEX
                                                                        : ROTATE_BROADCAST_NOTIFY_INDEX;
        return BroadcastOwner(arg, arg.ownerOffset, arg.ownerBytes, broadcastNotifyIndex, ctx);
    }

    CcuResult RunHierLocalDataPhase(const CcuKernelArgP12Hier8 &arg, uint32_t lane, uint32_t phase, FusedContext &ctx)
    {
        for (uint32_t ownedIndex = 0; ownedIndex < (arg.localSize == 4 ? 2U : 1U); ++ownedIndex) {
            uint16_t eventMask = 0;
            for (uint32_t i = 0; i < arg.channelCount; ++i) {
                if (arg.peerIsLocal[i] == 0 || arg.targetLane[i] != lane) {
                    continue;
                }
                const uint32_t shardId = arg.peerLocalIndex[i] + (ownedIndex == 0 ? 0U : 4U);
                const uint32_t shard = (arg.targetLanePosition[i] + phase) % arg.targetLaneSize[i];
                uint64_t shardOffset = 0;
                uint64_t shardBytes = 0;
                GetEqualShard(arg.shardBytes[shardId], arg.targetLaneSize[i], shard, shardOffset, shardBytes);
                const uint16_t bit = static_cast<uint16_t>(1U << i);
                ctx.localSource.addr = ctx.inputAddr;
                ctx.addressOffset = arg.shardOffset[shardId] + shardOffset;
                ctx.localSource.addr += ctx.addressOffset;
                ctx.localSource.token = ctx.inputToken;
                ctx.remoteDestination.addr = ctx.peerScratchAddr[i];
                ctx.addressOffset
                    = static_cast<uint64_t>(ownedIndex * HIER_LANE_SLOT_COUNT + lane) * arg.scratchStride + shardOffset;
                ctx.remoteDestination.addr += ctx.addressOffset;
                ctx.remoteDestination.token = ctx.peerScratchToken[i];
                ctx.transferBytes = shardBytes;
                if (phase == 0) {
                    CCU_CHK_RET(ccu::Write(arg.channels[i], ctx.remoteDestination, ctx.localSource, ctx.transferBytes,
                        ctx.dataEvent, bit));
                } else {
                    CCU_CHK_RET(ccu::WriteReduce(arg.channels[i], ctx.remoteDestination, ctx.localSource,
                        ctx.transferBytes, arg.dataType, arg.reduceType, ctx.dataEvent, bit));
                }
                eventMask = static_cast<uint16_t>(eventMask | bit);
            }
            if (eventMask != 0) {
                CCU_CHK_RET(ccu::EventWait(ctx.dataEvent, eventMask));
            }
        }

        constexpr uint16_t doneBit = 1;
        for (uint32_t i = 0; i < arg.channelCount; ++i) {
            if (arg.peerIsLocal[i] != 0 && arg.targetLane[i] == lane) {
                CCU_CHK_RET(ccu::NotifyRecord(arg.channels[i], HIER_LOCAL_DATA_NOTIFY_INDEX, doneBit));
            }
        }
        for (uint32_t i = 0; i < arg.channelCount; ++i) {
            if (arg.peerIsLocal[i] != 0 && arg.sourceLane[i] == lane) {
                CCU_CHK_RET(ccu::NotifyWait(arg.channels[i], HIER_LOCAL_DATA_NOTIFY_INDEX, doneBit));
            }
        }
        return CCU_SUCCESS;
    }

    CcuResult RunHierLocalReleasePhase(const CcuKernelArgP12Hier8 &arg, uint32_t lane)
    {
        constexpr uint16_t doneBit = 1;
        for (uint32_t i = 0; i < arg.channelCount; ++i) {
            if (arg.peerIsLocal[i] != 0 && arg.sourceLane[i] == lane) {
                CCU_CHK_RET(ccu::NotifyRecord(arg.channels[i], HIER_LOCAL_NEXT_NOTIFY_INDEX, doneBit));
            }
        }
        for (uint32_t i = 0; i < arg.channelCount; ++i) {
            if (arg.peerIsLocal[i] != 0 && arg.targetLane[i] == lane) {
                CCU_CHK_RET(ccu::NotifyWait(arg.channels[i], HIER_LOCAL_NEXT_NOTIFY_INDEX, doneBit));
            }
        }
        return CCU_SUCCESS;
    }

    CcuResult DoHierLocalReduce(const CcuKernelArgP12Hier8 &arg, FusedContext &ctx)
    {
        for (uint32_t ownedIndex = 0; ownedIndex < arg.ownedShardCount; ++ownedIndex) {
            const uint32_t shardId = arg.ownedShardId[ownedIndex];
            ctx.localDestination.addr = ctx.scratchAddr;
            ctx.addressOffset = static_cast<uint64_t>(ownedIndex * HIER_LANE_SLOT_COUNT) * arg.scratchStride;
            ctx.localDestination.addr += ctx.addressOffset;
            ctx.localDestination.token = ctx.scratchToken;
            ctx.transferBytes = arg.shardBytes[shardId];
            if (arg.localLaneCount == 2) {
                ctx.localSource.addr = ctx.scratchAddr;
                ctx.addressOffset = static_cast<uint64_t>(ownedIndex * HIER_LANE_SLOT_COUNT + 1U) * arg.scratchStride;
                ctx.localSource.addr += ctx.addressOffset;
                ctx.localSource.token = ctx.scratchToken;
                CCU_CHK_RET(ccu::LocalReduce(ctx.localDestination, ctx.localSource, ctx.transferBytes, arg.dataType,
                    arg.reduceType, ctx.dataEvent, 1));
                CCU_CHK_RET(ccu::EventWait(ctx.dataEvent, 1));
            }
            ctx.localSource.addr = ctx.inputAddr;
            ctx.addressOffset = arg.shardOffset[shardId];
            ctx.localSource.addr += ctx.addressOffset;
            ctx.localSource.token = ctx.inputToken;
            CCU_CHK_RET(ccu::LocalReduce(ctx.localDestination, ctx.localSource, ctx.transferBytes, arg.dataType,
                arg.reduceType, ctx.dataEvent, 1));
            CCU_CHK_RET(ccu::EventWait(ctx.dataEvent, 1));
        }
        return CCU_SUCCESS;
    }

    CcuResult CopyHierOwnedShardToOutput(const CcuKernelArgP12Hier8 &arg, uint32_t ownedIndex, FusedContext &ctx)
    {
        const uint32_t shardId = arg.ownedShardId[ownedIndex];
        ctx.localSource.addr = ctx.scratchAddr;
        ctx.addressOffset = static_cast<uint64_t>(ownedIndex * HIER_LANE_SLOT_COUNT) * arg.scratchStride;
        ctx.localSource.addr += ctx.addressOffset;
        ctx.localSource.token = ctx.scratchToken;
        ctx.localDestination.addr = ctx.outputAddr;
        ctx.addressOffset = arg.shardOffset[shardId];
        ctx.localDestination.addr += ctx.addressOffset;
        ctx.localDestination.token = ctx.outputToken;
        ctx.transferBytes = arg.shardBytes[shardId];
        CCU_CHK_RET(ccu::LocalCopy(ctx.localDestination, ctx.localSource, ctx.transferBytes, ctx.dataEvent, 1));
        CCU_CHK_RET(ccu::EventWait(ctx.dataEvent, 1));
        return CCU_SUCCESS;
    }

    CcuResult RunHierCross8Side(const CcuKernelArgP12Hier8 &arg, FusedContext &ctx)
    {
        constexpr uint16_t doneBit = 1;
        for (uint32_t i = 0; i < arg.channelCount; ++i) {
            if (arg.peerIsLocal[i] != 0) {
                continue;
            }
            CCU_CHK_RET(ccu::NotifyWait(arg.channels[i], HIER_CROSS_READY_NOTIFY_INDEX, doneBit));
            const uint32_t ownedIndex = arg.ownedShardId[0] >= 4 ? 1U : 0U;
            ctx.localSource.addr = ctx.scratchAddr;
            ctx.localSource.token = ctx.scratchToken;
            ctx.remoteDestination.addr = ctx.peerScratchAddr[i];
            ctx.addressOffset = static_cast<uint64_t>(ownedIndex * HIER_LANE_SLOT_COUNT) * arg.scratchStride;
            ctx.remoteDestination.addr += ctx.addressOffset;
            ctx.remoteDestination.token = ctx.peerScratchToken[i];
            ctx.transferBytes = arg.shardBytes[arg.ownedShardId[0]];
            CCU_CHK_RET(ccu::WriteReduce(arg.channels[i], ctx.remoteDestination, ctx.localSource, ctx.transferBytes,
                arg.dataType, arg.reduceType, ctx.dataEvent, 1));
            CCU_CHK_RET(ccu::EventWait(ctx.dataEvent, 1));
            CCU_CHK_RET(ccu::NotifyRecord(arg.channels[i], HIER_CROSS_DATA_NOTIFY_INDEX, doneBit));
            CCU_CHK_RET(ccu::NotifyWait(arg.channels[i], HIER_CROSS_RESULT_NOTIFY_INDEX, doneBit));
        }
        return CCU_SUCCESS;
    }

    CcuResult RunHierCross4Side(const CcuKernelArgP12Hier8 &arg, FusedContext &ctx)
    {
        constexpr uint16_t doneBit = 1;
        for (uint32_t i = 0; i < arg.channelCount; ++i) {
            if (arg.peerIsLocal[i] == 0) {
                CCU_CHK_RET(ccu::NotifyRecord(arg.channels[i], HIER_CROSS_READY_NOTIFY_INDEX, doneBit));
                CCU_CHK_RET(ccu::NotifyWait(arg.channels[i], HIER_CROSS_DATA_NOTIFY_INDEX, doneBit));
            }
        }
        for (uint32_t i = 0; i < arg.channelCount; ++i) {
            if (arg.peerIsLocal[i] != 0) {
                continue;
            }
            const uint32_t ownedIndex = arg.crossOwnedIndex[i];
            const uint32_t shardId = arg.ownedShardId[ownedIndex];
            CCU_CHK_RET(CopyHierOwnedShardToOutput(arg, ownedIndex, ctx));
            ctx.localSource.addr = ctx.outputAddr;
            ctx.addressOffset = arg.shardOffset[shardId];
            ctx.localSource.addr += ctx.addressOffset;
            ctx.localSource.token = ctx.outputToken;
            ctx.remoteDestination.addr = ctx.peerOutputAddr[i];
            ctx.addressOffset = arg.shardOffset[shardId];
            ctx.remoteDestination.addr += ctx.addressOffset;
            ctx.remoteDestination.token = ctx.peerOutputToken[i];
            ctx.transferBytes = arg.shardBytes[shardId];
            CCU_CHK_RET(ccu::Write(
                arg.channels[i], ctx.remoteDestination, ctx.localSource, ctx.transferBytes, ctx.dataEvent, 1));
            CCU_CHK_RET(ccu::EventWait(ctx.dataEvent, 1));
            CCU_CHK_RET(ccu::NotifyRecord(arg.channels[i], HIER_CROSS_RESULT_NOTIFY_INDEX, doneBit));
        }
        return CCU_SUCCESS;
    }

    CcuResult RunHierAllGather(const CcuKernelArgP12Hier8 &arg, FusedContext &ctx)
    {
        for (uint32_t ownedIndex = 0; ownedIndex < arg.ownedShardCount; ++ownedIndex) {
            const uint32_t shardId = arg.ownedShardId[ownedIndex];
            ctx.localSource.addr = ctx.outputAddr;
            ctx.addressOffset = arg.shardOffset[shardId];
            ctx.localSource.addr += ctx.addressOffset;
            ctx.localSource.token = ctx.outputToken;
            ctx.transferBytes = arg.shardBytes[shardId];
            uint16_t eventMask = 0;
            for (uint32_t i = 0; i < arg.channelCount; ++i) {
                if (arg.peerIsLocal[i] == 0) {
                    continue;
                }
                const uint16_t bit = static_cast<uint16_t>(1U << i);
                ctx.remoteDestination.addr = ctx.peerOutputAddr[i];
                ctx.addressOffset = arg.shardOffset[shardId];
                ctx.remoteDestination.addr += ctx.addressOffset;
                ctx.remoteDestination.token = ctx.peerOutputToken[i];
                CCU_CHK_RET(ccu::Write(
                    arg.channels[i], ctx.remoteDestination, ctx.localSource, ctx.transferBytes, ctx.dataEvent, bit));
                eventMask = static_cast<uint16_t>(eventMask | bit);
            }
            if (eventMask != 0) {
                CCU_CHK_RET(ccu::EventWait(ctx.dataEvent, eventMask));
            }
        }
        constexpr uint16_t doneBit = 1;
        for (uint32_t i = 0; i < arg.channelCount; ++i) {
            if (arg.peerIsLocal[i] != 0) {
                CCU_CHK_RET(ccu::NotifyRecord(arg.channels[i], HIER_LOCAL_DATA_NOTIFY_INDEX, doneBit));
            }
        }
        for (uint32_t i = 0; i < arg.channelCount; ++i) {
            if (arg.peerIsLocal[i] != 0) {
                CCU_CHK_RET(ccu::NotifyWait(arg.channels[i], HIER_LOCAL_DATA_NOTIFY_INDEX, doneBit));
            }
        }
        return CCU_SUCCESS;
    }

    CcuResult RunP12Hier8(const CcuKernelArgP12Hier8 &arg, FusedContext &ctx)
    {
        const uint32_t localPhaseCount = arg.localSize == 8 ? HIER_LOCAL_PHASE_COUNT : 3U;
        for (uint32_t logicalPhase = 0; logicalPhase < localPhaseCount; ++logicalPhase) {
            const uint32_t lane = arg.localSize == 8 && logicalPhase >= 4 ? 1U : 0U;
            const uint32_t lanePhase = lane == 0 ? logicalPhase : logicalPhase - 4U;
            CCU_IF(ctx.phase == static_cast<uint64_t>(2U * logicalPhase))
            {
                if (logicalPhase == 0) {
                    CCU_CHK_RET(FusedPreSync(arg, ctx));
                }
                CCU_CHK_RET(RunHierLocalDataPhase(arg, lane, lanePhase, ctx));
            }
            CCU_IF(ctx.phase == static_cast<uint64_t>(2U * logicalPhase + 1U))
            {
                CCU_CHK_RET(RunHierLocalReleasePhase(arg, lane));
            }
        }
        CCU_IF(ctx.phase == static_cast<uint64_t>(P12HierPhase::LOCAL_REDUCE))
        {
            if (arg.isPrimary != 0) {
                CCU_CHK_RET(DoHierLocalReduce(arg, ctx));
            }
        }
        CCU_IF(ctx.phase == static_cast<uint64_t>(P12HierPhase::CROSS_REDUCE))
        {
            CCU_CHK_RET(FusedPreSync(arg, ctx));
            if (arg.localSize == 8) {
                CCU_CHK_RET(RunHierCross8Side(arg, ctx));
            } else {
                CCU_CHK_RET(RunHierCross4Side(arg, ctx));
            }
        }
        CCU_IF(ctx.phase == static_cast<uint64_t>(P12HierPhase::LOCAL_ALLGATHER))
        {
            CCU_CHK_RET(FusedPreSync(arg, ctx));
            CCU_CHK_RET(RunHierAllGather(arg, ctx));
        }
        return CCU_SUCCESS;
    }
} // namespace

CcuResult CcuKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgOwnerRsag *>(arg);
    if (kernelArg == nullptr || kernelArg->channelCount == 0 || kernelArg->channelCount >= MAX_RANK_SIZE
        || kernelArg->kernelCount == 0 || kernelArg->kernelCount > 2) {
        HCCL_ERROR("[CcuKernel] invalid owner-RSAG registration argument");
        return CCU_E_PARA;
    }

    OwnerRsagContext ctx;
    ctx.arg = kernelArg;
    CCU_CHK_RET(InitResources(ctx));
    CCU_CHK_RET(LoadArgs(ctx));

    CCU_IF(ctx.phase == static_cast<uint64_t>(OwnerRsagPhase::PARTIAL_REDUCE))
    {
        CCU_CHK_RET(PreSync(ctx));
        CCU_IF(ctx.ownerBytes != 0)
        {
            CCU_CHK_RET(DoPartialReduce(ctx));
        }
        CCU_CHK_RET(PostSync(ctx, PARTIAL_DONE_NOTIFY_INDEX));
    }

    if (kernelArg->isPrimary != 0) {
        CCU_IF(ctx.phase == static_cast<uint64_t>(OwnerRsagPhase::COMBINE))
        {
            CCU_IF(ctx.ownerBytes != 0)
            {
                CCU_CHK_RET(FinalizeOwnerSlice(ctx));
            }
        }
    }

    CCU_IF(ctx.phase == static_cast<uint64_t>(OwnerRsagPhase::BROADCAST))
    {
        CCU_CHK_RET(PreSync(ctx));
        CCU_IF(ctx.ownerBytes != 0)
        {
            CCU_CHK_RET(BroadcastOwnerSlice(ctx));
        }
        CCU_CHK_RET(PostSync(ctx, BROADCAST_DONE_NOTIFY_INDEX));
    }

    return CCU_SUCCESS;
}

CcuResult CcuButterfly2x8Kernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgButterfly2x8 *>(arg);
    if (kernelArg == nullptr || kernelArg->channelCount == 0 || kernelArg->channelCount > 4
        || kernelArg->kernelCount == 0 || kernelArg->kernelCount > 2 || kernelArg->kernelIndex >= kernelArg->kernelCount
        || kernelArg->totalBytes == 0 || kernelArg->scratchStride < kernelArg->totalBytes
        || kernelArg->scratchAddr == 0) {
        HCCL_ERROR("[CcuButterfly2x8Kernel] invalid registration argument");
        return CCU_E_PARA;
    }
    for (uint32_t phase = 0; phase < 4; ++phase) {
        if (kernelArg->phaseOwnerKernel[phase] >= kernelArg->kernelCount) {
            HCCL_ERROR(
                "[CcuButterfly2x8Kernel] invalid owner %u for phase %u", kernelArg->phaseOwnerKernel[phase], phase);
            return CCU_E_PARA;
        }
    }
    HCCL_INFO("[CcuButterfly2x8Kernel] register rank=%u kernel=%u/%u die=%u channels=%u bytes=%lu", kernelArg->rankId,
        kernelArg->kernelIndex, kernelArg->kernelCount, kernelArg->actualDieId, kernelArg->channelCount,
        kernelArg->totalBytes);

    ButterflyContext ctx;
    ctx.arg = kernelArg;
    CCU_CHK_RET(InitButterfly(ctx));
    CCU_CHK_RET(ButterflyPreSync(ctx));
    for (uint32_t phase = 0; phase < 4; ++phase) {
        if (kernelArg->phaseOwnerKernel[phase] != kernelArg->kernelIndex) {
            continue;
        }
        bool foundChannel = false;
        for (uint32_t channelIndex = 0; channelIndex < kernelArg->channelCount; ++channelIndex) {
            if (kernelArg->phaseByChannel[channelIndex] != phase) {
                continue;
            }
            foundChannel = true;
            CCU_CHK_RET(ButterflyWaitPreviousKernel(*kernelArg, phase));
            CCU_CHK_RET(DoButterflyPhase(ctx, channelIndex, phase));
            CCU_CHK_RET(ButterflyRecordNextKernel(*kernelArg, phase));
            break;
        }
        if (!foundChannel) {
            HCCL_ERROR("[CcuButterfly2x8Kernel] phase %u is owned but has no channel", phase);
            return CCU_E_PARA;
        }
    }
    return CCU_SUCCESS;
}

CcuResult CcuCrossLane2x8Kernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgCrossLane2x8 *>(arg);
    const bool validLaneCount = kernelArg != nullptr
                                && ((kernelArg->meshLaneMode == 0 && kernelArg->localLaneCount >= 2
                                        && kernelArg->localLaneCount <= CROSS_LANE_LEGACY_MAX_COUNT)
                                    || (kernelArg->meshLaneMode == 1 && kernelArg->localLaneCount >= 4
                                        && kernelArg->localLaneCount <= MESH_LANE_MAX_COUNT));
    if (kernelArg == nullptr || kernelArg->channelCount == 0 || kernelArg->channelCount >= MAX_RANK_SIZE
        || kernelArg->kernelCount == 0 || kernelArg->kernelCount > 2 || !validLaneCount || kernelArg->meshLaneMode > 1
        || kernelArg->directOutputAccumulator > 1
        || (kernelArg->meshLaneMode != 0 && kernelArg->directOutputAccumulator != 0) || kernelArg->mode != 0
        || kernelArg->ownerBytes == 0
        || kernelArg->scratchStride < kernelArg->ownerBytes) {
        HCCL_ERROR("[CcuCrossLane2x8Kernel] invalid registration argument");
        return CCU_E_PARA;
    }
    HCCL_INFO("[CcuCrossLane2x8Kernel] register rank=%u die=%u channels=%u lanes=%u shardMask=0x%x", kernelArg->rankId,
        kernelArg->actualDieId, kernelArg->channelCount, kernelArg->localLaneCount, kernelArg->localShardMask);

    CrossLaneContext ctx;
    ctx.arg = kernelArg;
    CCU_CHK_RET(InitCrossLane(ctx));
    CCU_IF(ctx.phase == static_cast<uint64_t>(CrossLanePhase::REDUCE_SCATTER))
    {
        CCU_CHK_RET(DoCrossLaneReduceScatter(ctx));
    }
    CCU_IF(ctx.phase == static_cast<uint64_t>(CrossLanePhase::LOCAL_REDUCE))
    {
        CCU_CHK_RET(DoCrossLaneLocalReduceSerial(ctx));
    }
    CCU_IF(ctx.phase == static_cast<uint64_t>(CrossLanePhase::BROADCAST))
    {
        CCU_CHK_RET(DoCrossLaneBroadcast(ctx));
    }
    CCU_IF(ctx.phase == static_cast<uint64_t>(CrossLanePhase::MESH_LANE_0))
    {
        CCU_CHK_RET(DoMeshLanePhase(ctx, 0));
    }
    CCU_IF(ctx.phase == static_cast<uint64_t>(CrossLanePhase::MESH_LANE_1))
    {
        CCU_CHK_RET(DoMeshLanePhase(ctx, 1));
    }
    CCU_IF(ctx.phase == static_cast<uint64_t>(CrossLanePhase::MESH_LANE_2))
    {
        CCU_CHK_RET(DoMeshLanePhase(ctx, 2));
    }
    CCU_IF(ctx.phase == static_cast<uint64_t>(CrossLanePhase::MESH_LANE_3))
    {
        CCU_CHK_RET(DoMeshLanePhase(ctx, 3));
    }
    return CCU_SUCCESS;
}

CcuResult CcuCrossLaneFourLane512MKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgCrossLane2x8 *>(arg);
    if (kernelArg == nullptr || kernelArg->channelCount == 0 || kernelArg->channelCount >= MAX_RANK_SIZE
        || kernelArg->kernelCount == 0 || kernelArg->kernelCount > 2 || kernelArg->kernelIndex >= kernelArg->kernelCount
        || kernelArg->localLaneCount < 2 || kernelArg->localLaneCount > CROSS_LANE_MAX_COUNT
        || kernelArg->ownerBytes == 0 || kernelArg->scratchStride < kernelArg->ownerBytes
        || kernelArg->mode != CROSS_LANE_MODE_PARALLEL_LOCAL) {
        HCCL_ERROR("[CcuCrossLaneFourLane512MKernel] invalid registration argument");
        return CCU_E_PARA;
    }
    HCCL_INFO("[CcuCrossLaneFourLane512MKernel] register rank=%u die=%u channels=%u lanes=%u shardMask=0x%x mode=%u",
        kernelArg->rankId, kernelArg->actualDieId, kernelArg->channelCount, kernelArg->localLaneCount,
        kernelArg->localShardMask, kernelArg->mode);

    CrossLaneContext ctx;
    ctx.arg = kernelArg;
    CCU_CHK_RET(InitCrossLane(ctx));
    CCU_IF(ctx.phase == static_cast<uint64_t>(CrossLanePhase::REDUCE_SCATTER))
    {
        CCU_CHK_RET(DoCrossLaneReduceScatter(ctx));
    }
    CCU_IF(ctx.phase == static_cast<uint64_t>(CrossLanePhase::LOCAL_REDUCE))
    {
        CCU_CHK_RET(DoCrossLaneLocalReduceParallel(ctx));
    }
    CCU_IF(ctx.phase == static_cast<uint64_t>(CrossLanePhase::BROADCAST))
    {
        CCU_CHK_RET(DoCrossLaneBroadcast(ctx));
    }
    return CCU_SUCCESS;
}

CcuResult CcuDirectRsagKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgDirectRsag *>(arg);
    if (kernelArg == nullptr || kernelArg->channelCount == 0 || kernelArg->channelCount >= MAX_RANK_SIZE
        || (kernelArg->rankSize != 4 && kernelArg->rankSize != 12) || kernelArg->rankId >= kernelArg->rankSize
        || kernelArg->kernelCount == 0 || kernelArg->kernelCount > 2 || kernelArg->kernelIndex >= kernelArg->kernelCount
        || kernelArg->ownerBytes == 0 || kernelArg->scratchStride < kernelArg->ownerBytes
        || kernelArg->scratchAddr == 0 || kernelArg->preSyncOnce > 1 || kernelArg->earlyInit > 1) {
        HCCL_ERROR("[CcuDirectRsagKernel] invalid registration argument");
        return CCU_E_PARA;
    }
    HCCL_INFO("[CcuDirectRsagKernel] register rank=%u/%u kernel=%u/%u die=%u channels=%u preSyncOnce=%u "
              "earlyInit=%u",
        kernelArg->rankId, kernelArg->rankSize, kernelArg->kernelIndex, kernelArg->kernelCount,
        kernelArg->actualDieId, kernelArg->channelCount, kernelArg->preSyncOnce, kernelArg->earlyInit);
    FusedContext ctx;
    CCU_CHK_RET(InitFusedContext(*kernelArg, kernelArg->scratchAddr, kernelArg->scratchToken, ctx));
    return RunDirectRsag(*kernelArg, ctx);
}

CcuResult CcuP12AllPairsKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgP12AllPairs *>(arg);
    bool validSegments = kernelArg != nullptr;
    if (validSegments) {
        validSegments = kernelArg->segmentCount > 0 && kernelArg->segmentCount <= P12_ALLPAIRS_SEGMENT_COUNT;
        for (uint32_t segment = 0; validSegments && segment < kernelArg->segmentCount; ++segment) {
            validSegments = validSegments && kernelArg->segmentBytes[segment] != 0
                            && kernelArg->ownerBytes[segment] != 0
                            && kernelArg->ownerBytes[segment] < MAX_DATA_SIZE
                            && kernelArg->scratchStride >= kernelArg->ownerBytes[segment];
        }
    }
    if (!validSegments || kernelArg->channelCount == 0 || kernelArg->channelCount >= MAX_RANK_SIZE
        || kernelArg->rankSize != 12 || kernelArg->rankId >= kernelArg->rankSize || kernelArg->kernelCount == 0
        || kernelArg->kernelCount > 2 || kernelArg->kernelIndex >= kernelArg->kernelCount
        || kernelArg->balancedReduce > 1 || kernelArg->preSyncOnce > 1 || kernelArg->earlyInit > 1
        || kernelArg->closAccumulate > 1 || kernelArg->isClosOwner > 1 || kernelArg->scratchAddr == 0) {
        HCCL_ERROR("[CcuP12AllPairsKernel] invalid registration argument");
        return CCU_E_PARA;
    }
    if (kernelArg->balancedReduce != 0 || kernelArg->closAccumulate != 0) {
        bool seenRanks[MAX_RANK_SIZE]{};
        for (uint32_t i = 0; i < kernelArg->rankSize; ++i) {
            const uint32_t treeRank = kernelArg->treeRanks[i];
            if (treeRank >= kernelArg->rankSize || seenRanks[treeRank]) {
                HCCL_ERROR("[CcuP12AllPairsKernel] invalid balanced tree rank at index %u", i);
                return CCU_E_PARA;
            }
            seenRanks[treeRank] = true;
        }
    }
    for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
        if (kernelArg->peerIsLocal[i] > 1
            || (kernelArg->closAccumulate != 0 && kernelArg->peerIsLocal[i] != 0
                && kernelArg->targetSlot[i] >= CROSS_LANE_SAME_SLOT_COUNT)
            || (kernelArg->closAccumulate != 0 && kernelArg->peerIsLocal[i] == 0
                && kernelArg->peerClosPhase[i] >= P12_ALLPAIRS_A_RANK_COUNT)) {
            HCCL_ERROR("[CcuP12AllPairsKernel] invalid CLOS placement at channel %u", i);
            return CCU_E_PARA;
        }
        for (uint32_t segment = 0; segment < kernelArg->segmentCount; ++segment) {
            if (kernelArg->targetOwnerBytes[segment][i] == 0
                || kernelArg->targetOwnerBytes[segment][i] >= MAX_DATA_SIZE
                || kernelArg->scratchStride < kernelArg->targetOwnerBytes[segment][i]) {
                HCCL_ERROR("[CcuP12AllPairsKernel] invalid peer segment length");
                return CCU_E_PARA;
            }
        }
    }
    HCCL_INFO("[CcuP12AllPairsKernel] register rank=%u kernel=%u/%u die=%u channels=%u segments=%u stride=%lu "
              "balanced=%u preSyncOnce=%u earlyInit=%u closAccumulate=%u closOwner=%u",
        kernelArg->rankId, kernelArg->kernelIndex, kernelArg->kernelCount, kernelArg->actualDieId,
        kernelArg->channelCount, kernelArg->segmentCount, kernelArg->scratchStride, kernelArg->balancedReduce,
        kernelArg->preSyncOnce, kernelArg->earlyInit, kernelArg->closAccumulate, kernelArg->isClosOwner);
    FusedContext ctx;
    CCU_CHK_RET(InitFusedContext(*kernelArg, kernelArg->scratchAddr, kernelArg->scratchToken, ctx));
    CCU_CHK_RET(ccu::LoadArg(ctx.segment, P12_ALLPAIRS_SEGMENT_ARG_ID));
    return RunP12AllPairs(*kernelArg, ctx);
}

CcuResult CcuP12OutputTreeKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgP12OutputTree *>(arg);
    bool validSegments = kernelArg != nullptr;
    if (validSegments) {
        validSegments = kernelArg->segmentCount == 1U;
        for (uint32_t segment = 0; validSegments && segment < kernelArg->segmentCount; ++segment) {
            validSegments = kernelArg->segmentBytes[segment] != 0 && kernelArg->ownerBytes[segment] != 0
                            && kernelArg->ownerBytes[segment] < MAX_DATA_SIZE
                            && kernelArg->scratchStride >= kernelArg->ownerBytes[segment];
        }
    }
    if (!validSegments || kernelArg->channelCount == 0 || kernelArg->channelCount >= MAX_RANK_SIZE
        || kernelArg->rankSize != 12 || kernelArg->rankId >= kernelArg->rankSize || kernelArg->kernelCount != 2
        || kernelArg->kernelIndex >= kernelArg->kernelCount || kernelArg->balancedReduce != 1
        || kernelArg->preSyncOnce != 1 || kernelArg->compactOutputTree != 1 || kernelArg->dualGroupReduce != 1
        || kernelArg->rotatingReadReduce != 1 || kernelArg->groupOutputRoot > 1 || kernelArg->scratchAddr == 0) {
        HCCL_ERROR("[CcuP12OutputTreeKernel] invalid registration argument");
        return CCU_E_PARA;
    }
    const bool hasGroupRoot
        = std::find(kernelArg->peerRanks, kernelArg->peerRanks + kernelArg->channelCount, kernelArg->groupRootRank)
          != kernelArg->peerRanks + kernelArg->channelCount;
    if (!hasGroupRoot
        || (kernelArg->groupOutputRoot != 0 && kernelArg->groupRootRank != kernelArg->directOutputSourceRank)) {
        HCCL_ERROR("[CcuP12OutputTreeKernel] invalid dual group root");
        return CCU_E_PARA;
    }
    if (kernelArg->directOutputSourceRank >= kernelArg->rankSize
        || kernelArg->tempOutputSourceRank >= kernelArg->rankSize
        || kernelArg->directOutputSourceRank == kernelArg->rankId
        || kernelArg->tempOutputSourceRank == kernelArg->rankId
        || kernelArg->directOutputSourceRank == kernelArg->tempOutputSourceRank) {
        HCCL_ERROR("[CcuP12OutputTreeKernel] invalid output source ranks");
        return CCU_E_PARA;
    }
    bool seenRanks[MAX_RANK_SIZE]{};
    for (uint32_t i = 0; i < kernelArg->rankSize; ++i) {
        const uint32_t treeRank = kernelArg->treeRanks[i];
        if (treeRank >= kernelArg->rankSize || seenRanks[treeRank]) {
            HCCL_ERROR("[CcuP12OutputTreeKernel] invalid tree rank at index %u", i);
            return CCU_E_PARA;
        }
        seenRanks[treeRank] = true;
    }
    for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
        if (kernelArg->targetOwnerBytes[0][i] == 0 || kernelArg->targetOwnerBytes[0][i] >= MAX_DATA_SIZE
            || kernelArg->scratchStride < kernelArg->targetOwnerBytes[0][i]) {
            HCCL_ERROR("[CcuP12OutputTreeKernel] invalid peer segment length");
            return CCU_E_PARA;
        }
    }
    HCCL_INFO("[CcuP12OutputTreeKernel] register rank=%u kernel=%u/%u die=%u channels=%u stride=%lu outputRoot=%u",
        kernelArg->rankId, kernelArg->kernelIndex, kernelArg->kernelCount, kernelArg->actualDieId,
        kernelArg->channelCount, kernelArg->scratchStride, kernelArg->groupOutputRoot);
    FusedContext ctx;
    CCU_CHK_RET(v112_p12_detail::InitContext(*kernelArg, ctx));
    CCU_CHK_RET(ccu::LoadArg(ctx.segment, P12_ALLPAIRS_SEGMENT_ARG_ID));
    return v112_p12_detail::Run(*kernelArg, ctx);
}

CcuResult CcuP4Rotate3Kernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgP4Rotate3 *>(arg);
    if (kernelArg == nullptr || kernelArg->channelCount == 0 || kernelArg->channelCount > 3 || kernelArg->rankId >= 4
        || kernelArg->kernelCount == 0 || kernelArg->kernelCount > 2 || kernelArg->kernelIndex >= kernelArg->kernelCount
        || kernelArg->ownerBytes == 0 || kernelArg->scratchStride < kernelArg->ownerBytes || kernelArg->scratchAddr == 0
        || kernelArg->ownerPull > 1 || kernelArg->directOutputAccumulator > 1 || kernelArg->fullSliceAccumulate > 1
        || kernelArg->phaseChain > 1 || kernelArg->wave2Accumulate > 1
        || (kernelArg->ownerPull != 0 && kernelArg->directOutputAccumulator == 0)
        || (kernelArg->fullSliceAccumulate != 0
            && (kernelArg->directOutputAccumulator == 0 || kernelArg->ownerPull != 0))
        || (kernelArg->phaseChain != 0 && kernelArg->fullSliceAccumulate == 0)
        || (kernelArg->wave2Accumulate != 0
            && (kernelArg->kernelCount != 1 || kernelArg->isPrimary == 0 || kernelArg->directOutputAccumulator == 0
                || kernelArg->fullSliceAccumulate == 0 || kernelArg->phaseChain == 0
                || kernelArg->ownerPull != 0))) {
        HCCL_ERROR("[CcuP4Rotate3Kernel] invalid registration argument");
        return CCU_E_PARA;
    }
    HCCL_INFO("[CcuP4Rotate3Kernel] register rank=%u kernel=%u/%u die=%u channels=%u ownerBytes=%lu wave2=%u",
        kernelArg->rankId, kernelArg->kernelIndex, kernelArg->kernelCount, kernelArg->actualDieId,
        kernelArg->channelCount, kernelArg->ownerBytes, kernelArg->wave2Accumulate);
    FusedContext ctx;
    CCU_CHK_RET(InitFusedContext(*kernelArg, kernelArg->scratchAddr, kernelArg->scratchToken, ctx));
    return RunP4Rotate3(*kernelArg, ctx);
}

CcuResult CcuP12Hier8Kernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgP12Hier8 *>(arg);
    const bool validLocalShape
        = kernelArg != nullptr
          && ((kernelArg->localSize == 8 && kernelArg->localLaneCount == 2 && kernelArg->ownedShardCount == 1)
              || (kernelArg->localSize == 4 && kernelArg->localLaneCount == 1 && kernelArg->ownedShardCount == 2));
    if (!validLocalShape || kernelArg->channelCount == 0 || kernelArg->channelCount >= MAX_RANK_SIZE
        || kernelArg->rankId >= 12 || kernelArg->localIndex >= kernelArg->localSize || kernelArg->kernelCount == 0
        || kernelArg->kernelCount > 2 || kernelArg->kernelIndex >= kernelArg->kernelCount
        || kernelArg->scratchStride < kernelArg->shardBytes[kernelArg->ownedShardId[0]]
        || kernelArg->scratchAddr == 0) {
        HCCL_ERROR("[CcuP12Hier8Kernel] invalid registration argument");
        return CCU_E_PARA;
    }
    HCCL_INFO("[CcuP12Hier8Kernel] register rank=%u local=%u/%u kernel=%u/%u die=%u channels=%u owned=%u",
        kernelArg->rankId, kernelArg->localIndex, kernelArg->localSize, kernelArg->kernelIndex, kernelArg->kernelCount,
        kernelArg->actualDieId, kernelArg->channelCount, kernelArg->ownedShardCount);
    FusedContext ctx;
    CCU_CHK_RET(InitFusedContext(*kernelArg, kernelArg->scratchAddr, kernelArg->scratchToken, ctx));
    return RunP12Hier8(*kernelArg, ctx);
}

namespace v101_small_detail {

constexpr uint32_t XN_INPUT = 0U;
constexpr uint32_t XN_INPUT_TOKEN = 1U;
constexpr uint32_t XN_OUTPUT = 2U;
constexpr uint32_t XN_OUTPUT_TOKEN = 3U;
constexpr uint32_t CKE_PRE_SYNC = 0U;
constexpr uint16_t BIT_INPUT = 1U << 0U;
constexpr uint16_t BIT_INPUT_TOKEN = 1U << 1U;
constexpr uint16_t BIT_OUTPUT = 1U << 2U;
constexpr uint16_t BIT_OUTPUT_TOKEN = 1U << 3U;
constexpr uint16_t BIT_READY = 1U;
constexpr uint32_t MS_GROUP_SMALL_PARALLEL = 8U;
constexpr uint32_t MS_GROUP_R12_SMALL_PARALLEL = 10U;
constexpr uint32_t MS_GROUP_R12_ALIGNED_PARALLEL = 11U;
constexpr uint32_t MS_GROUP_INTERLEAVE = 8U;
constexpr uint32_t MS_TILE_BYTES = 4096U;
constexpr uint32_t R4_TILE_COUNT = 32U;
constexpr uint32_t R4_MS_PARALLEL = 16U;
constexpr uint32_t R4_INTERLEAVE = 4U;
constexpr uint64_t R4_WAVE_BYTES = static_cast<uint64_t>(R4_MS_PARALLEL) * MS_TILE_BYTES;

constexpr uint64_t SetBits(uint16_t width)
{
    return (uint64_t{1} << width) - uint64_t{1};
}

constexpr uint64_t EncodeLoopParam(uint64_t loopContext, uint64_t gsaOffset, uint64_t loopCount)
{
    return ((loopContext & SetBits(8U)) << 45U) | ((gsaOffset & SetBits(32U)) << 13U)
           | (loopCount & SetBits(13U));
}

constexpr uint64_t EncodeParallelParam(uint64_t repeatCount, uint64_t repeatLoopIndex, uint64_t totalLoopCount)
{
    return ((repeatCount & SetBits(7U)) << 55U) | ((repeatLoopIndex & SetBits(7U)) << 48U)
           | ((totalLoopCount & SetBits(7U)) << 41U);
}

constexpr uint64_t EncodeOffsetParam(uint64_t gsaOffset, uint64_t msOffset, uint64_t ckeOffset)
{
    return ((gsaOffset & SetBits(32U)) << 21U) | ((msOffset & SetBits(11U)) << 10U)
           | (ckeOffset & SetBits(10U));
}

uint16_t ChannelMask(uint32_t channelCount)
{
    return static_cast<uint16_t>((1U << channelCount) - 1U);
}

struct Context {
    const CcuKernelArgV101Small *arg{nullptr};
    ccu::Variable inputAddr;
    ccu::Variable outputAddr;
    ccu::Variable inputToken;
    ccu::Variable outputToken;
    ccu::Variable scratchAddr;
    ccu::Variable scratchToken;
    ccu::Variable stage;
    std::vector<ccu::Variable> remoteInput;
    std::vector<ccu::Variable> remoteInputToken;
    std::vector<ccu::Variable> remoteOutput;
    std::vector<ccu::Variable> remoteOutputToken;
    ccu::Event event;
    ccu::Event allGatherEvent;
};

CcuResult InitRemoteVariables(Context &ctx)
{
    const CcuKernelArgV101Small *arg = ctx.arg;
    if (arg->rankSize == 0U || arg->rankSize > V101_MAX_RANK_SIZE || arg->rankId >= arg->rankSize
        || arg->channelCount == 0U || arg->channelCount >= V101_MAX_RANK_SIZE) {
        return CCU_E_PARA;
    }
    ctx.remoteInput.resize(arg->rankSize);
    ctx.remoteInputToken.resize(arg->rankSize);
    ctx.remoteOutput.resize(arg->rankSize);
    ctx.remoteOutputToken.resize(arg->rankSize);
    for (uint32_t rank = 0; rank < arg->rankSize; ++rank) {
        if (rank == arg->rankId) {
            continue;
        }
        const uint32_t channelIndex = arg->channelIndexByRank[rank];
        if (channelIndex >= arg->channelCount) {
            continue;
        }
        ctx.remoteInput[rank] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIndex], XN_INPUT);
        ctx.remoteInputToken[rank]
            = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIndex], XN_INPUT_TOKEN);
        ctx.remoteOutput[rank] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIndex], XN_OUTPUT);
        ctx.remoteOutputToken[rank]
            = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIndex], XN_OUTPUT_TOKEN);
    }
    return CCU_SUCCESS;
}

CcuResult LoadCompactArguments(Context &ctx)
{
    uint32_t argIndex = 0U;
    CCU_CHK_RET(ccu::LoadArg(ctx.inputAddr, argIndex++));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputAddr, argIndex++));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputToken, argIndex++));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputToken, argIndex++));
    if (ctx.arg->directTwoDie != 0U) {
        CCU_CHK_RET(ccu::LoadArg(ctx.scratchAddr, argIndex++));
        CCU_CHK_RET(ccu::LoadArg(ctx.scratchToken, argIndex++));
        CCU_CHK_RET(ccu::LoadArg(ctx.stage, argIndex++));
    }
    return CCU_SUCCESS;
}

CcuResult ExchangeAddresses(Context &ctx, bool shareInputOutputToken)
{
    const CcuKernelArgV101Small *arg = ctx.arg;
    ctx.remoteInput[arg->rankId] = ctx.inputAddr;
    ctx.remoteInputToken[arg->rankId] = ctx.inputToken;
    ctx.remoteOutput[arg->rankId] = ctx.outputAddr;
    ctx.remoteOutputToken[arg->rankId] = ctx.outputToken;
    uint16_t syncMask = BIT_INPUT | BIT_INPUT_TOKEN | BIT_OUTPUT;
    if (!shareInputOutputToken) {
        syncMask = static_cast<uint16_t>(syncMask | BIT_OUTPUT_TOKEN);
    }
    for (uint32_t channelIndex = 0; channelIndex < arg->channelCount; ++channelIndex) {
        CCU_CHK_RET(ccu::WriteVariableWithNotify(
            arg->channels[channelIndex], ctx.inputAddr, XN_INPUT, CKE_PRE_SYNC, BIT_INPUT));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(
            arg->channels[channelIndex], ctx.inputToken, XN_INPUT_TOKEN, CKE_PRE_SYNC, BIT_INPUT_TOKEN));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(
            arg->channels[channelIndex], ctx.outputAddr, XN_OUTPUT, CKE_PRE_SYNC, BIT_OUTPUT));
        if (!shareInputOutputToken) {
            CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[channelIndex], ctx.outputToken, XN_OUTPUT_TOKEN,
                CKE_PRE_SYNC, BIT_OUTPUT_TOKEN));
        }
    }
    for (uint32_t channelIndex = 0; channelIndex < arg->channelCount; ++channelIndex) {
        CCU_CHK_RET(ccu::NotifyWait(arg->channels[channelIndex], CKE_PRE_SYNC, syncMask));
    }
    return CCU_SUCCESS;
}

CcuResult ValidateTwoDie(const CcuKernelArgV101Small *arg)
{
    if (arg->channelCount == 0U || arg->directSliceBytes == 0U || arg->directTileCount != 1U
        || arg->directTileBytes != arg->directSliceBytes || arg->directMergeBytes == 0U) {
        return CCU_E_PARA;
    }
    return CCU_SUCCESS;
}

CcuResult RunTwoDiePartialMs(Context &ctx)
{
    const CcuKernelArgV101Small *arg = ctx.arg;
    CCU_CHK_RET(ValidateTwoDie(arg));
    if (arg->directUseMsGroup == 0U) {
        return CCU_E_PARA;
    }
    const uint32_t msParallel = arg->directMsParallel;
    if (msParallel != MS_GROUP_SMALL_PARALLEL && msParallel != MS_GROUP_R12_SMALL_PARALLEL
        && msParallel != MS_GROUP_R12_ALIGNED_PARALLEL) {
        return CCU_E_PARA;
    }
    const uint64_t msWaveBytes = static_cast<uint64_t>(msParallel) * MS_TILE_BYTES;
    if (arg->directSliceBytes != msWaveBytes) {
        return CCU_E_PARA;
    }

    ccu::Variable inputOffset;
    inputOffset = arg->directSliceOffset;
    ccu::LocalAddr localInput;
    localInput.addr = ctx.inputAddr;
    localInput.addr += inputOffset;
    localInput.token = ctx.inputToken;

    ccu::LocalAddr partial;
    const bool partialZeroInOutput = arg->directPartialZeroInOutput != 0U && arg->directScratchOffset == 0U;
    if (partialZeroInOutput) {
        partial.addr = ctx.outputAddr;
        partial.addr += inputOffset;
        partial.token = ctx.outputToken;
    } else {
        partial.addr = ctx.scratchAddr;
        ccu::Variable partialOffset;
        partialOffset = arg->directScratchOffset;
        partial.addr += partialOffset;
        partial.token = ctx.scratchToken;
    }

    std::vector<ccu::RemoteAddr> remoteInputs;
    remoteInputs.reserve(arg->channelCount);
    for (uint32_t sourceRank = 0; sourceRank < arg->rankSize; ++sourceRank) {
        if (sourceRank == arg->rankId) {
            continue;
        }
        const uint32_t channelIndex = arg->channelIndexByRank[sourceRank];
        if (channelIndex >= arg->channelCount) {
            continue;
        }
        ccu::RemoteAddr remoteInput;
        remoteInput.addr = ctx.remoteInput[sourceRank];
        remoteInput.addr += inputOffset;
        remoteInput.token = ctx.remoteInputToken[sourceRank];
        remoteInputs.push_back(remoteInput);
    }
    const uint32_t sourceCount = static_cast<uint32_t>(remoteInputs.size()) + arg->directIncludeLocalInput;
    if (remoteInputs.size() != arg->channelCount || sourceCount == 0U || sourceCount > MS_GROUP_INTERLEAVE) {
        return CCU_E_PARA;
    }

    ccu::Array<ccu::Event> completedEvents(msParallel);
    ccu::Array<ccu::CcuBuffer> ccuBuffers(msParallel * MS_GROUP_INTERLEAVE);
    ccu::Variable tileBytes;
    tileBytes = MS_TILE_BYTES;
    const uint16_t sourceMask = static_cast<uint16_t>((1U << sourceCount) - 1U);
    ccu::Func partialBody([&]() {
        uint32_t sourceIndex = 0U;
        for (const ccu::RemoteAddr &remoteInput : remoteInputs) {
            (void)ccu::Read(arg->channels[sourceIndex], ccuBuffers[sourceIndex], remoteInput, tileBytes,
                completedEvents[0], static_cast<uint16_t>(1U << sourceIndex));
            ++sourceIndex;
        }
        if (arg->directIncludeLocalInput != 0U) {
            (void)ccu::LocalCopy(ccuBuffers[sourceIndex], localInput, tileBytes, completedEvents[0],
                static_cast<uint16_t>(1U << sourceIndex));
        }
        (void)ccu::EventWait(completedEvents[0], sourceMask);
        if (sourceCount > 1U) {
            (void)ccu::LocalReduce(ccuBuffers.data(), sourceCount, arg->dataType, arg->dataType, arg->reduceType,
                tileBytes, completedEvents[0], BIT_READY);
            (void)ccu::EventWait(completedEvents[0], BIT_READY);
        }
        (void)ccu::LocalCopy(partial, ccuBuffers[0], tileBytes, completedEvents[0], BIT_READY);
        (void)ccu::EventWait(completedEvents[0], BIT_READY);
    });

    ccu::Variable loopParam;
    ccu::Loop loop(loopParam, partialBody);
    loopParam = EncodeLoopParam(0U, msWaveBytes, 1U);
    ccu::Variable parallelParam;
    parallelParam = EncodeParallelParam(msParallel - 1U, 0U, 1U);
    ccu::Variable offsetParam;
    offsetParam = EncodeOffsetParam(MS_TILE_BYTES, MS_GROUP_INTERLEAVE, 1U);
    std::vector<ccu::Loop> loops{loop};
    ccu::LoopGroup group(parallelParam, offsetParam, msParallel, loops);
    CCU_CHK_RET(ccu::EventRecord(completedEvents[0], BIT_READY));
    return ccu::EventWait(completedEvents[0], BIT_READY);
}

CcuResult RunTwoDiePartialSerial(Context &ctx)
{
    const CcuKernelArgV101Small *arg = ctx.arg;
    CCU_CHK_RET(ValidateTwoDie(arg));
    ccu::Variable currentBytes;
    currentBytes = arg->directSliceBytes;
    ccu::Variable inputOffset;
    inputOffset = arg->directSliceOffset;

    ccu::LocalAddr localInput;
    localInput.addr = ctx.inputAddr;
    localInput.addr += inputOffset;
    localInput.token = ctx.inputToken;

    ccu::LocalAddr partial;
    partial.addr = ctx.scratchAddr;
    ccu::Variable partialOffset;
    partialOffset = arg->directScratchOffset;
    partial.addr += partialOffset;
    partial.token = ctx.scratchToken;

    bool partialInitialized = false;
    if (arg->directIncludeLocalInput != 0U) {
        CCU_CHK_RET(ccu::LocalCopy(partial, localInput, currentBytes, ctx.event, BIT_READY));
        CCU_CHK_RET(ccu::EventWait(ctx.event, BIT_READY));
        partialInitialized = true;
    }
    for (uint32_t sourceRank = 0; sourceRank < arg->rankSize; ++sourceRank) {
        if (sourceRank == arg->rankId) {
            continue;
        }
        const uint32_t channelIndex = arg->channelIndexByRank[sourceRank];
        if (channelIndex >= arg->channelCount) {
            continue;
        }
        ccu::RemoteAddr remoteInput;
        remoteInput.addr = ctx.remoteInput[sourceRank];
        remoteInput.addr += inputOffset;
        remoteInput.token = ctx.remoteInputToken[sourceRank];
        if (!partialInitialized) {
            CCU_CHK_RET(
                ccu::Read(arg->channels[channelIndex], partial, remoteInput, currentBytes, ctx.event, BIT_READY));
            partialInitialized = true;
        } else {
            CCU_CHK_RET(ccu::ReadReduce(arg->channels[channelIndex], partial, remoteInput, currentBytes,
                arg->dataType, arg->reduceType, ctx.event, BIT_READY));
        }
        CCU_CHK_RET(ccu::EventWait(ctx.event, BIT_READY));
    }
    return partialInitialized ? CCU_SUCCESS : CCU_E_PARA;
}

CcuResult RunTwoDieMerge(Context &ctx)
{
    const CcuKernelArgV101Small *arg = ctx.arg;
    CCU_CHK_RET(ValidateTwoDie(arg));
    ccu::Variable mergeBytes;
    mergeBytes = arg->directMergeBytes;
    ccu::Variable mergeOffset;
    mergeOffset = arg->directSliceOffset + arg->directMergeOffset;
    ccu::LocalAddr mergeOutput;
    mergeOutput.addr = ctx.outputAddr;
    mergeOutput.addr += mergeOffset;
    mergeOutput.token = ctx.outputToken;
    ccu::LocalAddr partialOne;
    partialOne.addr = ctx.scratchAddr;
    ccu::Variable partialOneOffset;
    partialOneOffset = arg->directTileBytes + arg->directMergeOffset;
    partialOne.addr += partialOneOffset;
    partialOne.token = ctx.scratchToken;
    if (arg->directPartialZeroInOutput == 0U) {
        ccu::LocalAddr partialZero;
        partialZero.addr = ctx.scratchAddr;
        partialZero.token = ctx.scratchToken;
        ccu::Variable partialMergeOffset;
        partialMergeOffset = arg->directMergeOffset;
        partialZero.addr += partialMergeOffset;
        CCU_CHK_RET(ccu::LocalCopy(mergeOutput, partialZero, mergeBytes, ctx.event, BIT_READY));
        CCU_CHK_RET(ccu::EventWait(ctx.event, BIT_READY));
    }
    CCU_CHK_RET(
        ccu::LocalReduce(mergeOutput, partialOne, mergeBytes, arg->dataType, arg->reduceType, ctx.event, BIT_READY));
    return ccu::EventWait(ctx.event, BIT_READY);
}

CcuResult RunTwoDieReplicatedMerge(Context &ctx)
{
    const CcuKernelArgV101Small *arg = ctx.arg;
    CCU_CHK_RET(ValidateTwoDie(arg));
    if (arg->directReplicatedMerge == 0U) {
        return CCU_E_PARA;
    }

    ccu::Variable mergeBytes;
    mergeBytes = arg->directSliceBytes;
    ccu::Variable outputOffset;
    outputOffset = arg->directSliceOffset;

    ccu::LocalAddr partialZero;
    if (arg->directPartialZeroInOutput != 0U) {
        partialZero.addr = ctx.outputAddr;
        partialZero.addr += outputOffset;
        partialZero.token = ctx.outputToken;
    } else {
        partialZero.addr = ctx.scratchAddr;
        partialZero.token = ctx.scratchToken;
    }

    ccu::LocalAddr partialOne;
    partialOne.addr = ctx.scratchAddr;
    ccu::Variable partialOneOffset;
    partialOneOffset = arg->directTileBytes;
    partialOne.addr += partialOneOffset;
    partialOne.token = ctx.scratchToken;

    const bool primaryDie = arg->directScratchOffset == 0U;
    ccu::LocalAddr result;
    if (primaryDie) {
        result.addr = ctx.outputAddr;
        result.addr += outputOffset;
        result.token = ctx.outputToken;
    } else {
        result.addr = ctx.scratchAddr;
        ccu::Variable replicaOffset;
        replicaOffset = 2U * arg->directTileBytes;
        result.addr += replicaOffset;
        result.token = ctx.scratchToken;
    }

    if (!(primaryDie && arg->directPartialZeroInOutput != 0U)) {
        CCU_CHK_RET(ccu::LocalCopy(result, partialZero, mergeBytes, ctx.event, BIT_READY));
        CCU_CHK_RET(ccu::EventWait(ctx.event, BIT_READY));
    }
    CCU_CHK_RET(
        ccu::LocalReduce(result, partialOne, mergeBytes, arg->dataType, arg->reduceType, ctx.event, BIT_READY));
    return ccu::EventWait(ctx.event, BIT_READY);
}

CcuResult RunTwoDieAllGather(Context &ctx)
{
    const CcuKernelArgV101Small *arg = ctx.arg;
    CCU_CHK_RET(ValidateTwoDie(arg));
    ccu::Variable currentBytes;
    currentBytes = arg->directSliceBytes;
    ccu::Variable inputOffset;
    inputOffset = arg->directSliceOffset;
    ccu::LocalAddr localOutput;
    if (arg->directReplicatedMerge != 0U && arg->directScratchOffset != 0U) {
        localOutput.addr = ctx.scratchAddr;
        ccu::Variable replicaOffset;
        replicaOffset = 2U * arg->directTileBytes;
        localOutput.addr += replicaOffset;
        localOutput.token = ctx.scratchToken;
    } else {
        localOutput.addr = ctx.outputAddr;
        localOutput.addr += inputOffset;
        localOutput.token = ctx.outputToken;
    }
    const uint16_t peerMask = ChannelMask(arg->channelCount);
    for (uint32_t remoteRank = 0; remoteRank < arg->rankSize; ++remoteRank) {
        if (remoteRank == arg->rankId) {
            continue;
        }
        const uint32_t channelIndex = arg->channelIndexByRank[remoteRank];
        if (channelIndex >= arg->channelCount) {
            continue;
        }
        ccu::RemoteAddr remoteOutput;
        remoteOutput.addr = ctx.remoteOutput[remoteRank];
        remoteOutput.addr += inputOffset;
        remoteOutput.token = ctx.remoteOutputToken[remoteRank];
        const uint16_t channelBit = static_cast<uint16_t>(1U << channelIndex);
        CCU_CHK_RET(ccu::Write(arg->channels[channelIndex], remoteOutput, localOutput, currentBytes,
            ctx.allGatherEvent, channelBit));
    }
    return ccu::EventWait(ctx.allGatherEvent, peerMask);
}

CcuResult RunR4MsDirect(Context &ctx)
{
    const CcuKernelArgV101Small *arg = ctx.arg;
    if (arg->directUseR4MsDirect == 0U || arg->rankSize != 4U || arg->channelCount != 3U
        || arg->directSliceBytes != static_cast<uint64_t>(R4_TILE_COUNT) * MS_TILE_BYTES) {
        return CCU_E_PARA;
    }
    ccu::LocalAddr localInput;
    localInput.addr = ctx.inputAddr;
    ccu::Variable sliceOffset;
    sliceOffset = arg->directSliceOffset;
    localInput.addr += sliceOffset;
    localInput.token = ctx.inputToken;

    ccu::LocalAddr localOutput;
    localOutput.addr = ctx.outputAddr;
    localOutput.addr += sliceOffset;
    localOutput.token = ctx.outputToken;

    std::vector<ccu::RemoteAddr> remoteInputs;
    std::vector<ccu::RemoteAddr> remoteOutputs;
    std::vector<uint32_t> remoteChannels;
    remoteInputs.reserve(arg->channelCount);
    remoteOutputs.reserve(arg->channelCount);
    remoteChannels.reserve(arg->channelCount);
    for (uint32_t remoteRank = 0; remoteRank < arg->rankSize; ++remoteRank) {
        if (remoteRank == arg->rankId) {
            continue;
        }
        const uint32_t channelIndex = arg->channelIndexByRank[remoteRank];
        if (channelIndex >= arg->channelCount) {
            return CCU_E_PARA;
        }
        ccu::RemoteAddr remoteInput;
        remoteInput.addr = ctx.remoteInput[remoteRank];
        remoteInput.addr += sliceOffset;
        remoteInput.token = ctx.remoteInputToken[remoteRank];
        remoteInputs.push_back(remoteInput);
        ccu::RemoteAddr remoteOutput;
        remoteOutput.addr = ctx.remoteOutput[remoteRank];
        remoteOutput.addr += sliceOffset;
        remoteOutput.token = ctx.remoteInputToken[remoteRank];
        remoteOutputs.push_back(remoteOutput);
        remoteChannels.push_back(channelIndex);
    }
    if (remoteInputs.size() != arg->channelCount || remoteOutputs.size() != arg->channelCount) {
        return CCU_E_PARA;
    }

    constexpr uint32_t bankStride = R4_MS_PARALLEL;
    constexpr uint32_t bufferCount = 5U * bankStride;
    ccu::Array<ccu::Event> tileEvents(R4_MS_PARALLEL);
    ccu::Array<ccu::CcuBuffer> tileBuffers(bufferCount);
    ccu::Variable tileBytes;
    tileBytes = MS_TILE_BYTES;
    constexpr uint16_t bankASourceMask = 0x000FU;
    constexpr uint16_t bankBSourceMask = 0x00F0U;
    constexpr uint16_t bothFanoutMask = 0x00FFU;

    ccu::Variable bankOffset;
    bankOffset = R4_WAVE_BYTES;
    ccu::LocalAddr bankBLocalInput;
    bankBLocalInput.addr = ctx.inputAddr;
    bankBLocalInput.addr += sliceOffset;
    bankBLocalInput.addr += bankOffset;
    bankBLocalInput.token = ctx.inputToken;
    ccu::LocalAddr bankBLocalOutput;
    bankBLocalOutput.addr = ctx.outputAddr;
    bankBLocalOutput.addr += sliceOffset;
    bankBLocalOutput.addr += bankOffset;
    bankBLocalOutput.token = ctx.outputToken;

    std::vector<ccu::RemoteAddr> bankBRemoteInputs;
    std::vector<ccu::RemoteAddr> bankBRemoteOutputs;
    bankBRemoteInputs.reserve(arg->channelCount);
    bankBRemoteOutputs.reserve(arg->channelCount);
    for (uint32_t remoteRank = 0; remoteRank < arg->rankSize; ++remoteRank) {
        if (remoteRank == arg->rankId) {
            continue;
        }
        ccu::RemoteAddr remoteInput;
        remoteInput.addr = ctx.remoteInput[remoteRank];
        remoteInput.addr += sliceOffset;
        remoteInput.addr += bankOffset;
        remoteInput.token = ctx.remoteInputToken[remoteRank];
        bankBRemoteInputs.push_back(remoteInput);

        ccu::RemoteAddr remoteOutput;
        remoteOutput.addr = ctx.remoteOutput[remoteRank];
        remoteOutput.addr += sliceOffset;
        remoteOutput.addr += bankOffset;
        remoteOutput.token = ctx.remoteInputToken[remoteRank];
        bankBRemoteOutputs.push_back(remoteOutput);
    }

    ccu::CcuBuffer bankABuffers[R4_INTERLEAVE] = {tileBuffers[0U], tileBuffers[bankStride],
        tileBuffers[2U * bankStride], tileBuffers[3U * bankStride]};
    ccu::CcuBuffer bankBBuffers[R4_INTERLEAVE] = {tileBuffers[bankStride], tileBuffers[2U * bankStride],
        tileBuffers[3U * bankStride], tileBuffers[4U * bankStride]};

    ccu::Func tileBody([&]() {
        uint32_t sourceIndex = 0U;
        for (uint32_t remoteIndex = 0; remoteIndex < remoteInputs.size(); ++remoteIndex) {
            (void)ccu::Read(arg->channels[remoteChannels[remoteIndex]], bankABuffers[sourceIndex],
                remoteInputs[remoteIndex], tileBytes, tileEvents[0], static_cast<uint16_t>(1U << sourceIndex));
            ++sourceIndex;
        }
        (void)ccu::LocalCopy(bankABuffers[sourceIndex], localInput, tileBytes, tileEvents[0],
            static_cast<uint16_t>(1U << sourceIndex));
        (void)ccu::EventWait(tileEvents[0], bankASourceMask);
        (void)ccu::LocalReduce(bankABuffers, R4_INTERLEAVE, arg->dataType, arg->dataType, arg->reduceType,
            tileBytes, tileEvents[0], BIT_READY);
        (void)ccu::EventWait(tileEvents[0], BIT_READY);

        (void)ccu::LocalCopy(localOutput, bankABuffers[0], tileBytes, tileEvents[0], BIT_READY);
        for (uint32_t remoteIndex = 0; remoteIndex < remoteOutputs.size(); ++remoteIndex) {
            (void)ccu::Write(arg->channels[remoteChannels[remoteIndex]], remoteOutputs[remoteIndex], bankABuffers[0],
                tileBytes, tileEvents[0], static_cast<uint16_t>(1U << (remoteIndex + 1U)));
        }

        sourceIndex = 0U;
        for (uint32_t remoteIndex = 0; remoteIndex < bankBRemoteInputs.size(); ++remoteIndex) {
            (void)ccu::Read(arg->channels[remoteChannels[remoteIndex]], bankBBuffers[sourceIndex],
                bankBRemoteInputs[remoteIndex], tileBytes, tileEvents[0],
                static_cast<uint16_t>(1U << (sourceIndex + 4U)));
            ++sourceIndex;
        }
        (void)ccu::LocalCopy(bankBBuffers[sourceIndex], bankBLocalInput, tileBytes, tileEvents[0],
            static_cast<uint16_t>(1U << (sourceIndex + 4U)));
        (void)ccu::EventWait(tileEvents[0], bankBSourceMask);
        (void)ccu::LocalReduce(bankBBuffers, R4_INTERLEAVE, arg->dataType, arg->dataType, arg->reduceType, tileBytes,
            tileEvents[0], static_cast<uint16_t>(1U << 4U));
        (void)ccu::EventWait(tileEvents[0], static_cast<uint16_t>(1U << 4U));

        (void)ccu::LocalCopy(bankBLocalOutput, bankBBuffers[0], tileBytes, tileEvents[0],
            static_cast<uint16_t>(1U << 4U));
        for (uint32_t remoteIndex = 0; remoteIndex < bankBRemoteOutputs.size(); ++remoteIndex) {
            (void)ccu::Write(arg->channels[remoteChannels[remoteIndex]], bankBRemoteOutputs[remoteIndex],
                bankBBuffers[0], tileBytes, tileEvents[0], static_cast<uint16_t>(1U << (remoteIndex + 5U)));
        }
        (void)ccu::EventWait(tileEvents[0], bothFanoutMask);
    });

    ccu::Variable loopParam;
    ccu::Loop loop(loopParam, tileBody);
    loopParam = EncodeLoopParam(0U, R4_WAVE_BYTES * 2U, 1U);
    ccu::Variable parallelParam;
    parallelParam = EncodeParallelParam(R4_MS_PARALLEL - 1U, 0U, 1U);
    ccu::Variable offsetParam;
    offsetParam = EncodeOffsetParam(MS_TILE_BYTES, 1U, 1U);
    std::vector<ccu::Loop> loops{loop};
    ccu::LoopGroup group(parallelParam, offsetParam, R4_MS_PARALLEL, loops);
    CCU_CHK_RET(ccu::EventRecord(tileEvents[0], BIT_READY));
    return ccu::EventWait(tileEvents[0], BIT_READY);
}

} // namespace v101_small_detail

CcuResult CcuV101SmallKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgV101Small *>(arg);
    if (kernelArg == nullptr || kernelArg->directUseCompactArgs == 0U) {
        return CCU_E_PARA;
    }
    v101_small_detail::Context ctx;
    ctx.arg = kernelArg;
    CCU_CHK_RET(v101_small_detail::InitRemoteVariables(ctx));
    CCU_CHK_RET(v101_small_detail::LoadCompactArguments(ctx));
    if (kernelArg->directTwoDie == 0U) {
        CCU_CHK_RET(v101_small_detail::ExchangeAddresses(ctx, true));
        return v101_small_detail::RunR4MsDirect(ctx);
    }
    CCU_IF(ctx.stage == static_cast<uint64_t>(V101DirectStage::PARTIAL))
    {
        CCU_CHK_RET(v101_small_detail::ExchangeAddresses(ctx, false));
        if (kernelArg->directUseMsGroup != 0U) {
            CCU_CHK_RET(v101_small_detail::RunTwoDiePartialMs(ctx));
        } else {
            CCU_CHK_RET(v101_small_detail::RunTwoDiePartialSerial(ctx));
        }
    }
    CCU_IF(ctx.stage == static_cast<uint64_t>(V101DirectStage::MERGE))
    {
        if (kernelArg->directReplicatedMerge != 0U) {
            CCU_CHK_RET(v101_small_detail::RunTwoDieReplicatedMerge(ctx));
            CCU_CHK_RET(v101_small_detail::RunTwoDieAllGather(ctx));
        } else {
            CCU_CHK_RET(v101_small_detail::RunTwoDieMerge(ctx));
        }
    }
    CCU_IF(ctx.stage == static_cast<uint64_t>(V101DirectStage::ALLGATHER))
    {
        CCU_CHK_RET(v101_small_detail::RunTwoDieAllGather(ctx));
    }
    return CCU_SUCCESS;
}

} // namespace ops_hccl
