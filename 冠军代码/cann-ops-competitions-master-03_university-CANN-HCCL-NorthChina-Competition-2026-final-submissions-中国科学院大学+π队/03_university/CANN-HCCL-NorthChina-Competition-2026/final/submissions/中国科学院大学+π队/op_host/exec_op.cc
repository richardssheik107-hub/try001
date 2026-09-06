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
#include <array>
#include <cstddef>
#include <limits>

#include <ccu/ccu_res.h>

#include "ccu_kernel.h"
#include "ccu_launch.h"
#include "custom.h"
#include "exec_op.h"
#include "log.h"

namespace ops_hccl {
namespace {

constexpr uint64_t HYBRID_ALIGNMENT = 32ULL * 1024ULL;
constexpr std::size_t DIRECT_ARG_COUNT = 12;
constexpr std::size_t DIRECT_NO_COPY_ARG_COUNT = 8;
constexpr std::size_t SMALL_DIRECT_ARG_COUNT = 9;
constexpr std::size_t FOUR_X_ONE_SMALL_ARG_COUNT = 5;
constexpr std::size_t RELAY_ARG_COUNT = 9;

constexpr uint64_t SetBits(uint16_t end)
{
    return (uint64_t(1) << (end + 1)) - uint64_t(1);
}

uint64_t GetMaxLoopIterNum()
{
    constexpr uint16_t loopNumBitNum = 12;
    return SetBits(loopNumBitNum);
}

uint64_t GetParallelParamHost(
    uint64_t repeatNum, uint64_t repeatLoopIndex, uint64_t totalLoopNum)
{
    constexpr uint16_t repeatBitNum = 7;
    constexpr uint16_t repeatNumShift = 55;
    constexpr uint16_t repeatLoopShift = 48;
    constexpr uint16_t totalLoopShift = 41;
    return ((repeatNum & SetBits(repeatBitNum)) << repeatNumShift) |
        ((repeatLoopIndex & SetBits(repeatBitNum)) << repeatLoopShift) |
        ((totalLoopNum & SetBits(repeatBitNum)) << totalLoopShift);
}

std::array<uint64_t, 4> CalculateGroupCopySize(
    uint64_t size, const LoopGroupConfig &config)
{
    const uint64_t loopSize = config.loopCount * config.memSlice;
    const uint64_t maxSize = loopSize * (GetMaxLoopIterNum() + 1);
    uint64_t m = size / loopSize;
    uint64_t n = (size - m * loopSize) / config.memSlice;
    uint64_t p = size - m * loopSize - n * config.memSlice;

    if (size == maxSize) {
        m = GetMaxLoopIterNum();
        n = config.loopCount - 1;
        p = config.memSlice;
    }

    const uint64_t offset = config.memSlice * config.loopCount * m;
    const uint64_t loopIteration = m;
    uint64_t loopExtend = 0;
    uint64_t tailSize = 0;
    if (n == 0 && p == 0) {
        loopExtend = 0;
    } else if (n != 0 && p == 0) {
        loopExtend = GetParallelParamHost(n - 1, 0, 1);
        tailSize = config.memSlice;
    } else if (n == 0) {
        loopExtend = GetParallelParamHost(0, 0, 1);
        tailSize = p;
    } else {
        loopExtend = GetParallelParamHost(n - 1, 1, 2);
        tailSize = p;
    }
    return {offset, loopIteration, loopExtend, tailSize};
}

HcclResult CheckKernel(const AlgResourceCtx &resource, uint32_t mask, const char *name)
{
    CHK_PRT_RET((resource.kernelMask & mask) == 0,
        HCCL_ERROR("[ExecOp] required kernel %s is unavailable", name), HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult StartLocalThread(const AlgResourceCtx &resource)
{
    CHK_PRT_RET(resource.threadCount != 2,
        HCCL_ERROR("[ExecOp] a local CCU thread is required"), HCCL_E_INTERNAL);
    CHK_RET(HcommThreadNotifyRecordOnThread(
        resource.mainThread, resource.localThread, 0));
    CHK_RET(HcommThreadNotifyWaitOnThread(
        resource.localThread, 0, CUSTOM_TIMEOUT));
    return HCCL_SUCCESS;
}

HcclResult JoinLocalThread(const AlgResourceCtx &resource)
{
    CHK_RET(HcommThreadNotifyRecordOnThread(
        resource.localThread, resource.mainThread, 0));
    CHK_RET(HcommThreadNotifyWaitOnThread(
        resource.mainThread, 0, CUSTOM_TIMEOUT));
    return HCCL_SUCCESS;
}

HcclResult WaitLocalForMain(const AlgResourceCtx &resource)
{
    CHK_PRT_RET(resource.threadCount != 2,
        HCCL_ERROR("[ExecOp] a local CCU thread is required"), HCCL_E_INTERNAL);
    CHK_RET(HcommThreadNotifyWaitOnThread(
        resource.localThread, 0, CUSTOM_TIMEOUT));
    return HCCL_SUCCESS;
}

HcclResult NotifyLocalFromMain(const AlgResourceCtx &resource)
{
    CHK_RET(HcommThreadNotifyRecordOnThread(
        resource.mainThread, resource.localThread, 0));
    return HCCL_SUCCESS;
}

std::array<uint64_t, DIRECT_ARG_COUNT> PrepareDirectArgs(
    const OpParam &param, uint64_t dataSize, uint64_t token,
    uint64_t sourceOffset, uint64_t transferSize, bool enableSelfCopy,
    bool skipDone = false)
{
    const uint64_t inputAddress =
        reinterpret_cast<uint64_t>(param.inputPtr) + sourceOffset;
    const uint64_t outputAddress = reinterpret_cast<uint64_t>(param.outputPtr);
    const uint64_t outputOffset = dataSize * param.myRank + sourceOffset;
    const uint64_t firstSize =
        std::min<uint64_t>(MAX_DATA_SIZE, transferSize);
    const uint64_t secondSize = transferSize - firstSize;
    const uint64_t copyFlag = enableSelfCopy &&
            inputAddress != outputAddress + outputOffset ?
        uint64_t(1) : uint64_t(0);

    LoopGroupConfig copyConfig;
    copyConfig.msInterleave = CCU_MS_INTERLEAVE;
    copyConfig.loopCount = CCU_MS_LOCAL_COPY_LOOP_COUNT;
    copyConfig.memSlice = CCU_MS_SIZE * CCU_LOCAL_COPY_MS_PER_LOOP;
    const auto copySize =
        CalculateGroupCopySize(transferSize, copyConfig);

    return {
        inputAddress,
        outputAddress,
        token,
        outputOffset,
        firstSize,
        secondSize,
        copyFlag,
        skipDone ? uint64_t(1) : uint64_t(0),
        copySize[0],
        copySize[1],
        copySize[2],
        copySize[3],
    };
}

std::array<uint64_t, SMALL_DIRECT_ARG_COUNT> PrepareSmallDirectArgs(
    const OpParam &param, uint64_t dataSize, uint64_t token)
{
    LoopGroupConfig copyConfig;
    copyConfig.msInterleave = CCU_MS_INTERLEAVE;
    copyConfig.loopCount = CCU_MS_LOCAL_COPY_LOOP_COUNT;
    copyConfig.memSlice = CCU_MS_SIZE * CCU_LOCAL_COPY_MS_PER_LOOP;
    const auto copySize = CalculateGroupCopySize(dataSize, copyConfig);

    return {
        reinterpret_cast<uint64_t>(param.inputPtr),
        reinterpret_cast<uint64_t>(param.outputPtr),
        token,
        dataSize * param.myRank,
        dataSize,
        copySize[0],
        copySize[1],
        copySize[2],
        copySize[3],
    };
}

std::array<uint64_t, FOUR_X_ONE_SMALL_ARG_COUNT> Prepare4x1SmallArgs(
    const OpParam &param, uint64_t dataSize, uint64_t token)
{
    return {
        reinterpret_cast<uint64_t>(param.outputPtr),
        token,
        reinterpret_cast<uint64_t>(param.inputPtr),
        dataSize * param.myRank,
        dataSize,
    };
}

std::array<uint64_t, RELAY_ARG_COUNT> PrepareRelayArgs(
    const OpParam &param, const AlgResourceCtx &resource,
    uint64_t dataSize, uint64_t relaySize, uint64_t token)
{
    const uint64_t firstSize =
        std::min<uint64_t>(MAX_DATA_SIZE, relaySize);
    return {
        reinterpret_cast<uint64_t>(param.inputPtr),
        reinterpret_cast<uint64_t>(param.outputPtr),
        token,
        dataSize * param.myRank,
        firstSize,
        relaySize - firstSize,
        uint64_t(0),
        dataSize * resource.relayRanks[0],
        dataSize * resource.relayRanks[1],
    };
}

uint64_t GetAlignedFraction(
    uint64_t dataSize, uint64_t numerator, uint64_t denominator)
{
    const uint64_t rawSize =
        (dataSize / denominator) * numerator +
        ((dataSize % denominator) * numerator) / denominator;
    return rawSize - rawSize % HYBRID_ALIGNMENT;
}

HcclResult GetHybridSizes(const AlgResourceCtx &resource, uint64_t dataSize,
    uint64_t &ownSeedSize, uint64_t &remoteRelaySize)
{
    if (resource.topology == TopologyKind::TOPO_2X8) {
        ownSeedSize = GetAlignedFraction(dataSize, 3, 8);
        remoteRelaySize = ownSeedSize;
    } else if (resource.topology == TopologyKind::TOPO_8P4 &&
        resource.role == TopologyRole::ROLE_8P4_FULL_SERVER) {
        // Minimax split for local-bandwidth 1 and per-rank network bandwidth
        // 4: full-side relay fraction a=2/7, partial-side fraction b=4/11.
        ownSeedSize = GetAlignedFraction(dataSize, 2, 7);
        remoteRelaySize = GetAlignedFraction(dataSize, 4, 11);
    } else if (resource.topology == TopologyKind::TOPO_8P4 &&
        resource.role == TopologyRole::ROLE_8P4_PARTIAL_SERVER) {
        ownSeedSize = GetAlignedFraction(dataSize, 4, 11);
        remoteRelaySize = GetAlignedFraction(dataSize, 2, 7);
    } else {
        HCCL_ERROR("[ExecOp] hybrid split requested for invalid topology/role %u/%u",
            static_cast<uint32_t>(resource.topology),
            static_cast<uint32_t>(resource.role));
        return HCCL_E_INTERNAL;
    }

    CHK_PRT_RET(ownSeedSize == 0 || ownSeedSize >= dataSize ||
            remoteRelaySize == 0 || remoteRelaySize >= dataSize,
        HCCL_ERROR("[ExecOp] invalid hybrid split %llu/%llu for %llu bytes",
            static_cast<unsigned long long>(ownSeedSize),
            static_cast<unsigned long long>(remoteRelaySize),
            static_cast<unsigned long long>(dataSize)),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult Launch4x1Small(const AlgResourceCtx &resource,
    const std::array<uint64_t, FOUR_X_ONE_SMALL_ARG_COUNT> &taskArgs)
{
    CHK_RET(CheckKernel(
        resource, KERNEL_NETWORK_SMALL, "network-small"));
    CHK_RET_CCU(HcommCcuKernelLaunch(resource.mainThread,
        resource.networkSmallKernel, taskArgs.data(), taskArgs.size()));
    return HCCL_SUCCESS;
}

HcclResult Launch4x1Large(const AlgResourceCtx &resource,
    const std::array<uint64_t, DIRECT_ARG_COUNT> &networkArgs)
{
    CHK_RET(CheckKernel(
        resource, KERNEL_NETWORK_DIRECT, "network-direct"));
    CHK_RET_CCU(HcommCcuKernelLaunch(resource.mainThread,
        resource.networkDirectKernel, networkArgs.data(), networkArgs.size()));
    return HCCL_SUCCESS;
}

HcclResult LaunchV04SmallDirect(const AlgResourceCtx &resource,
    const std::array<uint64_t, DIRECT_ARG_COUNT> &networkArgs,
    const std::array<uint64_t, DIRECT_ARG_COUNT> &localArgs)
{
    CHK_RET(CheckKernel(
        resource, KERNEL_NETWORK_DIRECT, "network-direct"));
    CHK_RET(CheckKernel(
        resource, KERNEL_LOCAL_DIRECT, "local-direct"));
    CHK_RET(StartLocalThread(resource));

    // Preserve the v0.4 launch order: submit the slower network layer first,
    // then run the local layer on the slave stream.
    CHK_RET_CCU(HcommCcuKernelLaunch(resource.mainThread,
        resource.networkDirectKernel, networkArgs.data(), networkArgs.size()));
    CHK_RET_CCU(HcommCcuKernelLaunch(resource.localThread,
        resource.localDirectKernel, localArgs.data(), localArgs.size()));
    return JoinLocalThread(resource);
}

HcclResult Launch8p4SmallDirect(const AlgResourceCtx &resource,
    const std::array<uint64_t, SMALL_DIRECT_ARG_COUNT> &taskArgs)
{
    CHK_RET(CheckKernel(
        resource, KERNEL_NETWORK_SMALL, "network-small"));
    CHK_RET(CheckKernel(
        resource, KERNEL_LOCAL_SMALL, "local-small"));
    // VM checker requires the first task on every slave stream to be a local
    // WAIT. This START/WAIT pair is therefore part of the execution contract,
    // even though peer READY already protects the communication dependency.
    CHK_RET(StartLocalThread(resource));

    // The main stream launches the layer that is most likely to determine the
    // critical path. Both layers still transfer the original input directly.
    if (resource.topology == TopologyKind::TOPO_8P4 &&
        resource.role == TopologyRole::ROLE_8P4_FULL_SERVER) {
        CHK_RET_CCU(HcommCcuKernelLaunch(resource.mainThread,
            resource.localSmallKernel, taskArgs.data(), taskArgs.size()));
        CHK_RET_CCU(HcommCcuKernelLaunch(resource.localThread,
            resource.networkSmallKernel, taskArgs.data(), taskArgs.size()));
    } else {
        CHK_RET_CCU(HcommCcuKernelLaunch(resource.mainThread,
            resource.networkSmallKernel, taskArgs.data(), taskArgs.size()));
        CHK_RET_CCU(HcommCcuKernelLaunch(resource.localThread,
            resource.localSmallKernel, taskArgs.data(), taskArgs.size()));
    }
    return JoinLocalThread(resource);
}

HcclResult LaunchHierarchicalHybrid(const AlgResourceCtx &resource,
    const std::array<uint64_t, DIRECT_ARG_COUNT> &seedArgs,
    const std::array<uint64_t, DIRECT_ARG_COUNT> &suffixArgs,
    const std::array<uint64_t, DIRECT_ARG_COUNT> &localArgs,
    const std::array<uint64_t, RELAY_ARG_COUNT> &relayArgs)
{
    CHK_RET(CheckKernel(resource, KERNEL_NETWORK_SEED, "network-seed"));
    CHK_RET(CheckKernel(resource, KERNEL_NETWORK_DIRECT, "network-suffix"));
    CHK_RET(CheckKernel(resource, KERNEL_LOCAL_DIRECT, "local-own"));
    CHK_RET(CheckKernel(resource, KERNEL_LOCAL_RELAY, "local-relay"));
    CHK_RET(StartLocalThread(resource));

    // Main thread: seed-prefix -> release relay -> direct suffix.
    // Local thread: own all-to-all -> wait seed -> relay prefix.
    // Thus the network suffix remains active while the local mesh relays the
    // prefix. Prefix and suffix write disjoint OUTPUT ranges.
    CHK_RET_CCU(HcommCcuKernelLaunch(resource.mainThread,
        resource.networkSeedKernel, seedArgs.data(),
        DIRECT_NO_COPY_ARG_COUNT));
    CHK_RET_CCU(HcommCcuKernelLaunch(resource.localThread,
        resource.localDirectKernel, localArgs.data(), localArgs.size()));
    CHK_RET(NotifyLocalFromMain(resource));
    CHK_RET_CCU(HcommCcuKernelLaunch(resource.mainThread,
        resource.networkDirectKernel, suffixArgs.data(),
        DIRECT_NO_COPY_ARG_COUNT));
    CHK_RET(WaitLocalForMain(resource));
    CHK_RET_CCU(HcommCcuKernelLaunch(resource.localThread,
        resource.localRelayKernel, relayArgs.data(), relayArgs.size()));
    return JoinLocalThread(resource);
}

} // namespace

HcclResult ExecOp(const OpParam &param)
{
    CHK_PRT_RET(param.resCtx == nullptr || param.ctxSize != sizeof(AlgResourceCtx),
        HCCL_ERROR("[ExecOp] invalid resource context addr[%p] size[%llu]",
            param.resCtx, static_cast<unsigned long long>(param.ctxSize)),
        HCCL_E_PTR);

    const AlgResourceCtx &resource =
        *static_cast<const AlgResourceCtx *>(param.resCtx);
    CHK_PRT_RET(resource.magic != AlgResourceCtx::MAGIC ||
            resource.version != AlgResourceCtx::VERSION,
        HCCL_ERROR("[ExecOp] invalid resource header 0x%x/%u",
            resource.magic, resource.version),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(resource.threadCount == 0 || resource.threadCount > 2,
        HCCL_ERROR("[ExecOp] invalid CCU thread count %u", resource.threadCount),
        HCCL_E_INTERNAL);

    CHK_PRT_RET(param.dataType != HCCL_DATA_TYPE_FP32,
        HCCL_ERROR("[ExecOp] unsupported data type %d", static_cast<int>(param.dataType)),
        HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(param.count >
            std::numeric_limits<uint64_t>::max() / sizeof(float),
        HCCL_ERROR("[ExecOp] input byte size overflows"), HCCL_E_PARA);
    CHK_PRT_RET(param.rankSize == 0 || param.myRank >= param.rankSize,
        HCCL_ERROR("[ExecOp] invalid rank configuration %u/%u",
            param.myRank, param.rankSize),
        HCCL_E_PARA);

    const uint64_t dataSize = param.count * sizeof(float);
    if (dataSize == 0) {
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(dataSize > MAX_DATA_SIZE * 2ULL,
        HCCL_ERROR("[ExecOp] input size %llu exceeds the two-WQE design limit",
            static_cast<unsigned long long>(dataSize)),
        HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(dataSize >
            std::numeric_limits<uint64_t>::max() / param.rankSize,
        HCCL_ERROR("[ExecOp] output byte size overflows"), HCCL_E_PARA);

    const uint64_t inputAddress = reinterpret_cast<uint64_t>(param.inputPtr);
    uint64_t token = 0;
    CHK_RET_CCU(HcommCcuGetMemToken(inputAddress, dataSize, &token));

    if (resource.topology == TopologyKind::TOPO_4X1) {
        const bool isSmall = dataSize <= CUSTOM_SMALL_MESSAGE_LIMIT;
        if (isSmall) {
            const auto smallArgs =
                Prepare4x1SmallArgs(param, dataSize, token);
            return Launch4x1Small(resource, smallArgs);
        }
        const auto networkArgs =
            PrepareDirectArgs(param, dataSize, token, 0, dataSize, true, true);
        return Launch4x1Large(resource, networkArgs);
    }
    if (resource.topology == TopologyKind::TOPO_2X8 ||
        resource.topology == TopologyKind::TOPO_8P4) {
        CHK_PRT_RET(resource.threadCount != 2 || resource.relayCount > 2,
            HCCL_ERROR("[ExecOp] invalid hierarchical resources"), HCCL_E_INTERNAL);
        if (dataSize <= CUSTOM_SMALL_MESSAGE_LIMIT) {
            if (resource.topology == TopologyKind::TOPO_2X8) {
                const auto networkArgs = PrepareDirectArgs(param, dataSize, token,
                    0, dataSize, resource.selfCopyOnNetwork != 0, true);
                const auto localArgs = PrepareDirectArgs(param, dataSize, token,
                    0, dataSize, resource.selfCopyOnNetwork == 0, true);
                return LaunchV04SmallDirect(
                    resource, networkArgs, localArgs);
            }
            const auto smallArgs =
                PrepareSmallDirectArgs(param, dataSize, token);
            return Launch8p4SmallDirect(resource, smallArgs);
        }

        uint64_t ownSeedSize = 0;
        uint64_t remoteRelaySize = 0;
        CHK_RET(GetHybridSizes(
            resource, dataSize, ownSeedSize, remoteRelaySize));
        const auto seedArgs = PrepareDirectArgs(
            param, dataSize, token, 0, ownSeedSize, false);
        const bool skipSuffixDone =
            resource.topology == TopologyKind::TOPO_8P4;
        const auto suffixArgs = PrepareDirectArgs(param, dataSize, token,
            ownSeedSize, dataSize - ownSeedSize, false, skipSuffixDone);
        const auto localArgs =
            PrepareDirectArgs(param, dataSize, token, 0, dataSize, true);
        const auto relayArgs = PrepareRelayArgs(
            param, resource, dataSize, remoteRelaySize, token);
        return LaunchHierarchicalHybrid(
            resource, seedArgs, suffixArgs, localArgs, relayArgs);
    }

    HCCL_ERROR("[ExecOp] unknown topology %u",
        static_cast<uint32_t>(resource.topology));
    return HCCL_E_NOT_SUPPORT;
}

} // namespace ops_hccl
