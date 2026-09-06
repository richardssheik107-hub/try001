/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#include <hccl/hccl_res_expt.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_diag.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <limits>
#include <memory>
#include <vector>

#include <ccu/ccu_launch.h>
#include <hccl/hccl_ccu_res.h>

#include "log.h"
#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "exec_op.h"
#include "ccu_kernel.h"

namespace {

struct IoDieChannelGroup {
    std::vector<ChannelHandle> channels;
    std::vector<uint32_t> peerRanks;
};

HcclResult FillChannelDesc(HcclComm comm, uint32_t srcRank, uint32_t dstRank, HcclChannelDesc &desc)
{
    CommLink selectedLink{};
    bool found = false;
    for (uint32_t layer = 0; layer < 2 && !found; ++layer) {
        CommLink *links = nullptr;
        uint32_t linkNum = 0;
        HcclResult ret = HcclRankGraphGetLinks(comm, layer, srcRank, dstRank, &links, &linkNum);
        if (ret != HCCL_SUCCESS || links == nullptr) {
            continue;
        }
        for (uint32_t i = 0; i < linkNum; ++i) {
            if (links[i].linkAttr.linkProtocol == CommProtocol::COMM_PROTOCOL_UBC_CTP) {
                selectedLink = links[i];
                found = true;
                break;
            }
        }
    }
    CHK_PRT_RET(!found,
        HCCL_ERROR("No UBC_CTP link between rank[%u] and rank[%u]", srcRank, dstRank), HCCL_E_NOT_FOUND);
    CHK_RET(HcclChannelDescInit(&desc, 1));
    desc.remoteRank = dstRank;
    desc.notifyNum = custom_rs::CHANNEL_NOTIFY_NUM;
    desc.channelProtocol = selectedLink.linkAttr.linkProtocol;
    desc.localEndpoint = selectedLink.srcEndpointDesc;
    desc.remoteEndpoint = selectedLink.dstEndpointDesc;
    return HCCL_SUCCESS;
}

HcclResult BuildServerGroups(HcclComm comm, const OpParam &param,
    std::vector<uint32_t> &localRanks, std::vector<uint32_t> &remoteRanks)
{
    uint32_t *ranks = nullptr;
    uint32_t rankNum = 0;
    CHK_RET(HcclRankGraphGetRanksByLayer(comm, 0, &ranks, &rankNum));
    CHK_PRT_RET(ranks == nullptr || rankNum == 0,
        HCCL_ERROR("Invalid layer-0 rank group"), HCCL_E_INTERNAL);
    localRanks.assign(ranks, ranks + rankNum);
    std::sort(localRanks.begin(), localRanks.end());
    CHK_PRT_RET(std::find(localRanks.begin(), localRanks.end(), param.myRank) == localRanks.end(),
        HCCL_ERROR("Rank[%u] is absent from its layer-0 group", param.myRank), HCCL_E_INTERNAL);
    std::vector<bool> isLocal(param.rankSize, false);
    for (uint32_t rank : localRanks) {
        CHK_PRT_RET(rank >= param.rankSize,
            HCCL_ERROR("Invalid local rank[%u]", rank), HCCL_E_INTERNAL);
        isLocal[rank] = true;
    }
    for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
        if (!isLocal[rank]) {
            remoteRanks.push_back(rank);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AcquireChannels(HcclComm comm, const OpParam &param,
    std::vector<IoDieChannelGroup> &groups, std::vector<uint32_t> &peerIoDies)
{
    groups.resize(custom_rs::IO_DIE_NUM);
    peerIoDies.assign(param.rankSize, custom_rs::INVALID_IO_DIE);
    if (param.rankSize <= 1) {
        return HCCL_SUCCESS;
    }
    std::vector<HcclChannelDesc> descs(param.rankSize - 1);
    std::vector<ChannelHandle> channels(param.rankSize - 1);
    std::vector<uint32_t> ioDies(param.rankSize - 1);
    uint32_t idx = 0;
    for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
        if (peer == param.myRank) {
            continue;
        }
        CHK_RET(FillChannelDesc(comm, param.myRank, peer, descs[idx]));
        CHK_RET(HcclRankGraphGetEndpointInfo(comm, param.myRank, &descs[idx].localEndpoint,
            ENDPOINT_ATTR_DIE_ID, sizeof(ioDies[idx]), &ioDies[idx]));
        CHK_PRT_RET(ioDies[idx] >= custom_rs::IO_DIE_NUM,
            HCCL_ERROR("Invalid IO die[%u] for peer[%u]", ioDies[idx], peer), HCCL_E_INTERNAL);
        peerIoDies[peer] = ioDies[idx];
        ++idx;
    }
    CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CCU,
        descs.data(), static_cast<uint32_t>(descs.size()), channels.data()));
    for (uint32_t i = 0; i < channels.size(); ++i) {
        groups[ioDies[i]].channels.push_back(channels[i]);
        groups[ioDies[i]].peerRanks.push_back(descs[i].remoteRank);
    }
    return HCCL_SUCCESS;
}

void FillChannels(const IoDieChannelGroup &group, ops_hccl::ReduceScatterKernelArg &arg)
{
    arg.channelCount = static_cast<uint32_t>(group.channels.size());
    for (uint32_t i = 0; i < arg.channelCount; ++i) {
        arg.channels[i] = group.channels[i];
        arg.peerRanks[i] = group.peerRanks[i];
    }
}

void FillMembers(const std::vector<uint32_t> &members, uint32_t myRank, uint32_t selfIoDie,
    uint32_t ioDieId, const std::vector<uint32_t> &peerIoDies,
    uint32_t *memberRanks, uint32_t &memberCount)
{
    for (uint32_t rank : members) {
        if (rank == myRank) {
            if (ioDieId == selfIoDie) {
                memberRanks[memberCount++] = rank;
            }
        } else if (peerIoDies[rank] == ioDieId) {
            memberRanks[memberCount++] = rank;
        }
    }
    std::sort(memberRanks, memberRanks + memberCount);
}

void FillTargets(const std::vector<custom_rs::AggregateTask> &targets, uint32_t ioDieId,
    const std::vector<uint32_t> &peerIoDies, ops_hccl::ReduceScatterKernelArg &arg)
{
    for (uint32_t slot = 0; slot < targets.size(); ++slot) {
        if (peerIoDies[targets[slot].rank] == ioDieId) {
            arg.publishTargetRanks[arg.publishTargetCount] = targets[slot].rank;
            ++arg.publishTargetCount;
        }
    }
}

HcclResult RegisterOne(CcuInsHandle insHandle, uint32_t dieId, const char *name, void *func,
    const std::shared_ptr<ops_hccl::ReduceScatterKernelArg> &arg, CcuKernelHandle &handle)
{
    const void *args[] = {arg.get()};
    CcuResult ret = HcommCcuKernelRegister(insHandle, dieId, name, func, args, 1, &handle);
    if (ret != CCU_SUCCESS) {
        HCCL_ERROR("Register kernel[%s] on die[%u] failed, ret[%d]", name, dieId, static_cast<int>(ret));
    }
    return ConvertCcuToHccl(ret);
}

struct KernelRegistrationTopology {
    bool mixedTopology = false;
    bool isSymmetricTwoServerTopology = false;
    bool smallBufferKernel = false;
};

using KernelArgPtr = std::shared_ptr<ops_hccl::ReduceScatterKernelArg>;
using KernelArgArray = std::array<KernelArgPtr, custom_rs::IO_DIE_NUM>;

HcclResult DetectKernelRegistrationTopology(const OpParam &param,
    const std::vector<IoDieChannelGroup> &groups, bool smallDataKernel,
    AlgResourceCtx &resCtx, KernelRegistrationTopology &topology)
{
    resCtx.activeIoDieMask = 0;
    for (uint32_t die = 0; die < custom_rs::IO_DIE_NUM; ++die) {
        if (!groups[die].channels.empty()) {
            resCtx.activeIoDieMask |= 1U << die;
        }
    }
    CHK_PRT_RET(resCtx.activeIoDieMask == 0,
        HCCL_ERROR("No active IO die"), HCCL_E_INTERNAL);
    resCtx.meshIoDie = custom_rs::INVALID_IO_DIE;
    resCtx.closIoDie = custom_rs::INVALID_IO_DIE;
    for (uint32_t rank : resCtx.localServerRanks) {
        if (rank != param.myRank) {
            resCtx.meshIoDie = resCtx.peerIoDies[rank];
            break;
        }
    }
    for (uint32_t rank : resCtx.remoteServerRanks) {
        resCtx.closIoDie = resCtx.peerIoDies[rank];
        break;
    }
    CHK_PRT_RET(resCtx.meshIoDie != custom_rs::INVALID_IO_DIE &&
            resCtx.closIoDie != custom_rs::INVALID_IO_DIE &&
            resCtx.meshIoDie == resCtx.closIoDie,
        HCCL_ERROR("Layer-0 and layer-1 channels unexpectedly share IO die[%u]",
            resCtx.meshIoDie), HCCL_E_INTERNAL);
    resCtx.resultIoDie = resCtx.meshIoDie != custom_rs::INVALID_IO_DIE
        ? resCtx.meshIoDie : ((resCtx.activeIoDieMask & 1U) ? 0U : 1U);
    topology.mixedTopology = !smallDataKernel &&
        resCtx.meshIoDie != custom_rs::INVALID_IO_DIE &&
        resCtx.closIoDie != custom_rs::INVALID_IO_DIE;
    topology.isSymmetricTwoServerTopology = topology.mixedTopology &&
        resCtx.localServerRanks.size() == 8U &&
        resCtx.remoteServerRanks.size() == 8U;
    topology.smallBufferKernel = smallDataKernel;
    return HCCL_SUCCESS;
}

HcclResult BuildCommunicationKernelArgs(const OpParam &param,
    const std::vector<IoDieChannelGroup> &groups,
    const KernelRegistrationTopology &topology,
    AlgResourceCtx &resCtx, KernelArgArray &commArgs)
{
    const uint32_t localSelfDie = resCtx.resultIoDie;
    uint32_t fusedSelfDie = resCtx.resultIoDie;
    if (topology.smallBufferKernel) {
        uint32_t bestPeerCount = MAX_RANK_SIZE;
        fusedSelfDie = custom_rs::INVALID_IO_DIE;
        for (uint32_t die = 0; die < custom_rs::IO_DIE_NUM; ++die) {
            if ((resCtx.activeIoDieMask & (1U << die)) == 0) continue;
            const uint32_t peerCount = static_cast<uint32_t>(groups[die].channels.size());
            if (peerCount >= custom_rs::CCU_BUFFER_REDUCE_MAX_INPUTS) continue;
            if (fusedSelfDie == custom_rs::INVALID_IO_DIE || peerCount < bestPeerCount ||
                (peerCount == bestPeerCount && die == resCtx.resultIoDie)) {
                fusedSelfDie = die;
                bestPeerCount = peerCount;
            }
        }
        if (fusedSelfDie == custom_rs::INVALID_IO_DIE) {
            fusedSelfDie = resCtx.resultIoDie;
        }
    }
    resCtx.localMemberCounts.fill(0);
    resCtx.directMemberCounts.fill(0);
    resCtx.fusedMemberCounts.fill(0);

    for (uint32_t die = 0; die < custom_rs::IO_DIE_NUM; ++die) {
        if ((resCtx.activeIoDieMask & (1U << die)) == 0) {
            continue;
        }
        commArgs[die] = std::make_shared<ops_hccl::ReduceScatterKernelArg>();
        auto &arg = *commArgs[die];
        arg.rankId = param.myRank;
        arg.smallBufferFuseSync = param.rankSize != 12U;
        if (topology.isSymmetricTwoServerTopology) {
            arg.commRole = die == resCtx.meshIoDie
                ? ops_hccl::CommKernelRole::MESH : ops_hccl::CommKernelRole::CLOS;
        } else {
            // Asymmetric two-server and 4x1 routes intentionally keep the
            // general translated shape; runtime still launches only the
            // modes used by the selected scheduler.
            arg.commRole = ops_hccl::CommKernelRole::GENERAL;
        }
        FillChannels(groups[die], arg);
        if (topology.mixedTopology) {
            for (const auto &source : resCtx.aggregateSources) {
                if (source.rank < resCtx.peerIoDies.size() &&
                    resCtx.peerIoDies[source.rank] == die) {
                    const uint32_t sourceIdx = arg.aggregateSourceCount++;
                    arg.aggregateSourceRanks[sourceIdx] = source.rank;
                    arg.aggregateSourceRemoteOffsets[sourceIdx] = source.remoteOffset;
                    arg.aggregateSourceLocalOffsets[sourceIdx] = source.localOffset;
                    arg.aggregateSourceBytes[sourceIdx] = source.bytes;
                }
            }
        }
        std::array<uint32_t, MAX_RANK_SIZE> localMemberRanks{};
        FillMembers(resCtx.localServerRanks, param.myRank, localSelfDie, die, resCtx.peerIoDies,
            localMemberRanks.data(), arg.localMemberCount);
        for (uint32_t memberIdx = 0; memberIdx < arg.localMemberCount; ++memberIdx) {
            const uint32_t rank = localMemberRanks[memberIdx];
            if (rank != param.myRank) {
                arg.remoteLocalMemberRanks[arg.remoteLocalMemberCount++] = rank;
            }
        }
        if (topology.mixedTopology) {
            FillTargets(resCtx.aggregateTargets, die, resCtx.peerIoDies, arg);
        }
        for (uint32_t rank : resCtx.remoteServerRanks) {
            if (resCtx.peerIoDies[rank] == die) {
                arg.directMemberRanks[arg.directMemberCount++] = rank;
            }
        }
        for (uint32_t rank : resCtx.localServerRanks) {
            if (rank != param.myRank && resCtx.peerIoDies[rank] == die) {
                arg.directMemberRanks[arg.directMemberCount++] = rank;
            }
        }
        std::sort(arg.directMemberRanks, arg.directMemberRanks + arg.directMemberCount);
        for (uint32_t memberIdx = 0; memberIdx < arg.directMemberCount; ++memberIdx) {
            const uint32_t rank = arg.directMemberRanks[memberIdx];
            arg.fusedMemberRanks[arg.fusedMemberCount++] = rank;
        }
        // FETCH_CLOS_REMOTE_SLOTS is remote-only. Small Buffer reduction and
        // the 4x1 segmented fetch retain self in this independent table.
        if (die == fusedSelfDie) {
            arg.fusedMemberRanks[arg.fusedMemberCount++] = param.myRank;
            std::sort(arg.fusedMemberRanks, arg.fusedMemberRanks + arg.fusedMemberCount);
        }
        if (topology.smallBufferKernel) {
            CHK_PRT_RET(arg.fusedMemberCount == 0 ||
                    arg.fusedMemberCount > custom_rs::CCU_BUFFER_REDUCE_MAX_INPUTS,
                HCCL_ERROR("Unsupported small Buffer member count die[%u] members[%u]",
                    die, arg.fusedMemberCount), HCCL_E_NOT_SUPPORT);
            arg.smallBufferCapacityBlocksPerSource =
                custom_rs::GetSmallBufferCapacityBlocksPerSource(
                    param.rankSize, arg.fusedMemberCount);
            CHK_PRT_RET(arg.smallBufferCapacityBlocksPerSource == 0 ||
                    arg.fusedMemberCount * arg.smallBufferCapacityBlocksPerSource >
                        custom_rs::CCU_SCHED_MS_BLOCKS_PER_DIE,
                HCCL_ERROR("Invalid small MS layout die[%u] members[%u] stride[%u]",
                    die, arg.fusedMemberCount, arg.smallBufferCapacityBlocksPerSource),
                HCCL_E_NOT_SUPPORT);
        }
        resCtx.localMemberCounts[die] = arg.localMemberCount;
        resCtx.directMemberCounts[die] = arg.directMemberCount;
        resCtx.fusedMemberCounts[die] = arg.fusedMemberCount;
    }
    return HCCL_SUCCESS;
}

KernelArgPtr BuildReduceKernelArg(uint32_t die,
    const std::vector<IoDieChannelGroup> &groups,
    const KernelRegistrationTopology &topology,
    const KernelArgArray &commArgs, const AlgResourceCtx &resCtx)
{
    auto reduceArg = std::make_shared<ops_hccl::ReduceScatterKernelArg>();
    // HcommCcuKernelRegister's dieId is currently reserved; the translator
    // selects a die from the channels referenced by the kernel. Give a local-
    // only Mission one channel as a placement anchor. It never communicates
    // through that channel.
    if (topology.mixedTopology && die == resCtx.closIoDie) {
        // Clos reduction performs A-partial ReadReduce and therefore needs every
        // source/target channel, rather than only a placement anchor.
        FillChannels(groups[die], *reduceArg);
    } else {
        reduceArg->channelCount = 1;
        reduceArg->channels[0] = groups[die].channels[0];
        reduceArg->peerRanks[0] = groups[die].peerRanks[0];
    }
    const uint32_t sourceMemberCount =
        die == resCtx.meshIoDie && commArgs[die]->localMemberCount != 0
        ? commArgs[die]->localMemberCount : commArgs[die]->directMemberCount;
    reduceArg->localMemberCount = die == resCtx.meshIoDie
        ? commArgs[die]->remoteLocalMemberCount : sourceMemberCount;
    reduceArg->remoteLocalMemberCount = commArgs[die]->remoteLocalMemberCount;

    // Only MixedRoute launches READ_REDUCE_CLOS_PARTIALS. Keeping these tables
    // empty for 4x1 prevents its segmented Reduce Mission from translating unused
    // channel lookups through the single placement-anchor channel.
    if (topology.mixedTopology) {
        reduceArg->publishTargetCount = commArgs[die]->publishTargetCount;
        for (uint32_t targetIdx = 0;
            targetIdx < reduceArg->publishTargetCount; ++targetIdx) {
            reduceArg->publishTargetRanks[targetIdx] =
                commArgs[die]->publishTargetRanks[targetIdx];
        }
        reduceArg->aggregateSourceCount = commArgs[die]->aggregateSourceCount;
        for (uint32_t sourceIdx = 0;
            sourceIdx < reduceArg->aggregateSourceCount; ++sourceIdx) {
            reduceArg->aggregateSourceRanks[sourceIdx] =
                commArgs[die]->aggregateSourceRanks[sourceIdx];
            reduceArg->aggregateSourceRemoteOffsets[sourceIdx] =
                commArgs[die]->aggregateSourceRemoteOffsets[sourceIdx];
            reduceArg->aggregateSourceLocalOffsets[sourceIdx] =
                commArgs[die]->aggregateSourceLocalOffsets[sourceIdx];
            reduceArg->aggregateSourceBytes[sourceIdx] =
                commArgs[die]->aggregateSourceBytes[sourceIdx];
        }
    }
    if (topology.isSymmetricTwoServerTopology) {
        reduceArg->reduceRole = die == resCtx.meshIoDie
            ? ops_hccl::ReduceKernelRole::MESH : ops_hccl::ReduceKernelRole::CLOS;
    } else {
        reduceArg->reduceRole = ops_hccl::ReduceKernelRole::GENERAL;
    }
    return reduceArg;
}

HcclResult RegisterKernelMissions(HcclComm comm, const OpParam &param,
    const std::vector<IoDieChannelGroup> &groups, bool smallDataKernel,
    const KernelRegistrationTopology &topology,
    const KernelArgArray &commArgs, AlgResourceCtx &resCtx)
{

    CcuInsHandle insHandle = 0;
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum != 1, HCCL_ERROR("Unexpected CCU instance count[%u]", insNum), HCCL_E_INTERNAL);
    CcuResult ret = HcommCcuKernelRegisterStart(insHandle);
    if (ret != CCU_SUCCESS) {
        return ConvertCcuToHccl(ret);
    }
    resCtx.ccuKernels.fill(0);
    // MixedRoute uses one communication and one reduction Mission on each active
    // die. 4x1 also uses exactly two Missions: communication performs the two
    // recursive-halving rounds, while reduction consumes their static per-piece
    // readiness DAG.
    for (uint32_t die = 0; die < custom_rs::IO_DIE_NUM; ++die) {
        if ((resCtx.activeIoDieMask & (1U << die)) == 0) {
            continue;
        }
        const bool kernel4X1 = !smallDataKernel && param.rankSize == 4U;
        const bool small2X8 = smallDataKernel && param.rankSize == 16U;
        const char *commKernelName = small2X8
            ? "CcuReduceScatterKernel_Comm_Small_2X8"
            : (smallDataKernel
            ? "CcuReduceScatterKernel_Comm_Small"
            : (kernel4X1
                ? "CcuReduceScatterKernel_Comm_4X1"
                : "CcuReduceScatterKernel_Comm_General"));
        void *commKernelFunc = small2X8
            ? reinterpret_cast<void *>(ops_hccl::CcuReduceScatterKernel_Comm_Small_2X8)
            : (smallDataKernel
            ? reinterpret_cast<void *>(ops_hccl::CcuReduceScatterKernel_Comm_Small)
            : (kernel4X1
                ? reinterpret_cast<void *>(ops_hccl::CcuReduceScatterKernel_Comm_4X1)
                : reinterpret_cast<void *>(ops_hccl::CcuReduceScatterKernel_Comm_General)));
        CHK_RET(RegisterOne(insHandle, die, commKernelName, commKernelFunc, commArgs[die],
            resCtx.ccuKernels[custom_rs::GetCommKernelIndex(die)]));

        // The fused small route performs all source-slot reduction on the
        // communication Mission. A second reduction Mission is needed only when
        // the other IO die contributes a partial that must be merged on the
        // result die.
        const bool smallNeedsLocalMission = resCtx.activeIoDieMask == 3U &&
            die == resCtx.resultIoDie;
        if (smallDataKernel && !smallNeedsLocalMission) {
            continue;
        }
        KernelArgPtr reduceArg;
        const char *reduceKernelName = nullptr;
        void *reduceKernelFunc = nullptr;
        if (smallDataKernel) {
            reduceArg = BuildReduceKernelArg(
                die, groups, topology, commArgs, resCtx);
            reduceKernelName = small2X8
                ? "CcuReduceScatterKernel_Reduce_Small_2X8"
                : "CcuReduceScatterKernel_Reduce_Small";
            reduceKernelFunc = small2X8
                ? reinterpret_cast<void *>(ops_hccl::CcuReduceScatterKernel_Reduce_Small_2X8)
                : reinterpret_cast<void *>(ops_hccl::CcuReduceScatterKernel_Reduce_Small);
        } else if (kernel4X1) {
            reduceArg = std::make_shared<ops_hccl::ReduceScatterKernelArg>();
            reduceKernelName = "CcuReduceScatterKernel_Reduce_4X1";
            reduceKernelFunc = reinterpret_cast<void *>(
                ops_hccl::CcuReduceScatterKernel_Reduce_4X1);
        } else {
            reduceArg = BuildReduceKernelArg(
                die, groups, topology, commArgs, resCtx);
            reduceKernelName = "CcuReduceScatterKernel_Reduce_General";
            reduceKernelFunc = reinterpret_cast<void *>(
                ops_hccl::CcuReduceScatterKernel_Reduce_General);
        }
        CHK_RET(RegisterOne(insHandle, die, reduceKernelName,
            reduceKernelFunc, reduceArg,
            resCtx.ccuKernels[custom_rs::GetReduceKernelIndex(die)]));
    }
    ret = HcommCcuKernelRegisterEnd(insHandle);
    return ConvertCcuToHccl(ret);
}

HcclResult RegisterKernels(HcclComm comm, const OpParam &param,
    const std::vector<IoDieChannelGroup> &groups, bool smallDataKernel, AlgResourceCtx &resCtx)
{
    KernelRegistrationTopology topology;
    CHK_RET(DetectKernelRegistrationTopology(
        param, groups, smallDataKernel, resCtx, topology));

    KernelArgArray commArgs{};
    CHK_RET(BuildCommunicationKernelArgs(param, groups, topology, resCtx, commArgs));
    return RegisterKernelMissions(
        comm, param, groups, smallDataKernel, topology, commArgs, resCtx);
}

} // namespace

HcclResult HcclReduceScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType,
    HcclReduceOp op, HcclComm comm, aclrtStream stream)
{
    if (recvCount == 0) return HCCL_SUCCESS;
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);
    CHK_PRT_RET(dataType != HCCL_DATA_TYPE_FP32 || op != HCCL_REDUCE_SUM,
        HCCL_ERROR("Only FP32 SUM is supported"), HCCL_E_NOT_SUPPORT);

    OpParam param;
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = recvCount;
    param.dataType = dataType;
    param.reduceType = op;
    param.opType = HcclCMDType::HCCL_CMD_REDUCE_SCATTER;
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_PRT_RET(param.rankSize < 2 || param.rankSize > MAX_RANK_SIZE,
        HCCL_ERROR("Unsupported rankSize[%u]", param.rankSize), HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(recvCount > std::numeric_limits<uint64_t>::max() / sizeof(float) / param.rankSize,
        HCCL_ERROR("Input size overflow"), HCCL_E_PARA);
    const uint64_t totalBytes = recvCount * sizeof(float) * param.rankSize;
    const char *ctxTag = totalBytes <= custom_rs::SMALL_TOTAL_DATA_BYTES
        ? (param.rankSize == 16U
            ? "hccl_custom_reducescatter_small_2x8_s33_v1"
            : "hccl_custom_reducescatter_small_buffer_tile_v5")
        : (param.rankSize == 4U
            ? "hccl_custom_reducescatter_4x1_recursive_halving_fused_v5"
            : "hccl_custom_reducescatter_mixed_linear_serial_clos_v6");
    std::sprintf(param.tag, "%s", ctxTag);

    HcclDfxOpInfo dfxInfo;
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    CommEngine engine = CommEngine::COMM_ENGINE_CCU;
    CHK_RET(HcclThreadAcquireWithStream(
        comm, engine, stream, custom_rs::MAIN_THREAD_NOTIFY_NUM, &param.cpuThread));
    void *ctx = nullptr;
    uint64_t size = 0;
    if (HcclEngineCtxGet(comm, param.tag, engine, &ctx, &size) == HCCL_SUCCESS) {
        param.resCtx = ctx;
        param.ctxSize = size;
    } else {
        AlgResourceCtx resCtx;
        void *buffer = nullptr;
        uint64_t bufferSize = 0;
        CHK_RET(HcclGetHcclBuffer(comm, &buffer, &bufferSize));
        resCtx.localBuffer = CommBuffer{buffer, bufferSize};
        CHK_RET(BuildServerGroups(comm, param, resCtx.localServerRanks, resCtx.remoteServerRanks));
        const uint32_t localSize = static_cast<uint32_t>(resCtx.localServerRanks.size());
        const uint32_t remoteSize = static_cast<uint32_t>(resCtx.remoteServerRanks.size());
        resCtx.sendRoutePlan = custom_rs::GetRoutePlan(localSize, remoteSize);
        resCtx.receiveRoutePlan = custom_rs::GetRoutePlan(remoteSize, localSize);
        const uint64_t recvBytes = recvCount * sizeof(float);
        const uint64_t sendAggregateBytes =
            custom_rs::GetAggregateBytes(recvBytes, resCtx.sendRoutePlan);
        const uint64_t receiveAggregateBytes =
            custom_rs::GetAggregateBytes(recvBytes, resCtx.receiveRoutePlan);
        const auto localIt = std::find(
            resCtx.localServerRanks.begin(), resCtx.localServerRanks.end(), param.myRank);
        const uint32_t localIndex = static_cast<uint32_t>(localIt - resCtx.localServerRanks.begin());
        // On the 8-card side of 8+4, every local rank reduces one byte half:
        // ranks 0..3 own the first halves and ranks 4..7 own the second halves.
        // Each 4-card receiver waits for both leaders, so Clos traffic remains A.
        const bool splitEightToFour = localSize == 8 && remoteSize == 4;
        if (splitEightToFour) {
            const uint64_t firstHalf = (sendAggregateBytes / 2U) /
                custom_rs::SLOT_ALIGNMENT_BYTES * custom_rs::SLOT_ALIGNMENT_BYTES;
            const uint32_t targetIndex = localIndex % remoteSize;
            const bool secondHalf = localIndex >= remoteSize;
            resCtx.aggregateTargets.push_back(custom_rs::AggregateTask{
                resCtx.remoteServerRanks[targetIndex],
                secondHalf ? firstHalf : 0,
                secondHalf ? sendAggregateBytes - firstHalf : firstHalf});
        } else {
            for (uint32_t targetIndex = 0;
                targetIndex < resCtx.remoteServerRanks.size(); ++targetIndex) {
                if (targetIndex % localSize == localIndex) {
                    resCtx.aggregateTargets.push_back(custom_rs::AggregateTask{
                        resCtx.remoteServerRanks[targetIndex], 0, sendAggregateBytes});
                }
            }
        }
        if (!resCtx.remoteServerRanks.empty()) {
            if (localSize == 4 && remoteSize == 8) {
                const uint64_t firstHalf = (receiveAggregateBytes / 2U) /
                    custom_rs::SLOT_ALIGNMENT_BYTES * custom_rs::SLOT_ALIGNMENT_BYTES;
                resCtx.aggregateSources.push_back(custom_rs::AggregateSourceTask{
                    resCtx.remoteServerRanks[localIndex], 0, 0, firstHalf});
                resCtx.aggregateSources.push_back(custom_rs::AggregateSourceTask{
                    resCtx.remoteServerRanks[localIndex + localSize], 0,
                    firstHalf, receiveAggregateBytes - firstHalf});
            } else {
                const uint64_t sourceSlot = localSize == 8 && remoteSize == 4
                    ? localIndex / remoteSize : 0U;
                resCtx.aggregateSources.push_back(custom_rs::AggregateSourceTask{
                    resCtx.remoteServerRanks[localIndex % remoteSize],
                    sourceSlot * receiveAggregateBytes, 0,
                    custom_rs::GetAggregateBytes(recvBytes, resCtx.receiveRoutePlan)});
            }
        }
        std::vector<IoDieChannelGroup> groups;
        CHK_RET(AcquireChannels(comm, param, groups, resCtx.peerIoDies));
        // Two-layer routes need worker Threads for the second IO die and local
        // reduction Missions. Large 4x1 needs one worker for segmented reduction;
        // its single-die small path still runs entirely on the main Thread.
        resCtx.threads[0] = param.cpuThread;
        const bool smallDataKernel = recvBytes * param.rankSize <= custom_rs::SMALL_TOTAL_DATA_BYTES;
        const uint32_t workerCount = param.rankSize == 4
            ? (smallDataKernel ? 0U : 1U)
            : custom_rs::REQUIRED_THREAD_NUM - 1U;
        if (workerCount != 0) {
            CHK_RET(HcclThreadAcquire(comm, engine, workerCount,
                custom_rs::WORKER_THREAD_NOTIFY_NUM, &resCtx.threads[1]));
        }
        CHK_RET(RegisterKernels(comm, param, groups, smallDataKernel, resCtx));
        uint32_t next = 0;
        if (resCtx.activeIoDieMask & 1U) resCtx.ioDie0ThreadIndex = next++;
        if (resCtx.activeIoDieMask & 2U) resCtx.ioDie1ThreadIndex = next++;
        if (resCtx.activeIoDieMask & 1U) resCtx.ioDie0ReduceThreadIndex = next++;
        if (resCtx.activeIoDieMask & 2U) resCtx.ioDie1ReduceThreadIndex = next;
        // A one-layer topology still needs separate communication and local
        // streams.  Unused acquired Threads remain idle but preserve stable
        // serialized indices for every topology.
        if ((resCtx.activeIoDieMask & 1U) && !(resCtx.activeIoDieMask & 2U)) {
            resCtx.ioDie0ReduceThreadIndex = 1U;
        }
        if ((resCtx.activeIoDieMask & 2U) && !(resCtx.activeIoDieMask & 1U)) {
            resCtx.ioDie1ReduceThreadIndex = 1U;
        }
        std::vector<char> seq = resCtx.Serialize();
        param.ctxSize = seq.size();
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, engine, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, engine, param.tag, seq.data(), seq.size(), 0));
    }
    return ops_hccl::ExecOp(param);
}
