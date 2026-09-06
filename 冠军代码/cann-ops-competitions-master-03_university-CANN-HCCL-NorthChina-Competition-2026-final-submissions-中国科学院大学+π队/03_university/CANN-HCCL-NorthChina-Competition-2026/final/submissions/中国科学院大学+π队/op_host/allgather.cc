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
#include <cstddef>
#include <cstdio>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include <hccl/hccl_ccu_res.h>
#include <hccl/hccl_diag.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_res_expt.h>

#include "ccu_kernel.h"
#include "ccu_launch.h"
#include "common.h"
#include "custom.h"
#include "exec_op.h"
#include "hccl.h"
#include "log.h"

namespace {

constexpr uint32_t CHANNEL_NOTIFY_NUM = 3;
constexpr uint32_t NETWORK_LAYER_NUM = 2;
constexpr uint32_t KERNEL_ARG_NUM = 1;
constexpr uint32_t RESERVED_DIE_ID = 0;

struct SelectedLink {
    uint32_t layer = 0;
    CommLink link{};
};

struct PeerChannel {
    uint32_t remoteRank = 0;
    ChannelHandle channel{};
};

struct TopologyPlan {
    TopologyKind topology = TopologyKind::TOPO_UNKNOWN;
    TopologyRole role = TopologyRole::ROLE_COMMON;
    uint32_t localIndex = 0;
    std::vector<uint32_t> localRanks;
    std::vector<PeerChannel> layerChannels[NETWORK_LAYER_NUM];
    std::vector<PeerChannel> networkKernelChannels;
    uint32_t networkWriteMask = 0;
    uint32_t networkReuseMask = 0;
    uint32_t networkSyncMask = 0;
    uint32_t networkSyncWaitMask = 0;
    uint32_t relayCount = 0;
    uint32_t relayRanks[2]{};
};

struct PendingKernel {
    CcuKernelInfo info;
    CcuKernelHandle *destination = nullptr;
    uint32_t mask = 0;
};

HcclResult SelectLink(HcclComm comm, uint32_t localRank, uint32_t remoteRank, SelectedLink &selected)
{
    for (uint32_t layer = 0; layer < NETWORK_LAYER_NUM; ++layer) {
        CommLink *links = nullptr;
        uint32_t linkCount = 0;
        HcclResult result =
            HcclRankGraphGetLinks(comm, layer, localRank, remoteRank, &links, &linkCount);
        if (result != HCCL_SUCCESS || links == nullptr || linkCount == 0) {
            continue;
        }

        constexpr CommProtocol supportedProtocols[] = {
            CommProtocol::COMM_PROTOCOL_UBC_CTP,
            CommProtocol::COMM_PROTOCOL_UBC_TP,
        };
        for (CommProtocol protocol : supportedProtocols) {
            for (uint32_t index = 0; index < linkCount; ++index) {
                if (links[index].linkAttr.linkProtocol == protocol) {
                    selected.layer = layer;
                    selected.link = links[index];
                    return HCCL_SUCCESS;
                }
            }
        }
    }

    HCCL_ERROR("[SelectLink] no usable link between rank %u and rank %u", localRank, remoteRank);
    return HCCL_E_NOT_FOUND;
}

HcclResult AcquireChannel(HcclComm comm, uint32_t localRank, uint32_t remoteRank,
    SelectedLink &selected, ChannelHandle &channel)
{
    CHK_RET(SelectLink(comm, localRank, remoteRank, selected));

    HcclChannelDesc desc;
    CHK_RET(HcclChannelDescInit(&desc, 1));
    desc.remoteRank = remoteRank;
    desc.notifyNum = CHANNEL_NOTIFY_NUM;
    desc.channelProtocol = selected.link.linkAttr.linkProtocol;
    desc.localEndpoint.protocol = selected.link.srcEndpointDesc.protocol;
    desc.localEndpoint.commAddr = selected.link.srcEndpointDesc.commAddr;
    desc.localEndpoint.loc = selected.link.srcEndpointDesc.loc;
    desc.remoteEndpoint.protocol = selected.link.dstEndpointDesc.protocol;
    desc.remoteEndpoint.commAddr = selected.link.dstEndpointDesc.commAddr;
    desc.remoteEndpoint.loc = selected.link.dstEndpointDesc.loc;

    return HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CCU, &desc, 1, &channel);
}

HcclResult QueryLocalRanks(HcclComm comm, std::vector<uint32_t> &localRanks)
{
    uint32_t *ranks = nullptr;
    uint32_t rankCount = 0;
    CHK_RET(HcclRankGraphGetRanksByLayer(comm, 0, &ranks, &rankCount));
    CHK_PRT_RET(ranks == nullptr || rankCount == 0,
        HCCL_ERROR("[QueryLocalRanks] empty layer-0 rank group"), HCCL_E_INTERNAL);
    localRanks.assign(ranks, ranks + rankCount);
    std::sort(localRanks.begin(), localRanks.end());
    return HCCL_SUCCESS;
}

const PeerChannel *FindPeerChannel(
    const std::vector<PeerChannel> &channels, uint32_t remoteRank)
{
    auto it = std::find_if(channels.begin(), channels.end(),
        [remoteRank](const PeerChannel &peerChannel) {
            return peerChannel.remoteRank == remoteRank;
        });
    return it == channels.end() ? nullptr : &*it;
}

HcclResult AddNetworkPeer(TopologyPlan &plan, uint32_t remoteRank)
{
    const PeerChannel *peer = FindPeerChannel(plan.layerChannels[1], remoteRank);
    CHK_PRT_RET(peer == nullptr,
        HCCL_ERROR("[AddNetworkPeer] missing layer-1 channel to rank %u", remoteRank),
        HCCL_E_INTERNAL);
    plan.networkKernelChannels.push_back(*peer);
    return HCCL_SUCCESS;
}

HcclResult GetNetworkPeerBit(
    const TopologyPlan &plan, uint32_t remoteRank, uint32_t &peerBit)
{
    const PeerChannel *peer =
        FindPeerChannel(plan.layerChannels[1], remoteRank);
    CHK_PRT_RET(peer == nullptr,
        HCCL_ERROR("[GetNetworkPeerBit] missing layer-1 peer %u", remoteRank),
        HCCL_E_INTERNAL);
    const auto peerIt = std::find_if(
        plan.layerChannels[1].begin(), plan.layerChannels[1].end(),
        [remoteRank](const PeerChannel &channel) {
            return channel.remoteRank == remoteRank;
        });
    const uint32_t peerIndex =
        static_cast<uint32_t>(peerIt - plan.layerChannels[1].begin());
    CHK_PRT_RET(peerIndex >= MAX_RANK_SIZE,
        HCCL_ERROR("[GetNetworkPeerBit] invalid peer index %u", peerIndex),
        HCCL_E_INTERNAL);
    peerBit = 1U << peerIndex;
    return HCCL_SUCCESS;
}

HcclResult BuildHierarchicalPlan(const OpParam &param, TopologyPlan &plan)
{
    auto localIt = std::lower_bound(
        plan.localRanks.begin(), plan.localRanks.end(), param.myRank);
    CHK_PRT_RET(localIt == plan.localRanks.end() || *localIt != param.myRank,
        HCCL_ERROR("[BuildHierarchicalPlan] local rank %u is missing", param.myRank),
        HCCL_E_INTERNAL);
    const uint32_t localIndex =
        static_cast<uint32_t>(localIt - plan.localRanks.begin());
    plan.localIndex = localIndex;

    std::vector<uint32_t> remoteRanks;
    remoteRanks.reserve(plan.layerChannels[1].size());
    for (const PeerChannel &channel : plan.layerChannels[1]) {
        remoteRanks.push_back(channel.remoteRank);
    }
    std::sort(remoteRanks.begin(), remoteRanks.end());

    if (plan.topology == TopologyKind::TOPO_2X8) {
        CHK_PRT_RET(remoteRanks.size() != plan.localRanks.size(),
            HCCL_ERROR("[BuildHierarchicalPlan] invalid 2x8 remote group size %zu",
                remoteRanks.size()),
            HCCL_E_INTERNAL);
        const uint32_t crossPeer = remoteRanks[localIndex];
        CHK_RET(AddNetworkPeer(plan, crossPeer));
        uint32_t crossPeerBit = 0;
        CHK_RET(GetNetworkPeerBit(plan, crossPeer, crossPeerBit));
        plan.networkWriteMask = 1;
        plan.networkReuseMask = crossPeerBit;
        plan.networkSyncMask = 1;
        plan.networkSyncWaitMask = 1;
        plan.relayCount = 1;
        plan.relayRanks[0] = crossPeer;
        return HCCL_SUCCESS;
    }

    if (plan.role == TopologyRole::ROLE_8P4_FULL_SERVER) {
        constexpr uint32_t partialGroupSize = 4;
        CHK_PRT_RET(remoteRanks.size() != partialGroupSize,
            HCCL_ERROR("[BuildHierarchicalPlan] invalid partial group size %zu",
                remoteRanks.size()),
            HCCL_E_INTERNAL);
        const uint32_t remoteIndex = localIndex % partialGroupSize;
        const uint32_t crossPeer = remoteRanks[remoteIndex];
        CHK_RET(AddNetworkPeer(plan, crossPeer));
        uint32_t crossPeerBit = 0;
        CHK_RET(GetNetworkPeerBit(plan, crossPeer, crossPeerBit));
        plan.networkWriteMask = crossPeerBit;
        plan.networkReuseMask = crossPeerBit;
        plan.networkSyncMask = crossPeerBit;
        // Ranks 4..7 only export their seed to the partial server; unlike
        // ranks 0..3, they never relay an inbound partial-side seed locally.
        plan.networkSyncWaitMask =
            localIndex < partialGroupSize ? crossPeerBit : 0;
        if (localIndex < partialGroupSize) {
            plan.relayCount = 1;
            plan.relayRanks[0] = crossPeer;
        }
        return HCCL_SUCCESS;
    }

    CHK_PRT_RET(plan.role != TopologyRole::ROLE_8P4_PARTIAL_SERVER ||
            remoteRanks.size() != 8 || localIndex >= 4,
        HCCL_ERROR("[BuildHierarchicalPlan] invalid 8+4 partial-side resources"),
        HCCL_E_INTERNAL);
    const uint32_t firstPeer = remoteRanks[localIndex];
    const uint32_t secondPeer = remoteRanks[localIndex + 4];
    CHK_RET(AddNetworkPeer(plan, firstPeer));
    CHK_RET(AddNetworkPeer(plan, secondPeer));
    // Only the first full-server peer receives this partial-server input.
    // The second channel remains active in the opposite direction so that all
    // eight full-server inputs cross the inter-server cut exactly once.
    uint32_t firstPeerBit = 0;
    uint32_t secondPeerBit = 0;
    CHK_RET(GetNetworkPeerBit(plan, firstPeer, firstPeerBit));
    CHK_RET(GetNetworkPeerBit(plan, secondPeer, secondPeerBit));
    plan.networkWriteMask = firstPeerBit;
    plan.networkReuseMask = firstPeerBit | secondPeerBit;
    // Only the first full-side peer consumes the partial-side seed.  Both
    // full-side peers still record completion because the partial side needs
    // both inbound prefixes before starting its local relay.
    plan.networkSyncMask = firstPeerBit;
    plan.networkSyncWaitMask = firstPeerBit | secondPeerBit;
    plan.relayCount = 2;
    plan.relayRanks[0] = firstPeer;
    plan.relayRanks[1] = secondPeer;
    return HCCL_SUCCESS;
}

HcclResult BuildTopologyPlan(HcclComm comm, const OpParam &param, TopologyPlan &plan)
{
    CHK_RET(QueryLocalRanks(comm, plan.localRanks));

    for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
        if (remoteRank == param.myRank) {
            continue;
        }
        SelectedLink selected;
        ChannelHandle channel;
        CHK_RET(AcquireChannel(comm, param.myRank, remoteRank, selected, channel));
        CHK_PRT_RET(selected.layer >= NETWORK_LAYER_NUM,
            HCCL_ERROR("[BuildTopologyPlan] invalid layer %u", selected.layer), HCCL_E_INTERNAL);
        plan.layerChannels[selected.layer].push_back(PeerChannel{remoteRank, channel});
    }

    const uint32_t localPeerCount = static_cast<uint32_t>(plan.layerChannels[0].size());
    const uint32_t networkPeerCount = static_cast<uint32_t>(plan.layerChannels[1].size());
    const uint32_t localGroupSize = static_cast<uint32_t>(plan.localRanks.size());

    if (param.rankSize == 16 && localGroupSize == 8 &&
        localPeerCount == 7 && networkPeerCount == 8) {
        plan.topology = TopologyKind::TOPO_2X8;
    } else if (param.rankSize == 4 && localGroupSize == 1 &&
        localPeerCount == 0 && networkPeerCount == 3) {
        plan.topology = TopologyKind::TOPO_4X1;
    } else if (param.rankSize == 12 && localGroupSize == 8 &&
        localPeerCount == 7 && networkPeerCount == 4) {
        plan.topology = TopologyKind::TOPO_8P4;
        plan.role = TopologyRole::ROLE_8P4_FULL_SERVER;
    } else if (param.rankSize == 12 && localGroupSize == 4 &&
        localPeerCount == 3 && networkPeerCount == 8) {
        plan.topology = TopologyKind::TOPO_8P4;
        plan.role = TopologyRole::ROLE_8P4_PARTIAL_SERVER;
    } else {
        HCCL_ERROR("[BuildTopologyPlan] unsupported topology: rankSize=%u localGroup=%u "
                   "layer0Peers=%u layer1Peers=%u",
            param.rankSize, localGroupSize, localPeerCount, networkPeerCount);
        return HCCL_E_NOT_SUPPORT;
    }

    if (plan.topology == TopologyKind::TOPO_4X1) {
        plan.networkKernelChannels = plan.layerChannels[1];
        plan.networkWriteMask =
            (1U << static_cast<uint32_t>(plan.networkKernelChannels.size())) - 1U;
        return HCCL_SUCCESS;
    }
    return BuildHierarchicalPlan(param, plan);
}

std::vector<ChannelHandle> GetHandles(const std::vector<PeerChannel> &peerChannels)
{
    std::vector<ChannelHandle> handles;
    handles.reserve(peerChannels.size());
    for (const auto &peerChannel : peerChannels) {
        handles.push_back(peerChannel.channel);
    }
    return handles;
}

HcclResult AddDirectKernel(const char *name, const std::vector<ChannelHandle> &channels,
    uint32_t doLocalCopy, uint32_t writeMask,
    CcuKernelHandle &destination, uint32_t mask,
    std::vector<PendingKernel> &kernels,
    void *kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuDirectKernel),
    uint32_t publishMask = std::numeric_limits<uint32_t>::max(),
    uint32_t readyWaitMask = 0,
    uint32_t syncMask = 0,
    uint32_t syncWaitMask = std::numeric_limits<uint32_t>::max(),
    uint32_t waitWriteCq = 1)
{
    CHK_PRT_RET(channels.empty() || channels.size() >= MAX_RANK_SIZE,
        HCCL_ERROR("[AddDirectKernel] invalid channel count %zu for %s", channels.size(), name),
        HCCL_E_PARA);

    PendingKernel pending;
    int written =
        std::snprintf(pending.info.kernelFuncName, sizeof(pending.info.kernelFuncName), "%s", name);
    CHK_PRT_RET(written <= 0 ||
            static_cast<std::size_t>(written) >= sizeof(pending.info.kernelFuncName),
        HCCL_ERROR("[AddDirectKernel] invalid kernel name %s", name), HCCL_E_INTERNAL);
    pending.info.kernelFunc = kernelFunc;

    auto kernelArg = std::make_shared<ops_hccl::CcuKernelArgDirect>();
    kernelArg->channelCount = static_cast<uint32_t>(channels.size());
    kernelArg->doLocalCopy = doLocalCopy;
    const uint32_t channelMask = (1U << kernelArg->channelCount) - 1U;
    CHK_PRT_RET(writeMask == 0 || (writeMask & ~channelMask) != 0,
        HCCL_ERROR("[AddDirectKernel] invalid write mask 0x%x for %s",
            writeMask, name),
        HCCL_E_PARA);
    kernelArg->writeMask = writeMask;
    kernelArg->publishMask =
        publishMask == std::numeric_limits<uint32_t>::max() ?
        channelMask : publishMask;
    kernelArg->readyWaitMask =
        readyWaitMask == 0 ? channelMask : readyWaitMask;
    kernelArg->syncMask = syncMask == 0 ? channelMask : syncMask;
    kernelArg->syncWaitMask =
        syncWaitMask == std::numeric_limits<uint32_t>::max() ?
        kernelArg->syncMask : syncWaitMask;
    kernelArg->waitWriteCq = waitWriteCq;
    CHK_PRT_RET((kernelArg->publishMask & ~channelMask) != 0 ||
            (kernelArg->readyWaitMask & ~channelMask) != 0 ||
            (kernelArg->syncMask & ~channelMask) != 0 ||
            (kernelArg->syncWaitMask & ~channelMask) != 0,
        HCCL_ERROR("[AddDirectKernel] invalid publish/ready/sync masks "
            "0x%x/0x%x/0x%x/0x%x for %s", kernelArg->publishMask,
            kernelArg->readyWaitMask, kernelArg->syncMask,
            kernelArg->syncWaitMask, name),
        HCCL_E_PARA);
    for (uint32_t index = 0; index < kernelArg->channelCount; ++index) {
        kernelArg->channels[index] = channels[index];
    }
    pending.info.SetKernelArg(kernelArg);
    pending.destination = &destination;
    pending.mask = mask;
    kernels.push_back(std::move(pending));
    return HCCL_SUCCESS;
}

HcclResult AddSmallDirectKernel(const char *name,
    const std::vector<ChannelHandle> &channels, uint32_t doLocalCopy,
    CcuKernelHandle &destination, uint32_t mask,
    std::vector<PendingKernel> &kernels,
    void *kernelFunc =
        reinterpret_cast<void *>(ops_hccl::CcuSmallDirectKernel),
    uint32_t waitWriteCq = 1)
{
    CHK_PRT_RET(channels.empty() || channels.size() >= MAX_RANK_SIZE,
        HCCL_ERROR("[AddSmallDirectKernel] invalid channel count %zu for %s",
            channels.size(), name),
        HCCL_E_PARA);

    PendingKernel pending;
    int written =
        std::snprintf(pending.info.kernelFuncName,
            sizeof(pending.info.kernelFuncName), "%s", name);
    CHK_PRT_RET(written <= 0 ||
            static_cast<std::size_t>(written) >= sizeof(pending.info.kernelFuncName),
        HCCL_ERROR("[AddSmallDirectKernel] invalid kernel name %s", name),
        HCCL_E_INTERNAL);
    pending.info.kernelFunc = kernelFunc;

    auto kernelArg = std::make_shared<ops_hccl::CcuKernelArgSmallDirect>();
    kernelArg->channelCount = static_cast<uint32_t>(channels.size());
    kernelArg->doLocalCopy = doLocalCopy;
    kernelArg->waitWriteCq = waitWriteCq;
    for (uint32_t index = 0; index < kernelArg->channelCount; ++index) {
        kernelArg->channels[index] = channels[index];
    }
    pending.info.SetKernelArg(kernelArg);
    pending.destination = &destination;
    pending.mask = mask;
    kernels.push_back(std::move(pending));
    return HCCL_SUCCESS;
}

HcclResult AddRelayKernel(const TopologyPlan &plan,
    const std::vector<ChannelHandle> &localChannels,
    CcuKernelHandle &destination, std::vector<PendingKernel> &kernels)
{
    CHK_PRT_RET(localChannels.empty() || localChannels.size() >= MAX_RANK_SIZE ||
            plan.relayCount > 2,
        HCCL_ERROR("[AddRelayKernel] invalid channel/relay count %zu/%u",
            localChannels.size(), plan.relayCount),
        HCCL_E_PARA);
    PendingKernel pending;
    int written = std::snprintf(pending.info.kernelFuncName,
        sizeof(pending.info.kernelFuncName), "%s", "CcuAllGatherLocalRelay");
    CHK_PRT_RET(written <= 0 ||
            static_cast<std::size_t>(written) >= sizeof(pending.info.kernelFuncName),
        HCCL_ERROR("[AddRelayKernel] invalid kernel name"), HCCL_E_INTERNAL);
    pending.info.kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuRelayKernel);

    auto kernelArg = std::make_shared<ops_hccl::CcuKernelArgRelay>();
    kernelArg->channelCount = static_cast<uint32_t>(localChannels.size());
    kernelArg->relayCount = plan.relayCount;
    // The preceding local-direct kernel has already published and consumed
    // OUTPUT/TOKEN on these exact local channels in both hierarchical
    // topologies.  Reuse the initialized channel variables for relay.
    kernelArg->reuseDirectAddresses = 1;
    // Restore the v1.0 2x8 relay completion barrier.  The current 8+4 path is
    // intentionally left unchanged.
    kernelArg->postSync =
        plan.topology == TopologyKind::TOPO_2X8 ? 1U : 0U;
    for (uint32_t index = 0; index < kernelArg->channelCount; ++index) {
        kernelArg->channels[index] = localChannels[index];
    }
    pending.info.SetKernelArg(kernelArg);
    pending.destination = &destination;
    pending.mask = KERNEL_LOCAL_RELAY;
    kernels.push_back(std::move(pending));
    return HCCL_SUCCESS;
}

HcclResult RegisterKernels(
    HcclComm comm, std::vector<PendingKernel> &kernels, AlgResourceCtx &resource)
{
    CHK_PRT_RET(kernels.empty(),
        HCCL_ERROR("[RegisterKernels] no kernel requested"), HCCL_E_INTERNAL);

    CcuInsHandle insHandle{0};
    uint32_t insCount = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insCount));
    CHK_PRT_RET(insCount != 1,
        HCCL_ERROR("[RegisterKernels] expected one CCU instruction instance, got %u", insCount),
        HCCL_E_INTERNAL);

    CHK_RET_CCU(HcommCcuKernelRegisterStart(insHandle));
    for (auto &pending : kernels) {
        const void *kernelArgs[] = {pending.info.kernelArg};
        CcuKernelHandle kernelHandle;
        CHK_RET_CCU(HcommCcuKernelRegister(insHandle, RESERVED_DIE_ID,
            pending.info.kernelFuncName, pending.info.kernelFunc,
            kernelArgs, KERNEL_ARG_NUM, &kernelHandle));
        *pending.destination = kernelHandle;
        resource.kernelMask |= pending.mask;
    }
    CHK_RET_CCU(HcommCcuKernelRegisterEnd(insHandle));
    return HCCL_SUCCESS;
}

HcclResult CreateResource(
    HcclComm comm, aclrtStream stream, OpParam &param, AlgResourceCtx &resource)
{
    TopologyPlan plan;
    CHK_RET(BuildTopologyPlan(comm, param, plan));
    resource.topology = plan.topology;
    resource.role = plan.role;
    resource.relayCount = plan.relayCount;
    for (uint32_t index = 0; index < resource.relayCount; ++index) {
        resource.relayRanks[index] = plan.relayRanks[index];
    }
    if (plan.topology == TopologyKind::TOPO_8P4) {
        resource.networkPeerCount =
            static_cast<uint32_t>(plan.layerChannels[1].size());
        CHK_PRT_RET(resource.networkPeerCount >= MAX_RANK_SIZE,
            HCCL_ERROR("[CreateResource] invalid network peer count %u",
                resource.networkPeerCount),
            HCCL_E_INTERNAL);
        for (uint32_t index = 0; index < resource.networkPeerCount; ++index) {
            resource.networkPeerRanks[index] =
                plan.layerChannels[1][index].remoteRank;
        }
    }

    const uint64_t dataSize = param.count * sizeof(float);
    const bool isSmall = dataSize <= CUSTOM_SMALL_MESSAGE_LIMIT;
    CHK_RET(HcclThreadAcquireWithStream(
        comm, CommEngine::COMM_ENGINE_CCU, stream, 1, &param.cpuThread));
    resource.mainThread = param.cpuThread;
    resource.threadCount =
        (!plan.layerChannels[0].empty() &&
                !plan.layerChannels[1].empty()) ?
        2 : 1;
    if (resource.threadCount == 2) {
        CHK_RET(HcclThreadAcquire(
            comm, CommEngine::COMM_ENGINE_CCU, 1, 1, &resource.localThread));
    }

    const std::vector<ChannelHandle> localChannels = GetHandles(plan.layerChannels[0]);
    const std::vector<ChannelHandle> networkChannels =
        GetHandles(plan.layerChannels[1]);
    const std::vector<ChannelHandle> selectedSeedChannels =
        GetHandles(plan.networkKernelChannels);
    const std::vector<ChannelHandle> &seedChannels =
        plan.topology == TopologyKind::TOPO_8P4 ?
        networkChannels : selectedSeedChannels;
    resource.selfCopyOnNetwork =
        localChannels.empty() || localChannels.size() > networkChannels.size() ? 1U : 0U;
    const bool is8p4Small =
        plan.topology == TopologyKind::TOPO_8P4 && isSmall;
    std::vector<PendingKernel> kernels;
    kernels.reserve(is8p4Small ? 2 : 4);

    if (is8p4Small) {
        const uint32_t copyOnNetwork = resource.selfCopyOnNetwork;
        CHK_RET(AddSmallDirectKernel("CcuAllGatherNetworkSmall",
            networkChannels, copyOnNetwork,
            resource.networkSmallKernel, KERNEL_NETWORK_SMALL, kernels,
            reinterpret_cast<void *>(ops_hccl::CcuSmallDirectKernel), 0));
        CHK_RET(AddSmallDirectKernel("CcuAllGatherLocalSmall",
            localChannels, copyOnNetwork == 0 ? 1U : 0U,
            resource.localSmallKernel, KERNEL_LOCAL_SMALL, kernels,
            reinterpret_cast<void *>(ops_hccl::CcuSmallDirectKernel), 0));
    } else if (plan.topology == TopologyKind::TOPO_4X1) {
        if (isSmall) {
            CHK_RET(AddSmallDirectKernel("CcuAllGather4x1NetworkSmall",
                networkChannels, 1, resource.networkSmallKernel,
                KERNEL_NETWORK_SMALL, kernels,
                reinterpret_cast<void *>(
                    ops_hccl::Ccu4x1SmallDirectKernel)));
        } else {
            CHK_RET(AddDirectKernel("CcuAllGather4x1NetworkDirect",
                networkChannels, 1,
                (1U << static_cast<uint32_t>(networkChannels.size())) - 1U,
                resource.networkDirectKernel, KERNEL_NETWORK_DIRECT, kernels,
                reinterpret_cast<void *>(ops_hccl::Ccu4x1DirectKernel)));
        }
    } else {
        const char *seedName = plan.topology == TopologyKind::TOPO_2X8 ?
            "CcuAllGather2x8NetworkSeed" : "CcuAllGather8p4NetworkSeed";
        const char *networkName = plan.topology == TopologyKind::TOPO_2X8 ?
            "CcuAllGather2x8NetworkDirect" : "CcuAllGather8p4NetworkDirect";
        const char *localName = plan.topology == TopologyKind::TOPO_2X8 ?
            "CcuAllGather2x8LocalDirect" : "CcuAllGather8p4LocalDirect";
        void *largeDirectKernel =
            reinterpret_cast<void *>(ops_hccl::CcuLargeDirectKernel);
        if (plan.topology == TopologyKind::TOPO_8P4) {
            const uint32_t networkMask =
                (1U << static_cast<uint32_t>(networkChannels.size())) - 1U;
            CHK_RET(AddDirectKernel(seedName, seedChannels, 0,
                plan.networkWriteMask, resource.networkSeedKernel,
                KERNEL_NETWORK_SEED, kernels,
                largeDirectKernel,
                plan.networkReuseMask, plan.networkWriteMask,
                plan.networkSyncMask, plan.networkSyncWaitMask));
            CHK_RET(AddDirectKernel(networkName, networkChannels, 0,
                networkMask, resource.networkDirectKernel,
                KERNEL_NETWORK_DIRECT, kernels,
                largeDirectKernel,
                networkMask ^ plan.networkReuseMask,
                networkMask ^ plan.networkWriteMask, networkMask));
        } else {
            const uint32_t networkMask =
                (1U << static_cast<uint32_t>(networkChannels.size())) - 1U;
            void *directKernel = isSmall ?
                reinterpret_cast<void *>(ops_hccl::CcuDirectKernel) :
                largeDirectKernel;
            CHK_RET(AddDirectKernel(seedName, seedChannels, 0,
                plan.networkWriteMask,
                resource.networkSeedKernel, KERNEL_NETWORK_SEED, kernels,
                directKernel));
            CHK_RET(AddDirectKernel(networkName, networkChannels,
                isSmall ? resource.selfCopyOnNetwork : 0U, networkMask,
                resource.networkDirectKernel, KERNEL_NETWORK_DIRECT, kernels,
                directKernel,
                isSmall ? networkMask : networkMask ^ plan.networkReuseMask,
                isSmall ? networkMask : networkMask ^ plan.networkReuseMask,
                networkMask, networkMask, isSmall ? 0U : 1U));
        }
        const uint32_t localMask =
            (1U << static_cast<uint32_t>(localChannels.size())) - 1U;
        CHK_RET(AddDirectKernel(localName, localChannels,
            isSmall ? (resource.selfCopyOnNetwork == 0 ? 1U : 0U) : 1U,
            localMask,
            resource.localDirectKernel, KERNEL_LOCAL_DIRECT, kernels,
            isSmall ?
                reinterpret_cast<void *>(ops_hccl::CcuDirectKernel) :
                largeDirectKernel,
            std::numeric_limits<uint32_t>::max(), 0, 0,
            std::numeric_limits<uint32_t>::max(), isSmall ? 0U : 1U));
        CHK_RET(AddRelayKernel(
            plan, localChannels, resource.localRelayKernel, kernels));
    }

    return RegisterKernels(comm, kernels, resource);
}

HcclResult InitResourceContext(HcclComm comm, aclrtStream stream, OpParam &param)
{
    void *ctx = nullptr;
    uint64_t size = 0;
    if (HcclEngineCtxGet(
            comm, param.tag, CommEngine::COMM_ENGINE_CCU, &ctx, &size) == HCCL_SUCCESS) {
        param.resCtx = ctx;
        param.ctxSize = size;
        return HCCL_SUCCESS;
    }

    AlgResourceCtx resource{};
    CHK_RET(CreateResource(comm, stream, param, resource));
    param.ctxSize = sizeof(resource);
    CHK_RET(HcclEngineCtxCreate(
        comm, param.tag, CommEngine::COMM_ENGINE_CCU, param.ctxSize, &param.resCtx));
    CHK_RET(HcclEngineCtxCopy(comm, CommEngine::COMM_ENGINE_CCU,
        param.tag, &resource, sizeof(resource), 0));
    return HCCL_SUCCESS;
}

} // namespace

HcclResult HcclAllGather(
    void *sendBuf, void *recvBuf, uint64_t sendCount,
    HcclDataType dataType, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    CHK_PRT_RET(dataType != HCCL_DATA_TYPE_FP32,
        HCCL_ERROR("[HcclAllGather] only FP32 is supported, dataType=%d",
            static_cast<int>(dataType)),
        HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(sendCount >
            std::numeric_limits<uint64_t>::max() / sizeof(float),
        HCCL_ERROR("[HcclAllGather] sendCount overflows byte size"), HCCL_E_PARA);

    OpParam param;
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = sendCount;
    param.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_ALLGATHER;

    HcclDfxOpInfo dfxInfo{};
    char commName[COMM_INDENTIFIER_MAX_LENGTH]{};
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_PRT_RET(param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE ||
            param.myRank >= param.rankSize,
        HCCL_ERROR("[HcclAllGather] invalid rank configuration: rank=%u size=%u",
            param.myRank, param.rankSize),
        HCCL_E_PARA);

    const uint64_t dataSize = sendCount * sizeof(float);
    const bool isSmall = dataSize <= CUSTOM_SMALL_MESSAGE_LIMIT;
    const char *resourceTag = nullptr;
    if (param.rankSize == 12) {
        resourceTag = isSmall ?
            "hccl_custom_allgather_ccu_topo_v10_8p4_small" :
            "hccl_custom_allgather_ccu_topo_v9_8p4_large";
    } else if (param.rankSize == 4) {
        resourceTag = isSmall ?
            "hccl_custom_allgather_ccu_topo_v10_4x1_small" :
            "hccl_custom_allgather_ccu_topo_v10_4x1_large";
    } else {
        resourceTag = isSmall ?
            "hccl_custom_allgather_ccu_topo_v10_2x8_small" :
            "hccl_custom_allgather_ccu_topo_v10_2x8_large";
    }
    int written = std::snprintf(
        param.tag, sizeof(param.tag), "%s", resourceTag);
    CHK_PRT_RET(written <= 0 ||
            static_cast<std::size_t>(written) >= sizeof(param.tag),
        HCCL_ERROR("[HcclAllGather] failed to construct resource tag"), HCCL_E_INTERNAL);

    CHK_PRT_RET(dataSize > std::numeric_limits<uint64_t>::max() / param.rankSize,
        HCCL_ERROR("[HcclAllGather] output byte size overflows"), HCCL_E_PARA);

    CHK_RET(InitResourceContext(comm, stream, param));
    return ops_hccl::ExecOp(param);
}
