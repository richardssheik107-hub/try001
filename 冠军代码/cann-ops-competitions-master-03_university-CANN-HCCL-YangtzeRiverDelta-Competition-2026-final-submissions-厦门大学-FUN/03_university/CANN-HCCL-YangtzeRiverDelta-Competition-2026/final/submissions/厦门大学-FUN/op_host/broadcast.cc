/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Integrated Broadcast resource management and dispatch.
 */
#include <array>
#include <cstdio>
#include <limits>
#include <memory>
#include <vector>
#include <hccl/hccl_diag.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_res_expt.h>
#include <hccl/hccl_types.h>
#include "ccu_launch.h"
#include "hccl_ccu_res.h"
#include "ccu_kernel.h"
#include "common.h"
#include "custom.h"
#include "exec_op.h"
#include "hccl.h"
#include "log.h"

namespace bcast2x8_small_final {
namespace {

constexpr uint64_t SUPPORTED_512KB_SIZE = 512ULL * 1024ULL;

// Kernels use BUFFER=bit1, TOKEN=bit2 and DONE=bit3.
constexpr uint32_t CHANNEL_NOTIFY_NUM = 5;
constexpr uint32_t THREAD_NOTIFY_NUM = 1;
constexpr CommProtocol REQUIRED_PROTOCOL = CommProtocol::COMM_PROTOCOL_UBC_CTP;
constexpr const char *BASE_TAG = "hccl_bcast_final_v4_2x8_base";
constexpr const char *ROOT_TAG_PREFIX = "hccl_bcast_final_v4_2x8_small_root";
constexpr const char *LOG_TAG = "BCAST_FINAL_V4_2X8_SMALL";

struct ChannelSet {
    Broadcast2x8ChannelGroup groupType = BCAST_2X8_GROUP_INTRA;
    std::vector<ChannelHandle> channels;
    std::vector<uint32_t> peers;
    std::array<uint32_t, MAX_RANK_SIZE> channelIndexByRank{};
    uint32_t netLayer = INVALID_VALUE_RANKID;

    ChannelSet()
    {
        channelIndexByRank.fill(INVALID_VALUE_RANKID);
    }
};

uint32_t GetServerId(uint32_t rank)
{
    return rank < BCAST_2X8_SERVER1_BASE ? 0U : 1U;
}

uint32_t GetServerBase(uint32_t serverId)
{
    return serverId == 0U ? BCAST_2X8_SERVER0_BASE : BCAST_2X8_SERVER1_BASE;
}

uint32_t GetServerRankNum(uint32_t serverId)
{
    return serverId == 0U ? BCAST_2X8_SERVER0_RANK_NUM : BCAST_2X8_SERVER1_RANK_NUM;
}

bool GroupContainsRoot(uint32_t myRank, uint32_t root,
    Broadcast2x8ChannelGroup groupType)
{
    const bool sameServer = GetServerId(myRank) == GetServerId(root);
    return groupType == BCAST_2X8_GROUP_INTRA ? sameServer : !sameServer;
}

HcclResult FindLinkForPeer(HcclComm comm, uint32_t myRank, uint32_t remoteRank,
    CommLink &selectedLink, uint32_t &selectedLayer)
{
    uint32_t *layers = nullptr;
    uint32_t layerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layers, &layerNum));

    for (uint32_t layerIndex = 0; layerIndex < layerNum; ++layerIndex) {
        CommLink *linkList = nullptr;
        uint32_t listSize = 0;
        HcclResult ret = HcclRankGraphGetLinks(
            comm, layers[layerIndex], myRank, remoteRank, &linkList, &listSize);
        if (ret != HCCL_SUCCESS || linkList == nullptr) {
            continue;
        }
        for (uint32_t linkIndex = 0; linkIndex < listSize; ++linkIndex) {
            if (linkList[linkIndex].linkAttr.linkProtocol == REQUIRED_PROTOCOL) {
                selectedLink = linkList[linkIndex];
                selectedLayer = layers[layerIndex];
                return HCCL_SUCCESS;
            }
        }
    }

    HCCL_ERROR("[%s] no UBC_CTP link found, myRank=%u, peer=%u",
        LOG_TAG, myRank, remoteRank);
    return HCCL_E_NOT_FOUND;
}

HcclResult AcquireOneChannel(HcclComm comm, const OpParam &param, uint32_t remoteRank,
    ChannelSet &set, std::array<bool, MAX_RANK_SIZE> &peerAcquired)
{
    CHK_PRT_RET(remoteRank >= param.rankSize || remoteRank == param.myRank,
        HCCL_ERROR("[%s] invalid remoteRank=%u for myRank=%u",
            LOG_TAG, remoteRank, param.myRank),
        HCCL_E_PARA);
    CHK_PRT_RET(peerAcquired[remoteRank],
        HCCL_ERROR("[%s] peer=%u would receive more than one channel", LOG_TAG, remoteRank),
        HCCL_E_PARA);
    CHK_PRT_RET(set.channelIndexByRank[remoteRank] != INVALID_VALUE_RANKID,
        HCCL_ERROR("[%s] duplicated channel in group, peer=%u", LOG_TAG, remoteRank),
        HCCL_E_PARA);

    CommLink link{};
    uint32_t netLayer = INVALID_VALUE_RANKID;
    CHK_RET(FindLinkForPeer(comm, param.myRank, remoteRank, link, netLayer));

    if (set.netLayer == INVALID_VALUE_RANKID) {
        set.netLayer = netLayer;
    } else {
        CHK_PRT_RET(set.netLayer != netLayer,
            HCCL_ERROR("[%s] one channel group spans multiple layers: previous=%u, current=%u, peer=%u",
                LOG_TAG, set.netLayer, netLayer, remoteRank),
            HCCL_E_NOT_SUPPORT);
    }

    HcclChannelDesc desc{};
    CHK_RET(HcclChannelDescInit(&desc, 1));
    desc.remoteRank = remoteRank;
    desc.notifyNum = CHANNEL_NOTIFY_NUM;
    desc.channelProtocol = link.linkAttr.linkProtocol;
    desc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
    desc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
    desc.localEndpoint.loc = link.srcEndpointDesc.loc;
    desc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
    desc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
    desc.remoteEndpoint.loc = link.dstEndpointDesc.loc;

    ChannelHandle channel{};
    CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CCU, &desc, 1, &channel));

    const uint32_t channelIndex = static_cast<uint32_t>(set.channels.size());
    set.channels.push_back(channel);
    set.peers.push_back(remoteRank);
    set.channelIndexByRank[remoteRank] = channelIndex;
    peerAcquired[remoteRank] = true;
    return HCCL_SUCCESS;
}

HcclResult Acquire2x8Channels(HcclComm comm, const OpParam &param,
    ChannelSet &intraSet, ChannelSet &interSet)
{
    intraSet.groupType = BCAST_2X8_GROUP_INTRA;
    interSet.groupType = BCAST_2X8_GROUP_INTER;

    std::array<bool, MAX_RANK_SIZE> peerAcquired{};
    peerAcquired.fill(false);

    const uint32_t myServer = GetServerId(param.myRank);
    const uint32_t remoteServer = 1U - myServer;
    const uint32_t sameServerBase = GetServerBase(myServer);
    const uint32_t sameServerRankNum = GetServerRankNum(myServer);
    const uint32_t remoteServerBase = GetServerBase(remoteServer);
    const uint32_t remoteServerRankNum = GetServerRankNum(remoteServer);

    for (uint32_t local = 0; local < sameServerRankNum; ++local) {
        const uint32_t peer = sameServerBase + local;
        if (peer == param.myRank) {
            continue;
        }
        CHK_RET(AcquireOneChannel(comm, param, peer, intraSet, peerAcquired));
    }

    for (uint32_t local = 0; local < remoteServerRankNum; ++local) {
        CHK_RET(AcquireOneChannel(
            comm, param, remoteServerBase + local, interSet, peerAcquired));
    }

    const size_t expectedIntra = static_cast<size_t>(sameServerRankNum - 1U);
    const size_t expectedInter = static_cast<size_t>(remoteServerRankNum);
    CHK_PRT_RET(intraSet.channels.size() != expectedIntra ||
            interSet.channels.size() != expectedInter,
        HCCL_ERROR("[%s] unexpected channel counts, intra=%zu/%zu, inter=%zu/%zu",
            LOG_TAG, intraSet.channels.size(), expectedIntra,
            interSet.channels.size(), expectedInter),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(intraSet.netLayer == INVALID_VALUE_RANKID ||
            interSet.netLayer == INVALID_VALUE_RANKID,
        HCCL_ERROR("[%s] failed to resolve topology layers, intra=%u, inter=%u",
            LOG_TAG, intraSet.netLayer, interSet.netLayer),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(intraSet.netLayer == interSet.netLayer,
        HCCL_ERROR("[%s] intra and inter channels resolved to the same layer=%u; "
            "one CCU kernel must not span both IO Dies",
            LOG_TAG, intraSet.netLayer),
        HCCL_E_NOT_SUPPORT);

    for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
        if (peer != param.myRank) {
            CHK_PRT_RET(!peerAcquired[peer],
                HCCL_ERROR("[%s] missing channel to peer=%u", LOG_TAG, peer),
                HCCL_E_INTERNAL);
        }
    }
    return HCCL_SUCCESS;
}

void SaveChannelSets(const ChannelSet &intraSet, const ChannelSet &interSet,
    AlgResourceCtx &resCtx)
{
    resCtx.intraChannels = intraSet.channels;
    resCtx.intraPeers = intraSet.peers;
    resCtx.intraLayer = intraSet.netLayer;
    resCtx.interChannels = interSet.channels;
    resCtx.interPeers = interSet.peers;
    resCtx.interLayer = interSet.netLayer;
}

HcclResult RestoreOneChannelSet(const std::vector<ChannelHandle> &channels,
    const std::vector<uint32_t> &peers, uint32_t netLayer,
    Broadcast2x8ChannelGroup groupType, ChannelSet &set)
{
    CHK_PRT_RET(channels.size() != peers.size() || channels.empty(),
        HCCL_ERROR("[%s] invalid cached group=%u, channels=%zu, peers=%zu",
            LOG_TAG, static_cast<uint32_t>(groupType), channels.size(), peers.size()),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(netLayer == INVALID_VALUE_RANKID,
        HCCL_ERROR("[%s] invalid cached layer for group=%u",
            LOG_TAG, static_cast<uint32_t>(groupType)),
        HCCL_E_INTERNAL);

    set.groupType = groupType;
    set.channels = channels;
    set.peers = peers;
    set.netLayer = netLayer;
    set.channelIndexByRank.fill(INVALID_VALUE_RANKID);

    for (size_t index = 0; index < peers.size(); ++index) {
        const uint32_t peer = peers[index];
        CHK_PRT_RET(peer >= BCAST_2X8_RANK_SIZE,
            HCCL_ERROR("[%s] invalid cached peer=%u", LOG_TAG, peer),
            HCCL_E_INTERNAL);
        CHK_PRT_RET(set.channelIndexByRank[peer] != INVALID_VALUE_RANKID,
            HCCL_ERROR("[%s] duplicate cached peer=%u", LOG_TAG, peer),
            HCCL_E_INTERNAL);
        set.channelIndexByRank[peer] = static_cast<uint32_t>(index);
    }
    return HCCL_SUCCESS;
}

HcclResult RestoreChannelSets(const AlgResourceCtx &baseCtx,
    ChannelSet &intraSet, ChannelSet &interSet)
{
    CHK_RET(RestoreOneChannelSet(baseCtx.intraChannels, baseCtx.intraPeers,
        baseCtx.intraLayer, BCAST_2X8_GROUP_INTRA, intraSet));
    CHK_RET(RestoreOneChannelSet(baseCtx.interChannels, baseCtx.interPeers,
        baseCtx.interLayer, BCAST_2X8_GROUP_INTER, interSet));
    CHK_PRT_RET(intraSet.netLayer == interSet.netLayer,
        HCCL_ERROR("[%s] cached intra/inter groups share layer=%u",
            LOG_TAG, intraSet.netLayer),
        HCCL_E_NOT_SUPPORT);
    return HCCL_SUCCESS;
}

std::shared_ptr<Broadcast2x8SmallSpecializedKernelArg> BuildSpecializedKernelArg(
    const OpParam &param, uint32_t root, const ChannelSet &set)
{
    auto arg = std::make_shared<Broadcast2x8SmallSpecializedKernelArg>();
    arg->rankId = param.myRank;
    arg->rankSize = param.rankSize;
    arg->rootRank = root;
    arg->serverId = GetServerId(param.myRank);
    arg->groupType = static_cast<uint32_t>(set.groupType);
    arg->netLayer = set.netLayer;

    if (param.myRank == root) {
        arg->role = BCAST_2X8_ROLE_SENDER;
        arg->channelCount = static_cast<uint32_t>(set.channels.size());
        for (uint32_t index = 0; index < arg->channelCount; ++index) {
            arg->channels[index] = set.channels[index];
            arg->peerRanks[index] = set.peers[index];
        }
        return arg;
    }

    if (GroupContainsRoot(param.myRank, root, set.groupType)) {
        const uint32_t rootChannelIndex = set.channelIndexByRank[root];
        if (rootChannelIndex == INVALID_VALUE_RANKID ||
            rootChannelIndex >= set.channels.size()) {
            return nullptr;
        }
        arg->role = BCAST_2X8_ROLE_RECEIVER;
        arg->channelCount = 1;
        arg->channels[0] = set.channels[rootChannelIndex];
        arg->peerRanks[0] = root;
        return arg;
    }

    // The host never launches this layer on a non-root receiver. Keep one
    // channel solely so the translator can place this idle kernel on the
    // intended IO Die without binding the full unused channel group.
    if (set.channels.empty()) {
        return nullptr;
    }
    arg->role = BCAST_2X8_ROLE_IDLE;
    arg->channelCount = 1;
    arg->channels[0] = set.channels[0];
    arg->peerRanks[0] = set.peers[0];
    return arg;
}

HcclResult RegisterOneSpecializedKernel(CcuInsHandle insHandle,
    uint32_t kernelIndex, const char *kernelName, void *kernelFunc,
    const OpParam &param, uint32_t root, const ChannelSet &set,
    AlgResourceCtx &rootCtx)
{
    auto kernelArg = BuildSpecializedKernelArg(param, root, set);
    CHK_PTR_NULL(kernelArg);

    const void *kernelArgArray[] = {kernelArg.get()};
    CcuKernelHandle kernelHandle{};
    constexpr uint32_t dieId = 0;
    constexpr uint32_t kernelArgNum = 1;

    CcuResult ret = HcommCcuKernelRegister(insHandle, dieId,
        kernelName, kernelFunc, kernelArgArray, kernelArgNum, &kernelHandle);
    if (ret != CCU_SUCCESS) {
        HCCL_ERROR("[%s] kernel register failed, root=%u, rank=%u, index=%u, "
            "role=%u, group=%u, layer=%u, channels=%u, ccuRet=%d",
            LOG_TAG, root, param.myRank, kernelIndex, kernelArg->role,
            kernelArg->groupType, kernelArg->netLayer,
            kernelArg->channelCount, ret);
        return ConvertCcuToHccl(ret);
    }
    rootCtx.ccuKernels[kernelIndex] = kernelHandle;
    return HCCL_SUCCESS;
}

HcclResult RegisterRootSpecializedKernels(HcclComm comm, const OpParam &param,
    uint32_t root, const ChannelSet &intraSet, const ChannelSet &interSet,
    AlgResourceCtx &rootCtx)
{
    CcuInsHandle insHandle{0};
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum != 1,
        HCCL_ERROR("[%s] unexpected ccu insNum=%u", LOG_TAG, insNum),
        HCCL_E_INTERNAL);

    rootCtx.ccuKernels.assign(BCAST_2X8_SMALL_SPECIALIZED_KERNEL_NUM,
        CcuKernelHandle{});

    char intraName[128]{};
    char interName[128]{};
    int intraWritten = std::snprintf(intraName, sizeof(intraName),
        "Ccu2x8SmallRootSpecializedIntra_r%u", root);
    int interWritten = std::snprintf(interName, sizeof(interName),
        "Ccu2x8SmallRootSpecializedInter_r%u", root);
    CHK_PRT_RET(intraWritten <= 0 || static_cast<size_t>(intraWritten) >= sizeof(intraName) ||
            interWritten <= 0 || static_cast<size_t>(interWritten) >= sizeof(interName),
        HCCL_ERROR("[%s] failed to build specialized kernel names, root=%u", LOG_TAG, root),
        HCCL_E_INTERNAL);

    CHK_RET_CCU(HcommCcuKernelRegisterStart(insHandle));
    CHK_RET(RegisterOneSpecializedKernel(insHandle,
        BCAST_2X8_SMALL_SPECIALIZED_INTRA, intraName,
        reinterpret_cast<void *>(ops_hccl::Ccu2x8SmallRootSpecializedIntraKernel),
        param, root, intraSet, rootCtx));
    CHK_RET(RegisterOneSpecializedKernel(insHandle,
        BCAST_2X8_SMALL_SPECIALIZED_INTER, interName,
        reinterpret_cast<void *>(ops_hccl::Ccu2x8SmallRootSpecializedInterKernel),
        param, root, interSet, rootCtx));
    CHK_RET_CCU(HcommCcuKernelRegisterEnd(insHandle));
    return HCCL_SUCCESS;
}

HcclResult CreateBaseResources(HcclComm comm, const OpParam &param,
    AlgResourceCtx &baseCtx)
{
    const CommEngine engine = CommEngine::COMM_ENGINE_CCU;
    CHK_RET(HcclThreadAcquire(comm, engine, 1,
        THREAD_NOTIFY_NUM, &baseCtx.workerThread));

    ChannelSet intraSet{};
    ChannelSet interSet{};
    CHK_RET(Acquire2x8Channels(comm, param, intraSet, interSet));
    SaveChannelSets(intraSet, interSet, baseCtx);
    return HCCL_SUCCESS;
}

HcclResult GetOrCreateBaseContext(HcclComm comm, const OpParam &param,
    AlgResourceCtx &baseCtx)
{
    const CommEngine engine = CommEngine::COMM_ENGINE_CCU;
    void *ctx = nullptr;
    uint64_t ctxSize = 0;
    if (HcclEngineCtxGet(comm, BASE_TAG, engine, &ctx, &ctxSize) == HCCL_SUCCESS) {
        CHK_PTR_NULL(ctx);
        std::vector<char> sequence(static_cast<char *>(ctx),
            static_cast<char *>(ctx) + ctxSize);
        baseCtx.DeSerialize(sequence);
        return HCCL_SUCCESS;
    }

    CHK_RET(CreateBaseResources(comm, param, baseCtx));
    std::vector<char> sequence = baseCtx.Serialize();
    void *createdCtx = nullptr;
    CHK_RET(HcclEngineCtxCreate(
        comm, BASE_TAG, engine, sequence.size(), &createdCtx));
    CHK_RET(HcclEngineCtxCopy(
        comm, engine, BASE_TAG, sequence.data(), sequence.size(), 0));
    return HCCL_SUCCESS;
}

HcclResult GetOrCreateRootContext(HcclComm comm, const OpParam &param,
    const AlgResourceCtx &baseCtx, char *rootTag, size_t rootTagSize,
    void **ctx, uint64_t *ctxSize)
{
    CHK_PTR_NULL(rootTag);
    CHK_PTR_NULL(ctx);
    CHK_PTR_NULL(ctxSize);

    int written = std::snprintf(rootTag, rootTagSize, "%s_%u",
        ROOT_TAG_PREFIX, param.root);
    CHK_PRT_RET(written <= 0 || static_cast<size_t>(written) >= rootTagSize,
        HCCL_ERROR("[%s] failed to construct root tag, root=%u", LOG_TAG, param.root),
        HCCL_E_INTERNAL);

    const CommEngine engine = CommEngine::COMM_ENGINE_CCU;
    if (HcclEngineCtxGet(comm, rootTag, engine, ctx, ctxSize) == HCCL_SUCCESS) {
        return HCCL_SUCCESS;
    }

    ChannelSet intraSet{};
    ChannelSet interSet{};
    CHK_RET(RestoreChannelSets(baseCtx, intraSet, interSet));

    AlgResourceCtx rootCtx = baseCtx;
    rootCtx.ccuKernels.clear();
    CHK_RET(RegisterRootSpecializedKernels(
        comm, param, param.root, intraSet, interSet, rootCtx));

    std::vector<char> sequence = rootCtx.Serialize();
    CHK_RET(HcclEngineCtxCreate(
        comm, rootTag, engine, sequence.size(), ctx));
    CHK_RET(HcclEngineCtxCopy(
        comm, engine, rootTag, sequence.data(), sequence.size(), 0));
    *ctxSize = sequence.size();
    return HCCL_SUCCESS;
}

} // namespace

HcclResult HcclBroadcast2x8SmallRootSpecializedImpl(
    void *buf, uint64_t count, HcclDataType dataType, uint32_t root,
    HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(buf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    OpParam param{};
    param.inputPtr = buf;
    param.outputPtr = buf;
    param.count = count;
    param.root = root;
    param.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_BROADCAST;

    HcclDfxOpInfo dfxInfo{};
    char commName[COMM_INDENTIFIER_MAX_LENGTH]{};
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_PRT_RET(param.rankSize != BCAST_2X8_RANK_SIZE,
        HCCL_ERROR("[%s] only 2x8 rankSize=16 is supported, rankSize=%u",
            LOG_TAG, param.rankSize),
        HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(root >= param.rankSize,
        HCCL_ERROR("[%s] invalid root=%u, rankSize=%u", LOG_TAG, root, param.rankSize),
        HCCL_E_PARA);
    CHK_PRT_RET(dataType != HCCL_DATA_TYPE_FP32,
        HCCL_ERROR("[%s] only FP32 is supported, dataType=%d",
            LOG_TAG, static_cast<int>(dataType)),
        HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(count > std::numeric_limits<uint64_t>::max() / sizeof(float),
        HCCL_ERROR("[%s] count too large: %lu", LOG_TAG, count), HCCL_E_PARA);

    const uint64_t dataSize = count * sizeof(float);
    CHK_PRT_RET(dataSize != SUPPORTED_512KB_SIZE,
        HCCL_ERROR("[%s] only 512KB is supported, dataSize=%lu", LOG_TAG, dataSize),
        HCCL_E_NOT_SUPPORT);

    const CommEngine engine = CommEngine::COMM_ENGINE_CCU;
    CHK_RET(HcclThreadAcquireWithStream(comm, engine, stream,
        THREAD_NOTIFY_NUM, &param.cpuThread));

    AlgResourceCtx baseCtx{};
    CHK_RET(GetOrCreateBaseContext(comm, param, baseCtx));

    char rootTag[128]{};
    CHK_RET(GetOrCreateRootContext(comm, param, baseCtx,
        rootTag, sizeof(rootTag), &param.resCtx, &param.ctxSize));

    int tagWritten = std::snprintf(param.tag, sizeof(param.tag), "%s", rootTag);
    CHK_PRT_RET(tagWritten <= 0 || static_cast<size_t>(tagWritten) >= sizeof(param.tag),
        HCCL_ERROR("[%s] failed to copy root tag", LOG_TAG),
        HCCL_E_INTERNAL);

    return ops_hccl::ExecOp2x8SmallRootSpecializedBalanced(param);
}

} // namespace bcast2x8_small_final

namespace bcast2x8_large_final {
namespace {

constexpr uint64_t SUPPORTED_512MB_SIZE = 512ULL * 1024ULL * 1024ULL;
constexpr uint64_t SUPPORTED_400MB_4B_SIZE = 400ULL * 1024ULL * 1024ULL + 4ULL;

// The compact pipeline safely reuses bits only after the previous stage has
// completed its full Record -> Wait lifecycle. Highest bit is 4.
constexpr uint32_t CHANNEL_NOTIFY_NUM = 5;

// All stages reuse one thread-notify index. Queue order guarantees that the
// previous start/done Record -> Wait pair is consumed before the next Record.
constexpr uint32_t THREAD_NOTIFY_NUM = 1;
constexpr CommProtocol REQUIRED_PROTOCOL = CommProtocol::COMM_PROTOCOL_UBC_CTP;
constexpr const char *BASE_TAG = "hccl_bcast_final_v4_2x8_base";
constexpr const char *ROOT_TAG_PREFIX = "hccl_bcast_final_v4_2x8_large_root";
constexpr const char *LOG_TAG = "BCAST_FINAL_V4_2X8_LARGE";

struct ChannelSet {
    Broadcast2x8ChannelGroup groupType = BCAST_2X8_GROUP_INTRA;
    std::vector<ChannelHandle> channels;
    std::vector<uint32_t> peers;
    std::array<uint32_t, MAX_RANK_SIZE> channelIndexByRank{};
    uint32_t netLayer = INVALID_VALUE_RANKID;

    ChannelSet()
    {
        channelIndexByRank.fill(INVALID_VALUE_RANKID);
    }
};

bool IsServer0Rank(uint32_t rank)
{
    return rank < BCAST_2X8_SERVER0_RANK_NUM;
}

uint32_t GetServerId(uint32_t rank)
{
    return IsServer0Rank(rank) ? 0U : 1U;
}

uint32_t GetServerBase(uint32_t serverId)
{
    return serverId == 0U ? BCAST_2X8_SERVER0_BASE : BCAST_2X8_SERVER1_BASE;
}

uint32_t GetServerRankNum(uint32_t serverId)
{
    return serverId == 0U ? BCAST_2X8_SERVER0_RANK_NUM : BCAST_2X8_SERVER1_RANK_NUM;
}

bool IsSupportedLargeSize(uint64_t dataSize)
{
    return dataSize == SUPPORTED_512MB_SIZE || dataSize == SUPPORTED_400MB_4B_SIZE;
}

HcclResult FindLinkForPeer(HcclComm comm, uint32_t myRank, uint32_t remoteRank,
    CommLink &selectedLink, uint32_t &selectedLayer)
{
    uint32_t *layers = nullptr;
    uint32_t layerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layers, &layerNum));

    for (uint32_t layerIndex = 0; layerIndex < layerNum; ++layerIndex) {
        CommLink *linkList = nullptr;
        uint32_t listSize = 0;
        HcclResult ret = HcclRankGraphGetLinks(
            comm, layers[layerIndex], myRank, remoteRank, &linkList, &listSize);
        if (ret != HCCL_SUCCESS || linkList == nullptr) {
            continue;
        }
        for (uint32_t linkIndex = 0; linkIndex < listSize; ++linkIndex) {
            if (linkList[linkIndex].linkAttr.linkProtocol == REQUIRED_PROTOCOL) {
                selectedLink = linkList[linkIndex];
                selectedLayer = layers[layerIndex];
                return HCCL_SUCCESS;
            }
        }
    }

    HCCL_ERROR("[%s] no UBC_CTP link found, myRank=%u, peer=%u",
        LOG_TAG, myRank, remoteRank);
    return HCCL_E_NOT_FOUND;
}

HcclResult AcquireOneChannel(HcclComm comm, const OpParam &param, uint32_t remoteRank,
    ChannelSet &set, std::array<bool, MAX_RANK_SIZE> &peerAcquired)
{
    CHK_PRT_RET(remoteRank >= param.rankSize || remoteRank == param.myRank,
        HCCL_ERROR("[%s] invalid remoteRank=%u for myRank=%u",
            LOG_TAG, remoteRank, param.myRank),
        HCCL_E_PARA);
    CHK_PRT_RET(peerAcquired[remoteRank],
        HCCL_ERROR("[%s] peer=%u would receive more than one channel", LOG_TAG, remoteRank),
        HCCL_E_PARA);
    CHK_PRT_RET(set.channelIndexByRank[remoteRank] != INVALID_VALUE_RANKID,
        HCCL_ERROR("[%s] duplicated channel in group, peer=%u", LOG_TAG, remoteRank),
        HCCL_E_PARA);

    CommLink link{};
    uint32_t netLayer = INVALID_VALUE_RANKID;
    CHK_RET(FindLinkForPeer(comm, param.myRank, remoteRank, link, netLayer));

    if (set.netLayer == INVALID_VALUE_RANKID) {
        set.netLayer = netLayer;
    } else {
        CHK_PRT_RET(set.netLayer != netLayer,
            HCCL_ERROR("[%s] one channel group spans multiple layers: previous=%u, current=%u, peer=%u",
                LOG_TAG, set.netLayer, netLayer, remoteRank),
            HCCL_E_NOT_SUPPORT);
    }

    HcclChannelDesc desc{};
    CHK_RET(HcclChannelDescInit(&desc, 1));
    desc.remoteRank = remoteRank;
    desc.notifyNum = CHANNEL_NOTIFY_NUM;
    desc.channelProtocol = link.linkAttr.linkProtocol;
    desc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
    desc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
    desc.localEndpoint.loc = link.srcEndpointDesc.loc;
    desc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
    desc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
    desc.remoteEndpoint.loc = link.dstEndpointDesc.loc;

    ChannelHandle channel{};
    CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CCU, &desc, 1, &channel));

    const uint32_t channelIndex = static_cast<uint32_t>(set.channels.size());
    set.channels.push_back(channel);
    set.peers.push_back(remoteRank);
    set.channelIndexByRank[remoteRank] = channelIndex;
    peerAcquired[remoteRank] = true;
    return HCCL_SUCCESS;
}

HcclResult Acquire2x8Channels(HcclComm comm, const OpParam &param,
    ChannelSet &intraSet, ChannelSet &interSet)
{
    intraSet.groupType = BCAST_2X8_GROUP_INTRA;
    interSet.groupType = BCAST_2X8_GROUP_INTER;

    std::array<bool, MAX_RANK_SIZE> peerAcquired{};
    peerAcquired.fill(false);

    const uint32_t myServer = GetServerId(param.myRank);
    const uint32_t remoteServer = 1U - myServer;
    const uint32_t sameServerBase = GetServerBase(myServer);
    const uint32_t sameServerRankNum = GetServerRankNum(myServer);
    const uint32_t remoteServerBase = GetServerBase(remoteServer);
    const uint32_t remoteServerRankNum = GetServerRankNum(remoteServer);

    for (uint32_t local = 0; local < sameServerRankNum; ++local) {
        const uint32_t peer = sameServerBase + local;
        if (peer == param.myRank) {
            continue;
        }
        CHK_RET(AcquireOneChannel(comm, param, peer, intraSet, peerAcquired));
    }

    for (uint32_t local = 0; local < remoteServerRankNum; ++local) {
        CHK_RET(AcquireOneChannel(
            comm, param, remoteServerBase + local, interSet, peerAcquired));
    }

    const size_t expectedIntra = static_cast<size_t>(sameServerRankNum - 1U);
    const size_t expectedInter = static_cast<size_t>(remoteServerRankNum);
    CHK_PRT_RET(intraSet.channels.size() != expectedIntra ||
            interSet.channels.size() != expectedInter,
        HCCL_ERROR("[%s] unexpected channel counts, intra=%zu/%zu, inter=%zu/%zu",
            LOG_TAG, intraSet.channels.size(), expectedIntra,
            interSet.channels.size(), expectedInter),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(intraSet.netLayer == INVALID_VALUE_RANKID ||
            interSet.netLayer == INVALID_VALUE_RANKID,
        HCCL_ERROR("[%s] failed to resolve topology layers, intra=%u, inter=%u",
            LOG_TAG, intraSet.netLayer, interSet.netLayer),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(intraSet.netLayer == interSet.netLayer,
        HCCL_ERROR("[%s] intra and inter channels resolved to the same layer=%u; "
            "one CCU kernel must not span two IO Dies",
            LOG_TAG, intraSet.netLayer),
        HCCL_E_NOT_SUPPORT);

    for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
        if (peer != param.myRank) {
            CHK_PRT_RET(!peerAcquired[peer],
                HCCL_ERROR("[%s] missing channel to peer=%u", LOG_TAG, peer),
                HCCL_E_INTERNAL);
        }
    }
    return HCCL_SUCCESS;
}

void SaveChannelSets(const ChannelSet &intraSet, const ChannelSet &interSet,
    AlgResourceCtx &resCtxHost)
{
    resCtxHost.intraChannels = intraSet.channels;
    resCtxHost.intraPeers = intraSet.peers;
    resCtxHost.intraLayer = intraSet.netLayer;
    resCtxHost.interChannels = interSet.channels;
    resCtxHost.interPeers = interSet.peers;
    resCtxHost.interLayer = interSet.netLayer;
}

HcclResult RestoreOneChannelSet(const std::vector<ChannelHandle> &channels,
    const std::vector<uint32_t> &peers, uint32_t netLayer,
    Broadcast2x8ChannelGroup groupType, ChannelSet &set)
{
    CHK_PRT_RET(channels.size() != peers.size(),
        HCCL_ERROR("[%s] channel/peer size mismatch, channels=%zu, peers=%zu",
            LOG_TAG, channels.size(), peers.size()),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(netLayer == INVALID_VALUE_RANKID,
        HCCL_ERROR("[%s] invalid cached layer for group=%u",
            LOG_TAG, static_cast<uint32_t>(groupType)),
        HCCL_E_INTERNAL);

    set.groupType = groupType;
    set.channels = channels;
    set.peers = peers;
    set.netLayer = netLayer;
    set.channelIndexByRank.fill(INVALID_VALUE_RANKID);

    for (size_t index = 0; index < peers.size(); ++index) {
        const uint32_t peer = peers[index];
        CHK_PRT_RET(peer >= BCAST_2X8_RANK_SIZE,
            HCCL_ERROR("[%s] invalid cached peer=%u", LOG_TAG, peer),
            HCCL_E_INTERNAL);
        CHK_PRT_RET(set.channelIndexByRank[peer] != INVALID_VALUE_RANKID,
            HCCL_ERROR("[%s] duplicate cached peer=%u", LOG_TAG, peer),
            HCCL_E_INTERNAL);
        set.channelIndexByRank[peer] = static_cast<uint32_t>(index);
    }
    return HCCL_SUCCESS;
}

HcclResult RestoreChannelSets(const AlgResourceCtx &baseCtx,
    ChannelSet &intraSet, ChannelSet &interSet)
{
    CHK_RET(RestoreOneChannelSet(baseCtx.intraChannels, baseCtx.intraPeers,
        baseCtx.intraLayer, BCAST_2X8_GROUP_INTRA, intraSet));
    CHK_RET(RestoreOneChannelSet(baseCtx.interChannels, baseCtx.interPeers,
        baseCtx.interLayer, BCAST_2X8_GROUP_INTER, interSet));
    CHK_PRT_RET(intraSet.netLayer == interSet.netLayer,
        HCCL_ERROR("[%s] cached intra/inter layers are identical, layer=%u",
            LOG_TAG, intraSet.netLayer),
        HCCL_E_NOT_SUPPORT);
    return HCCL_SUCCESS;
}

std::shared_ptr<Broadcast2x8OwnerPipelineKernelArg> BuildKernelArg(
    const OpParam &param, const ChannelSet &set)
{
    auto kernelArg = std::make_shared<Broadcast2x8OwnerPipelineKernelArg>();
    kernelArg->rankId = param.myRank;
    kernelArg->rankSize = param.rankSize;
    kernelArg->rootRank = param.root;
    kernelArg->serverId = GetServerId(param.myRank);
    kernelArg->groupType = static_cast<uint32_t>(set.groupType);
    kernelArg->netLayer = set.netLayer;
    kernelArg->channelCount = static_cast<uint32_t>(set.channels.size());

    for (uint32_t index = 0; index < set.channels.size(); ++index) {
        kernelArg->channels[index] = set.channels[index];
        kernelArg->peerRanks[index] = set.peers[index];
    }
    for (uint32_t rank = 0; rank < MAX_RANK_SIZE; ++rank) {
        kernelArg->channelIndexByRank[rank] = set.channelIndexByRank[rank];
    }
    return kernelArg;
}

HcclResult RegisterOneKernel(CcuInsHandle insHandle, uint32_t kernelIndex,
    const char *kernelName, void *kernelFunc, const OpParam &param,
    const ChannelSet &set, AlgResourceCtx &resCtxHost)
{
    auto kernelArg = BuildKernelArg(param, set);
    const void *kernelArgArray[] = {kernelArg.get()};
    CcuKernelHandle kernelHandle{};
    constexpr uint32_t dieId = 0;
    constexpr uint32_t kernelArgNum = 1;

    CcuResult ret = HcommCcuKernelRegister(insHandle, dieId,
        kernelName, kernelFunc, kernelArgArray, kernelArgNum, &kernelHandle);
    if (ret != CCU_SUCCESS) {
        HCCL_ERROR("[%s] kernel register failed, index=%u, name=%s, group=%u, "
            "layer=%u, channelCount=%u, ccuRet=%d",
            LOG_TAG, kernelIndex, kernelName, kernelArg->groupType,
            kernelArg->netLayer, kernelArg->channelCount, ret);
        return ConvertCcuToHccl(ret);
    }
    resCtxHost.ccuKernels[kernelIndex] = kernelHandle;
    return HCCL_SUCCESS;
}

HcclResult Register2x8Kernels(HcclComm comm, const OpParam &param,
    const ChannelSet &intraSet, const ChannelSet &interSet,
    AlgResourceCtx &resCtxHost)
{
    CcuInsHandle insHandle{0};
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum != 1,
        HCCL_ERROR("[%s] unexpected ccu insNum=%u", LOG_TAG, insNum),
        HCCL_E_INTERNAL);

    resCtxHost.ccuKernels.resize(BCAST_2X8_PIPELINE_KERNEL_NUM);
    CHK_RET_CCU(HcommCcuKernelRegisterStart(insHandle));

    CHK_RET(RegisterOneKernel(insHandle,
        BCAST_2X8_PIPELINE_INTRA,
        "Ccu2x8OwnerPipeline2CompactIntra",
        reinterpret_cast<void *>(ops_hccl::Ccu2x8OwnerPipeline2CompactIntraKernel),
        param, intraSet, resCtxHost));
    CHK_RET(RegisterOneKernel(insHandle,
        BCAST_2X8_PIPELINE_INTER,
        "Ccu2x8OwnerPipeline2CompactInter",
        reinterpret_cast<void *>(ops_hccl::Ccu2x8OwnerPipeline2CompactInterKernel),
        param, interSet, resCtxHost));

    CHK_RET_CCU(HcommCcuKernelRegisterEnd(insHandle));
    return HCCL_SUCCESS;
}

void CopySharedResources(const AlgResourceCtx &baseCtx, AlgResourceCtx &algorithmCtx)
{
    algorithmCtx.workerThread = baseCtx.workerThread;
    algorithmCtx.intraChannels = baseCtx.intraChannels;
    algorithmCtx.intraPeers = baseCtx.intraPeers;
    algorithmCtx.intraLayer = baseCtx.intraLayer;
    algorithmCtx.interChannels = baseCtx.interChannels;
    algorithmCtx.interPeers = baseCtx.interPeers;
    algorithmCtx.interLayer = baseCtx.interLayer;
}

HcclResult StoreEngineCtx(HcclComm comm, CommEngine engine, const char *tag,
    const AlgResourceCtx &resCtxHost, void **ctxOut, uint64_t *ctxSizeOut)
{
    CHK_PTR_NULL(tag);
    CHK_PTR_NULL(ctxOut);
    CHK_PTR_NULL(ctxSizeOut);

    AlgResourceCtx serializableCtx = resCtxHost;
    std::vector<char> sequence = serializableCtx.Serialize();
    *ctxSizeOut = sequence.size();
    CHK_RET(HcclEngineCtxCreate(comm, tag, engine, *ctxSizeOut, ctxOut));
    CHK_RET(HcclEngineCtxCopy(
        comm, engine, tag, sequence.data(), sequence.size(), 0));
    return HCCL_SUCCESS;
}

HcclResult LoadEngineCtx(void *ctx, uint64_t ctxSize, AlgResourceCtx &resCtxHost)
{
    CHK_PTR_NULL(ctx);
    char *bytes = static_cast<char *>(ctx);
    std::vector<char> sequence(bytes, bytes + ctxSize);
    resCtxHost.DeSerialize(sequence);
    return HCCL_SUCCESS;
}

HcclResult GetOrCreateBaseResources(HcclComm comm, const OpParam &param,
    CommEngine engine, AlgResourceCtx &baseCtxHost)
{
    void *baseCtx = nullptr;
    uint64_t baseCtxSize = 0;
    if (HcclEngineCtxGet(comm, BASE_TAG, engine, &baseCtx, &baseCtxSize) == HCCL_SUCCESS) {
        return LoadEngineCtx(baseCtx, baseCtxSize, baseCtxHost);
    }

    CHK_RET(HcclThreadAcquire(comm, engine, 1,
        THREAD_NOTIFY_NUM, &baseCtxHost.workerThread));

    ChannelSet intraSet{};
    ChannelSet interSet{};
    CHK_RET(Acquire2x8Channels(comm, param, intraSet, interSet));
    SaveChannelSets(intraSet, interSet, baseCtxHost);
    baseCtxHost.ccuKernels.clear();

    return StoreEngineCtx(comm, engine, BASE_TAG,
        baseCtxHost, &baseCtx, &baseCtxSize);
}

HcclResult BuildRootSpecificTag(uint32_t root, char *tag, size_t tagSize)
{
    const int written = std::snprintf(tag, tagSize, "%s_%u", ROOT_TAG_PREFIX, root);
    CHK_PRT_RET(written <= 0 || static_cast<size_t>(written) >= tagSize,
        HCCL_ERROR("[%s] failed to construct root-specific EngineCtx tag", LOG_TAG),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult GetOrCreateAlgorithmResources(HcclComm comm, const OpParam &param,
    CommEngine engine, void **ctxOut, uint64_t *ctxSizeOut)
{
    CHK_PTR_NULL(ctxOut);
    CHK_PTR_NULL(ctxSizeOut);

    if (HcclEngineCtxGet(comm, param.tag, engine, ctxOut, ctxSizeOut) == HCCL_SUCCESS) {
        return HCCL_SUCCESS;
    }

    AlgResourceCtx baseCtxHost{};
    CHK_RET(GetOrCreateBaseResources(comm, param, engine, baseCtxHost));

    ChannelSet intraSet{};
    ChannelSet interSet{};
    CHK_RET(RestoreChannelSets(baseCtxHost, intraSet, interSet));

    AlgResourceCtx algorithmCtxHost{};
    CopySharedResources(baseCtxHost, algorithmCtxHost);
    CHK_RET(Register2x8Kernels(
        comm, param, intraSet, interSet, algorithmCtxHost));

    return StoreEngineCtx(comm, engine, param.tag,
        algorithmCtxHost, ctxOut, ctxSizeOut);
}

} // namespace

HcclResult HcclBroadcast2x8Large15OwnerPipeline2Impl(
    void *buf, uint64_t count, HcclDataType dataType, uint32_t root,
    HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(buf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    OpParam param{};
    param.inputPtr = buf;
    param.outputPtr = buf;
    param.count = count;
    param.root = root;
    param.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_BROADCAST;

    HcclDfxOpInfo dfxInfo{};
    char commName[COMM_INDENTIFIER_MAX_LENGTH]{};
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_PRT_RET(param.rankSize != BCAST_2X8_RANK_SIZE,
        HCCL_ERROR("[%s] only 2×8 rankSize=16 is supported, rankSize=%u",
            LOG_TAG, param.rankSize),
        HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(root >= param.rankSize,
        HCCL_ERROR("[%s] invalid root=%u, rankSize=%u", LOG_TAG, root, param.rankSize),
        HCCL_E_PARA);
    CHK_PRT_RET(dataType != HCCL_DATA_TYPE_FP32,
        HCCL_ERROR("[%s] only FP32 is supported, dataType=%d",
            LOG_TAG, static_cast<int>(dataType)),
        HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(count > std::numeric_limits<uint64_t>::max() / sizeof(float),
        HCCL_ERROR("[%s] count too large: %lu", LOG_TAG, count), HCCL_E_PARA);

    const uint64_t dataSize = count * sizeof(float);
    CHK_PRT_RET(!IsSupportedLargeSize(dataSize),
        HCCL_ERROR("[%s] only 512MB and 400MB+4B are supported, dataSize=%lu",
            LOG_TAG, dataSize),
        HCCL_E_NOT_SUPPORT);

    // Root is a registration-time constant in the two generic layer kernels.
    // The runtime stageId changes on each launch; all roots reuse BASE_TAG's
    // single channel to every peer and the same worker thread.
    CHK_RET(BuildRootSpecificTag(root, param.tag, sizeof(param.tag)));

    const CommEngine engine = CommEngine::COMM_ENGINE_CCU;
    CHK_RET(HcclThreadAcquireWithStream(comm, engine, stream,
        THREAD_NOTIFY_NUM, &param.cpuThread));

    CHK_RET(GetOrCreateAlgorithmResources(comm, param, engine,
        &param.resCtx, &param.ctxSize));

    return ops_hccl::ExecOp2x8Large15OwnerPipeline2Compact(param);
}

} // namespace bcast2x8_large_final

namespace bcast4x1_small_final {
namespace {

constexpr uint64_t SUPPORTED_512KB_SIZE = 512ULL * 1024ULL;

// The shared 4x1 base context is also reused by the large Pipeline15 path.
// The shared channel set has notify IDs 0..15 available. Small packets use
// IDs 1..3. Pipeline15 uses IDs 3..15 for chunks 0..12 and, only after the
// address exchange has completed, reuses IDs 1 and 2 for chunks 13 and 14.
constexpr uint32_t CHANNEL_NOTIFY_NUM = 16;
constexpr CommProtocol REQUIRED_PROTOCOL = CommProtocol::COMM_PROTOCOL_UBC_CTP;
constexpr const char *BASE_TAG = "hccl_bcast_final_v4_4x1_base";
constexpr const char *ROOT_TAG_PREFIX = "hccl_bcast_final_v4_4x1_small_root";
constexpr const char *LOG_TAG = "BCAST_FINAL_V4_4X1_SMALL";

struct ChannelSet {
    std::vector<ChannelHandle> channels;
    std::vector<uint32_t> peers;
    std::array<uint32_t, MAX_RANK_SIZE> channelIndexByRank{};
    uint32_t netLayer = INVALID_VALUE_RANKID;

    ChannelSet()
    {
        channelIndexByRank.fill(INVALID_VALUE_RANKID);
    }
};

HcclResult FindLinkForPeer(HcclComm comm, uint32_t myRank, uint32_t remoteRank,
    CommLink &selectedLink, uint32_t &selectedLayer)
{
    uint32_t *layers = nullptr;
    uint32_t layerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layers, &layerNum));

    for (uint32_t layerIndex = 0; layerIndex < layerNum; ++layerIndex) {
        CommLink *linkList = nullptr;
        uint32_t listSize = 0;
        HcclResult ret = HcclRankGraphGetLinks(
            comm, layers[layerIndex], myRank, remoteRank, &linkList, &listSize);
        if (ret != HCCL_SUCCESS || linkList == nullptr) {
            continue;
        }
        for (uint32_t linkIndex = 0; linkIndex < listSize; ++linkIndex) {
            if (linkList[linkIndex].linkAttr.linkProtocol == REQUIRED_PROTOCOL) {
                selectedLink = linkList[linkIndex];
                selectedLayer = layers[layerIndex];
                return HCCL_SUCCESS;
            }
        }
    }

    HCCL_ERROR("[%s] no UBC_CTP link found, myRank=%u, peer=%u",
        LOG_TAG, myRank, remoteRank);
    return HCCL_E_NOT_FOUND;
}

HcclResult AcquireChannels(HcclComm comm, const OpParam &param, ChannelSet &set)
{
    std::array<bool, MAX_RANK_SIZE> peerAcquired{};
    peerAcquired.fill(false);

    for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
        if (remoteRank == param.myRank) {
            continue;
        }

        CHK_PRT_RET(peerAcquired[remoteRank],
            HCCL_ERROR("[%s] peer=%u would receive more than one channel",
                LOG_TAG, remoteRank),
            HCCL_E_PARA);
        CHK_PRT_RET(set.channelIndexByRank[remoteRank] != INVALID_VALUE_RANKID,
            HCCL_ERROR("[%s] duplicate channel mapping for peer=%u",
                LOG_TAG, remoteRank),
            HCCL_E_PARA);

        CommLink link{};
        uint32_t netLayer = INVALID_VALUE_RANKID;
        CHK_RET(FindLinkForPeer(comm, param.myRank, remoteRank, link, netLayer));

        if (set.netLayer == INVALID_VALUE_RANKID) {
            set.netLayer = netLayer;
        } else {
            CHK_PRT_RET(set.netLayer != netLayer,
                HCCL_ERROR("[%s] 4x1 channels span layers %u/%u, peer=%u",
                    LOG_TAG, set.netLayer, netLayer, remoteRank),
                HCCL_E_NOT_SUPPORT);
        }

        HcclChannelDesc desc{};
        CHK_RET(HcclChannelDescInit(&desc, 1));
        desc.remoteRank = remoteRank;
        desc.notifyNum = CHANNEL_NOTIFY_NUM;
        desc.channelProtocol = link.linkAttr.linkProtocol;
        desc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
        desc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
        desc.localEndpoint.loc = link.srcEndpointDesc.loc;
        desc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
        desc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
        desc.remoteEndpoint.loc = link.dstEndpointDesc.loc;

        ChannelHandle channel{};
        CHK_RET(HcclChannelAcquire(
            comm, CommEngine::COMM_ENGINE_CCU, &desc, 1, &channel));

        const uint32_t channelIndex = static_cast<uint32_t>(set.channels.size());
        set.channels.push_back(channel);
        set.peers.push_back(remoteRank);
        set.channelIndexByRank[remoteRank] = channelIndex;
        peerAcquired[remoteRank] = true;
    }

    CHK_PRT_RET(set.channels.size() != BCAST_4X1_RANK_SIZE - 1U,
        HCCL_ERROR("[%s] unexpected channel count=%zu, expected=3",
            LOG_TAG, set.channels.size()),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(set.netLayer == INVALID_VALUE_RANKID,
        HCCL_ERROR("[%s] failed to resolve common 4x1 network layer", LOG_TAG),
        HCCL_E_INTERNAL);

    for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
        if (peer != param.myRank) {
            CHK_PRT_RET(!peerAcquired[peer],
                HCCL_ERROR("[%s] missing channel to peer=%u", LOG_TAG, peer),
                HCCL_E_INTERNAL);
        }
    }
    return HCCL_SUCCESS;
}

void SaveChannelSet(const ChannelSet &set, AlgResourceCtx &resCtx)
{
    resCtx.channels = set.channels;
    resCtx.peers = set.peers;
    resCtx.netLayer = set.netLayer;
}

HcclResult RestoreChannelSet(const AlgResourceCtx &baseCtx, ChannelSet &set)
{
    CHK_PRT_RET(baseCtx.channels.size() != BCAST_4X1_RANK_SIZE - 1U ||
            baseCtx.channels.size() != baseCtx.peers.size(),
        HCCL_ERROR("[%s] invalid cached channel set, channels=%zu, peers=%zu",
            LOG_TAG, baseCtx.channels.size(), baseCtx.peers.size()),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(baseCtx.netLayer == INVALID_VALUE_RANKID,
        HCCL_ERROR("[%s] invalid cached netLayer", LOG_TAG),
        HCCL_E_INTERNAL);

    set.channels = baseCtx.channels;
    set.peers = baseCtx.peers;
    set.netLayer = baseCtx.netLayer;
    set.channelIndexByRank.fill(INVALID_VALUE_RANKID);

    for (size_t index = 0; index < set.peers.size(); ++index) {
        const uint32_t peer = set.peers[index];
        CHK_PRT_RET(peer >= BCAST_4X1_RANK_SIZE,
            HCCL_ERROR("[%s] invalid cached peer=%u", LOG_TAG, peer),
            HCCL_E_INTERNAL);
        CHK_PRT_RET(set.channelIndexByRank[peer] != INVALID_VALUE_RANKID,
            HCCL_ERROR("[%s] duplicate cached peer=%u", LOG_TAG, peer),
            HCCL_E_INTERNAL);
        set.channelIndexByRank[peer] = static_cast<uint32_t>(index);
    }
    return HCCL_SUCCESS;
}

std::shared_ptr<Broadcast4x1SmallSpecializedKernelArg>
BuildSpecializedKernelArg(const OpParam &param, uint32_t root,
    const ChannelSet &set)
{
    auto arg = std::make_shared<Broadcast4x1SmallSpecializedKernelArg>();
    arg->rankId = param.myRank;
    arg->rankSize = param.rankSize;
    arg->rootRank = root;
    arg->netLayer = set.netLayer;

    if (param.myRank == root) {
        arg->role = BCAST_4X1_ROLE_SENDER;
        arg->channelCount = static_cast<uint32_t>(set.channels.size());
        for (uint32_t index = 0; index < arg->channelCount; ++index) {
            arg->channels[index] = set.channels[index];
            arg->peerRanks[index] = set.peers[index];
        }
        return arg;
    }

    const uint32_t rootChannelIndex = set.channelIndexByRank[root];
    if (rootChannelIndex == INVALID_VALUE_RANKID ||
        rootChannelIndex >= set.channels.size()) {
        return nullptr;
    }

    arg->role = BCAST_4X1_ROLE_RECEIVER;
    arg->channelCount = 1;
    arg->channels[0] = set.channels[rootChannelIndex];
    arg->peerRanks[0] = root;
    return arg;
}

HcclResult RegisterRootSpecializedKernel(HcclComm comm, const OpParam &param,
    uint32_t root, const ChannelSet &set, AlgResourceCtx &rootCtx)
{
    CcuInsHandle insHandle{0};
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum != 1,
        HCCL_ERROR("[%s] unexpected ccu insNum=%u", LOG_TAG, insNum),
        HCCL_E_INTERNAL);

    auto kernelArg = BuildSpecializedKernelArg(param, root, set);
    CHK_PTR_NULL(kernelArg);

    char kernelName[128]{};
    int written = std::snprintf(kernelName, sizeof(kernelName),
        "Ccu4x1SmallRootSpecialized_r%u", root);
    CHK_PRT_RET(written <= 0 || static_cast<size_t>(written) >= sizeof(kernelName),
        HCCL_ERROR("[%s] failed to build kernel name, root=%u", LOG_TAG, root),
        HCCL_E_INTERNAL);

    rootCtx.ccuKernels.assign(
        BCAST_4X1_SMALL_ROOT_SPECIALIZED_KERNEL_NUM, CcuKernelHandle{});

    const void *kernelArgArray[] = {kernelArg.get()};
    CcuKernelHandle kernelHandle{};
    constexpr uint32_t dieId = 0;
    constexpr uint32_t kernelArgNum = 1;

    CHK_RET_CCU(HcommCcuKernelRegisterStart(insHandle));
    CcuResult ccuRet = HcommCcuKernelRegister(
        insHandle,
        dieId,
        kernelName,
        reinterpret_cast<void *>(ops_hccl::Ccu4x1SmallRootSpecializedKernel),
        kernelArgArray,
        kernelArgNum,
        &kernelHandle);
    if (ccuRet != CCU_SUCCESS) {
        HCCL_ERROR("[%s] kernel register failed, root=%u, rank=%u, role=%u, "
            "layer=%u, channels=%u, ccuRet=%d",
            LOG_TAG, root, param.myRank, kernelArg->role,
            kernelArg->netLayer, kernelArg->channelCount, ccuRet);
        return ConvertCcuToHccl(ccuRet);
    }
    rootCtx.ccuKernels[BCAST_4X1_SMALL_ROOT_SPECIALIZED] = kernelHandle;
    CHK_RET_CCU(HcommCcuKernelRegisterEnd(insHandle));
    return HCCL_SUCCESS;
}

HcclResult CreateBaseResources(HcclComm comm, const OpParam &param,
    AlgResourceCtx &baseCtx)
{
    ChannelSet set{};
    CHK_RET(AcquireChannels(comm, param, set));
    SaveChannelSet(set, baseCtx);
    return HCCL_SUCCESS;
}

HcclResult GetOrCreateBaseContext(HcclComm comm, const OpParam &param,
    AlgResourceCtx &baseCtx)
{
    const CommEngine engine = CommEngine::COMM_ENGINE_CCU;
    void *ctx = nullptr;
    uint64_t ctxSize = 0;
    if (HcclEngineCtxGet(comm, BASE_TAG, engine, &ctx, &ctxSize) == HCCL_SUCCESS) {
        CHK_PTR_NULL(ctx);
        std::vector<char> sequence(
            static_cast<char *>(ctx), static_cast<char *>(ctx) + ctxSize);
        baseCtx.DeSerialize(sequence);
        return HCCL_SUCCESS;
    }

    CHK_RET(CreateBaseResources(comm, param, baseCtx));
    std::vector<char> sequence = baseCtx.Serialize();
    void *createdCtx = nullptr;
    CHK_RET(HcclEngineCtxCreate(
        comm, BASE_TAG, engine, sequence.size(), &createdCtx));
    CHK_RET(HcclEngineCtxCopy(
        comm, engine, BASE_TAG, sequence.data(), sequence.size(), 0));
    return HCCL_SUCCESS;
}

HcclResult GetOrCreateRootContext(HcclComm comm, const OpParam &param,
    const AlgResourceCtx &baseCtx, char *rootTag, size_t rootTagSize,
    void **ctx, uint64_t *ctxSize)
{
    CHK_PTR_NULL(rootTag);
    CHK_PTR_NULL(ctx);
    CHK_PTR_NULL(ctxSize);

    int written = std::snprintf(
        rootTag, rootTagSize, "%s_%u", ROOT_TAG_PREFIX, param.root);
    CHK_PRT_RET(written <= 0 || static_cast<size_t>(written) >= rootTagSize,
        HCCL_ERROR("[%s] failed to construct root tag, root=%u",
            LOG_TAG, param.root),
        HCCL_E_INTERNAL);

    const CommEngine engine = CommEngine::COMM_ENGINE_CCU;
    if (HcclEngineCtxGet(comm, rootTag, engine, ctx, ctxSize) == HCCL_SUCCESS) {
        return HCCL_SUCCESS;
    }

    ChannelSet set{};
    CHK_RET(RestoreChannelSet(baseCtx, set));

    AlgResourceCtx rootCtx = baseCtx;
    rootCtx.ccuKernels.clear();
    CHK_RET(RegisterRootSpecializedKernel(
        comm, param, param.root, set, rootCtx));

    std::vector<char> sequence = rootCtx.Serialize();
    CHK_RET(HcclEngineCtxCreate(
        comm, rootTag, engine, sequence.size(), ctx));
    CHK_RET(HcclEngineCtxCopy(
        comm, engine, rootTag, sequence.data(), sequence.size(), 0));
    *ctxSize = sequence.size();
    return HCCL_SUCCESS;
}

} // namespace

HcclResult HcclBroadcast4x1SmallRootSpecializedImpl(
    void *buf, uint64_t count, HcclDataType dataType, uint32_t root,
    HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(buf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    OpParam param{};
    param.inputPtr = buf;
    param.outputPtr = buf;
    param.count = count;
    param.root = root;
    param.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_BROADCAST;

    HcclDfxOpInfo dfxInfo{};
    char commName[COMM_INDENTIFIER_MAX_LENGTH]{};
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(
        commName, reinterpret_cast<void *>(&dfxInfo)));

    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_PRT_RET(param.rankSize != BCAST_4X1_RANK_SIZE,
        HCCL_ERROR("[%s] expected rankSize=4, actual=%u", LOG_TAG, param.rankSize),
        HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(root >= param.rankSize,
        HCCL_ERROR("[%s] invalid root=%u", LOG_TAG, root),
        HCCL_E_PARA);
    CHK_PRT_RET(dataType != HCCL_DATA_TYPE_FP32,
        HCCL_ERROR("[%s] only FP32 is supported, dataType=%d",
            LOG_TAG, static_cast<int>(dataType)),
        HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(count > std::numeric_limits<uint64_t>::max() / sizeof(float),
        HCCL_ERROR("[%s] count too large=%lu", LOG_TAG, count),
        HCCL_E_PARA);

    const uint64_t dataSize = count * sizeof(float);
    CHK_PRT_RET(dataSize != SUPPORTED_512KB_SIZE,
        HCCL_ERROR("[%s] only 512KB is supported, dataSize=%lu",
            LOG_TAG, dataSize),
        HCCL_E_NOT_SUPPORT);

    const CommEngine engine = CommEngine::COMM_ENGINE_CCU;
    CHK_RET(HcclThreadAcquireWithStream(comm, engine, stream, 0, &param.cpuThread));

    AlgResourceCtx baseCtx{};
    CHK_RET(GetOrCreateBaseContext(comm, param, baseCtx));

    char rootTag[128]{};
    CHK_RET(GetOrCreateRootContext(
        comm, param, baseCtx, rootTag, sizeof(rootTag),
        &param.resCtx, &param.ctxSize));

    int tagWritten = std::snprintf(param.tag, sizeof(param.tag), "%s", rootTag);
    CHK_PRT_RET(tagWritten <= 0 ||
            static_cast<size_t>(tagWritten) >= sizeof(param.tag),
        HCCL_ERROR("[%s] failed to copy root tag", LOG_TAG),
        HCCL_E_INTERNAL);

    return ops_hccl::ExecOp4x1SmallRootSpecialized(param);
}

} // namespace bcast4x1_small_final

namespace bcast8p4_small_final {
namespace {

constexpr uint64_t SUPPORTED_512KB_SIZE = 512ULL * 1024ULL;

// Kernels use BUFFER=bit1, TOKEN=bit2 and DONE=bit3.
constexpr uint32_t CHANNEL_NOTIFY_NUM = 5;
constexpr uint32_t THREAD_NOTIFY_NUM = 1;
constexpr CommProtocol REQUIRED_PROTOCOL = CommProtocol::COMM_PROTOCOL_UBC_CTP;
constexpr const char *BASE_TAG = "hccl_bcast_final_v4_8p4_base";
constexpr const char *ROOT_TAG_PREFIX = "hccl_bcast_final_v4_8p4_small_root";
constexpr const char *LOG_TAG = "BCAST_FINAL_V4_8P4_SMALL";

struct ChannelSet {
    Broadcast8p4ChannelGroup groupType = BCAST_8P4_GROUP_INTRA;
    std::vector<ChannelHandle> channels;
    std::vector<uint32_t> peers;
    std::array<uint32_t, MAX_RANK_SIZE> channelIndexByRank{};
    uint32_t netLayer = INVALID_VALUE_RANKID;

    ChannelSet()
    {
        channelIndexByRank.fill(INVALID_VALUE_RANKID);
    }
};

uint32_t GetServerId(uint32_t rank)
{
    return rank < BCAST_8P4_SERVER1_BASE ? 0U : 1U;
}

uint32_t GetServerBase(uint32_t serverId)
{
    return serverId == 0U ? BCAST_8P4_SERVER0_BASE : BCAST_8P4_SERVER1_BASE;
}

uint32_t GetServerRankNum(uint32_t serverId)
{
    return serverId == 0U ? BCAST_8P4_SERVER0_RANK_NUM : BCAST_8P4_SERVER1_RANK_NUM;
}

bool GroupContainsRoot(uint32_t myRank, uint32_t root,
    Broadcast8p4ChannelGroup groupType)
{
    const bool sameServer = GetServerId(myRank) == GetServerId(root);
    return groupType == BCAST_8P4_GROUP_INTRA ? sameServer : !sameServer;
}

HcclResult FindLinkForPeer(HcclComm comm, uint32_t myRank, uint32_t remoteRank,
    CommLink &selectedLink, uint32_t &selectedLayer)
{
    uint32_t *layers = nullptr;
    uint32_t layerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layers, &layerNum));

    for (uint32_t layerIndex = 0; layerIndex < layerNum; ++layerIndex) {
        CommLink *linkList = nullptr;
        uint32_t listSize = 0;
        HcclResult ret = HcclRankGraphGetLinks(
            comm, layers[layerIndex], myRank, remoteRank, &linkList, &listSize);
        if (ret != HCCL_SUCCESS || linkList == nullptr) {
            continue;
        }
        for (uint32_t linkIndex = 0; linkIndex < listSize; ++linkIndex) {
            if (linkList[linkIndex].linkAttr.linkProtocol == REQUIRED_PROTOCOL) {
                selectedLink = linkList[linkIndex];
                selectedLayer = layers[layerIndex];
                return HCCL_SUCCESS;
            }
        }
    }

    HCCL_ERROR("[%s] no UBC_CTP link found, myRank=%u, peer=%u",
        LOG_TAG, myRank, remoteRank);
    return HCCL_E_NOT_FOUND;
}

HcclResult AcquireOneChannel(HcclComm comm, const OpParam &param, uint32_t remoteRank,
    ChannelSet &set, std::array<bool, MAX_RANK_SIZE> &peerAcquired)
{
    CHK_PRT_RET(remoteRank >= param.rankSize || remoteRank == param.myRank,
        HCCL_ERROR("[%s] invalid remoteRank=%u for myRank=%u",
            LOG_TAG, remoteRank, param.myRank),
        HCCL_E_PARA);
    CHK_PRT_RET(peerAcquired[remoteRank],
        HCCL_ERROR("[%s] peer=%u would receive more than one channel", LOG_TAG, remoteRank),
        HCCL_E_PARA);
    CHK_PRT_RET(set.channelIndexByRank[remoteRank] != INVALID_VALUE_RANKID,
        HCCL_ERROR("[%s] duplicated channel in group, peer=%u", LOG_TAG, remoteRank),
        HCCL_E_PARA);

    CommLink link{};
    uint32_t netLayer = INVALID_VALUE_RANKID;
    CHK_RET(FindLinkForPeer(comm, param.myRank, remoteRank, link, netLayer));

    if (set.netLayer == INVALID_VALUE_RANKID) {
        set.netLayer = netLayer;
    } else {
        CHK_PRT_RET(set.netLayer != netLayer,
            HCCL_ERROR("[%s] one channel group spans multiple layers: previous=%u, current=%u, peer=%u",
                LOG_TAG, set.netLayer, netLayer, remoteRank),
            HCCL_E_NOT_SUPPORT);
    }

    HcclChannelDesc desc{};
    CHK_RET(HcclChannelDescInit(&desc, 1));
    desc.remoteRank = remoteRank;
    desc.notifyNum = CHANNEL_NOTIFY_NUM;
    desc.channelProtocol = link.linkAttr.linkProtocol;
    desc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
    desc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
    desc.localEndpoint.loc = link.srcEndpointDesc.loc;
    desc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
    desc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
    desc.remoteEndpoint.loc = link.dstEndpointDesc.loc;

    ChannelHandle channel{};
    CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CCU, &desc, 1, &channel));

    const uint32_t channelIndex = static_cast<uint32_t>(set.channels.size());
    set.channels.push_back(channel);
    set.peers.push_back(remoteRank);
    set.channelIndexByRank[remoteRank] = channelIndex;
    peerAcquired[remoteRank] = true;
    return HCCL_SUCCESS;
}

HcclResult Acquire8p4Channels(HcclComm comm, const OpParam &param,
    ChannelSet &intraSet, ChannelSet &interSet)
{
    intraSet.groupType = BCAST_8P4_GROUP_INTRA;
    interSet.groupType = BCAST_8P4_GROUP_INTER;

    std::array<bool, MAX_RANK_SIZE> peerAcquired{};
    peerAcquired.fill(false);

    const uint32_t myServer = GetServerId(param.myRank);
    const uint32_t remoteServer = 1U - myServer;
    const uint32_t sameServerBase = GetServerBase(myServer);
    const uint32_t sameServerRankNum = GetServerRankNum(myServer);
    const uint32_t remoteServerBase = GetServerBase(remoteServer);
    const uint32_t remoteServerRankNum = GetServerRankNum(remoteServer);

    for (uint32_t local = 0; local < sameServerRankNum; ++local) {
        const uint32_t peer = sameServerBase + local;
        if (peer == param.myRank) {
            continue;
        }
        CHK_RET(AcquireOneChannel(comm, param, peer, intraSet, peerAcquired));
    }

    for (uint32_t local = 0; local < remoteServerRankNum; ++local) {
        CHK_RET(AcquireOneChannel(
            comm, param, remoteServerBase + local, interSet, peerAcquired));
    }

    const size_t expectedIntra = static_cast<size_t>(sameServerRankNum - 1U);
    const size_t expectedInter = static_cast<size_t>(remoteServerRankNum);
    CHK_PRT_RET(intraSet.channels.size() != expectedIntra ||
            interSet.channels.size() != expectedInter,
        HCCL_ERROR("[%s] unexpected channel counts, intra=%zu/%zu, inter=%zu/%zu",
            LOG_TAG, intraSet.channels.size(), expectedIntra,
            interSet.channels.size(), expectedInter),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(intraSet.netLayer == INVALID_VALUE_RANKID ||
            interSet.netLayer == INVALID_VALUE_RANKID,
        HCCL_ERROR("[%s] failed to resolve topology layers, intra=%u, inter=%u",
            LOG_TAG, intraSet.netLayer, interSet.netLayer),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(intraSet.netLayer == interSet.netLayer,
        HCCL_ERROR("[%s] intra and inter channels resolved to the same layer=%u; "
            "one CCU kernel must not span both IO Dies",
            LOG_TAG, intraSet.netLayer),
        HCCL_E_NOT_SUPPORT);

    for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
        if (peer != param.myRank) {
            CHK_PRT_RET(!peerAcquired[peer],
                HCCL_ERROR("[%s] missing channel to peer=%u", LOG_TAG, peer),
                HCCL_E_INTERNAL);
        }
    }
    return HCCL_SUCCESS;
}

void SaveChannelSets(const ChannelSet &intraSet, const ChannelSet &interSet,
    AlgResourceCtx &resCtx)
{
    resCtx.intraChannels = intraSet.channels;
    resCtx.intraPeers = intraSet.peers;
    resCtx.intraLayer = intraSet.netLayer;
    resCtx.interChannels = interSet.channels;
    resCtx.interPeers = interSet.peers;
    resCtx.interLayer = interSet.netLayer;
}

HcclResult RestoreOneChannelSet(const std::vector<ChannelHandle> &channels,
    const std::vector<uint32_t> &peers, uint32_t netLayer,
    Broadcast8p4ChannelGroup groupType, ChannelSet &set)
{
    CHK_PRT_RET(channels.size() != peers.size() || channels.empty(),
        HCCL_ERROR("[%s] invalid cached group=%u, channels=%zu, peers=%zu",
            LOG_TAG, static_cast<uint32_t>(groupType), channels.size(), peers.size()),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(netLayer == INVALID_VALUE_RANKID,
        HCCL_ERROR("[%s] invalid cached layer for group=%u",
            LOG_TAG, static_cast<uint32_t>(groupType)),
        HCCL_E_INTERNAL);

    set.groupType = groupType;
    set.channels = channels;
    set.peers = peers;
    set.netLayer = netLayer;
    set.channelIndexByRank.fill(INVALID_VALUE_RANKID);

    for (size_t index = 0; index < peers.size(); ++index) {
        const uint32_t peer = peers[index];
        CHK_PRT_RET(peer >= BCAST_8P4_RANK_SIZE,
            HCCL_ERROR("[%s] invalid cached peer=%u", LOG_TAG, peer),
            HCCL_E_INTERNAL);
        CHK_PRT_RET(set.channelIndexByRank[peer] != INVALID_VALUE_RANKID,
            HCCL_ERROR("[%s] duplicate cached peer=%u", LOG_TAG, peer),
            HCCL_E_INTERNAL);
        set.channelIndexByRank[peer] = static_cast<uint32_t>(index);
    }
    return HCCL_SUCCESS;
}

HcclResult RestoreChannelSets(const AlgResourceCtx &baseCtx,
    ChannelSet &intraSet, ChannelSet &interSet)
{
    CHK_RET(RestoreOneChannelSet(baseCtx.intraChannels, baseCtx.intraPeers,
        baseCtx.intraLayer, BCAST_8P4_GROUP_INTRA, intraSet));
    CHK_RET(RestoreOneChannelSet(baseCtx.interChannels, baseCtx.interPeers,
        baseCtx.interLayer, BCAST_8P4_GROUP_INTER, interSet));
    CHK_PRT_RET(intraSet.netLayer == interSet.netLayer,
        HCCL_ERROR("[%s] cached intra/inter groups share layer=%u",
            LOG_TAG, intraSet.netLayer),
        HCCL_E_NOT_SUPPORT);
    return HCCL_SUCCESS;
}

std::shared_ptr<Broadcast8p4SmallSpecializedKernelArg> BuildSpecializedKernelArg(
    const OpParam &param, uint32_t root, const ChannelSet &set)
{
    auto arg = std::make_shared<Broadcast8p4SmallSpecializedKernelArg>();
    arg->rankId = param.myRank;
    arg->rankSize = param.rankSize;
    arg->rootRank = root;
    arg->serverId = GetServerId(param.myRank);
    arg->groupType = static_cast<uint32_t>(set.groupType);
    arg->netLayer = set.netLayer;

    if (param.myRank == root) {
        arg->role = BCAST_8P4_ROLE_SENDER;
        arg->channelCount = static_cast<uint32_t>(set.channels.size());
        for (uint32_t index = 0; index < arg->channelCount; ++index) {
            arg->channels[index] = set.channels[index];
            arg->peerRanks[index] = set.peers[index];
        }
        return arg;
    }

    if (GroupContainsRoot(param.myRank, root, set.groupType)) {
        const uint32_t rootChannelIndex = set.channelIndexByRank[root];
        if (rootChannelIndex == INVALID_VALUE_RANKID ||
            rootChannelIndex >= set.channels.size()) {
            return nullptr;
        }
        arg->role = BCAST_8P4_ROLE_RECEIVER;
        arg->channelCount = 1;
        arg->channels[0] = set.channels[rootChannelIndex];
        arg->peerRanks[0] = root;
        return arg;
    }

    // The host never launches this layer on a non-root receiver. Keep one
    // channel solely so the translator can place this idle kernel on the
    // intended IO Die without binding the full unused channel group.
    if (set.channels.empty()) {
        return nullptr;
    }
    arg->role = BCAST_8P4_ROLE_IDLE;
    arg->channelCount = 1;
    arg->channels[0] = set.channels[0];
    arg->peerRanks[0] = set.peers[0];
    return arg;
}

HcclResult RegisterOneSpecializedKernel(CcuInsHandle insHandle,
    uint32_t kernelIndex, const char *kernelName, void *kernelFunc,
    const OpParam &param, uint32_t root, const ChannelSet &set,
    AlgResourceCtx &rootCtx)
{
    auto kernelArg = BuildSpecializedKernelArg(param, root, set);
    CHK_PTR_NULL(kernelArg);

    const void *kernelArgArray[] = {kernelArg.get()};
    CcuKernelHandle kernelHandle{};
    constexpr uint32_t dieId = 0;
    constexpr uint32_t kernelArgNum = 1;

    CcuResult ret = HcommCcuKernelRegister(insHandle, dieId,
        kernelName, kernelFunc, kernelArgArray, kernelArgNum, &kernelHandle);
    if (ret != CCU_SUCCESS) {
        HCCL_ERROR("[%s] kernel register failed, root=%u, rank=%u, index=%u, "
            "role=%u, group=%u, layer=%u, channels=%u, ccuRet=%d",
            LOG_TAG, root, param.myRank, kernelIndex, kernelArg->role,
            kernelArg->groupType, kernelArg->netLayer,
            kernelArg->channelCount, ret);
        return ConvertCcuToHccl(ret);
    }
    rootCtx.ccuKernels[kernelIndex] = kernelHandle;
    return HCCL_SUCCESS;
}

HcclResult RegisterRootSpecializedKernels(HcclComm comm, const OpParam &param,
    uint32_t root, const ChannelSet &intraSet, const ChannelSet &interSet,
    AlgResourceCtx &rootCtx)
{
    CcuInsHandle insHandle{0};
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum != 1,
        HCCL_ERROR("[%s] unexpected ccu insNum=%u", LOG_TAG, insNum),
        HCCL_E_INTERNAL);

    rootCtx.ccuKernels.assign(BCAST_8P4_SMALL_SPECIALIZED_KERNEL_NUM,
        CcuKernelHandle{});

    char intraName[128]{};
    char interName[128]{};
    int intraWritten = std::snprintf(intraName, sizeof(intraName),
        "Ccu8p4SmallRootSpecializedIntra_r%u", root);
    int interWritten = std::snprintf(interName, sizeof(interName),
        "Ccu8p4SmallRootSpecializedInter_r%u", root);
    CHK_PRT_RET(intraWritten <= 0 || static_cast<size_t>(intraWritten) >= sizeof(intraName) ||
            interWritten <= 0 || static_cast<size_t>(interWritten) >= sizeof(interName),
        HCCL_ERROR("[%s] failed to build specialized kernel names, root=%u", LOG_TAG, root),
        HCCL_E_INTERNAL);

    CHK_RET_CCU(HcommCcuKernelRegisterStart(insHandle));
    CHK_RET(RegisterOneSpecializedKernel(insHandle,
        BCAST_8P4_SMALL_SPECIALIZED_INTRA, intraName,
        reinterpret_cast<void *>(ops_hccl::Ccu8p4SmallRootSpecializedIntraKernel),
        param, root, intraSet, rootCtx));
    CHK_RET(RegisterOneSpecializedKernel(insHandle,
        BCAST_8P4_SMALL_SPECIALIZED_INTER, interName,
        reinterpret_cast<void *>(ops_hccl::Ccu8p4SmallRootSpecializedInterKernel),
        param, root, interSet, rootCtx));
    CHK_RET_CCU(HcommCcuKernelRegisterEnd(insHandle));
    return HCCL_SUCCESS;
}

HcclResult CreateBaseResources(HcclComm comm, const OpParam &param,
    AlgResourceCtx &baseCtx)
{
    const CommEngine engine = CommEngine::COMM_ENGINE_CCU;
    CHK_RET(HcclThreadAcquire(comm, engine, 1,
        THREAD_NOTIFY_NUM, &baseCtx.workerThread));

    ChannelSet intraSet{};
    ChannelSet interSet{};
    CHK_RET(Acquire8p4Channels(comm, param, intraSet, interSet));
    SaveChannelSets(intraSet, interSet, baseCtx);
    return HCCL_SUCCESS;
}

HcclResult GetOrCreateBaseContext(HcclComm comm, const OpParam &param,
    AlgResourceCtx &baseCtx)
{
    const CommEngine engine = CommEngine::COMM_ENGINE_CCU;
    void *ctx = nullptr;
    uint64_t ctxSize = 0;
    if (HcclEngineCtxGet(comm, BASE_TAG, engine, &ctx, &ctxSize) == HCCL_SUCCESS) {
        CHK_PTR_NULL(ctx);
        std::vector<char> sequence(static_cast<char *>(ctx),
            static_cast<char *>(ctx) + ctxSize);
        baseCtx.DeSerialize(sequence);
        return HCCL_SUCCESS;
    }

    CHK_RET(CreateBaseResources(comm, param, baseCtx));
    std::vector<char> sequence = baseCtx.Serialize();
    void *createdCtx = nullptr;
    CHK_RET(HcclEngineCtxCreate(
        comm, BASE_TAG, engine, sequence.size(), &createdCtx));
    CHK_RET(HcclEngineCtxCopy(
        comm, engine, BASE_TAG, sequence.data(), sequence.size(), 0));
    return HCCL_SUCCESS;
}

HcclResult GetOrCreateRootContext(HcclComm comm, const OpParam &param,
    const AlgResourceCtx &baseCtx, char *rootTag, size_t rootTagSize,
    void **ctx, uint64_t *ctxSize)
{
    CHK_PTR_NULL(rootTag);
    CHK_PTR_NULL(ctx);
    CHK_PTR_NULL(ctxSize);

    int written = std::snprintf(rootTag, rootTagSize, "%s_%u",
        ROOT_TAG_PREFIX, param.root);
    CHK_PRT_RET(written <= 0 || static_cast<size_t>(written) >= rootTagSize,
        HCCL_ERROR("[%s] failed to construct root tag, root=%u", LOG_TAG, param.root),
        HCCL_E_INTERNAL);

    const CommEngine engine = CommEngine::COMM_ENGINE_CCU;
    if (HcclEngineCtxGet(comm, rootTag, engine, ctx, ctxSize) == HCCL_SUCCESS) {
        return HCCL_SUCCESS;
    }

    ChannelSet intraSet{};
    ChannelSet interSet{};
    CHK_RET(RestoreChannelSets(baseCtx, intraSet, interSet));

    AlgResourceCtx rootCtx = baseCtx;
    rootCtx.ccuKernels.clear();
    CHK_RET(RegisterRootSpecializedKernels(
        comm, param, param.root, intraSet, interSet, rootCtx));

    std::vector<char> sequence = rootCtx.Serialize();
    CHK_RET(HcclEngineCtxCreate(
        comm, rootTag, engine, sequence.size(), ctx));
    CHK_RET(HcclEngineCtxCopy(
        comm, engine, rootTag, sequence.data(), sequence.size(), 0));
    *ctxSize = sequence.size();
    return HCCL_SUCCESS;
}

} // namespace

HcclResult HcclBroadcast8p4SmallRootSpecializedImpl(
    void *buf, uint64_t count, HcclDataType dataType, uint32_t root,
    HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(buf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    OpParam param{};
    param.inputPtr = buf;
    param.outputPtr = buf;
    param.count = count;
    param.root = root;
    param.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_BROADCAST;

    HcclDfxOpInfo dfxInfo{};
    char commName[COMM_INDENTIFIER_MAX_LENGTH]{};
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_PRT_RET(param.rankSize != BCAST_8P4_RANK_SIZE,
        HCCL_ERROR("[%s] only 8+4 rankSize=12 is supported, rankSize=%u",
            LOG_TAG, param.rankSize),
        HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(root >= param.rankSize,
        HCCL_ERROR("[%s] invalid root=%u, rankSize=%u", LOG_TAG, root, param.rankSize),
        HCCL_E_PARA);
    CHK_PRT_RET(dataType != HCCL_DATA_TYPE_FP32,
        HCCL_ERROR("[%s] only FP32 is supported, dataType=%d",
            LOG_TAG, static_cast<int>(dataType)),
        HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(count > std::numeric_limits<uint64_t>::max() / sizeof(float),
        HCCL_ERROR("[%s] count too large: %lu", LOG_TAG, count), HCCL_E_PARA);

    const uint64_t dataSize = count * sizeof(float);
    CHK_PRT_RET(dataSize != SUPPORTED_512KB_SIZE,
        HCCL_ERROR("[%s] only 512KB is supported, dataSize=%lu", LOG_TAG, dataSize),
        HCCL_E_NOT_SUPPORT);

    const CommEngine engine = CommEngine::COMM_ENGINE_CCU;
    CHK_RET(HcclThreadAcquireWithStream(comm, engine, stream,
        THREAD_NOTIFY_NUM, &param.cpuThread));

    AlgResourceCtx baseCtx{};
    CHK_RET(GetOrCreateBaseContext(comm, param, baseCtx));

    char rootTag[128]{};
    CHK_RET(GetOrCreateRootContext(comm, param, baseCtx,
        rootTag, sizeof(rootTag), &param.resCtx, &param.ctxSize));

    int tagWritten = std::snprintf(param.tag, sizeof(param.tag), "%s", rootTag);
    CHK_PRT_RET(tagWritten <= 0 || static_cast<size_t>(tagWritten) >= sizeof(param.tag),
        HCCL_ERROR("[%s] failed to copy root tag", LOG_TAG),
        HCCL_E_INTERNAL);

    return ops_hccl::ExecOp8p4SmallRootSpecializedBalanced(param);
}

} // namespace bcast8p4_small_final

namespace bcast8p4_large_final {
namespace {

constexpr uint64_t SUPPORTED_512MB_SIZE = 512ULL * 1024ULL * 1024ULL;
constexpr uint64_t SUPPORTED_400MB_4B_SIZE = 400ULL * 1024ULL * 1024ULL + 4ULL;

// The compact pipeline safely reuses bits only after the previous stage has
// completed its full Record -> Wait lifecycle. Highest bit is 4.
constexpr uint32_t CHANNEL_NOTIFY_NUM = 5;

// All stages reuse one thread-notify index. Queue order guarantees that the
// previous start/done Record -> Wait pair is consumed before the next Record.
constexpr uint32_t THREAD_NOTIFY_NUM = 1;
constexpr CommProtocol REQUIRED_PROTOCOL = CommProtocol::COMM_PROTOCOL_UBC_CTP;
constexpr const char *BASE_TAG = "hccl_bcast_final_v4_8p4_base";
constexpr const char *ROOT_TAG_PREFIX = "hccl_bcast_final_v4_8p4_large_root";
constexpr const char *LOG_TAG = "BCAST_FINAL_V4_8P4_LARGE";

struct ChannelSet {
    Broadcast8p4ChannelGroup groupType = BCAST_8P4_GROUP_INTRA;
    std::vector<ChannelHandle> channels;
    std::vector<uint32_t> peers;
    std::array<uint32_t, MAX_RANK_SIZE> channelIndexByRank{};
    uint32_t netLayer = INVALID_VALUE_RANKID;

    ChannelSet()
    {
        channelIndexByRank.fill(INVALID_VALUE_RANKID);
    }
};

bool IsServer0Rank(uint32_t rank)
{
    return rank < BCAST_8P4_SERVER0_RANK_NUM;
}

uint32_t GetServerId(uint32_t rank)
{
    return IsServer0Rank(rank) ? 0U : 1U;
}

uint32_t GetServerBase(uint32_t serverId)
{
    return serverId == 0U ? BCAST_8P4_SERVER0_BASE : BCAST_8P4_SERVER1_BASE;
}

uint32_t GetServerRankNum(uint32_t serverId)
{
    return serverId == 0U ? BCAST_8P4_SERVER0_RANK_NUM : BCAST_8P4_SERVER1_RANK_NUM;
}

bool IsSupportedLargeSize(uint64_t dataSize)
{
    return dataSize == SUPPORTED_512MB_SIZE || dataSize == SUPPORTED_400MB_4B_SIZE;
}

HcclResult FindLinkForPeer(HcclComm comm, uint32_t myRank, uint32_t remoteRank,
    CommLink &selectedLink, uint32_t &selectedLayer)
{
    uint32_t *layers = nullptr;
    uint32_t layerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layers, &layerNum));

    for (uint32_t layerIndex = 0; layerIndex < layerNum; ++layerIndex) {
        CommLink *linkList = nullptr;
        uint32_t listSize = 0;
        HcclResult ret = HcclRankGraphGetLinks(
            comm, layers[layerIndex], myRank, remoteRank, &linkList, &listSize);
        if (ret != HCCL_SUCCESS || linkList == nullptr) {
            continue;
        }
        for (uint32_t linkIndex = 0; linkIndex < listSize; ++linkIndex) {
            if (linkList[linkIndex].linkAttr.linkProtocol == REQUIRED_PROTOCOL) {
                selectedLink = linkList[linkIndex];
                selectedLayer = layers[layerIndex];
                return HCCL_SUCCESS;
            }
        }
    }

    HCCL_ERROR("[%s] no UBC_CTP link found, myRank=%u, peer=%u",
        LOG_TAG, myRank, remoteRank);
    return HCCL_E_NOT_FOUND;
}

HcclResult AcquireOneChannel(HcclComm comm, const OpParam &param, uint32_t remoteRank,
    ChannelSet &set, std::array<bool, MAX_RANK_SIZE> &peerAcquired)
{
    CHK_PRT_RET(remoteRank >= param.rankSize || remoteRank == param.myRank,
        HCCL_ERROR("[%s] invalid remoteRank=%u for myRank=%u",
            LOG_TAG, remoteRank, param.myRank),
        HCCL_E_PARA);
    CHK_PRT_RET(peerAcquired[remoteRank],
        HCCL_ERROR("[%s] peer=%u would receive more than one channel", LOG_TAG, remoteRank),
        HCCL_E_PARA);
    CHK_PRT_RET(set.channelIndexByRank[remoteRank] != INVALID_VALUE_RANKID,
        HCCL_ERROR("[%s] duplicated channel in group, peer=%u", LOG_TAG, remoteRank),
        HCCL_E_PARA);

    CommLink link{};
    uint32_t netLayer = INVALID_VALUE_RANKID;
    CHK_RET(FindLinkForPeer(comm, param.myRank, remoteRank, link, netLayer));

    if (set.netLayer == INVALID_VALUE_RANKID) {
        set.netLayer = netLayer;
    } else {
        CHK_PRT_RET(set.netLayer != netLayer,
            HCCL_ERROR("[%s] one channel group spans multiple layers: previous=%u, current=%u, peer=%u",
                LOG_TAG, set.netLayer, netLayer, remoteRank),
            HCCL_E_NOT_SUPPORT);
    }

    HcclChannelDesc desc{};
    CHK_RET(HcclChannelDescInit(&desc, 1));
    desc.remoteRank = remoteRank;
    desc.notifyNum = CHANNEL_NOTIFY_NUM;
    desc.channelProtocol = link.linkAttr.linkProtocol;
    desc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
    desc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
    desc.localEndpoint.loc = link.srcEndpointDesc.loc;
    desc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
    desc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
    desc.remoteEndpoint.loc = link.dstEndpointDesc.loc;

    ChannelHandle channel{};
    CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CCU, &desc, 1, &channel));

    const uint32_t channelIndex = static_cast<uint32_t>(set.channels.size());
    set.channels.push_back(channel);
    set.peers.push_back(remoteRank);
    set.channelIndexByRank[remoteRank] = channelIndex;
    peerAcquired[remoteRank] = true;
    return HCCL_SUCCESS;
}

HcclResult Acquire8p4Channels(HcclComm comm, const OpParam &param,
    ChannelSet &intraSet, ChannelSet &interSet)
{
    intraSet.groupType = BCAST_8P4_GROUP_INTRA;
    interSet.groupType = BCAST_8P4_GROUP_INTER;

    std::array<bool, MAX_RANK_SIZE> peerAcquired{};
    peerAcquired.fill(false);

    const uint32_t myServer = GetServerId(param.myRank);
    const uint32_t remoteServer = 1U - myServer;
    const uint32_t sameServerBase = GetServerBase(myServer);
    const uint32_t sameServerRankNum = GetServerRankNum(myServer);
    const uint32_t remoteServerBase = GetServerBase(remoteServer);
    const uint32_t remoteServerRankNum = GetServerRankNum(remoteServer);

    for (uint32_t local = 0; local < sameServerRankNum; ++local) {
        const uint32_t peer = sameServerBase + local;
        if (peer == param.myRank) {
            continue;
        }
        CHK_RET(AcquireOneChannel(comm, param, peer, intraSet, peerAcquired));
    }

    for (uint32_t local = 0; local < remoteServerRankNum; ++local) {
        CHK_RET(AcquireOneChannel(
            comm, param, remoteServerBase + local, interSet, peerAcquired));
    }

    const size_t expectedIntra = static_cast<size_t>(sameServerRankNum - 1U);
    const size_t expectedInter = static_cast<size_t>(remoteServerRankNum);
    CHK_PRT_RET(intraSet.channels.size() != expectedIntra ||
            interSet.channels.size() != expectedInter,
        HCCL_ERROR("[%s] unexpected channel counts, intra=%zu/%zu, inter=%zu/%zu",
            LOG_TAG, intraSet.channels.size(), expectedIntra,
            interSet.channels.size(), expectedInter),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(intraSet.netLayer == INVALID_VALUE_RANKID ||
            interSet.netLayer == INVALID_VALUE_RANKID,
        HCCL_ERROR("[%s] failed to resolve topology layers, intra=%u, inter=%u",
            LOG_TAG, intraSet.netLayer, interSet.netLayer),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(intraSet.netLayer == interSet.netLayer,
        HCCL_ERROR("[%s] intra and inter channels resolved to the same layer=%u; "
            "one CCU kernel must not span two IO Dies",
            LOG_TAG, intraSet.netLayer),
        HCCL_E_NOT_SUPPORT);

    for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
        if (peer != param.myRank) {
            CHK_PRT_RET(!peerAcquired[peer],
                HCCL_ERROR("[%s] missing channel to peer=%u", LOG_TAG, peer),
                HCCL_E_INTERNAL);
        }
    }
    return HCCL_SUCCESS;
}

void SaveChannelSets(const ChannelSet &intraSet, const ChannelSet &interSet,
    AlgResourceCtx &resCtxHost)
{
    resCtxHost.intraChannels = intraSet.channels;
    resCtxHost.intraPeers = intraSet.peers;
    resCtxHost.intraLayer = intraSet.netLayer;
    resCtxHost.interChannels = interSet.channels;
    resCtxHost.interPeers = interSet.peers;
    resCtxHost.interLayer = interSet.netLayer;
}

HcclResult RestoreOneChannelSet(const std::vector<ChannelHandle> &channels,
    const std::vector<uint32_t> &peers, uint32_t netLayer,
    Broadcast8p4ChannelGroup groupType, ChannelSet &set)
{
    CHK_PRT_RET(channels.size() != peers.size(),
        HCCL_ERROR("[%s] channel/peer size mismatch, channels=%zu, peers=%zu",
            LOG_TAG, channels.size(), peers.size()),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(netLayer == INVALID_VALUE_RANKID,
        HCCL_ERROR("[%s] invalid cached layer for group=%u",
            LOG_TAG, static_cast<uint32_t>(groupType)),
        HCCL_E_INTERNAL);

    set.groupType = groupType;
    set.channels = channels;
    set.peers = peers;
    set.netLayer = netLayer;
    set.channelIndexByRank.fill(INVALID_VALUE_RANKID);

    for (size_t index = 0; index < peers.size(); ++index) {
        const uint32_t peer = peers[index];
        CHK_PRT_RET(peer >= BCAST_8P4_RANK_SIZE,
            HCCL_ERROR("[%s] invalid cached peer=%u", LOG_TAG, peer),
            HCCL_E_INTERNAL);
        CHK_PRT_RET(set.channelIndexByRank[peer] != INVALID_VALUE_RANKID,
            HCCL_ERROR("[%s] duplicate cached peer=%u", LOG_TAG, peer),
            HCCL_E_INTERNAL);
        set.channelIndexByRank[peer] = static_cast<uint32_t>(index);
    }
    return HCCL_SUCCESS;
}

HcclResult RestoreChannelSets(const AlgResourceCtx &baseCtx,
    ChannelSet &intraSet, ChannelSet &interSet)
{
    CHK_RET(RestoreOneChannelSet(baseCtx.intraChannels, baseCtx.intraPeers,
        baseCtx.intraLayer, BCAST_8P4_GROUP_INTRA, intraSet));
    CHK_RET(RestoreOneChannelSet(baseCtx.interChannels, baseCtx.interPeers,
        baseCtx.interLayer, BCAST_8P4_GROUP_INTER, interSet));
    CHK_PRT_RET(intraSet.netLayer == interSet.netLayer,
        HCCL_ERROR("[%s] cached intra/inter layers are identical, layer=%u",
            LOG_TAG, intraSet.netLayer),
        HCCL_E_NOT_SUPPORT);
    return HCCL_SUCCESS;
}

std::shared_ptr<Broadcast8p4OwnerPipelineKernelArg> BuildKernelArg(
    const OpParam &param, const ChannelSet &set)
{
    auto kernelArg = std::make_shared<Broadcast8p4OwnerPipelineKernelArg>();
    kernelArg->rankId = param.myRank;
    kernelArg->rankSize = param.rankSize;
    kernelArg->rootRank = param.root;
    kernelArg->serverId = GetServerId(param.myRank);
    kernelArg->groupType = static_cast<uint32_t>(set.groupType);
    kernelArg->netLayer = set.netLayer;
    kernelArg->channelCount = static_cast<uint32_t>(set.channels.size());

    for (uint32_t index = 0; index < set.channels.size(); ++index) {
        kernelArg->channels[index] = set.channels[index];
        kernelArg->peerRanks[index] = set.peers[index];
    }
    for (uint32_t rank = 0; rank < MAX_RANK_SIZE; ++rank) {
        kernelArg->channelIndexByRank[rank] = set.channelIndexByRank[rank];
    }
    return kernelArg;
}

HcclResult RegisterOneKernel(CcuInsHandle insHandle, uint32_t kernelIndex,
    const char *kernelName, void *kernelFunc, const OpParam &param,
    const ChannelSet &set, AlgResourceCtx &resCtxHost)
{
    auto kernelArg = BuildKernelArg(param, set);
    const void *kernelArgArray[] = {kernelArg.get()};
    CcuKernelHandle kernelHandle{};
    constexpr uint32_t dieId = 0;
    constexpr uint32_t kernelArgNum = 1;

    CcuResult ret = HcommCcuKernelRegister(insHandle, dieId,
        kernelName, kernelFunc, kernelArgArray, kernelArgNum, &kernelHandle);
    if (ret != CCU_SUCCESS) {
        HCCL_ERROR("[%s] kernel register failed, index=%u, name=%s, group=%u, "
            "layer=%u, channelCount=%u, ccuRet=%d",
            LOG_TAG, kernelIndex, kernelName, kernelArg->groupType,
            kernelArg->netLayer, kernelArg->channelCount, ret);
        return ConvertCcuToHccl(ret);
    }
    resCtxHost.ccuKernels[kernelIndex] = kernelHandle;
    return HCCL_SUCCESS;
}

HcclResult Register8p4Kernels(HcclComm comm, const OpParam &param,
    const ChannelSet &intraSet, const ChannelSet &interSet,
    AlgResourceCtx &resCtxHost)
{
    CcuInsHandle insHandle{0};
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum != 1,
        HCCL_ERROR("[%s] unexpected ccu insNum=%u", LOG_TAG, insNum),
        HCCL_E_INTERNAL);

    resCtxHost.ccuKernels.resize(BCAST_8P4_PIPELINE_KERNEL_NUM);
    CHK_RET_CCU(HcommCcuKernelRegisterStart(insHandle));

    CHK_RET(RegisterOneKernel(insHandle,
        BCAST_8P4_PIPELINE_INTRA,
        "Ccu8p4OwnerPipeline2CompactIntra",
        reinterpret_cast<void *>(ops_hccl::Ccu8p4OwnerPipeline2CompactIntraKernel),
        param, intraSet, resCtxHost));
    CHK_RET(RegisterOneKernel(insHandle,
        BCAST_8P4_PIPELINE_INTER,
        "Ccu8p4OwnerPipeline2CompactInter",
        reinterpret_cast<void *>(ops_hccl::Ccu8p4OwnerPipeline2CompactInterKernel),
        param, interSet, resCtxHost));

    CHK_RET_CCU(HcommCcuKernelRegisterEnd(insHandle));
    return HCCL_SUCCESS;
}

void CopySharedResources(const AlgResourceCtx &baseCtx, AlgResourceCtx &algorithmCtx)
{
    algorithmCtx.workerThread = baseCtx.workerThread;
    algorithmCtx.intraChannels = baseCtx.intraChannels;
    algorithmCtx.intraPeers = baseCtx.intraPeers;
    algorithmCtx.intraLayer = baseCtx.intraLayer;
    algorithmCtx.interChannels = baseCtx.interChannels;
    algorithmCtx.interPeers = baseCtx.interPeers;
    algorithmCtx.interLayer = baseCtx.interLayer;
}

HcclResult StoreEngineCtx(HcclComm comm, CommEngine engine, const char *tag,
    const AlgResourceCtx &resCtxHost, void **ctxOut, uint64_t *ctxSizeOut)
{
    CHK_PTR_NULL(tag);
    CHK_PTR_NULL(ctxOut);
    CHK_PTR_NULL(ctxSizeOut);

    AlgResourceCtx serializableCtx = resCtxHost;
    std::vector<char> sequence = serializableCtx.Serialize();
    *ctxSizeOut = sequence.size();
    CHK_RET(HcclEngineCtxCreate(comm, tag, engine, *ctxSizeOut, ctxOut));
    CHK_RET(HcclEngineCtxCopy(
        comm, engine, tag, sequence.data(), sequence.size(), 0));
    return HCCL_SUCCESS;
}

HcclResult LoadEngineCtx(void *ctx, uint64_t ctxSize, AlgResourceCtx &resCtxHost)
{
    CHK_PTR_NULL(ctx);
    char *bytes = static_cast<char *>(ctx);
    std::vector<char> sequence(bytes, bytes + ctxSize);
    resCtxHost.DeSerialize(sequence);
    return HCCL_SUCCESS;
}

HcclResult GetOrCreateBaseResources(HcclComm comm, const OpParam &param,
    CommEngine engine, AlgResourceCtx &baseCtxHost)
{
    void *baseCtx = nullptr;
    uint64_t baseCtxSize = 0;
    if (HcclEngineCtxGet(comm, BASE_TAG, engine, &baseCtx, &baseCtxSize) == HCCL_SUCCESS) {
        return LoadEngineCtx(baseCtx, baseCtxSize, baseCtxHost);
    }

    CHK_RET(HcclThreadAcquire(comm, engine, 1,
        THREAD_NOTIFY_NUM, &baseCtxHost.workerThread));

    ChannelSet intraSet{};
    ChannelSet interSet{};
    CHK_RET(Acquire8p4Channels(comm, param, intraSet, interSet));
    SaveChannelSets(intraSet, interSet, baseCtxHost);
    baseCtxHost.ccuKernels.clear();

    return StoreEngineCtx(comm, engine, BASE_TAG,
        baseCtxHost, &baseCtx, &baseCtxSize);
}

HcclResult BuildRootSpecificTag(uint32_t root, char *tag, size_t tagSize)
{
    const int written = std::snprintf(tag, tagSize, "%s_%u", ROOT_TAG_PREFIX, root);
    CHK_PRT_RET(written <= 0 || static_cast<size_t>(written) >= tagSize,
        HCCL_ERROR("[%s] failed to construct root-specific EngineCtx tag", LOG_TAG),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult GetOrCreateAlgorithmResources(HcclComm comm, const OpParam &param,
    CommEngine engine, void **ctxOut, uint64_t *ctxSizeOut)
{
    CHK_PTR_NULL(ctxOut);
    CHK_PTR_NULL(ctxSizeOut);

    if (HcclEngineCtxGet(comm, param.tag, engine, ctxOut, ctxSizeOut) == HCCL_SUCCESS) {
        return HCCL_SUCCESS;
    }

    AlgResourceCtx baseCtxHost{};
    CHK_RET(GetOrCreateBaseResources(comm, param, engine, baseCtxHost));

    ChannelSet intraSet{};
    ChannelSet interSet{};
    CHK_RET(RestoreChannelSets(baseCtxHost, intraSet, interSet));

    AlgResourceCtx algorithmCtxHost{};
    CopySharedResources(baseCtxHost, algorithmCtxHost);
    CHK_RET(Register8p4Kernels(
        comm, param, intraSet, interSet, algorithmCtxHost));

    return StoreEngineCtx(comm, engine, param.tag,
        algorithmCtxHost, ctxOut, ctxSizeOut);
}

} // namespace

HcclResult HcclBroadcast8p4Large11OwnerPipeline2Impl(
    void *buf, uint64_t count, HcclDataType dataType, uint32_t root,
    HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(buf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    OpParam param{};
    param.inputPtr = buf;
    param.outputPtr = buf;
    param.count = count;
    param.root = root;
    param.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_BROADCAST;

    HcclDfxOpInfo dfxInfo{};
    char commName[COMM_INDENTIFIER_MAX_LENGTH]{};
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_PRT_RET(param.rankSize != BCAST_8P4_RANK_SIZE,
        HCCL_ERROR("[%s] only 8+4 rankSize=12 is supported, rankSize=%u",
            LOG_TAG, param.rankSize),
        HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(root >= param.rankSize,
        HCCL_ERROR("[%s] invalid root=%u, rankSize=%u", LOG_TAG, root, param.rankSize),
        HCCL_E_PARA);
    CHK_PRT_RET(dataType != HCCL_DATA_TYPE_FP32,
        HCCL_ERROR("[%s] only FP32 is supported, dataType=%d",
            LOG_TAG, static_cast<int>(dataType)),
        HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(count > std::numeric_limits<uint64_t>::max() / sizeof(float),
        HCCL_ERROR("[%s] count too large: %lu", LOG_TAG, count), HCCL_E_PARA);

    const uint64_t dataSize = count * sizeof(float);
    CHK_PRT_RET(!IsSupportedLargeSize(dataSize),
        HCCL_ERROR("[%s] only 512MB and 400MB+4B are supported, dataSize=%lu",
            LOG_TAG, dataSize),
        HCCL_E_NOT_SUPPORT);

    // Root is a registration-time constant in the two generic layer kernels.
    // The runtime stageId changes on each launch; all roots reuse BASE_TAG's
    // single channel to every peer and the same worker thread.
    CHK_RET(BuildRootSpecificTag(root, param.tag, sizeof(param.tag)));

    const CommEngine engine = CommEngine::COMM_ENGINE_CCU;
    CHK_RET(HcclThreadAcquireWithStream(comm, engine, stream,
        THREAD_NOTIFY_NUM, &param.cpuThread));

    CHK_RET(GetOrCreateAlgorithmResources(comm, param, engine,
        &param.resCtx, &param.ctxSize));

    return ops_hccl::ExecOp8p4Large11OwnerPipeline2Compact(param);
}

} // namespace bcast8p4_large_final

namespace bcast4x1_large_final {
namespace {

constexpr uint64_t SIZE_512MB = 512ULL * 1024ULL * 1024ULL;
constexpr uint64_t SIZE_400MB_4B = 400ULL * 1024ULL * 1024ULL + 4ULL;

// bit 1: buffer address, bit 2: memory token,
// bits 3..15: completion for chunks 0..12. After the address exchange has
// completed Record->Wait on bits 1 and 2, those two bits are reused exactly
// once for chunks 13 and 14. The base channel set is shared with the unchanged
// 4x1 small-packet path.
constexpr uint32_t CHANNEL_NOTIFY_NUM = 16;
constexpr CommProtocol REQUIRED_PROTOCOL = CommProtocol::COMM_PROTOCOL_UBC_CTP;
constexpr const char *BASE_TAG =
    "hccl_bcast_final_v4_4x1_base";
constexpr const char *ROOT_TAG_PREFIX =
    "hccl_bcast_final_v8_4x1_large_v2_native_pipeline15_root";
constexpr const char *LOG_TAG =
    "BCAST_FINAL_V8_4X1_LARGE_V2_NATIVE_PIPELINE15";

struct ChannelSet {
    std::vector<ChannelHandle> channels;
    std::vector<uint32_t> peers;
    std::array<uint32_t, MAX_RANK_SIZE> channelIndexByRank{};
    uint32_t netLayer = INVALID_VALUE_RANKID;

    ChannelSet()
    {
        channelIndexByRank.fill(INVALID_VALUE_RANKID);
    }
};

bool IsSupportedSize(uint64_t bytes)
{
    return bytes == SIZE_512MB || bytes == SIZE_400MB_4B;
}

HcclResult FindLinkForPeer(HcclComm comm, uint32_t myRank, uint32_t peerRank,
    CommLink &selectedLink, uint32_t &selectedLayer)
{
    uint32_t *layers = nullptr;
    uint32_t layerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layers, &layerNum));

    for (uint32_t layerIndex = 0; layerIndex < layerNum; ++layerIndex) {
        CommLink *links = nullptr;
        uint32_t linkNum = 0;
        HcclResult ret = HcclRankGraphGetLinks(
            comm, layers[layerIndex], myRank, peerRank, &links, &linkNum);
        if (ret != HCCL_SUCCESS || links == nullptr) {
            continue;
        }
        for (uint32_t linkIndex = 0; linkIndex < linkNum; ++linkIndex) {
            if (links[linkIndex].linkAttr.linkProtocol == REQUIRED_PROTOCOL) {
                selectedLink = links[linkIndex];
                selectedLayer = layers[layerIndex];
                return HCCL_SUCCESS;
            }
        }
    }

    HCCL_ERROR("[%s] no UBC_CTP link, myRank=%u, peer=%u",
        LOG_TAG, myRank, peerRank);
    return HCCL_E_NOT_FOUND;
}

HcclResult AcquireChannels(HcclComm comm, const OpParam &param, ChannelSet &set)
{
    std::array<bool, BCAST_4X1_RANK_SIZE> acquired{};
    acquired.fill(false);

    for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
        if (peer == param.myRank) {
            continue;
        }

        CHK_PRT_RET(acquired[peer],
            HCCL_ERROR("[%s] duplicated peer=%u", LOG_TAG, peer),
            HCCL_E_INTERNAL);

        CommLink link{};
        uint32_t layer = INVALID_VALUE_RANKID;
        CHK_RET(FindLinkForPeer(comm, param.myRank, peer, link, layer));

        if (set.netLayer == INVALID_VALUE_RANKID) {
            set.netLayer = layer;
        } else {
            CHK_PRT_RET(set.netLayer != layer,
                HCCL_ERROR("[%s] 4x1 channels span layers: old=%u new=%u peer=%u",
                    LOG_TAG, set.netLayer, layer, peer),
                HCCL_E_NOT_SUPPORT);
        }

        HcclChannelDesc desc{};
        CHK_RET(HcclChannelDescInit(&desc, 1));
        desc.remoteRank = peer;
        desc.notifyNum = CHANNEL_NOTIFY_NUM;
        desc.channelProtocol = link.linkAttr.linkProtocol;
        desc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
        desc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
        desc.localEndpoint.loc = link.srcEndpointDesc.loc;
        desc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
        desc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
        desc.remoteEndpoint.loc = link.dstEndpointDesc.loc;

        ChannelHandle channel{};
        CHK_RET(HcclChannelAcquire(
            comm, CommEngine::COMM_ENGINE_CCU, &desc, 1, &channel));

        const uint32_t index = static_cast<uint32_t>(set.channels.size());
        set.channels.push_back(channel);
        set.peers.push_back(peer);
        set.channelIndexByRank[peer] = index;
        acquired[peer] = true;
    }

    constexpr uint32_t expectedChannelNum = BCAST_4X1_RANK_SIZE - 1U;
    CHK_PRT_RET(set.channels.size() != expectedChannelNum ||
            set.peers.size() != expectedChannelNum ||
            set.netLayer == INVALID_VALUE_RANKID,
        HCCL_ERROR("[%s] invalid channel set: channels=%zu peers=%zu layer=%u",
            LOG_TAG, set.channels.size(), set.peers.size(), set.netLayer),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

void SaveChannels(const ChannelSet &set, AlgResourceCtx &ctx)
{
    ctx.channels = set.channels;
    ctx.peers = set.peers;
    ctx.netLayer = set.netLayer;
}

HcclResult RestoreChannels(const AlgResourceCtx &ctx, ChannelSet &set)
{
    constexpr uint32_t expectedChannelNum = BCAST_4X1_RANK_SIZE - 1U;
    CHK_PRT_RET(ctx.channels.size() != expectedChannelNum ||
            ctx.peers.size() != expectedChannelNum ||
            ctx.netLayer == INVALID_VALUE_RANKID,
        HCCL_ERROR("[%s] invalid cached base context", LOG_TAG),
        HCCL_E_INTERNAL);

    set.channels = ctx.channels;
    set.peers = ctx.peers;
    set.netLayer = ctx.netLayer;
    set.channelIndexByRank.fill(INVALID_VALUE_RANKID);

    for (uint32_t index = 0; index < expectedChannelNum; ++index) {
        const uint32_t peer = set.peers[index];
        CHK_PRT_RET(peer >= BCAST_4X1_RANK_SIZE,
            HCCL_ERROR("[%s] invalid cached peer=%u", LOG_TAG, peer),
            HCCL_E_INTERNAL);
        CHK_PRT_RET(set.channelIndexByRank[peer] != INVALID_VALUE_RANKID,
            HCCL_ERROR("[%s] duplicate cached peer=%u", LOG_TAG, peer),
            HCCL_E_INTERNAL);
        set.channelIndexByRank[peer] = index;
    }
    return HCCL_SUCCESS;
}

std::shared_ptr<Broadcast4x1LargePipeline15KernelArg> BuildKernelArg(
    const OpParam &param, const ChannelSet &set)
{
    auto arg = std::make_shared<Broadcast4x1LargePipeline15KernelArg>();
    arg->rankId = param.myRank;
    arg->rankSize = param.rankSize;
    arg->channelCount = static_cast<uint32_t>(set.channels.size());

    for (uint32_t index = 0; index < arg->channelCount; ++index) {
        arg->channels[index] = set.channels[index];
        arg->peerRanks[index] = set.peers[index];
    }
    for (uint32_t rank = 0; rank < MAX_RANK_SIZE; ++rank) {
        arg->channelIndexByRank[rank] = set.channelIndexByRank[rank];
    }
    return arg;
}

HcclResult StoreContext(HcclComm comm, const char *tag,
    const AlgResourceCtx &hostCtx, void **deviceCtx, uint64_t *ctxSize)
{
    AlgResourceCtx copy = hostCtx;
    std::vector<char> data = copy.Serialize();
    *ctxSize = data.size();
    CHK_RET(HcclEngineCtxCreate(comm, tag, CommEngine::COMM_ENGINE_CCU,
        *ctxSize, deviceCtx));
    CHK_RET(HcclEngineCtxCopy(comm, CommEngine::COMM_ENGINE_CCU,
        tag, data.data(), data.size(), 0));
    return HCCL_SUCCESS;
}

HcclResult LoadContext(void *deviceCtx, uint64_t ctxSize, AlgResourceCtx &hostCtx)
{
    CHK_PTR_NULL(deviceCtx);
    char *data = static_cast<char *>(deviceCtx);
    std::vector<char> sequence(data, data + ctxSize);
    hostCtx.DeSerialize(sequence);
    return HCCL_SUCCESS;
}

HcclResult GetOrCreateBaseContext(
    HcclComm comm, const OpParam &param, AlgResourceCtx &baseCtx)
{
    void *deviceCtx = nullptr;
    uint64_t ctxSize = 0;
    HcclResult getRet = HcclEngineCtxGet(
        comm, BASE_TAG, CommEngine::COMM_ENGINE_CCU, &deviceCtx, &ctxSize);
    if (getRet == HCCL_SUCCESS) {
        return LoadContext(deviceCtx, ctxSize, baseCtx);
    }

    ChannelSet set{};
    CHK_RET(AcquireChannels(comm, param, set));
    SaveChannels(set, baseCtx);
    baseCtx.ccuKernels.clear();
    return StoreContext(comm, BASE_TAG, baseCtx, &deviceCtx, &ctxSize);
}

HcclResult MakeRootTag(uint32_t root, char *tag, size_t capacity)
{
    const int written = std::snprintf(
        tag, capacity, "%s_%u", ROOT_TAG_PREFIX, root);
    CHK_PRT_RET(written <= 0 || static_cast<size_t>(written) >= capacity,
        HCCL_ERROR("[%s] failed to create root tag", LOG_TAG),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult GetOrCreateAlgorithmContext(HcclComm comm, const OpParam &param,
    void **deviceCtx, uint64_t *ctxSize)
{
    HcclResult getRet = HcclEngineCtxGet(
        comm, param.tag, CommEngine::COMM_ENGINE_CCU, deviceCtx, ctxSize);
    if (getRet == HCCL_SUCCESS) {
        return HCCL_SUCCESS;
    }

    AlgResourceCtx baseCtx{};
    CHK_RET(GetOrCreateBaseContext(comm, param, baseCtx));

    ChannelSet set{};
    CHK_RET(RestoreChannels(baseCtx, set));
    auto kernelArg = BuildKernelArg(param, set);

    CcuInsHandle ccuIns{0};
    uint32_t ccuInsNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &ccuIns, &ccuInsNum));
    CHK_PRT_RET(ccuInsNum != 1,
        HCCL_ERROR("[%s] expected one CCU instance, got=%u", LOG_TAG, ccuInsNum),
        HCCL_E_INTERNAL);

    AlgResourceCtx algorithmCtx = baseCtx;
    algorithmCtx.ccuKernels.resize(BCAST_4X1_LARGE_KERNEL_NUM);
    const void *kernelArgs[] = {kernelArg.get()};
    CcuKernelHandle kernel{};

    CHK_RET_CCU(HcommCcuKernelRegisterStart(ccuIns));
    CcuResult registerRet = HcommCcuKernelRegister(
        ccuIns, 0,
        "Ccu4x1LargePipeline15",
        reinterpret_cast<void *>(ops_hccl::Ccu4x1LargePipeline15Kernel),
        kernelArgs, 1, &kernel);
    if (registerRet != CCU_SUCCESS) {
        return ConvertCcuToHccl(registerRet);
    }
    algorithmCtx.ccuKernels[BCAST_4X1_LARGE_PIPELINE15] = kernel;
    CHK_RET_CCU(HcommCcuKernelRegisterEnd(ccuIns));

    return StoreContext(
        comm, param.tag, algorithmCtx, deviceCtx, ctxSize);
}

} // namespace

HcclResult HcclBroadcast4x1LargePipeline15Impl(void *buf, uint64_t count,
    HcclDataType dataType, uint32_t root, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(buf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    OpParam param{};
    param.inputPtr = buf;
    param.outputPtr = buf;
    param.count = count;
    param.root = root;
    param.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_BROADCAST;

    HcclDfxOpInfo dfxInfo{};
    char commName[COMM_INDENTIFIER_MAX_LENGTH]{};
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(
        commName, reinterpret_cast<void *>(&dfxInfo)));
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));

    CHK_PRT_RET(param.rankSize != BCAST_4X1_RANK_SIZE,
        HCCL_ERROR("[%s] rankSize=%u", LOG_TAG, param.rankSize),
        HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(root >= BCAST_4X1_RANK_SIZE,
        HCCL_ERROR("[%s] invalid root=%u", LOG_TAG, root),
        HCCL_E_PARA);
    CHK_PRT_RET(dataType != HCCL_DATA_TYPE_FP32,
        HCCL_ERROR("[%s] only FP32 is supported", LOG_TAG),
        HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(count > std::numeric_limits<uint64_t>::max() / sizeof(float),
        HCCL_ERROR("[%s] count overflow", LOG_TAG),
        HCCL_E_PARA);

    const uint64_t bytes = count * sizeof(float);
    CHK_PRT_RET(!IsSupportedSize(bytes),
        HCCL_ERROR("[%s] unsupported bytes=%lu", LOG_TAG, bytes),
        HCCL_E_NOT_SUPPORT);

    CHK_RET(MakeRootTag(root, param.tag, sizeof(param.tag)));
    CHK_RET(HcclThreadAcquireWithStream(
        comm, CommEngine::COMM_ENGINE_CCU, stream, 0, &param.cpuThread));
    CHK_RET(GetOrCreateAlgorithmContext(
        comm, param, &param.resCtx, &param.ctxSize));
    return ops_hccl::ExecOp4x1LargePipeline15(param);
}

} // namespace bcast4x1_large_final

HcclResult HcclBroadcast(
    void *buf, uint64_t count, HcclDataType dataType, uint32_t root,
    HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(comm);
    uint32_t rankSize = 0;
    CHK_RET(HcclGetRankSize(comm, &rankSize));
    CHK_PRT_RET(count > std::numeric_limits<uint64_t>::max() / sizeof(float),
        HCCL_ERROR("[BCAST_FINAL_V4] count too large=%lu", count), HCCL_E_PARA);
    const uint64_t dataSize = count * sizeof(float);
    constexpr uint64_t SIZE_512KB = 512ULL * 1024ULL;
    constexpr uint64_t SIZE_512MB = 512ULL * 1024ULL * 1024ULL;
    constexpr uint64_t SIZE_400MB_4B = 400ULL * 1024ULL * 1024ULL + 4ULL;
    const bool isLarge = dataSize == SIZE_512MB || dataSize == SIZE_400MB_4B;

    switch (rankSize) {
        case BCAST_4X1_RANK_SIZE:
            if (dataSize == SIZE_512KB) {
                return bcast4x1_small_final::
                    HcclBroadcast4x1SmallRootSpecializedImpl(
                        buf, count, dataType, root, comm, stream);
            }
            if (isLarge) {
                return bcast4x1_large_final::
                    HcclBroadcast4x1LargePipeline15Impl(
                        buf, count, dataType, root, comm, stream);
            }
            break;
        case BCAST_8P4_RANK_SIZE:
            if (dataSize == SIZE_512KB) {
                return bcast8p4_small_final::
                    HcclBroadcast8p4SmallRootSpecializedImpl(
                        buf, count, dataType, root, comm, stream);
            }
            if (isLarge) {
                return bcast8p4_large_final::
                    HcclBroadcast8p4Large11OwnerPipeline2Impl(buf, count, dataType, root, comm, stream);
            }
            break;
        case BCAST_2X8_RANK_SIZE:
            if (dataSize == SIZE_512KB) {
                return bcast2x8_small_final::
                    HcclBroadcast2x8SmallRootSpecializedImpl(
                        buf, count, dataType, root, comm, stream);
            }
            if (isLarge) {
                return bcast2x8_large_final::
                    HcclBroadcast2x8Large15OwnerPipeline2Impl(buf, count, dataType, root, comm, stream);
            }
            break;
        default:
            HCCL_ERROR("[BCAST_FINAL_V4] unsupported rankSize=%u", rankSize);
            return HCCL_E_NOT_SUPPORT;
    }
    HCCL_ERROR("[BCAST_FINAL_V4] unsupported dataSize=%lu for rankSize=%u",
        dataSize, rankSize);
    return HCCL_E_NOT_SUPPORT;
}
