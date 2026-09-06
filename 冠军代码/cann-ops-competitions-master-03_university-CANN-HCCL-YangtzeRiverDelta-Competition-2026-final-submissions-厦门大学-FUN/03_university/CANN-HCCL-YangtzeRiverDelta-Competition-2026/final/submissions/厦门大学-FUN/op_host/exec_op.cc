/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Integrated host execution for the Broadcast scheme matrix.
 */
#include "exec_op.h"
#include <array>
#include <cstdint>
#include <limits>
#include <vector>
#include "ccu_launch.h"
#include "ccu_res.h"
#include "custom.h"
#include "log.h"

namespace ops_hccl {
namespace impl_2x8_small {

constexpr uint64_t SUPPORTED_512KB_SIZE = 512ULL * 1024ULL;
constexpr uint32_t THREAD_NOTIFY_INDEX = 0;
constexpr uint32_t THREAD_NOTIFY_TIMEOUT_MS = 0;
constexpr const char *LOG_TAG = "BCAST_FINAL_V4_2X8_SMALL";

uint32_t GetServerId(uint32_t rank)
{
    return rank < BCAST_2X8_SERVER1_BASE ? 0U : 1U;
}

HcclResult LaunchKernel(ThreadHandle thread, const AlgResourceCtx &resCtx,
    uint32_t kernelIndex, const std::vector<uint64_t> &taskArgs)
{
    CHK_PRT_RET(kernelIndex >= resCtx.ccuKernels.size(),
        HCCL_ERROR("[%s] kernel index %u out of range, kernelNum=%zu",
            LOG_TAG, kernelIndex, resCtx.ccuKernels.size()),
        HCCL_E_INTERNAL);

    CcuResult ret = HcommCcuKernelLaunch(
        thread, resCtx.ccuKernels[kernelIndex], taskArgs.data(), taskArgs.size());
    if (ret != CCU_SUCCESS) {
        HCCL_ERROR("[%s] HcommCcuKernelLaunch failed, kernelIndex=%u, ccuRet=%d",
            LOG_TAG, kernelIndex, ret);
        return ConvertCcuToHccl(ret);
    }
    return HCCL_SUCCESS;
}

HcclResult ThreadNotifyRecord(ThreadHandle srcThread, ThreadHandle dstThread,
    uint32_t notifyIndex)
{
    int32_t ret = HcommThreadNotifyRecordOnThread(srcThread, dstThread, notifyIndex);
    CHK_PRT_RET(ret != 0,
        HCCL_ERROR("[%s] HcommThreadNotifyRecordOnThread failed, notifyIndex=%u, ret=%d",
            LOG_TAG, notifyIndex, ret),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult ThreadNotifyWait(ThreadHandle thread, uint32_t notifyIndex)
{
    int32_t ret = HcommThreadNotifyWaitOnThread(
        thread, notifyIndex, THREAD_NOTIFY_TIMEOUT_MS);
    CHK_PRT_RET(ret != 0,
        HCCL_ERROR("[%s] HcommThreadNotifyWaitOnThread failed, notifyIndex=%u, ret=%d",
            LOG_TAG, notifyIndex, ret),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult ValidateResource(const AlgResourceCtx &resCtx)
{
    CHK_PRT_RET(resCtx.ccuKernels.size() !=
            BCAST_2X8_SMALL_SPECIALIZED_KERNEL_NUM,
        HCCL_ERROR("[%s] unexpected kernelNum=%zu, expected=%u",
            LOG_TAG, resCtx.ccuKernels.size(),
            BCAST_2X8_SMALL_SPECIALIZED_KERNEL_NUM),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult RunRootDualDie(const OpParam &param,
    const AlgResourceCtx &resCtx, const std::vector<uint64_t> &taskArgs)
{
    const ThreadHandle mainThread = param.cpuThread;
    const ThreadHandle workerThread = resCtx.workerThread;

    // In 2x8, every root has seven same-server receivers and eight remote-
    // server receivers. Inter is therefore always the heavier branch and is
    // launched directly on the stream-bound main Thread. Intra runs on the
    // worker Thread after the stream-ordering START notify.
    CHK_RET(ThreadNotifyRecord(mainThread, workerThread, THREAD_NOTIFY_INDEX));

    CHK_RET(ThreadNotifyWait(workerThread, THREAD_NOTIFY_INDEX));
    CHK_RET(LaunchKernel(workerThread, resCtx,
        BCAST_2X8_SMALL_SPECIALIZED_INTRA, taskArgs));
    CHK_RET(ThreadNotifyRecord(workerThread, mainThread, THREAD_NOTIFY_INDEX));

    CHK_RET(LaunchKernel(mainThread, resCtx,
        BCAST_2X8_SMALL_SPECIALIZED_INTER, taskArgs));
    CHK_RET(ThreadNotifyWait(mainThread, THREAD_NOTIFY_INDEX));
    return HCCL_SUCCESS;
}

HcclResult RunReceiverSingleChannel(const OpParam &param,
    const AlgResourceCtx &resCtx, const std::vector<uint64_t> &taskArgs)
{
    const bool sameServerAsRoot =
        GetServerId(param.myRank) == GetServerId(param.root);
    const uint32_t kernelIndex = sameServerAsRoot ?
        BCAST_2X8_SMALL_SPECIALIZED_INTRA :
        BCAST_2X8_SMALL_SPECIALIZED_INTER;

    // The selected root-specialized receiver kernel contains exactly one
    // channel: the channel to root. The unrelated layer kernel is not launched.
    return LaunchKernel(param.cpuThread, resCtx, kernelIndex, taskArgs);
}

} // namespace impl_2x8_small

HcclResult ExecOp2x8SmallRootSpecializedBalanced(const OpParam &param)
{
    using namespace impl_2x8_small;
    if (param.count == 0 || param.rankSize == 1) {
        return HCCL_SUCCESS;
    }

    auto sizeIter = SIZE_TABLE.find(param.dataType);
    CHK_PRT_RET(sizeIter == SIZE_TABLE.end(),
        HCCL_ERROR("[%s] unsupported dataType=%d", LOG_TAG, param.dataType),
        HCCL_E_NOT_SUPPORT);
    const uint64_t typeSize = sizeIter->second;
    CHK_PRT_RET(param.count > std::numeric_limits<uint64_t>::max() / typeSize,
        HCCL_ERROR("[%s] count too large: %lu", LOG_TAG, param.count), HCCL_E_PARA);
    const uint64_t dataSize = param.count * typeSize;

    CHK_PRT_RET(param.rankSize != BCAST_2X8_RANK_SIZE,
        HCCL_ERROR("[%s] only 2x8 rankSize=16 is supported, rankSize=%u",
            LOG_TAG, param.rankSize),
        HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(param.root >= param.rankSize,
        HCCL_ERROR("[%s] invalid root=%u, rankSize=%u",
            LOG_TAG, param.root, param.rankSize),
        HCCL_E_PARA);
    CHK_PRT_RET(dataSize != SUPPORTED_512KB_SIZE,
        HCCL_ERROR("[%s] only 512KB is supported, dataSize=%lu", LOG_TAG, dataSize),
        HCCL_E_NOT_SUPPORT);

    char *ctx = static_cast<char *>(param.resCtx);
    CHK_PTR_NULL(ctx);
    std::vector<char> sequence(ctx, ctx + param.ctxSize);
    AlgResourceCtx resCtx{};
    resCtx.DeSerialize(sequence);
    CHK_RET(ValidateResource(resCtx));

    const uint64_t baseAddr = reinterpret_cast<uint64_t>(param.inputPtr);
    uint64_t token = 0;
    CHK_RET_CCU(HcommCcuGetMemToken(baseAddr, dataSize, &token));

    // rootRank and channel selection were fixed at kernel registration time.
    const std::vector<uint64_t> taskArgs = {
        baseAddr,
        token,
        dataSize,
    };

    if (param.myRank == param.root) {
        return RunRootDualDie(param, resCtx, taskArgs);
    }
    return RunReceiverSingleChannel(param, resCtx, taskArgs);
}

} // namespace ops_hccl


namespace ops_hccl {
namespace impl_2x8_large {

constexpr uint64_t SUPPORTED_512MB_SIZE = 512ULL * 1024ULL * 1024ULL;
constexpr uint64_t SUPPORTED_400MB_4B_SIZE = 400ULL * 1024ULL * 1024ULL + 4ULL;
constexpr uint32_t THREAD_NOTIFY_STAGE = 0;
constexpr uint32_t THREAD_NOTIFY_TIMEOUT_MS = 0;
constexpr uint64_t PIPE_STAGE_SCATTER_R0 = 0;
constexpr uint64_t PIPE_STAGE_SCATTER_R1_FANOUT_R0 = 1;
constexpr uint64_t PIPE_STAGE_FANOUT_R1 = 2;
constexpr const char *LOG_TAG = "BCAST_FINAL_V4_2X8_LARGE";

bool IsSupportedLargeSize(uint64_t dataSize)
{
    return dataSize == SUPPORTED_512MB_SIZE || dataSize == SUPPORTED_400MB_4B_SIZE;
}

HcclResult LaunchKernel(ThreadHandle thread, const AlgResourceCtx &resCtx,
    uint32_t kernelIndex, const std::vector<uint64_t> &taskArgs)
{
    CHK_PRT_RET(kernelIndex >= resCtx.ccuKernels.size(),
        HCCL_ERROR("[%s] kernel index %u out of range, kernelNum=%zu",
            LOG_TAG, kernelIndex, resCtx.ccuKernels.size()),
        HCCL_E_INTERNAL);
    CcuResult ret = HcommCcuKernelLaunch(
        thread, resCtx.ccuKernels[kernelIndex], taskArgs.data(), taskArgs.size());
    if (ret != CCU_SUCCESS) {
        HCCL_ERROR("[%s] HcommCcuKernelLaunch failed, kernelIndex=%u, ccuRet=%d",
            LOG_TAG, kernelIndex, ret);
        return ConvertCcuToHccl(ret);
    }
    return HCCL_SUCCESS;
}

HcclResult ThreadNotifyRecord(ThreadHandle srcThread, ThreadHandle dstThread,
    uint32_t notifyIndex)
{
    int32_t ret = HcommThreadNotifyRecordOnThread(srcThread, dstThread, notifyIndex);
    CHK_PRT_RET(ret != 0,
        HCCL_ERROR("[%s] HcommThreadNotifyRecordOnThread failed, notifyIndex=%u, ret=%d",
            LOG_TAG, notifyIndex, ret),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult ThreadNotifyWait(ThreadHandle thread, uint32_t notifyIndex)
{
    int32_t ret = HcommThreadNotifyWaitOnThread(
        thread, notifyIndex, THREAD_NOTIFY_TIMEOUT_MS);
    CHK_PRT_RET(ret != 0,
        HCCL_ERROR("[%s] HcommThreadNotifyWaitOnThread failed, notifyIndex=%u, ret=%d",
            LOG_TAG, notifyIndex, ret),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult ValidateResource(const AlgResourceCtx &resCtx)
{
    CHK_PRT_RET(resCtx.ccuKernels.size() != BCAST_2X8_PIPELINE_KERNEL_NUM,
        HCCL_ERROR("[%s] unexpected kernelNum=%zu, expected=%u",
            LOG_TAG, resCtx.ccuKernels.size(), BCAST_2X8_PIPELINE_KERNEL_NUM),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult EnqueueParallelStage(ThreadHandle mainThread, ThreadHandle interThread,
    const AlgResourceCtx &resCtx, const std::vector<uint64_t> &taskArgs)
{
    CHK_RET(ThreadNotifyRecord(mainThread, interThread, THREAD_NOTIFY_STAGE));

    CHK_RET(ThreadNotifyWait(interThread, THREAD_NOTIFY_STAGE));
    CHK_RET(LaunchKernel(interThread, resCtx,
        BCAST_2X8_PIPELINE_INTER, taskArgs));
    CHK_RET(ThreadNotifyRecord(interThread, mainThread, THREAD_NOTIFY_STAGE));

    CHK_RET(LaunchKernel(mainThread, resCtx,
        BCAST_2X8_PIPELINE_INTRA, taskArgs));
    CHK_RET(ThreadNotifyWait(mainThread, THREAD_NOTIFY_STAGE));
    return HCCL_SUCCESS;
}

std::vector<uint64_t> BuildStageTaskArgs(uint64_t baseAddr, uint64_t token,
    uint64_t baseSliceBytes, uint64_t baseRound0Bytes,
    uint64_t baseRound1Bytes, uint64_t lastRound0Bytes,
    uint64_t lastRound1Bytes, uint64_t stageId)
{
    return {
        baseAddr,
        token,
        baseSliceBytes,
        baseRound0Bytes,
        baseRound1Bytes,
        lastRound0Bytes,
        lastRound1Bytes,
        stageId,
    };
}

HcclResult RunPipeline2Compact(const OpParam &param, const AlgResourceCtx &resCtx,
    uint64_t baseAddr, uint64_t token, uint64_t typeSize)
{
    const uint64_t baseSliceCount = param.count / BCAST_2X8_OWNER_NUM;
    const uint64_t remainCount = param.count % BCAST_2X8_OWNER_NUM;
    const uint64_t lastSliceCount = baseSliceCount + remainCount;

    const uint64_t baseRound0Count = baseSliceCount / BCAST_2X8_PIPELINE_ROUND_NUM;
    const uint64_t baseRound1Count = baseSliceCount - baseRound0Count;
    const uint64_t lastRound0Count = lastSliceCount / BCAST_2X8_PIPELINE_ROUND_NUM;
    const uint64_t lastRound1Count = lastSliceCount - lastRound0Count;

    CHK_PRT_RET(baseRound0Count == 0 || baseRound1Count == 0 ||
            lastRound0Count == 0 || lastRound1Count == 0,
        HCCL_ERROR("[%s] invalid two-round split, base=(%lu,%lu), last=(%lu,%lu)",
            LOG_TAG, baseRound0Count, baseRound1Count,
            lastRound0Count, lastRound1Count),
        HCCL_E_INTERNAL);

    const uint64_t baseSliceBytes = baseSliceCount * typeSize;
    const uint64_t baseRound0Bytes = baseRound0Count * typeSize;
    const uint64_t baseRound1Bytes = baseRound1Count * typeSize;
    const uint64_t lastRound0Bytes = lastRound0Count * typeSize;
    const uint64_t lastRound1Bytes = lastRound1Count * typeSize;

    const ThreadHandle mainIntraThread = param.cpuThread;
    const ThreadHandle interThread = resCtx.workerThread;

    const std::vector<uint64_t> stage0Args = BuildStageTaskArgs(
        baseAddr, token, baseSliceBytes, baseRound0Bytes, baseRound1Bytes,
        lastRound0Bytes, lastRound1Bytes, PIPE_STAGE_SCATTER_R0);
    CHK_RET(EnqueueParallelStage(
        mainIntraThread, interThread, resCtx, stage0Args));

    const std::vector<uint64_t> stage1Args = BuildStageTaskArgs(
        baseAddr, token, baseSliceBytes, baseRound0Bytes, baseRound1Bytes,
        lastRound0Bytes, lastRound1Bytes, PIPE_STAGE_SCATTER_R1_FANOUT_R0);
    CHK_RET(EnqueueParallelStage(
        mainIntraThread, interThread, resCtx, stage1Args));

    const std::vector<uint64_t> stage2Args = BuildStageTaskArgs(
        baseAddr, token, baseSliceBytes, baseRound0Bytes, baseRound1Bytes,
        lastRound0Bytes, lastRound1Bytes, PIPE_STAGE_FANOUT_R1);
    CHK_RET(EnqueueParallelStage(
        mainIntraThread, interThread, resCtx, stage2Args));

    return HCCL_SUCCESS;
}

} // namespace impl_2x8_large

HcclResult ExecOp2x8Large15OwnerPipeline2Compact(const OpParam &param)
{
    using namespace impl_2x8_large;
    if (param.count == 0 || param.rankSize == 1) {
        return HCCL_SUCCESS;
    }

    auto sizeIter = SIZE_TABLE.find(param.dataType);
    CHK_PRT_RET(sizeIter == SIZE_TABLE.end(),
        HCCL_ERROR("[%s] unsupported dataType=%d", LOG_TAG, param.dataType),
        HCCL_E_NOT_SUPPORT);
    const uint64_t typeSize = sizeIter->second;
    CHK_PRT_RET(param.count > std::numeric_limits<uint64_t>::max() / typeSize,
        HCCL_ERROR("[%s] count too large: %lu", LOG_TAG, param.count), HCCL_E_PARA);
    const uint64_t dataSize = param.count * typeSize;

    CHK_PRT_RET(param.rankSize != BCAST_2X8_RANK_SIZE,
        HCCL_ERROR("[%s] only 2×8 rankSize=16 is supported, rankSize=%u",
            LOG_TAG, param.rankSize),
        HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(param.root >= param.rankSize,
        HCCL_ERROR("[%s] invalid root=%u, rankSize=%u",
            LOG_TAG, param.root, param.rankSize),
        HCCL_E_PARA);
    CHK_PRT_RET(!IsSupportedLargeSize(dataSize),
        HCCL_ERROR("[%s] only 512MB and 400MB+4B are supported, dataSize=%lu",
            LOG_TAG, dataSize),
        HCCL_E_NOT_SUPPORT);

    char *ctx = static_cast<char *>(param.resCtx);
    CHK_PTR_NULL(ctx);
    std::vector<char> sequence(ctx, ctx + param.ctxSize);
    AlgResourceCtx resCtx{};
    resCtx.DeSerialize(sequence);
    CHK_RET(ValidateResource(resCtx));

    const uint64_t baseAddr = reinterpret_cast<uint64_t>(param.inputPtr);
    uint64_t token = 0;
    CHK_RET_CCU(HcommCcuGetMemToken(baseAddr, dataSize, &token));

    return RunPipeline2Compact(param, resCtx, baseAddr, token, typeSize);
}

} // namespace ops_hccl


namespace ops_hccl {
namespace impl_4x1_small {

constexpr uint64_t SUPPORTED_512KB_SIZE = 512ULL * 1024ULL;
constexpr const char *LOG_TAG = "BCAST_FINAL_V4_4X1_SMALL";

HcclResult ValidateResource(const AlgResourceCtx &resCtx)
{
    CHK_PRT_RET(resCtx.ccuKernels.size() !=
            BCAST_4X1_SMALL_ROOT_SPECIALIZED_KERNEL_NUM,
        HCCL_ERROR("[%s] unexpected kernelNum=%zu, expected=%u",
            LOG_TAG, resCtx.ccuKernels.size(),
            BCAST_4X1_SMALL_ROOT_SPECIALIZED_KERNEL_NUM),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult LaunchKernel(ThreadHandle thread, const AlgResourceCtx &resCtx,
    const std::vector<uint64_t> &taskArgs)
{
    CcuResult ret = HcommCcuKernelLaunch(
        thread,
        resCtx.ccuKernels[BCAST_4X1_SMALL_ROOT_SPECIALIZED],
        taskArgs.data(),
        taskArgs.size());
    if (ret != CCU_SUCCESS) {
        HCCL_ERROR("[%s] HcommCcuKernelLaunch failed, ccuRet=%d", LOG_TAG, ret);
        return ConvertCcuToHccl(ret);
    }
    return HCCL_SUCCESS;
}

} // namespace impl_4x1_small

HcclResult ExecOp4x1SmallRootSpecialized(const OpParam &param)
{
    using namespace impl_4x1_small;
    if (param.count == 0 || param.rankSize == 1) {
        return HCCL_SUCCESS;
    }

    auto sizeIter = SIZE_TABLE.find(param.dataType);
    CHK_PRT_RET(sizeIter == SIZE_TABLE.end(),
        HCCL_ERROR("[%s] unsupported dataType=%d", LOG_TAG, param.dataType),
        HCCL_E_NOT_SUPPORT);
    const uint64_t typeSize = sizeIter->second;
    CHK_PRT_RET(param.count > std::numeric_limits<uint64_t>::max() / typeSize,
        HCCL_ERROR("[%s] count too large=%lu", LOG_TAG, param.count),
        HCCL_E_PARA);
    const uint64_t dataSize = param.count * typeSize;

    CHK_PRT_RET(param.rankSize != BCAST_4X1_RANK_SIZE,
        HCCL_ERROR("[%s] expected rankSize=4, actual=%u", LOG_TAG, param.rankSize),
        HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(param.root >= param.rankSize,
        HCCL_ERROR("[%s] invalid root=%u", LOG_TAG, param.root),
        HCCL_E_PARA);
    CHK_PRT_RET(dataSize != SUPPORTED_512KB_SIZE,
        HCCL_ERROR("[%s] only 512KB is supported, dataSize=%lu", LOG_TAG, dataSize),
        HCCL_E_NOT_SUPPORT);

    char *ctx = static_cast<char *>(param.resCtx);
    CHK_PTR_NULL(ctx);
    std::vector<char> sequence(ctx, ctx + param.ctxSize);
    AlgResourceCtx resCtx{};
    resCtx.DeSerialize(sequence);
    CHK_RET(ValidateResource(resCtx));

    const uint64_t baseAddr = reinterpret_cast<uint64_t>(param.inputPtr);
    uint64_t token = 0;
    CHK_RET_CCU(HcommCcuGetMemToken(baseAddr, dataSize, &token));

    // Root and role were fixed at kernel registration time.
    const std::vector<uint64_t> taskArgs = {
        baseAddr,
        token,
        dataSize,
    };
    return LaunchKernel(param.cpuThread, resCtx, taskArgs);
}

} // namespace ops_hccl


namespace ops_hccl {
namespace impl_4x1_large_pipeline15 {

constexpr uint64_t SIZE_512MB = 512ULL * 1024ULL * 1024ULL;
constexpr uint64_t SIZE_400MB_4B = 400ULL * 1024ULL * 1024ULL + 4ULL;
constexpr const char *LOG_TAG =
    "BCAST_FINAL_V8_4X1_LARGE_V2_NATIVE_PIPELINE15";

HcclResult ValidateResource(const AlgResourceCtx &ctx)
{
    CHK_PRT_RET(ctx.ccuKernels.size() != BCAST_4X1_LARGE_KERNEL_NUM,
        HCCL_ERROR("[%s] invalid kernel count=%zu, expected=%u",
            LOG_TAG, ctx.ccuKernels.size(), BCAST_4X1_LARGE_KERNEL_NUM),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult LaunchPipeline15(ThreadHandle thread, const AlgResourceCtx &ctx,
    const std::vector<uint64_t> &args)
{
    CHK_RET(ValidateResource(ctx));
    CcuResult ret = HcommCcuKernelLaunch(
        thread, ctx.ccuKernels[BCAST_4X1_LARGE_PIPELINE15],
        args.data(), args.size());
    if (ret != CCU_SUCCESS) {
        HCCL_ERROR("[%s] Pipeline15 kernel launch failed, ccuRet=%d",
            LOG_TAG, ret);
        return ConvertCcuToHccl(ret);
    }
    return HCCL_SUCCESS;
}

} // namespace impl_4x1_large_pipeline15

HcclResult ExecOp4x1LargePipeline15(const OpParam &param)
{
    using namespace impl_4x1_large_pipeline15;
    if (param.count == 0) {
        return HCCL_SUCCESS;
    }

    auto typeIt = SIZE_TABLE.find(param.dataType);
    CHK_PRT_RET(typeIt == SIZE_TABLE.end(),
        HCCL_ERROR("[%s] unsupported data type=%d", LOG_TAG, param.dataType),
        HCCL_E_NOT_SUPPORT);
    const uint64_t typeSize = typeIt->second;

    CHK_PRT_RET(param.count > std::numeric_limits<uint64_t>::max() / typeSize,
        HCCL_ERROR("[%s] count overflow", LOG_TAG),
        HCCL_E_PARA);
    const uint64_t bytes = param.count * typeSize;
    CHK_PRT_RET(bytes != SIZE_512MB && bytes != SIZE_400MB_4B,
        HCCL_ERROR("[%s] unsupported bytes=%lu", LOG_TAG, bytes),
        HCCL_E_NOT_SUPPORT);

    char *rawCtx = static_cast<char *>(param.resCtx);
    CHK_PTR_NULL(rawCtx);
    std::vector<char> sequence(rawCtx, rawCtx + param.ctxSize);
    AlgResourceCtx ctx{};
    ctx.DeSerialize(sequence);
    CHK_RET(ValidateResource(ctx));

    const uint64_t baseAddr = reinterpret_cast<uint64_t>(param.inputPtr);
    uint64_t token = 0;
    CHK_RET_CCU(HcommCcuGetMemToken(baseAddr, bytes, &token));

    const uint64_t baseChunkCount =
        param.count / BCAST_4X1_PIPELINE_CHUNK_NUM;
    const uint64_t remainCount =
        param.count % BCAST_4X1_PIPELINE_CHUNK_NUM;
    CHK_PRT_RET(baseChunkCount == 0,
        HCCL_ERROR("[%s] baseChunkCount is zero", LOG_TAG),
        HCCL_E_INTERNAL);

    const uint64_t baseChunkBytes = baseChunkCount * typeSize;
    const uint64_t lastChunkBytes =
        (baseChunkCount + remainCount) * typeSize;

    // TaskArg order must match Ccu4x1LargePipeline15Kernel::LoadArg exactly.
    const std::vector<uint64_t> taskArgs = {
        baseAddr,
        token,
        static_cast<uint64_t>(param.root),
        baseChunkBytes,
        lastChunkBytes,
    };
    return LaunchPipeline15(param.cpuThread, ctx, taskArgs);
}

} // namespace ops_hccl


namespace ops_hccl {
namespace impl_8p4_small {

constexpr uint64_t SUPPORTED_512KB_SIZE = 512ULL * 1024ULL;
constexpr uint32_t THREAD_NOTIFY_INDEX = 0;
constexpr uint32_t THREAD_NOTIFY_TIMEOUT_MS = 0;
constexpr const char *LOG_TAG = "BCAST_FINAL_V4_8P4_SMALL";

uint32_t GetServerId(uint32_t rank)
{
    return rank < BCAST_8P4_SERVER1_BASE ? 0U : 1U;
}

uint32_t GetIntraTargetCount(uint32_t root)
{
    return GetServerId(root) == 0U ?
        BCAST_8P4_SERVER0_RANK_NUM - 1U :
        BCAST_8P4_SERVER1_RANK_NUM - 1U;
}

uint32_t GetInterTargetCount(uint32_t root)
{
    return GetServerId(root) == 0U ?
        BCAST_8P4_SERVER1_RANK_NUM :
        BCAST_8P4_SERVER0_RANK_NUM;
}

HcclResult LaunchKernel(ThreadHandle thread, const AlgResourceCtx &resCtx,
    uint32_t kernelIndex, const std::vector<uint64_t> &taskArgs)
{
    CHK_PRT_RET(kernelIndex >= resCtx.ccuKernels.size(),
        HCCL_ERROR("[%s] kernel index %u out of range, kernelNum=%zu",
            LOG_TAG, kernelIndex, resCtx.ccuKernels.size()),
        HCCL_E_INTERNAL);

    CcuResult ret = HcommCcuKernelLaunch(
        thread, resCtx.ccuKernels[kernelIndex], taskArgs.data(), taskArgs.size());
    if (ret != CCU_SUCCESS) {
        HCCL_ERROR("[%s] HcommCcuKernelLaunch failed, kernelIndex=%u, ccuRet=%d",
            LOG_TAG, kernelIndex, ret);
        return ConvertCcuToHccl(ret);
    }
    return HCCL_SUCCESS;
}

HcclResult ThreadNotifyRecord(ThreadHandle srcThread, ThreadHandle dstThread,
    uint32_t notifyIndex)
{
    int32_t ret = HcommThreadNotifyRecordOnThread(srcThread, dstThread, notifyIndex);
    CHK_PRT_RET(ret != 0,
        HCCL_ERROR("[%s] HcommThreadNotifyRecordOnThread failed, notifyIndex=%u, ret=%d",
            LOG_TAG, notifyIndex, ret),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult ThreadNotifyWait(ThreadHandle thread, uint32_t notifyIndex)
{
    int32_t ret = HcommThreadNotifyWaitOnThread(
        thread, notifyIndex, THREAD_NOTIFY_TIMEOUT_MS);
    CHK_PRT_RET(ret != 0,
        HCCL_ERROR("[%s] HcommThreadNotifyWaitOnThread failed, notifyIndex=%u, ret=%d",
            LOG_TAG, notifyIndex, ret),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult ValidateResource(const AlgResourceCtx &resCtx)
{
    CHK_PRT_RET(resCtx.ccuKernels.size() !=
            BCAST_8P4_SMALL_SPECIALIZED_KERNEL_NUM,
        HCCL_ERROR("[%s] unexpected kernelNum=%zu, expected=%u",
            LOG_TAG, resCtx.ccuKernels.size(),
            BCAST_8P4_SMALL_SPECIALIZED_KERNEL_NUM),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult RunRootBalancedDualDie(const OpParam &param,
    const AlgResourceCtx &resCtx, const std::vector<uint64_t> &taskArgs)
{
    const uint32_t intraTargetCount = GetIntraTargetCount(param.root);
    const uint32_t interTargetCount = GetInterTargetCount(param.root);

    const bool interIsHeavier = interTargetCount > intraTargetCount;
    const uint32_t mainKernelIndex = interIsHeavier ?
        BCAST_8P4_SMALL_SPECIALIZED_INTER :
        BCAST_8P4_SMALL_SPECIALIZED_INTRA;
    const uint32_t workerKernelIndex = interIsHeavier ?
        BCAST_8P4_SMALL_SPECIALIZED_INTRA :
        BCAST_8P4_SMALL_SPECIALIZED_INTER;

    const ThreadHandle mainThread = param.cpuThread;
    const ThreadHandle workerThread = resCtx.workerThread;

    // Queue the worker start after all earlier user-stream work.
    CHK_RET(ThreadNotifyRecord(mainThread, workerThread, THREAD_NOTIFY_INDEX));

    // Worker queue: wait for stream ordering, execute the lighter branch, and
    // signal completion to the main thread.
    CHK_RET(ThreadNotifyWait(workerThread, THREAD_NOTIFY_INDEX));
    CHK_RET(LaunchKernel(workerThread, resCtx, workerKernelIndex, taskArgs));
    CHK_RET(ThreadNotifyRecord(workerThread, mainThread, THREAD_NOTIFY_INDEX));

    // The heavier layer is launched directly on the stream-bound main Thread,
    // so its critical path does not include the worker wake-up delay.
    CHK_RET(LaunchKernel(mainThread, resCtx, mainKernelIndex, taskArgs));
    CHK_RET(ThreadNotifyWait(mainThread, THREAD_NOTIFY_INDEX));
    return HCCL_SUCCESS;
}

HcclResult RunReceiverSingleChannel(const OpParam &param,
    const AlgResourceCtx &resCtx, const std::vector<uint64_t> &taskArgs)
{
    const bool sameServerAsRoot = GetServerId(param.myRank) == GetServerId(param.root);
    const uint32_t kernelIndex = sameServerAsRoot ?
        BCAST_8P4_SMALL_SPECIALIZED_INTRA :
        BCAST_8P4_SMALL_SPECIALIZED_INTER;

    // The root-specialized receiver kernel contains exactly one channel: the
    // channel to root. The unrelated layer kernel is never launched.
    return LaunchKernel(param.cpuThread, resCtx, kernelIndex, taskArgs);
}

} // namespace impl_8p4_small

HcclResult ExecOp8p4SmallRootSpecializedBalanced(const OpParam &param)
{
    using namespace impl_8p4_small;
    if (param.count == 0 || param.rankSize == 1) {
        return HCCL_SUCCESS;
    }

    auto sizeIter = SIZE_TABLE.find(param.dataType);
    CHK_PRT_RET(sizeIter == SIZE_TABLE.end(),
        HCCL_ERROR("[%s] unsupported dataType=%d", LOG_TAG, param.dataType),
        HCCL_E_NOT_SUPPORT);
    const uint64_t typeSize = sizeIter->second;
    CHK_PRT_RET(param.count > std::numeric_limits<uint64_t>::max() / typeSize,
        HCCL_ERROR("[%s] count too large: %lu", LOG_TAG, param.count), HCCL_E_PARA);
    const uint64_t dataSize = param.count * typeSize;

    CHK_PRT_RET(param.rankSize != BCAST_8P4_RANK_SIZE,
        HCCL_ERROR("[%s] only 8+4 rankSize=12 is supported, rankSize=%u",
            LOG_TAG, param.rankSize),
        HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(param.root >= param.rankSize,
        HCCL_ERROR("[%s] invalid root=%u, rankSize=%u",
            LOG_TAG, param.root, param.rankSize),
        HCCL_E_PARA);
    CHK_PRT_RET(dataSize != SUPPORTED_512KB_SIZE,
        HCCL_ERROR("[%s] only 512KB is supported, dataSize=%lu", LOG_TAG, dataSize),
        HCCL_E_NOT_SUPPORT);

    char *ctx = static_cast<char *>(param.resCtx);
    CHK_PTR_NULL(ctx);
    std::vector<char> sequence(ctx, ctx + param.ctxSize);
    AlgResourceCtx resCtx{};
    resCtx.DeSerialize(sequence);
    CHK_RET(ValidateResource(resCtx));

    const uint64_t baseAddr = reinterpret_cast<uint64_t>(param.inputPtr);
    uint64_t token = 0;
    CHK_RET_CCU(HcommCcuGetMemToken(baseAddr, dataSize, &token));

    // rootRank is absent: it was fixed at kernel registration time.
    const std::vector<uint64_t> taskArgs = {
        baseAddr,
        token,
        dataSize,
    };

    if (param.myRank == param.root) {
        return RunRootBalancedDualDie(param, resCtx, taskArgs);
    }
    return RunReceiverSingleChannel(param, resCtx, taskArgs);
}

} // namespace ops_hccl


namespace ops_hccl {
namespace impl_8p4_large {

constexpr uint64_t SUPPORTED_512MB_SIZE = 512ULL * 1024ULL * 1024ULL;
constexpr uint64_t SUPPORTED_400MB_4B_SIZE = 400ULL * 1024ULL * 1024ULL + 4ULL;
constexpr uint32_t THREAD_NOTIFY_STAGE = 0;
constexpr uint32_t THREAD_NOTIFY_TIMEOUT_MS = 0;
constexpr uint64_t PIPE_STAGE_SCATTER_R0 = 0;
constexpr uint64_t PIPE_STAGE_SCATTER_R1_FANOUT_R0 = 1;
constexpr uint64_t PIPE_STAGE_FANOUT_R1 = 2;
constexpr const char *LOG_TAG = "BCAST_FINAL_V4_8P4_LARGE";

bool IsSupportedLargeSize(uint64_t dataSize)
{
    return dataSize == SUPPORTED_512MB_SIZE || dataSize == SUPPORTED_400MB_4B_SIZE;
}

HcclResult LaunchKernel(ThreadHandle thread, const AlgResourceCtx &resCtx,
    uint32_t kernelIndex, const std::vector<uint64_t> &taskArgs)
{
    CHK_PRT_RET(kernelIndex >= resCtx.ccuKernels.size(),
        HCCL_ERROR("[%s] kernel index %u out of range, kernelNum=%zu",
            LOG_TAG, kernelIndex, resCtx.ccuKernels.size()),
        HCCL_E_INTERNAL);
    CcuResult ret = HcommCcuKernelLaunch(
        thread, resCtx.ccuKernels[kernelIndex], taskArgs.data(), taskArgs.size());
    if (ret != CCU_SUCCESS) {
        HCCL_ERROR("[%s] HcommCcuKernelLaunch failed, kernelIndex=%u, ccuRet=%d",
            LOG_TAG, kernelIndex, ret);
        return ConvertCcuToHccl(ret);
    }
    return HCCL_SUCCESS;
}

HcclResult ThreadNotifyRecord(ThreadHandle srcThread, ThreadHandle dstThread,
    uint32_t notifyIndex)
{
    int32_t ret = HcommThreadNotifyRecordOnThread(srcThread, dstThread, notifyIndex);
    CHK_PRT_RET(ret != 0,
        HCCL_ERROR("[%s] HcommThreadNotifyRecordOnThread failed, notifyIndex=%u, ret=%d",
            LOG_TAG, notifyIndex, ret),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult ThreadNotifyWait(ThreadHandle thread, uint32_t notifyIndex)
{
    int32_t ret = HcommThreadNotifyWaitOnThread(
        thread, notifyIndex, THREAD_NOTIFY_TIMEOUT_MS);
    CHK_PRT_RET(ret != 0,
        HCCL_ERROR("[%s] HcommThreadNotifyWaitOnThread failed, notifyIndex=%u, ret=%d",
            LOG_TAG, notifyIndex, ret),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult ValidateResource(const AlgResourceCtx &resCtx)
{
    CHK_PRT_RET(resCtx.ccuKernels.size() != BCAST_8P4_PIPELINE_KERNEL_NUM,
        HCCL_ERROR("[%s] unexpected kernelNum=%zu, expected=%u",
            LOG_TAG, resCtx.ccuKernels.size(), BCAST_8P4_PIPELINE_KERNEL_NUM),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult EnqueueParallelStage(ThreadHandle mainThread, ThreadHandle interThread,
    const AlgResourceCtx &resCtx, const std::vector<uint64_t> &taskArgs)
{
    CHK_RET(ThreadNotifyRecord(mainThread, interThread, THREAD_NOTIFY_STAGE));

    CHK_RET(ThreadNotifyWait(interThread, THREAD_NOTIFY_STAGE));
    CHK_RET(LaunchKernel(interThread, resCtx,
        BCAST_8P4_PIPELINE_INTER, taskArgs));
    CHK_RET(ThreadNotifyRecord(interThread, mainThread, THREAD_NOTIFY_STAGE));

    CHK_RET(LaunchKernel(mainThread, resCtx,
        BCAST_8P4_PIPELINE_INTRA, taskArgs));
    CHK_RET(ThreadNotifyWait(mainThread, THREAD_NOTIFY_STAGE));
    return HCCL_SUCCESS;
}

std::vector<uint64_t> BuildStageTaskArgs(uint64_t baseAddr, uint64_t token,
    uint64_t baseSliceBytes, uint64_t baseRound0Bytes,
    uint64_t baseRound1Bytes, uint64_t lastRound0Bytes,
    uint64_t lastRound1Bytes, uint64_t stageId)
{
    return {
        baseAddr,
        token,
        baseSliceBytes,
        baseRound0Bytes,
        baseRound1Bytes,
        lastRound0Bytes,
        lastRound1Bytes,
        stageId,
    };
}

HcclResult RunPipeline2Compact(const OpParam &param, const AlgResourceCtx &resCtx,
    uint64_t baseAddr, uint64_t token, uint64_t typeSize)
{
    const uint64_t baseSliceCount = param.count / BCAST_8P4_OWNER_NUM;
    const uint64_t remainCount = param.count % BCAST_8P4_OWNER_NUM;
    const uint64_t lastSliceCount = baseSliceCount + remainCount;

    const uint64_t baseRound0Count = baseSliceCount / BCAST_8P4_PIPELINE_ROUND_NUM;
    const uint64_t baseRound1Count = baseSliceCount - baseRound0Count;
    const uint64_t lastRound0Count = lastSliceCount / BCAST_8P4_PIPELINE_ROUND_NUM;
    const uint64_t lastRound1Count = lastSliceCount - lastRound0Count;

    CHK_PRT_RET(baseRound0Count == 0 || baseRound1Count == 0 ||
            lastRound0Count == 0 || lastRound1Count == 0,
        HCCL_ERROR("[%s] invalid two-round split, base=(%lu,%lu), last=(%lu,%lu)",
            LOG_TAG, baseRound0Count, baseRound1Count,
            lastRound0Count, lastRound1Count),
        HCCL_E_INTERNAL);

    const uint64_t baseSliceBytes = baseSliceCount * typeSize;
    const uint64_t baseRound0Bytes = baseRound0Count * typeSize;
    const uint64_t baseRound1Bytes = baseRound1Count * typeSize;
    const uint64_t lastRound0Bytes = lastRound0Count * typeSize;
    const uint64_t lastRound1Bytes = lastRound1Count * typeSize;

    const ThreadHandle mainIntraThread = param.cpuThread;
    const ThreadHandle interThread = resCtx.workerThread;

    const std::vector<uint64_t> stage0Args = BuildStageTaskArgs(
        baseAddr, token, baseSliceBytes, baseRound0Bytes, baseRound1Bytes,
        lastRound0Bytes, lastRound1Bytes, PIPE_STAGE_SCATTER_R0);
    CHK_RET(EnqueueParallelStage(
        mainIntraThread, interThread, resCtx, stage0Args));

    const std::vector<uint64_t> stage1Args = BuildStageTaskArgs(
        baseAddr, token, baseSliceBytes, baseRound0Bytes, baseRound1Bytes,
        lastRound0Bytes, lastRound1Bytes, PIPE_STAGE_SCATTER_R1_FANOUT_R0);
    CHK_RET(EnqueueParallelStage(
        mainIntraThread, interThread, resCtx, stage1Args));

    const std::vector<uint64_t> stage2Args = BuildStageTaskArgs(
        baseAddr, token, baseSliceBytes, baseRound0Bytes, baseRound1Bytes,
        lastRound0Bytes, lastRound1Bytes, PIPE_STAGE_FANOUT_R1);
    CHK_RET(EnqueueParallelStage(
        mainIntraThread, interThread, resCtx, stage2Args));

    return HCCL_SUCCESS;
}

} // namespace impl_8p4_large

HcclResult ExecOp8p4Large11OwnerPipeline2Compact(const OpParam &param)
{
    using namespace impl_8p4_large;
    if (param.count == 0 || param.rankSize == 1) {
        return HCCL_SUCCESS;
    }

    auto sizeIter = SIZE_TABLE.find(param.dataType);
    CHK_PRT_RET(sizeIter == SIZE_TABLE.end(),
        HCCL_ERROR("[%s] unsupported dataType=%d", LOG_TAG, param.dataType),
        HCCL_E_NOT_SUPPORT);
    const uint64_t typeSize = sizeIter->second;
    CHK_PRT_RET(param.count > std::numeric_limits<uint64_t>::max() / typeSize,
        HCCL_ERROR("[%s] count too large: %lu", LOG_TAG, param.count), HCCL_E_PARA);
    const uint64_t dataSize = param.count * typeSize;

    CHK_PRT_RET(param.rankSize != BCAST_8P4_RANK_SIZE,
        HCCL_ERROR("[%s] only 8+4 rankSize=12 is supported, rankSize=%u",
            LOG_TAG, param.rankSize),
        HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(param.root >= param.rankSize,
        HCCL_ERROR("[%s] invalid root=%u, rankSize=%u",
            LOG_TAG, param.root, param.rankSize),
        HCCL_E_PARA);
    CHK_PRT_RET(!IsSupportedLargeSize(dataSize),
        HCCL_ERROR("[%s] only 512MB and 400MB+4B are supported, dataSize=%lu",
            LOG_TAG, dataSize),
        HCCL_E_NOT_SUPPORT);

    char *ctx = static_cast<char *>(param.resCtx);
    CHK_PTR_NULL(ctx);
    std::vector<char> sequence(ctx, ctx + param.ctxSize);
    AlgResourceCtx resCtx{};
    resCtx.DeSerialize(sequence);
    CHK_RET(ValidateResource(resCtx));

    const uint64_t baseAddr = reinterpret_cast<uint64_t>(param.inputPtr);
    uint64_t token = 0;
    CHK_RET_CCU(HcommCcuGetMemToken(baseAddr, dataSize, &token));

    return RunPipeline2Compact(param, resCtx, baseAddr, token, typeSize);
}

} // namespace ops_hccl
