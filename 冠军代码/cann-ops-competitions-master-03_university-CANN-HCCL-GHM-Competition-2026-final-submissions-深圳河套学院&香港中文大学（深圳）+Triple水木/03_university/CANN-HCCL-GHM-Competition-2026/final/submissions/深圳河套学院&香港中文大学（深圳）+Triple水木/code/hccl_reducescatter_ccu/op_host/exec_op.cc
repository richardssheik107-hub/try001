/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#include <algorithm>
#include <array>
#include <vector>

#include <ccu/ccu_launch.h>
#include <ccu/ccu_res.h>

#include "log.h"
#include "custom.h"
#include "exec_op.h"

namespace {

void *OffsetAddress(void *base, uint64_t offset)
{
    return static_cast<void *>(static_cast<uint8_t *>(base) + offset);
}

HcclResult RecordNotify(ThreadHandle src, ThreadHandle dst, uint32_t id)
{
    return static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(src, dst, id));
}

HcclResult WaitNotify(ThreadHandle thread, uint32_t id)
{
    return static_cast<HcclResult>(
        HcommThreadNotifyWaitOnThread(thread, id, CUSTOM_TIMEOUT));
}

ThreadHandle GetCommThread(const OpParam &param, const AlgResourceCtx &resCtx, uint32_t die)
{
    const uint32_t index = die == 0
        ? resCtx.ioDie0ThreadIndex : resCtx.ioDie1ThreadIndex;
    return index == 0 ? param.cpuThread : resCtx.threads[index];
}

ThreadHandle GetReduceThread(const AlgResourceCtx &resCtx, uint32_t die)
{
    const uint32_t index = die == 0
        ? resCtx.ioDie0ReduceThreadIndex : resCtx.ioDie1ReduceThreadIndex;
    return resCtx.threads[index];
}

template <size_t N>
HcclResult LaunchKernel(ThreadHandle thread, CcuKernelHandle kernel,
    const std::array<uint64_t, N> &args)
{
    return ConvertCcuToHccl(
        HcommCcuKernelLaunch(thread, kernel, args.data(), args.size()));
}

HcclResult LaunchReduceKernel(ThreadHandle thread, CcuKernelHandle kernel,
    custom_rs::ReduceKernelMode mode, void *dst, uint64_t dstToken,
    void *src, uint64_t srcToken, uint64_t bytes)
{
    const std::array<uint64_t, 6> args = {static_cast<uint64_t>(mode),
        reinterpret_cast<uint64_t>(dst), dstToken,
        reinterpret_cast<uint64_t>(src), srcToken, bytes};
    return LaunchKernel(thread, kernel, args);
}

HcclResult GetDataTokens(const OpParam &param, const AlgResourceCtx &resCtx,
    uint64_t recvBytes, uint64_t scratchBytes,
    uint64_t &inputToken, uint64_t &outputToken, uint64_t &scratchToken)
{
    CcuResult ret = HcommCcuGetMemToken(reinterpret_cast<uint64_t>(param.inputPtr),
        recvBytes * param.rankSize, &inputToken);
    if (ret != CCU_SUCCESS) return ConvertCcuToHccl(ret);
    ret = HcommCcuGetMemToken(reinterpret_cast<uint64_t>(param.outputPtr),
        recvBytes, &outputToken);
    if (ret != CCU_SUCCESS) return ConvertCcuToHccl(ret);
    if (scratchBytes == 0) {
        scratchToken = 0;
        return HCCL_SUCCESS;
    }
    ret = HcommCcuGetMemToken(reinterpret_cast<uint64_t>(resCtx.localBuffer.addr),
        scratchBytes, &scratchToken);
    return ConvertCcuToHccl(ret);
}

bool ContainsThread(const std::array<ThreadHandle, custom_rs::REQUIRED_THREAD_NUM> &workers,
    uint32_t workerCount, ThreadHandle thread)
{
    return std::find(workers.begin(), workers.begin() + workerCount, thread) !=
        workers.begin() + workerCount;
}

HcclResult StartWorker(ThreadHandle mainThread, ThreadHandle worker,
    std::array<ThreadHandle, custom_rs::REQUIRED_THREAD_NUM> &workers,
    uint32_t &workerCount)
{
    if (worker == mainThread || ContainsThread(workers, workerCount, worker)) {
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(workerCount >= workers.size(),
        HCCL_ERROR("Too many worker Threads"), HCCL_E_INTERNAL);
    workers[workerCount++] = worker;
    CHK_RET(RecordNotify(mainThread, worker, custom_rs::COMM_LIFECYCLE_NOTIFY));
    return WaitNotify(worker, custom_rs::COMM_LIFECYCLE_NOTIFY);
}

HcclResult JoinWorkers(ThreadHandle mainThread,
    const std::array<ThreadHandle, custom_rs::REQUIRED_THREAD_NUM> &workers,
    uint32_t workerCount)
{
    for (uint32_t workerId = 0; workerId < workerCount; ++workerId) {
        const uint32_t id = custom_rs::MAIN_LOCAL_DONE_BASE + workerId;
        CHK_RET(WaitNotify(mainThread, id));
        CHK_RET(RecordNotify(workers[workerId], mainThread, id));
    }
    return HCCL_SUCCESS;
}

} // namespace

namespace ops_hccl {
namespace {
// Large two-layer topologies use the coarse A/B route:
//   A: local-server pre-reduce on Mesh, then publish one partial through Clos.
//   B: direct A2A on Mesh and Clos, reduced through two ping-pong pipelines.
// Keeping this scheduler separate from ExecOp makes the exact task order easier
// to audit without mixing it with resource restoration and route selection.
HcclResult ExecuteMixedRoute(const OpParam &param, const AlgResourceCtx &resCtx,
    uint64_t recvBytes)
{
    // A. Derive the logical A/B split and the number of source slots owned
    // by each network layer.
    const uint32_t meshDie = resCtx.meshIoDie;
    const uint32_t closDie = resCtx.closIoDie;
    const uint64_t localCount = resCtx.localMemberCounts[meshDie];
    const uint64_t remoteCount = resCtx.directMemberCounts[closDie];
    uint64_t outboundBytes = 0;
    for (const auto &target : resCtx.aggregateTargets) {
        outboundBytes += target.bytes;
    }
    CHK_PRT_RET(localCount == 0 || remoteCount == 0,
        HCCL_ERROR("Invalid mixed route dimensions local[%llu] remote[%llu]",
            static_cast<unsigned long long>(localCount),
            static_cast<unsigned long long>(remoteCount)), HCCL_E_INTERNAL);

    // B. Derive the complete scratch layout once from immutable topology data.
    const auto layout = custom_rs::GetMixedScratchLayout(
        recvBytes, resCtx.localBuffer.size,
        static_cast<uint32_t>(localCount), static_cast<uint32_t>(remoteCount),
        outboundBytes, resCtx.sendRoutePlan, resCtx.receiveRoutePlan);
    CHK_PRT_RET(!layout.enabled,
        HCCL_ERROR("CCL buffer[%llu] cannot hold MixedRoute scratch",
            static_cast<unsigned long long>(resCtx.localBuffer.size)), HCCL_E_INTERNAL);
    const uint64_t receiveAggregateBytes = layout.receiveAggregateBytes;
    const uint64_t directOffset = receiveAggregateBytes;
    const uint64_t directBytes = layout.directBytes;

    // C. Register the three HBM regions with CCU. An address selects a byte
    // location; its token authorizes/translates access to the containing
    // registered region and remains valid for all offsets inside it.
    uint64_t inputToken = 0;
    uint64_t outputToken = 0;
    uint64_t scratchToken = 0;
    CHK_RET(GetDataTokens(param, resCtx, recvBytes, layout.totalBytes,
        inputToken, outputToken, scratchToken));

    // D. Resolve the four logical Missions and define thin launch helpers.
    ThreadHandle meshComm = GetCommThread(param, resCtx, meshDie);
    ThreadHandle closComm = GetCommThread(param, resCtx, closDie);
    ThreadHandle meshReduce = GetReduceThread(resCtx, meshDie);
    ThreadHandle closReduce = GetReduceThread(resCtx, closDie);

    auto launchComm = [&](uint32_t die, ThreadHandle thread,
        const auto &args) -> HcclResult {
        return LaunchKernel(thread,
            resCtx.ccuKernels[custom_rs::GetCommKernelIndex(die)], args);
    };
    auto launchReduce = [&](uint32_t die, ThreadHandle thread,
        custom_rs::ReduceKernelMode mode, void *dst, uint64_t dstToken,
        void *src, uint64_t srcToken, uint64_t bytes) -> HcclResult {
        return LaunchReduceKernel(thread,
            resCtx.ccuKernels[custom_rs::GetReduceKernelIndex(die)],
            mode, dst, dstToken, src, srcToken, bytes);
    };

    // E. Start every worker under the user-stream Thread's control.
    std::array<ThreadHandle, custom_rs::REQUIRED_THREAD_NUM> workers{};
    uint32_t workerCount = 0;
    for (ThreadHandle thread : {meshComm, closComm, meshReduce, closReduce}) {
        CHK_RET(StartWorker(param.cpuThread, thread, workers, workerCount));
    }

    // Initialize the complete result slice once. This is queued before the
    // address exchange and first Mesh fetch, so its HBM copy overlaps startup
    // communication instead of appearing between M0 and the B pipeline.
    CHK_RET(launchReduce(meshDie, meshReduce, custom_rs::ReduceKernelMode::COPY_RANGE,
        param.outputPtr, outputToken,
        OffsetAddress(param.inputPtr, recvBytes * param.myRank), inputToken,
        recvBytes));

    // F. Exchange input and outbound-A address/token pairs once.
    for (uint32_t die : {meshDie, closDie}) {
        const std::array<uint64_t, 7> args = {
            static_cast<uint64_t>(custom_rs::CommKernelMode::EXCHANGE_ADDRESSES),
            reinterpret_cast<uint64_t>(OffsetAddress(
                resCtx.localBuffer.addr, layout.outboundOffset)),
            scratchToken, reinterpret_cast<uint64_t>(param.inputPtr), inputToken, 0, 0};
        CHK_RET(launchComm(die, GetCommThread(param, resCtx, die), args));
    }

    uint64_t meshTask = 0;
    const uint64_t meshTaskCount = resCtx.aggregateTargets.size() + 1U +
        layout.directTileCount;
    auto enqueueMeshFetch = [&](uint64_t sourceOffset, uint64_t bytes,
        void *dst, uint64_t dstToken,
        custom_rs::ReduceKernelMode reduceMode) -> HcclResult {
        const uint32_t bank = static_cast<uint32_t>(meshTask % custom_rs::PIPELINE_BANK_NUM);
        if (meshTask >= custom_rs::PIPELINE_BANK_NUM) {
            CHK_RET(WaitNotify(meshComm, custom_rs::MIXED_BANK_REUSE_BASE + bank));
        }
        const uint64_t bankOffset = layout.meshBankOffset[bank];
        CHK_PRT_RET(bytes > layout.meshBankStride[bank],
            HCCL_ERROR("Mesh task[%llu] exceeds bank[%u] stride[%llu] bytes[%llu]",
                static_cast<unsigned long long>(meshTask), bank,
                static_cast<unsigned long long>(layout.meshBankStride[bank]),
                static_cast<unsigned long long>(bytes)), HCCL_E_INTERNAL);
        const std::array<uint64_t, 7> fetchArgs = {
            static_cast<uint64_t>(custom_rs::CommKernelMode::FETCH_MESH_REMOTE_SLOTS),
            reinterpret_cast<uint64_t>(OffsetAddress(resCtx.localBuffer.addr, bankOffset)),
            scratchToken, reinterpret_cast<uint64_t>(param.inputPtr), inputToken,
            sourceOffset, bytes};
        CHK_RET(launchComm(meshDie, meshComm, fetchArgs));
        CHK_RET(RecordNotify(meshComm, meshReduce, custom_rs::MIXED_OUT_READY_BASE + bank));

        CHK_RET(WaitNotify(meshReduce, custom_rs::MIXED_OUT_READY_BASE + bank));
        CHK_RET(launchReduce(meshDie, meshReduce, reduceMode,
            dst, dstToken, OffsetAddress(resCtx.localBuffer.addr, bankOffset),
            scratchToken, bytes));
        // Only post a reuse notification when this bank will actually be
        // consumed again two Mesh tasks later.  Posting for either of the
        // final two tasks leaves an unmatched Record in the checker graph.
        if (meshTask + custom_rs::PIPELINE_BANK_NUM < meshTaskCount) {
            CHK_RET(RecordNotify(meshReduce, meshComm,
                custom_rs::MIXED_BANK_REUSE_BASE + bank));
        }
        ++meshTask;
        return HCCL_SUCCESS;
    };

    // G. M0: produce remote-target A partials first. Their reduction overlaps
    // the next Mesh fetch and finishes long before Clos consumes them.
    uint64_t outboundTaskOffset = 0;
    for (uint32_t targetSlot = 0; targetSlot < resCtx.aggregateTargets.size(); ++targetSlot) {
        const auto &target = resCtx.aggregateTargets[targetSlot];
        CHK_RET(launchReduce(meshDie, meshReduce, custom_rs::ReduceKernelMode::COPY_RANGE,
            OffsetAddress(resCtx.localBuffer.addr,
                layout.outboundOffset + outboundTaskOffset), scratchToken,
            OffsetAddress(param.inputPtr, recvBytes * target.rank + target.offset),
            inputToken, target.bytes));
        CHK_RET(enqueueMeshFetch(recvBytes * target.rank + target.offset, target.bytes,
            OffsetAddress(resCtx.localBuffer.addr,
                layout.outboundOffset + outboundTaskOffset), scratchToken,
            custom_rs::ReduceKernelMode::REDUCE_MESH_SLOTS_TO_DST));
        outboundTaskOffset += target.bytes;
    }

    // H. Reduce local-target A directly into the initialized output prefix.
    CHK_RET(enqueueMeshFetch(recvBytes * param.myRank, receiveAggregateBytes,
        param.outputPtr, outputToken,
        custom_rs::ReduceKernelMode::REDUCE_MESH_SLOTS_TO_DST));
    CHK_RET(RecordNotify(meshReduce, closReduce, custom_rs::MIXED_LOCAL_A_READY_NOTIFY));

    auto enqueuePartialReadReduce = [&]() -> HcclResult {
        // All outbound A reductions and local A reduction are ordered on the
        // same Mesh-reduce Thread. One final readiness edge therefore covers
        // every outbound partial; per-target ready Record/Wait pairs are redundant.
        CHK_RET(WaitNotify(closReduce, custom_rs::MIXED_LOCAL_A_READY_NOTIFY));
        return launchReduce(closDie, closReduce,
            custom_rs::ReduceKernelMode::READ_REDUCE_CLOS_PARTIALS,
            param.outputPtr, outputToken,
            OffsetAddress(resCtx.localBuffer.addr, layout.outboundOffset),
            scratchToken, 0);
    };

    // I. C0 and M2: remote and local B travel concurrently on different dies.
    // Each network layer has two large ping-pong banks; reductions consume a
    // completed bank while the communication Mission fills the other one.
    for (uint64_t tile = 0; tile < layout.directTileCount; ++tile) {
        if (tile + 1U == layout.directTileCount) {
            CHK_RET(enqueuePartialReadReduce());
        }
        const uint32_t bank = static_cast<uint32_t>(tile % custom_rs::PIPELINE_BANK_NUM);
        const uint64_t tileOffset = layout.asymmetricTwoPass
            ? (tile == 0 ? 0 : layout.directBankBytes[0])
            : tile * layout.directTileBytes;
        const uint64_t tileBytes = layout.asymmetricTwoPass
            ? layout.directBankBytes[static_cast<uint32_t>(tile)]
            : std::min(layout.directTileBytes, directBytes - tileOffset);

        if (tile >= custom_rs::PIPELINE_BANK_NUM) {
            CHK_RET(WaitNotify(closComm, custom_rs::MIXED_BANK_REUSE_BASE + bank));
        }
        const std::array<uint64_t, 7> closArgs = {
            static_cast<uint64_t>(custom_rs::CommKernelMode::FETCH_CLOS_REMOTE_SLOTS),
            reinterpret_cast<uint64_t>(OffsetAddress(
                resCtx.localBuffer.addr, layout.closBankOffset[bank])),
            scratchToken, reinterpret_cast<uint64_t>(param.inputPtr), inputToken,
            recvBytes * param.myRank + directOffset + tileOffset, tileBytes};
        CHK_RET(launchComm(closDie, closComm, closArgs));
        CHK_RET(RecordNotify(closComm, closReduce,
            custom_rs::MIXED_REMOTE_B_READY_BASE + bank));
        CHK_RET(WaitNotify(closReduce, custom_rs::MIXED_REMOTE_B_READY_BASE + bank));
        void *closSlotBase = OffsetAddress(
            resCtx.localBuffer.addr, layout.closBankOffset[bank]);
        CHK_RET(launchReduce(closDie, closReduce,
            custom_rs::ReduceKernelMode::FOLD_SLOTS_IN_PLACE,
            closSlotBase, scratchToken, closSlotBase, scratchToken, tileBytes));
        CHK_RET(RecordNotify(closReduce, meshReduce,
            custom_rs::MIXED_REMOTE_PARTIAL_READY_BASE + bank));

        CHK_RET(enqueueMeshFetch(
            recvBytes * param.myRank + directOffset + tileOffset, tileBytes,
            OffsetAddress(param.outputPtr, directOffset + tileOffset), outputToken,
            custom_rs::ReduceKernelMode::REDUCE_MESH_SLOTS_TO_DST));
        CHK_RET(WaitNotify(meshReduce,
            custom_rs::MIXED_REMOTE_PARTIAL_READY_BASE + bank));
        CHK_RET(launchReduce(meshDie, meshReduce,
            custom_rs::ReduceKernelMode::REDUCE_RANGE,
            OffsetAddress(param.outputPtr, directOffset + tileOffset), outputToken,
            closSlotBase, scratchToken, tileBytes));
        // As above, the final two Clos banks have no future producer that
        // needs to wait for reuse, so they must not emit terminal Records.
        if (tile + custom_rs::PIPELINE_BANK_NUM < layout.directTileCount) {
            CHK_RET(RecordNotify(meshReduce, closComm,
                custom_rs::MIXED_BANK_REUSE_BASE + bank));
        }
    }

    // J. The completed output gates both communication streams. FINISH keeps
    // the cross-rank lifetime barrier without repeating address publication.
    // Only after it is queued do worker streams append their terminal
    // Record back to the user-stream Thread.
    ThreadHandle resultThread = meshReduce;
    for (uint32_t die : {meshDie, closDie}) {
        ThreadHandle thread = GetCommThread(param, resCtx, die);
        if (thread != resultThread) {
            CHK_RET(RecordNotify(resultThread, thread, custom_rs::MIXED_FINAL_READY_NOTIFY));
            CHK_RET(WaitNotify(thread, custom_rs::MIXED_FINAL_READY_NOTIFY));
        }
        const std::array<uint64_t, 7> args = {
            static_cast<uint64_t>(custom_rs::CommKernelMode::FINISH_BARRIER),
            0, 0, 0, 0, 0, 0};
        CHK_RET(launchComm(die, thread, args));
    }

    // Join all workers.  These Records are intentionally their last tasks,
    // as required by the CCU Thread lifecycle contract.
    return JoinWorkers(param.cpuThread, workers, workerCount);
}

// Final-pro small-message route. Each active IO die reduces its assigned
// source ranks directly in CCU Buffer; a dual-die topology leaves one partial
// in scratch and combines it with one dedicated local two-input kernel.
HcclResult ExecuteSmallBufferTileRoute2X8(const OpParam &param,
    const AlgResourceCtx &resCtx, uint64_t recvBytes,
    const custom_rs::SmallBufferTilePlan &plan)
{
    CHK_PRT_RET((resCtx.activeIoDieMask & 3U) != 3U,
        HCCL_ERROR("2x8 small route requires both IO dies"), HCCL_E_INTERNAL);
    const uint32_t resultDie = resCtx.resultIoDie;
    const uint32_t otherDie = resultDie ^ 1U;
    CHK_PRT_RET(recvBytes > resCtx.localBuffer.size,
        HCCL_ERROR("2x8 small scratch[%llu] exceeds CCL buffer[%llu]",
            static_cast<unsigned long long>(recvBytes),
            static_cast<unsigned long long>(resCtx.localBuffer.size)), HCCL_E_INTERNAL);

    uint64_t inputToken = 0;
    uint64_t outputToken = 0;
    uint64_t scratchToken = 0;
    CHK_RET(GetDataTokens(param, resCtx, recvBytes, recvBytes,
        inputToken, outputToken, scratchToken));

    auto launchComm = [&](uint32_t die,
        const std::array<uint64_t, 6> &args) -> HcclResult {
        return LaunchKernel(GetCommThread(param, resCtx, die),
            resCtx.ccuKernels[custom_rs::GetCommKernelIndex(die)], args);
    };

    std::array<ThreadHandle, custom_rs::REQUIRED_THREAD_NUM> workers{};
    uint32_t workerCount = 0;
    for (uint32_t die = 0; die < custom_rs::IO_DIE_NUM; ++die) {
        CHK_RET(StartWorker(param.cpuThread,
            GetCommThread(param, resCtx, die), workers, workerCount));
    }
    const ThreadHandle resultThread = GetCommThread(param, resCtx, resultDie);

    CHK_PRT_RET(plan.tileBytes == 0,
        HCCL_ERROR("2x8 small Buffer tile plan has zero tile size"), HCCL_E_INTERNAL);
    for (uint64_t tileOffset = 0; tileOffset < recvBytes; tileOffset += plan.tileBytes) {
        const uint64_t tileBytes = std::min(plan.tileBytes, recvBytes - tileOffset);
        for (uint32_t die = 0; die < custom_rs::IO_DIE_NUM; ++die) {
            const bool result = die == resultDie;
            const std::array<uint64_t, 6> args = {
                reinterpret_cast<uint64_t>(OffsetAddress(
                    result ? param.outputPtr : resCtx.localBuffer.addr, tileOffset)),
                result ? outputToken : scratchToken,
                reinterpret_cast<uint64_t>(param.inputPtr), inputToken,
                recvBytes * param.myRank + tileOffset, tileBytes};
            CHK_RET(launchComm(die, args));
            if (!result) {
                CHK_RET(RecordNotify(GetCommThread(param, resCtx, die), resultThread,
                    custom_rs::FUSED_READY_BASE + die));
            }
        }

        // The result partial is ordered before this wait on the same Thread;
        // only the other die needs an explicit readiness notification.
        CHK_RET(WaitNotify(resultThread, custom_rs::FUSED_READY_BASE + otherDie));
        const std::array<uint64_t, 5> combineArgs = {
            reinterpret_cast<uint64_t>(OffsetAddress(param.outputPtr, tileOffset)),
            outputToken,
            reinterpret_cast<uint64_t>(OffsetAddress(resCtx.localBuffer.addr, tileOffset)),
            scratchToken, tileBytes};
        CHK_RET(LaunchKernel(resultThread,
            resCtx.ccuKernels[custom_rs::GetReduceKernelIndex(resultDie)], combineArgs));
    }

    return JoinWorkers(param.cpuThread, workers, workerCount);
}

HcclResult ExecuteSmallBufferTileRoute(const OpParam &param, const AlgResourceCtx &resCtx,
    uint64_t recvBytes, const custom_rs::SmallBufferTilePlan &plan)
{
    const bool dualDie = (resCtx.activeIoDieMask & 3U) == 3U;
    const uint32_t resultDie = resCtx.resultIoDie;
    const uint64_t scratchBytes = dualDie ? recvBytes : 0;

    CHK_PRT_RET(scratchBytes > resCtx.localBuffer.size,
        HCCL_ERROR("Small Buffer tile scratch[%llu] exceeds CCL buffer[%llu]",
            static_cast<unsigned long long>(scratchBytes),
            static_cast<unsigned long long>(resCtx.localBuffer.size)), HCCL_E_INTERNAL);
    uint64_t inputToken = 0;
    uint64_t outputToken = 0;
    uint64_t scratchToken = 0;
    CHK_RET(GetDataTokens(param, resCtx, recvBytes, scratchBytes,
        inputToken, outputToken, scratchToken));

    auto launchComm = [&](uint32_t die,
        const std::array<uint64_t, 8> &args) -> HcclResult {
        return LaunchKernel(GetCommThread(param, resCtx, die),
            resCtx.ccuKernels[custom_rs::GetCommKernelIndex(die)], args);
    };

    std::array<uint32_t, custom_rs::IO_DIE_NUM> activeDies{};
    uint32_t activeDieCount = 0;
    std::array<ThreadHandle, custom_rs::REQUIRED_THREAD_NUM> workers{};
    uint32_t workerCount = 0;
    for (uint32_t die = 0; die < custom_rs::IO_DIE_NUM; ++die) {
        if ((resCtx.activeIoDieMask & (1U << die)) == 0 ||
            resCtx.fusedMemberCounts[die] == 0) continue;
        activeDies[activeDieCount++] = die;
        CHK_RET(StartWorker(param.cpuThread,
            GetCommThread(param, resCtx, die), workers, workerCount));
    }
    CHK_PRT_RET(activeDieCount == 0,
        HCCL_ERROR("Small Buffer tile has no active die"), HCCL_E_INTERNAL);

    ThreadHandle resultReduceThread = 0;
    if (dualDie) {
        resultReduceThread = GetReduceThread(resCtx, resultDie);
        CHK_RET(StartWorker(param.cpuThread,
            resultReduceThread, workers, workerCount));
    }

    const bool externalSync = param.rankSize == 12U;
    auto syncDie = [&](uint32_t die) -> HcclResult {
        const std::array<uint64_t, 8> args = {
            static_cast<uint64_t>(custom_rs::CommKernelMode::EXCHANGE_ADDRESSES),
            dualDie ? reinterpret_cast<uint64_t>(resCtx.localBuffer.addr)
                    : reinterpret_cast<uint64_t>(param.outputPtr),
            dualDie ? scratchToken : outputToken,
            reinterpret_cast<uint64_t>(param.inputPtr), inputToken,
            0, 0, recvBytes};
        return launchComm(die, args);
    };
    if (externalSync) {
        for (uint32_t index = 0; index < activeDieCount; ++index) {
            CHK_RET(syncDie(activeDies[index]));
        }
    }

    CHK_PRT_RET(plan.tileBytes == 0,
        HCCL_ERROR("Small Buffer tile plan has zero tile size"), HCCL_E_INTERNAL);
    for (uint64_t tileOffset = 0; tileOffset < recvBytes; tileOffset += plan.tileBytes) {
        const uint64_t tileBytes = std::min(plan.tileBytes, recvBytes - tileOffset);
        for (uint32_t index = 0; index < activeDieCount; ++index) {
            const uint32_t die = activeDies[index];
            const bool result = die == resultDie;
            const std::array<uint64_t, 8> args = {
                static_cast<uint64_t>(custom_rs::CommKernelMode::SMALL_BUFFER_TILE),
                result ? reinterpret_cast<uint64_t>(param.outputPtr)
                       : reinterpret_cast<uint64_t>(resCtx.localBuffer.addr),
                result ? outputToken : scratchToken,
                reinterpret_cast<uint64_t>(param.inputPtr), inputToken,
                tileOffset, tileBytes, recvBytes};
            CHK_RET(launchComm(die, args));
            if (dualDie) {
                CHK_RET(RecordNotify(GetCommThread(param, resCtx, die), resultReduceThread,
                    custom_rs::FUSED_READY_BASE + die));
            }
        }

        if (dualDie) {
            for (uint32_t index = 0; index < activeDieCount; ++index) {
                CHK_RET(WaitNotify(resultReduceThread,
                    custom_rs::FUSED_READY_BASE + activeDies[index]));
            }
            CHK_RET(LaunchReduceKernel(resultReduceThread,
                resCtx.ccuKernels[custom_rs::GetReduceKernelIndex(resultDie)],
                custom_rs::ReduceKernelMode::REDUCE_RANGE,
                OffsetAddress(param.outputPtr, tileOffset), outputToken,
                OffsetAddress(resCtx.localBuffer.addr, tileOffset), scratchToken,
                tileBytes));
        }
    }

    if (externalSync) {
        for (uint32_t index = 0; index < activeDieCount; ++index) {
            CHK_RET(syncDie(activeDies[index]));
        }
    }
    return JoinWorkers(param.cpuThread, workers, workerCount);
}

// 4x1 large messages use two-round recursive halving. Round 1 exchanges two
// output blocks with rank^2 in eight pieces; only after all eight network reads
// finish does round 2 exchange one partial block with rank^1 in nine fade-out
// pieces. The reduction Mission consumes the two static readiness DAGs.
HcclResult Execute4X1RecursiveHalving(const OpParam &param,
    const AlgResourceCtx &resCtx,
    uint64_t recvBytes)
{
    const uint32_t resultDie = resCtx.resultIoDie;
    const auto round2Bytes = custom_rs::Get4X1SegmentBytes(recvBytes);
    const uint64_t round2UnitBytes = custom_rs::Get4X1SegmentUnitBytes(recvBytes);
    const uint64_t round1TileBytes = recvBytes /
        custom_rs::ROUND1_4X1_TILES_PER_BLOCK /
        custom_rs::SLOT_ALIGNMENT_BYTES * custom_rs::SLOT_ALIGNMENT_BYTES;
    const uint64_t round1TailBytes = recvBytes -
        round1TileBytes * (custom_rs::ROUND1_4X1_TILES_PER_BLOCK - 1U);
    CHK_PRT_RET(round1TileBytes == 0 || round1TailBytes == 0 ||
            round2Bytes.front() == 0 ||
            round2Bytes.back() == 0,
        HCCL_ERROR("Invalid 4x1 RH layout recv[%llu] round1[%llu/%llu] round2[%llu/%llu]",
            static_cast<unsigned long long>(recvBytes),
            static_cast<unsigned long long>(round1TileBytes),
            static_cast<unsigned long long>(round1TailBytes),
            static_cast<unsigned long long>(round2Bytes.front()),
            static_cast<unsigned long long>(round2Bytes.back())), HCCL_E_INTERNAL);
    CHK_PRT_RET(resCtx.fusedMemberCounts[resultDie] != 4U,
        HCCL_ERROR("4x1 RH route expected four fused members, got[%u]",
            resCtx.fusedMemberCounts[resultDie]), HCCL_E_INTERNAL);
    const uint64_t scratchBytes = recvBytes;
    CHK_PRT_RET(scratchBytes > resCtx.localBuffer.size,
        HCCL_ERROR("CCL buffer[%llu] cannot hold 4x1 RH scratch[%llu]",
            static_cast<unsigned long long>(resCtx.localBuffer.size),
            static_cast<unsigned long long>(scratchBytes)), HCCL_E_INTERNAL);

    uint64_t inputToken = 0;
    uint64_t outputToken = 0;
    uint64_t scratchToken = 0;
    CHK_RET(GetDataTokens(param, resCtx, recvBytes, scratchBytes,
        inputToken, outputToken, scratchToken));

    ThreadHandle commThread = GetCommThread(param, resCtx, resultDie);
    ThreadHandle reduceThread = GetReduceThread(resCtx, resultDie);
    auto launchComm = [&](const std::array<uint64_t, 10> &args) -> HcclResult {
        return LaunchKernel(commThread,
            resCtx.ccuKernels[custom_rs::GetCommKernelIndex(resultDie)], args);
    };
    auto launchReduce = [&](void *sendInput, void *keepInput) -> HcclResult {
        const std::array<uint64_t, 10> args = {
            reinterpret_cast<uint64_t>(param.outputPtr), outputToken,
            reinterpret_cast<uint64_t>(resCtx.localBuffer.addr), scratchToken,
            reinterpret_cast<uint64_t>(sendInput),
            reinterpret_cast<uint64_t>(keepInput), inputToken, recvBytes,
            round1TileBytes, round1TailBytes};
        return LaunchKernel(reduceThread,
            resCtx.ccuKernels[custom_rs::GetReduceKernelIndex(resultDie)], args);
    };
    std::array<ThreadHandle, custom_rs::REQUIRED_THREAD_NUM> workers{};
    uint32_t workerCount = 0;
    CHK_RET(StartWorker(param.cpuThread, reduceThread, workers, workerCount));

    auto launchBarrier = [&](custom_rs::CommKernelMode mode) -> HcclResult {
        const std::array<uint64_t, 10> args = {
            static_cast<uint64_t>(mode),
            reinterpret_cast<uint64_t>(resCtx.localBuffer.addr), scratchToken,
            reinterpret_cast<uint64_t>(param.inputPtr), inputToken,
            recvBytes, round1TileBytes, round1TailBytes,
            round2UnitBytes, round2Bytes.back()};
        return launchComm(args);
    };
    CHK_RET(launchBarrier(custom_rs::CommKernelMode::EXCHANGE_ADDRESSES));

    // Queue reduction before communication. It immediately waits on round-1
    // readiness, so the first completed network piece can be consumed without
    // paying a later kernel-launch reaction gap.
    const uint32_t sendBlock = param.myRank ^ 1U;
    CHK_RET(launchReduce(
        OffsetAddress(param.inputPtr, recvBytes * sendBlock),
        OffsetAddress(param.inputPtr, recvBytes * param.myRank)));

    const std::array<uint64_t, 10> fetchArgs = {
        static_cast<uint64_t>(
            custom_rs::CommKernelMode::RECURSIVE_HALVING_4X1),
        reinterpret_cast<uint64_t>(resCtx.localBuffer.addr), scratchToken,
        reinterpret_cast<uint64_t>(param.outputPtr), outputToken,
        recvBytes, round1TileBytes, round1TailBytes,
        round2UnitBytes, round2Bytes.back()};
    CHK_RET(launchComm(fetchArgs));
    // All remote reads end when the two-round kernel completes. The reduction
    // Mission may still be finishing local HBM work while the channel lifetime
    // barrier executes.
    CHK_RET(launchBarrier(custom_rs::CommKernelMode::FINISH_BARRIER));

    return JoinWorkers(param.cpuThread, workers, workerCount);
}

} // namespace

HcclResult ExecOp(const OpParam &param)
{
    // 1. Restore the immutable topology/resource description created by
    // reduce_scatter.cc. No communication is submitted in this step.
    AlgResourceCtx resCtx;
    std::vector<char> seq(static_cast<char *>(param.resCtx),
        static_cast<char *>(param.resCtx) + param.ctxSize);
    resCtx.DeSerialize(seq);

    // 2. Select one complete scheduler. Small messages keep A2A's parallel
    // reads but fuse the slot reduction into the communication Kernel.  The
    // 4x1 large cases use dedicated segmented communication/reduction.
    const uint64_t recvBytes = param.count * sizeof(float);
    const uint64_t totalBytes = recvBytes * param.rankSize;
    const bool smallData = totalBytes <= custom_rs::SMALL_TOTAL_DATA_BYTES;
    if (smallData) {
        uint32_t maxMembersPerDie = 0;
        for (uint32_t count : resCtx.fusedMemberCounts) {
            maxMembersPerDie = std::max(maxMembersPerDie, count);
        }
        const auto smallBufferPlan = custom_rs::GetSmallBufferTilePlan(
            param.rankSize, recvBytes, maxMembersPerDie);
        CHK_PRT_RET(!smallBufferPlan.enabled,
            HCCL_ERROR("Small Buffer tile plan is unavailable"), HCCL_E_NOT_SUPPORT);
        if (param.rankSize == 16U) {
            return ExecuteSmallBufferTileRoute2X8(
                param, resCtx, recvBytes, smallBufferPlan);
        }
        return ExecuteSmallBufferTileRoute(param, resCtx, recvBytes, smallBufferPlan);
    }
    if (param.rankSize == 4) {
        return Execute4X1RecursiveHalving(param, resCtx, recvBytes);
    }
    const bool useMixedRoute = recvBytes >= custom_rs::MIXED_ROUTE_MIN_RECV_BYTES &&
        resCtx.meshIoDie != custom_rs::INVALID_IO_DIE &&
        resCtx.closIoDie != custom_rs::INVALID_IO_DIE &&
        resCtx.meshIoDie != resCtx.closIoDie;

    CHK_PRT_RET(!useMixedRoute,
        HCCL_ERROR("No scheduler for rankSize[%u]", param.rankSize), HCCL_E_NOT_SUPPORT);
    return ExecuteMixedRoute(param, resCtx, recvBytes);
}
} // namespace ops_hccl
