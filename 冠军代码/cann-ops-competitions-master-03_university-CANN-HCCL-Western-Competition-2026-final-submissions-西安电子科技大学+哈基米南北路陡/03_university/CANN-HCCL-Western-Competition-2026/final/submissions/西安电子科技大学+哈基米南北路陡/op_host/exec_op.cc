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
#include <cstdint>
#include <limits>
#include <vector>

#include <ccu/ccu_launch.h>
#include <ccu/ccu_res.h>
#include <hcomm/hcomm_primitives.h>

#include "custom.h"
#include "exec_op.h"
#include "log.h"

namespace ops_hccl {
namespace {
    constexpr uint64_t SCRATCH_ALIGNMENT = 512;
    constexpr uint64_t FP32_BYTES = sizeof(float);
    constexpr uint64_t BYTES_512K = 524288ULL;
    constexpr uint64_t BYTES_512M = 536870912ULL;
    constexpr uint64_t BYTES_400M4B = 419430404ULL;

    Algorithm2x8 ExpectedAlgorithm(const OpParam &param, uint64_t totalBytes)
    {
        if (param.rankSize == 4) {
            if (totalBytes == BYTES_512K) {
                return Algorithm2x8::V109_DIRECT_MESH_R16_512K;
            }
            if (totalBytes == BYTES_512M) {
                return Algorithm2x8::P4_OUTPUT_ACCUMULATE_CHAIN_512M;
            }
            if (totalBytes == BYTES_400M4B) {
                return Algorithm2x8::P4_OUTPUT_ACCUMULATE_CHAIN_400M4B;
            }
            return Algorithm2x8::OWNER_RSAG;
        }
        if (param.rankSize == 12) {
            if (totalBytes == BYTES_512K) {
                return Algorithm2x8::V101_DIRECT_MESH_512K;
            }
            if (totalBytes == BYTES_512M) {
                return param.inputPtr == param.outputPtr ? Algorithm2x8::P12_ALLPAIRS_EARLY_INIT_512M
                                                        : Algorithm2x8::P12_ALLPAIRS_1SEG_OUTPUT_TREE_512M;
            }
            if (totalBytes == BYTES_400M4B) {
                return param.inputPtr == param.outputPtr ? Algorithm2x8::P12_ALLPAIRS_1SEG_400M4B
                                                        : Algorithm2x8::P12_ALLPAIRS_1SEG_OUTPUT_TREE_400M4B;
            }
            return Algorithm2x8::OWNER_RSAG;
        }
        if (param.rankSize == 16) {
            if (totalBytes == BYTES_512K) {
                return Algorithm2x8::V109_DIRECT_MESH_R16_512K;
            }
            if (totalBytes == BYTES_512M) {
                return Algorithm2x8::CROSS_LANE_512M;
            }
            if (totalBytes == BYTES_400M4B) {
                return param.inputPtr == param.outputPtr ? Algorithm2x8::CROSS_LANE_400M4B
                                                        : Algorithm2x8::CROSS_LANE_DIRECT_OUTPUT_400M4B;
            }
        }
        return Algorithm2x8::OWNER_RSAG;
    }

    const char *AlgorithmName(Algorithm2x8 algorithm)
    {
        switch (algorithm) {
            case Algorithm2x8::BFLY16_CLOS_4STEP:
                return "BFLY16_CLOS_4STEP";
            case Algorithm2x8::CROSS_LANE_512M:
                return "CROSS_LANE_512M";
            case Algorithm2x8::CROSS_LANE_400M4B:
                return "CROSS_LANE_400M4B";
            case Algorithm2x8::CROSS_LANE_DIRECT_OUTPUT_400M4B:
                return "CROSS_LANE_DIRECT_OUTPUT_400M4B";
            case Algorithm2x8::P4_DIRECT_RSAG_512K:
                return "P4_DIRECT_RSAG_512K";
            case Algorithm2x8::P4_ROTATE3_512M:
                return "P4_ROTATE3_512M";
            case Algorithm2x8::P4_ROTATE3_400M4B:
                return "P4_ROTATE3_400M4B";
            case Algorithm2x8::P4_DIRECT_SLOT3_512M:
                return "P4_DIRECT_SLOT3_512M";
            case Algorithm2x8::P4_DIRECT_SLOT3_400M4B:
                return "P4_DIRECT_SLOT3_400M4B";
            case Algorithm2x8::P4_ROTATE3_UNIQUE_NOTIFY_512M:
                return "P4_ROTATE3_UNIQUE_NOTIFY_512M";
            case Algorithm2x8::P4_ROTATE3_UNIQUE_NOTIFY_400M4B:
                return "P4_ROTATE3_UNIQUE_NOTIFY_400M4B";
            case Algorithm2x8::P4_ROTATE3_DIRECT_OUTPUT_512M:
                return "P4_ROTATE3_DIRECT_OUTPUT_512M";
            case Algorithm2x8::P4_ROTATE3_DIRECT_OUTPUT_400M4B:
                return "P4_ROTATE3_DIRECT_OUTPUT_400M4B";
            case Algorithm2x8::P4_PULL_R3_512M:
                return "P4_PULL_R3_512M";
            case Algorithm2x8::P4_PULL_R3_400M4B:
                return "P4_PULL_R3_400M4B";
            case Algorithm2x8::P12_DIRECT_RSAG_512K:
                return "P12_DIRECT_RSAG_512K";
            case Algorithm2x8::P12_HIER8_512M:
                return "P12_HIER8_512M";
            case Algorithm2x8::P12_HIER8_400M4B:
                return "P12_HIER8_400M4B";
            case Algorithm2x8::P12_ALLPAIRS_512M:
                return "P12_ALLPAIRS_512M";
            case Algorithm2x8::P12_ALLPAIRS_400M4B:
                return "P12_ALLPAIRS_400M4B";
            case Algorithm2x8::P12_ALLPAIRS_BALANCED_512M:
                return "P12_ALLPAIRS_BALANCED_512M";
            case Algorithm2x8::P12_ALLPAIRS_BALANCED_400M4B:
                return "P12_ALLPAIRS_BALANCED_400M4B";
            case Algorithm2x8::P12_DIRECT_RSAG_PRESYNC_ONCE_512K:
                return "P12_DIRECT_RSAG_PRESYNC_ONCE_512K";
            case Algorithm2x8::P12_ALLPAIRS_BALANCED_PRESYNC_ONCE_512M:
                return "P12_ALLPAIRS_BALANCED_PRESYNC_ONCE_512M";
            case Algorithm2x8::P12_ALLPAIRS_BALANCED_PRESYNC_ONCE_400M4B:
                return "P12_ALLPAIRS_BALANCED_PRESYNC_ONCE_400M4B";
            case Algorithm2x8::P12_DIRECT_RSAG_EARLY_INIT_512K:
                return "P12_DIRECT_RSAG_EARLY_INIT_512K";
            case Algorithm2x8::P12_ALLPAIRS_EARLY_INIT_512M:
                return "P12_ALLPAIRS_EARLY_INIT_512M";
            case Algorithm2x8::P12_ALLPAIRS_1SEG_400M4B:
                return "P12_ALLPAIRS_1SEG_400M4B";
            case Algorithm2x8::P12_ALLPAIRS_1SEG_OUTPUT_TREE_512M:
                return "P12_ROTATING_READREDUCE_512M";
            case Algorithm2x8::P12_ALLPAIRS_1SEG_OUTPUT_TREE_400M4B:
                return "P12_ROTATING_READREDUCE_400M4B";
            case Algorithm2x8::P16_MESH_LANE_512M:
                return "P16_MESH_LANE_512M";
            case Algorithm2x8::P16_MESH_LANE_400M4B:
                return "P16_MESH_LANE_400M4B";
            case Algorithm2x8::P4_OUTPUT_ACCUMULATE_512M:
                return "P4_OUTPUT_ACCUMULATE_512M";
            case Algorithm2x8::P4_OUTPUT_ACCUMULATE_400M4B:
                return "P4_OUTPUT_ACCUMULATE_400M4B";
            case Algorithm2x8::P12_CLOS_ACCUMULATE_512M:
                return "P12_CLOS_ACCUMULATE_512M";
            case Algorithm2x8::P12_CLOS_ACCUMULATE_400M4B:
                return "P12_CLOS_ACCUMULATE_400M4B";
            case Algorithm2x8::P4_OUTPUT_ACCUMULATE_CHAIN_512M:
                return "P4_OUTPUT_ACCUMULATE_CHAIN_512M";
            case Algorithm2x8::P4_OUTPUT_ACCUMULATE_CHAIN_400M4B:
                return "P4_OUTPUT_ACCUMULATE_CHAIN_400M4B";
            case Algorithm2x8::P4_OUTPUT_ACCUMULATE_WAVE2_512M:
                return "P4_OUTPUT_ACCUMULATE_WAVE2_512M";
            case Algorithm2x8::P4_OUTPUT_ACCUMULATE_WAVE2_400M4B:
                return "P4_OUTPUT_ACCUMULATE_WAVE2_400M4B";
            case Algorithm2x8::V101_DIRECT_MESH_512K:
                return "V101_DIRECT_MESH_512K";
            case Algorithm2x8::V109_DIRECT_MESH_R16_512K:
                return "V109_DIRECT_MESH_R16_512K";
            default:
                return "OWNER_RSAG_V6";
        }
    }

    bool IsP12DirectRsagAlgorithm(Algorithm2x8 algorithm)
    {
        return algorithm == Algorithm2x8::P12_DIRECT_RSAG_512K
               || algorithm == Algorithm2x8::P12_DIRECT_RSAG_PRESYNC_ONCE_512K
               || algorithm == Algorithm2x8::P12_DIRECT_RSAG_EARLY_INIT_512K;
    }

    bool IsV101SmallAlgorithm(Algorithm2x8 algorithm)
    {
        return algorithm == Algorithm2x8::V101_DIRECT_MESH_512K
               || algorithm == Algorithm2x8::V109_DIRECT_MESH_R16_512K;
    }

    bool IsV109R16SmallAlgorithm(Algorithm2x8 algorithm)
    {
        return algorithm == Algorithm2x8::V109_DIRECT_MESH_R16_512K;
    }

    bool IsP4PullR3Algorithm(Algorithm2x8 algorithm)
    {
        return algorithm == Algorithm2x8::P4_PULL_R3_512M || algorithm == Algorithm2x8::P4_PULL_R3_400M4B;
    }

    bool IsP4OutputAccumulateAlgorithm(Algorithm2x8 algorithm)
    {
        return algorithm == Algorithm2x8::P4_OUTPUT_ACCUMULATE_512M
               || algorithm == Algorithm2x8::P4_OUTPUT_ACCUMULATE_400M4B
               || algorithm == Algorithm2x8::P4_OUTPUT_ACCUMULATE_CHAIN_512M
               || algorithm == Algorithm2x8::P4_OUTPUT_ACCUMULATE_CHAIN_400M4B
               || algorithm == Algorithm2x8::P4_OUTPUT_ACCUMULATE_WAVE2_512M
               || algorithm == Algorithm2x8::P4_OUTPUT_ACCUMULATE_WAVE2_400M4B;
    }

    bool IsP4OutputAccumulateChainAlgorithm(Algorithm2x8 algorithm)
    {
        return algorithm == Algorithm2x8::P4_OUTPUT_ACCUMULATE_CHAIN_512M
               || algorithm == Algorithm2x8::P4_OUTPUT_ACCUMULATE_CHAIN_400M4B
               || algorithm == Algorithm2x8::P4_OUTPUT_ACCUMULATE_WAVE2_512M
               || algorithm == Algorithm2x8::P4_OUTPUT_ACCUMULATE_WAVE2_400M4B;
    }

    bool IsP4OutputAccumulateWave2Algorithm(Algorithm2x8 algorithm)
    {
        return algorithm == Algorithm2x8::P4_OUTPUT_ACCUMULATE_WAVE2_512M
               || algorithm == Algorithm2x8::P4_OUTPUT_ACCUMULATE_WAVE2_400M4B;
    }

    bool IsP16MeshLaneAlgorithm(Algorithm2x8 algorithm)
    {
        return algorithm == Algorithm2x8::P16_MESH_LANE_512M || algorithm == Algorithm2x8::P16_MESH_LANE_400M4B;
    }

    bool IsCrossLaneAlgorithm(Algorithm2x8 algorithm)
    {
        return algorithm == Algorithm2x8::CROSS_LANE_512M || algorithm == Algorithm2x8::CROSS_LANE_400M4B
               || algorithm == Algorithm2x8::CROSS_LANE_DIRECT_OUTPUT_400M4B;
    }

    bool IsCrossLaneDirectOutputAlgorithm(Algorithm2x8 algorithm)
    {
        return algorithm == Algorithm2x8::CROSS_LANE_DIRECT_OUTPUT_400M4B;
    }

    bool IsP12ClosAccumulateAlgorithm(Algorithm2x8 algorithm)
    {
        return algorithm == Algorithm2x8::P12_CLOS_ACCUMULATE_512M
               || algorithm == Algorithm2x8::P12_CLOS_ACCUMULATE_400M4B;
    }

    bool IsP12PreSyncOnceAlgorithm(Algorithm2x8 algorithm)
    {
        return algorithm == Algorithm2x8::P12_DIRECT_RSAG_PRESYNC_ONCE_512K
               || algorithm == Algorithm2x8::P12_ALLPAIRS_BALANCED_PRESYNC_ONCE_512M
               || algorithm == Algorithm2x8::P12_ALLPAIRS_BALANCED_PRESYNC_ONCE_400M4B
               || algorithm == Algorithm2x8::P12_DIRECT_RSAG_EARLY_INIT_512K
               || algorithm == Algorithm2x8::P12_ALLPAIRS_EARLY_INIT_512M
               || algorithm == Algorithm2x8::P12_ALLPAIRS_1SEG_400M4B
               || algorithm == Algorithm2x8::P12_ALLPAIRS_1SEG_OUTPUT_TREE_512M
               || algorithm == Algorithm2x8::P12_ALLPAIRS_1SEG_OUTPUT_TREE_400M4B
               || IsP12ClosAccumulateAlgorithm(algorithm);
    }

    bool IsP12AllPairsAlgorithm(Algorithm2x8 algorithm)
    {
        return algorithm == Algorithm2x8::P12_ALLPAIRS_512M || algorithm == Algorithm2x8::P12_ALLPAIRS_400M4B
               || algorithm == Algorithm2x8::P12_ALLPAIRS_BALANCED_512M
               || algorithm == Algorithm2x8::P12_ALLPAIRS_BALANCED_400M4B
               || algorithm == Algorithm2x8::P12_ALLPAIRS_BALANCED_PRESYNC_ONCE_512M
               || algorithm == Algorithm2x8::P12_ALLPAIRS_BALANCED_PRESYNC_ONCE_400M4B
               || algorithm == Algorithm2x8::P12_ALLPAIRS_EARLY_INIT_512M
               || algorithm == Algorithm2x8::P12_ALLPAIRS_1SEG_400M4B
               || algorithm == Algorithm2x8::P12_ALLPAIRS_1SEG_OUTPUT_TREE_512M
               || algorithm == Algorithm2x8::P12_ALLPAIRS_1SEG_OUTPUT_TREE_400M4B
               || IsP12ClosAccumulateAlgorithm(algorithm);
    }

    bool IsP12AllPairs512MAlgorithm(Algorithm2x8 algorithm)
    {
        return algorithm == Algorithm2x8::P12_ALLPAIRS_512M || algorithm == Algorithm2x8::P12_ALLPAIRS_BALANCED_512M
               || algorithm == Algorithm2x8::P12_ALLPAIRS_BALANCED_PRESYNC_ONCE_512M
               || algorithm == Algorithm2x8::P12_ALLPAIRS_EARLY_INIT_512M
               || algorithm == Algorithm2x8::P12_CLOS_ACCUMULATE_512M
               || algorithm == Algorithm2x8::P12_ALLPAIRS_1SEG_OUTPUT_TREE_512M;
    }

    bool IsP12AllPairsBalancedAlgorithm(Algorithm2x8 algorithm)
    {
        return algorithm == Algorithm2x8::P12_ALLPAIRS_BALANCED_512M
               || algorithm == Algorithm2x8::P12_ALLPAIRS_BALANCED_400M4B
               || algorithm == Algorithm2x8::P12_ALLPAIRS_BALANCED_PRESYNC_ONCE_512M
               || algorithm == Algorithm2x8::P12_ALLPAIRS_BALANCED_PRESYNC_ONCE_400M4B
               || algorithm == Algorithm2x8::P12_ALLPAIRS_EARLY_INIT_512M
               || algorithm == Algorithm2x8::P12_ALLPAIRS_1SEG_400M4B
               || algorithm == Algorithm2x8::P12_ALLPAIRS_1SEG_OUTPUT_TREE_512M
               || algorithm == Algorithm2x8::P12_ALLPAIRS_1SEG_OUTPUT_TREE_400M4B;
    }

    bool IsP12AllPairsOneSegmentAlgorithm(Algorithm2x8 algorithm)
    {
        return algorithm == Algorithm2x8::P12_ALLPAIRS_1SEG_400M4B
               || algorithm == Algorithm2x8::P12_ALLPAIRS_1SEG_OUTPUT_TREE_512M
               || algorithm == Algorithm2x8::P12_ALLPAIRS_1SEG_OUTPUT_TREE_400M4B;
    }

    bool IsP12OutputTreeAlgorithm(Algorithm2x8 algorithm)
    {
        return algorithm == Algorithm2x8::P12_ALLPAIRS_1SEG_OUTPUT_TREE_512M
               || algorithm == Algorithm2x8::P12_ALLPAIRS_1SEG_OUTPUT_TREE_400M4B;
    }

    HcclResult ValidateResource(
        const OpParam &param, uint64_t totalBytes, Algorithm2x8 algorithm, const AlgResourceCtx &resource)
    {
        uint32_t expectedVersion = AlgResourceCtx::OWNER_VERSION;
        if (IsV109R16SmallAlgorithm(algorithm)) {
            expectedVersion = AlgResourceCtx::V109_DIRECT_MESH_R16_512K_VERSION;
        } else if (IsV101SmallAlgorithm(algorithm)) {
            expectedVersion = AlgResourceCtx::V101_DIRECT_MESH_512K_VERSION;
        } else if (algorithm == Algorithm2x8::P12_ALLPAIRS_1SEG_OUTPUT_TREE_512M) {
            expectedVersion = AlgResourceCtx::P12_ALLPAIRS_1SEG_OUTPUT_TREE_512M_VERSION;
        } else if (algorithm == Algorithm2x8::P12_ALLPAIRS_1SEG_OUTPUT_TREE_400M4B) {
            expectedVersion = AlgResourceCtx::P12_ALLPAIRS_1SEG_OUTPUT_TREE_400M4B_VERSION;
        } else if (IsP4OutputAccumulateWave2Algorithm(algorithm)) {
            expectedVersion = algorithm == Algorithm2x8::P4_OUTPUT_ACCUMULATE_WAVE2_400M4B
                                  ? AlgResourceCtx::P4_OUTPUT_ACCUMULATE_WAVE2_400M4B_VERSION
                                  : AlgResourceCtx::P4_OUTPUT_ACCUMULATE_WAVE2_VERSION;
        } else if (IsP4OutputAccumulateChainAlgorithm(algorithm)) {
            expectedVersion = AlgResourceCtx::P4_OUTPUT_ACCUMULATE_CHAIN_VERSION;
        } else if (IsP4OutputAccumulateAlgorithm(algorithm)) {
            expectedVersion = AlgResourceCtx::P4_OUTPUT_ACCUMULATE_VERSION;
        } else if (IsP16MeshLaneAlgorithm(algorithm)) {
            expectedVersion = AlgResourceCtx::P16_MESH_LANE_VERSION;
        } else if (IsP12ClosAccumulateAlgorithm(algorithm)) {
            expectedVersion = AlgResourceCtx::P12_CLOS_ACCUMULATE_VERSION;
        } else if (IsP4PullR3Algorithm(algorithm)) {
            expectedVersion = AlgResourceCtx::P4_PULL_R3_VERSION;
        } else if (algorithm == Algorithm2x8::P12_DIRECT_RSAG_EARLY_INIT_512K) {
            expectedVersion = AlgResourceCtx::P12_DIRECT_RSAG_EARLY_INIT_512K_VERSION;
        } else if (algorithm == Algorithm2x8::P12_ALLPAIRS_EARLY_INIT_512M) {
            expectedVersion = AlgResourceCtx::P12_ALLPAIRS_EARLY_INIT_512M_VERSION;
        } else if (algorithm == Algorithm2x8::P12_ALLPAIRS_1SEG_400M4B) {
            expectedVersion = AlgResourceCtx::P12_ALLPAIRS_1SEG_400M4B_VERSION;
        } else if (IsP12PreSyncOnceAlgorithm(algorithm)) {
            expectedVersion = AlgResourceCtx::P12_PRESYNC_ONCE_VERSION;
        } else if (algorithm == Algorithm2x8::BFLY16_CLOS_4STEP) {
            expectedVersion = AlgResourceCtx::BFLY_FUSED_VERSION;
        } else if (algorithm == Algorithm2x8::CROSS_LANE_512M) {
            expectedVersion = AlgResourceCtx::CROSS_LANE_V82_512M_VERSION;
        } else if (IsCrossLaneAlgorithm(algorithm)) {
            expectedVersion = IsCrossLaneDirectOutputAlgorithm(algorithm)
                                  ? AlgResourceCtx::CROSS_LANE_DIRECT_OUTPUT_VERSION
                                  : AlgResourceCtx::CROSS_LANE_VERSION;
        } else if (algorithm == Algorithm2x8::P4_DIRECT_RSAG_512K || algorithm == Algorithm2x8::P12_DIRECT_RSAG_512K) {
            expectedVersion = AlgResourceCtx::DIRECT_RSAG_VERSION;
        } else if (algorithm == Algorithm2x8::P4_ROTATE3_512M || algorithm == Algorithm2x8::P4_ROTATE3_400M4B) {
            expectedVersion = AlgResourceCtx::P4_ROTATE3_VERSION;
        } else if (algorithm == Algorithm2x8::P4_DIRECT_SLOT3_512M
                   || algorithm == Algorithm2x8::P4_DIRECT_SLOT3_400M4B) {
            expectedVersion = AlgResourceCtx::P4_DIRECT_SLOT3_VERSION;
        } else if (algorithm == Algorithm2x8::P4_ROTATE3_UNIQUE_NOTIFY_512M
                   || algorithm == Algorithm2x8::P4_ROTATE3_UNIQUE_NOTIFY_400M4B) {
            expectedVersion = AlgResourceCtx::P4_ROTATE3_UNIQUE_NOTIFY_VERSION;
        } else if (algorithm == Algorithm2x8::P4_ROTATE3_DIRECT_OUTPUT_512M
                   || algorithm == Algorithm2x8::P4_ROTATE3_DIRECT_OUTPUT_400M4B) {
            expectedVersion = AlgResourceCtx::P4_ROTATE3_DIRECT_OUTPUT_VERSION;
        } else if (algorithm == Algorithm2x8::P12_HIER8_512M || algorithm == Algorithm2x8::P12_HIER8_400M4B) {
            expectedVersion = AlgResourceCtx::P12_HIER8_VERSION;
        } else if (IsP12AllPairsBalancedAlgorithm(algorithm)) {
            expectedVersion = AlgResourceCtx::P12_ALLPAIRS_BALANCED_VERSION;
        } else if (IsP12AllPairsAlgorithm(algorithm)) {
            expectedVersion = AlgResourceCtx::P12_ALLPAIRS_VERSION;
        }
        CHK_PRT_RET(resource.magic != AlgResourceCtx::MAGIC || resource.version != expectedVersion,
            HCCL_ERROR("[ValidateResource] invalid EngineCtx magic/version"), HCCL_E_INTERNAL);
        CHK_PRT_RET(resource.rankSize != param.rankSize,
            HCCL_ERROR("[ValidateResource] cached rank size %u does not match %u", resource.rankSize, param.rankSize),
            HCCL_E_INTERNAL);
        FinalTopology expectedTopology;
        if (param.rankSize == 4) {
            expectedTopology = FinalTopology::TOPOLOGY_4X1;
        } else if (param.rankSize == 12) {
            expectedTopology = FinalTopology::TOPOLOGY_8X4;
        } else if (param.rankSize == 16) {
            expectedTopology = FinalTopology::TOPOLOGY_2X8;
        } else {
            HCCL_ERROR("[ValidateResource] unsupported cached rank size %u", param.rankSize);
            return HCCL_E_NOT_SUPPORT;
        }
        CHK_PRT_RET(resource.topology != expectedTopology,
            HCCL_ERROR("[ValidateResource] cached topology does not match rank size %u", param.rankSize),
            HCCL_E_INTERNAL);
        CHK_PRT_RET(resource.localBuffer.addr == nullptr || resource.localBuffer.size == 0,
            HCCL_ERROR("[ValidateResource] invalid HCCL scratch buffer"), HCCL_E_INTERNAL);
        CHK_PRT_RET(resource.ccuKernels.empty() || resource.ccuKernels.size() > 2
                        || resource.ccuKernels.size() != resource.kernelMeta.size(),
            HCCL_ERROR("[ValidateResource] invalid kernel metadata"), HCCL_E_INTERNAL);
        CHK_PRT_RET(resource.extraThreads.size() + 1 != resource.ccuKernels.size(),
            HCCL_ERROR("[ValidateResource] kernel/thread count mismatch"), HCCL_E_INTERNAL);
        uint32_t primaryCount = 0;
        uint32_t totalChannelCount = 0;
        for (size_t i = 0; i < resource.kernelMeta.size(); ++i) {
            CHK_PRT_RET(resource.kernelMeta[i].actualDieId >= 2,
                HCCL_ERROR("[ValidateResource] kernel %zu has invalid die %u", i, resource.kernelMeta[i].actualDieId),
                HCCL_E_INTERNAL);
            if (i > 0) {
                CHK_PRT_RET(resource.kernelMeta[i - 1].actualDieId >= resource.kernelMeta[i].actualDieId,
                    HCCL_ERROR("[ValidateResource] kernels are not strictly ordered by actual die"), HCCL_E_INTERNAL);
            }
            CHK_PRT_RET(
                resource.kernelMeta[i].channelCount == 0 || resource.kernelMeta[i].channelCount >= MAX_RANK_SIZE,
                HCCL_ERROR("[ValidateResource] kernel %zu has invalid channel count %u", i,
                    resource.kernelMeta[i].channelCount),
                HCCL_E_INTERNAL);
            totalChannelCount += resource.kernelMeta[i].channelCount;
            primaryCount += (resource.kernelMeta[i].roleFlags & KERNEL_ROLE_PRIMARY) != 0 ? 1U : 0U;
        }
        CHK_PRT_RET(primaryCount != 1,
            HCCL_ERROR("[ValidateResource] expected exactly one primary kernel, got %u", primaryCount),
            HCCL_E_INTERNAL);
        uint32_t expectedChannelCount = param.rankSize - 1U;
        if (algorithm == Algorithm2x8::BFLY16_CLOS_4STEP) {
            expectedChannelCount = 4U;
        }
        if (algorithm == Algorithm2x8::P12_HIER8_512M || algorithm == Algorithm2x8::P12_HIER8_400M4B) {
            CHK_PRT_RET(totalChannelCount != 5U && totalChannelCount != 8U,
                HCCL_ERROR("[ValidateResource] hierarchy channel count %u is neither 5 nor 8", totalChannelCount),
                HCCL_E_INTERNAL);
        } else {
            CHK_PRT_RET(totalChannelCount != expectedChannelCount,
                HCCL_ERROR("[ValidateResource] channel count %u does not match algorithm %s expectation %u",
                    totalChannelCount, AlgorithmName(algorithm), expectedChannelCount),
                HCCL_E_INTERNAL);
        }
        if (algorithm == Algorithm2x8::BFLY16_CLOS_4STEP) {
            uint32_t phaseMask = 0;
            for (const auto &meta : resource.kernelMeta) {
                const uint32_t kernelMask = (meta.roleFlags >> KERNEL_ROLE_BFLY_PHASE_SHIFT) & 0xFU;
                CHK_PRT_RET((phaseMask & kernelMask) != 0,
                    HCCL_ERROR("[ValidateResource] duplicate butterfly phase ownership"), HCCL_E_INTERNAL);
                phaseMask |= kernelMask;
            }
            CHK_PRT_RET(phaseMask != 0xFU || totalBytes != BYTES_512K,
                HCCL_ERROR("[ValidateResource] invalid butterfly phase mask or byte size"), HCCL_E_INTERNAL);
        } else if (IsCrossLaneAlgorithm(algorithm) || IsP16MeshLaneAlgorithm(algorithm)) {
            const uint32_t laneCount = resource.kernelMeta[0].roleFlags >> KERNEL_ROLE_LANE_COUNT_SHIFT;
            const bool validLaneCount
                = algorithm == Algorithm2x8::CROSS_LANE_512M
                      ? laneCount == CROSS_LANE_MAX_COUNT
                      : (IsP16MeshLaneAlgorithm(algorithm) ? laneCount >= 4U && laneCount <= MESH_LANE_MAX_COUNT
                                                           : laneCount == 2U);
            CHK_PRT_RET(!validLaneCount, HCCL_ERROR("[ValidateResource] invalid cross-lane count %u", laneCount),
                HCCL_E_INTERNAL);
            for (const auto &meta : resource.kernelMeta) {
                CHK_PRT_RET((meta.roleFlags >> KERNEL_ROLE_LANE_COUNT_SHIFT) != laneCount,
                    HCCL_ERROR("[ValidateResource] inconsistent cross-lane count"), HCCL_E_INTERNAL);
            }
        }
        return HCCL_SUCCESS;
    }

    HcclResult GetMemToken(uint64_t address, uint64_t size, uint64_t &token)
    {
        CcuResult ret = HcommCcuGetMemToken(address, size, &token);
        if (ret != CCU_SUCCESS) {
            HCCL_ERROR("[GetMemToken] failed to obtain CCU memory token, ret %d", ret);
            return ConvertCcuToHccl(ret);
        }
        return HCCL_SUCCESS;
    }

    HcclResult SyncThreads(ThreadHandle producerThread, ThreadHandle consumerThread, uint32_t notifyIndex)
    {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(producerThread, consumerThread, notifyIndex)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(consumerThread, notifyIndex, CUSTOM_TIMEOUT)));
        return HCCL_SUCCESS;
    }

    ThreadHandle KernelThread(const OpParam &param, const AlgResourceCtx &resource, size_t kernelIndex)
    {
        return kernelIndex == 0 ? param.cpuThread : resource.extraThreads[kernelIndex - 1];
    }

    HcclResult LaunchKernel(ThreadHandle thread, CcuKernelHandle kernel, const uint64_t *taskArgs,
        uint32_t taskArgCount, const char *algorithm, size_t kernelIndex, uint64_t phase)
    {
        const CcuResult launchRet = HcommCcuKernelLaunch(thread, kernel, taskArgs, taskArgCount);
        if (launchRet != CCU_SUCCESS) {
            HCCL_ERROR("[LaunchKernel] algorithm=%s kernel=%zu phase=%lu failed, ret %d", algorithm, kernelIndex, phase,
                launchRet);
            return ConvertCcuToHccl(launchRet);
        }
        return HCCL_SUCCESS;
    }

    HcclResult ExecButterfly2x8(const OpParam &param, const AlgResourceCtx &resource, uint64_t totalBytes,
        uint64_t inputAddr, uint64_t outputAddr, uint64_t inputToken, uint64_t outputToken)
    {
        const uint64_t scratchStride = (totalBytes + SCRATCH_ALIGNMENT - 1) & ~(SCRATCH_ALIGNMENT - 1);
        CHK_PRT_RET(
            scratchStride > std::numeric_limits<uint64_t>::max() / 3 || scratchStride * 3 > resource.localBuffer.size,
            HCCL_ERROR("[ExecButterfly2x8] scratch requires %lu bytes, only %lu available", scratchStride * 3,
                resource.localBuffer.size),
            HCCL_E_MEMORY);
        const std::array<uint64_t, 4> taskArgs = {inputAddr, outputAddr, inputToken, outputToken};
        if (resource.ccuKernels.size() == 1) {
            return LaunchKernel(param.cpuThread, resource.ccuKernels[0], taskArgs.data(),
                static_cast<uint32_t>(taskArgs.size()), "BFLY16_CLOS_4STEP_FUSED", 0, 0);
        }

        const ThreadHandle secondary = resource.extraThreads[0];
        CHK_RET(SyncThreads(param.cpuThread, secondary, 0));
        CHK_RET(LaunchKernel(param.cpuThread, resource.ccuKernels[0], taskArgs.data(),
            static_cast<uint32_t>(taskArgs.size()), "BFLY16_CLOS_4STEP_FUSED", 0, 0));
        CHK_RET(LaunchKernel(secondary, resource.ccuKernels[1], taskArgs.data(), static_cast<uint32_t>(taskArgs.size()),
            "BFLY16_CLOS_4STEP_FUSED", 1, 0));
        CHK_RET(SyncThreads(secondary, param.cpuThread, 1));
        return HCCL_SUCCESS;
    }

    HcclResult ExecV101Small(const OpParam &param, const AlgResourceCtx &resource, uint64_t inputAddr,
        uint64_t outputAddr, uint64_t inputToken, uint64_t outputToken)
    {
        std::array<uint64_t, 7> taskArgs{inputAddr, outputAddr, inputToken, outputToken, 0U, 0U, 0U};
        if (param.rankSize == 4U) {
            CHK_PRT_RET(resource.ccuKernels.size() != 1U,
                HCCL_ERROR("[ExecV101Small] R4 requires one kernel"), HCCL_E_INTERNAL);
            return LaunchKernel(param.cpuThread, resource.ccuKernels[0], taskArgs.data(), 4U,
                "V101_DIRECT_MESH_512K", 0U, 0U);
        }

        CHK_PRT_RET((param.rankSize != 12U && param.rankSize != 16U) || resource.ccuKernels.size() != 2U
                        || resource.extraThreads.size() != 1U,
            HCCL_ERROR("[ExecV101Small] R12/R16 requires two kernels and two threads"), HCCL_E_INTERNAL);
        taskArgs[4] = reinterpret_cast<uint64_t>(resource.localBuffer.addr);
        CHK_RET(GetMemToken(taskArgs[4], resource.localBuffer.size, taskArgs[5]));
        const ThreadHandle secondary = resource.extraThreads[0];
        CHK_RET(SyncThreads(param.cpuThread, secondary, 0U));
        const uint64_t finalStage = param.rankSize == 16U && param.inputPtr != param.outputPtr
                                        ? static_cast<uint64_t>(V101DirectStage::MERGE)
                                        : static_cast<uint64_t>(V101DirectStage::ALLGATHER);
        for (uint64_t stage = static_cast<uint64_t>(V101DirectStage::PARTIAL);
             stage <= finalStage; ++stage) {
            taskArgs[6] = stage;
            CHK_RET(LaunchKernel(param.cpuThread, resource.ccuKernels[0], taskArgs.data(),
                static_cast<uint32_t>(taskArgs.size()), "V101_DIRECT_MESH_512K", 0U, stage));
            CHK_RET(LaunchKernel(secondary, resource.ccuKernels[1], taskArgs.data(),
                static_cast<uint32_t>(taskArgs.size()), "V101_DIRECT_MESH_512K", 1U, stage));
            CHK_RET(SyncThreads(secondary, param.cpuThread, 0U));
            if (stage != finalStage) {
                CHK_RET(SyncThreads(param.cpuThread, secondary, 0U));
            }
        }
        return HCCL_SUCCESS;
    }

    bool IsFusedTopologyAlgorithm(Algorithm2x8 algorithm)
    {
        return algorithm == Algorithm2x8::P4_DIRECT_RSAG_512K || algorithm == Algorithm2x8::P4_DIRECT_SLOT3_512M
               || algorithm == Algorithm2x8::P4_DIRECT_SLOT3_400M4B || algorithm == Algorithm2x8::P4_ROTATE3_512M
               || algorithm == Algorithm2x8::P4_ROTATE3_400M4B
               || algorithm == Algorithm2x8::P4_ROTATE3_UNIQUE_NOTIFY_512M
               || algorithm == Algorithm2x8::P4_ROTATE3_UNIQUE_NOTIFY_400M4B
               || algorithm == Algorithm2x8::P4_ROTATE3_DIRECT_OUTPUT_512M
               || algorithm == Algorithm2x8::P4_ROTATE3_DIRECT_OUTPUT_400M4B || IsP4OutputAccumulateAlgorithm(algorithm)
               || IsP4PullR3Algorithm(algorithm) || IsP12DirectRsagAlgorithm(algorithm)
               || algorithm == Algorithm2x8::P12_HIER8_512M || algorithm == Algorithm2x8::P12_HIER8_400M4B
               || IsP12AllPairsAlgorithm(algorithm);
    }

    HcclResult ValidateFusedScratch(const OpParam &param, const AlgResourceCtx &resource, Algorithm2x8 algorithm)
    {
        uint64_t sliceCount = param.rankSize;
        uint64_t slotCount = param.rankSize - 1U;
        if (algorithm == Algorithm2x8::P4_ROTATE3_512M || algorithm == Algorithm2x8::P4_ROTATE3_400M4B) {
            slotCount = 1;
        } else if (algorithm == Algorithm2x8::P4_ROTATE3_UNIQUE_NOTIFY_512M
                   || algorithm == Algorithm2x8::P4_ROTATE3_UNIQUE_NOTIFY_400M4B) {
            slotCount = 1;
        } else if (algorithm == Algorithm2x8::P4_ROTATE3_DIRECT_OUTPUT_512M
                   || algorithm == Algorithm2x8::P4_ROTATE3_DIRECT_OUTPUT_400M4B) {
            slotCount = 1;
        } else if (IsP4OutputAccumulateAlgorithm(algorithm)) {
            slotCount = 1;
        } else if (IsP4PullR3Algorithm(algorithm)) {
            slotCount = 1;
        } else if (algorithm == Algorithm2x8::P12_HIER8_512M || algorithm == Algorithm2x8::P12_HIER8_400M4B) {
            sliceCount = HIER_GLOBAL_SHARD_COUNT;
            slotCount = HIER_TOTAL_SLOT_COUNT;
        } else if (IsP12ClosAccumulateAlgorithm(algorithm)) {
            slotCount = P12_ALLPAIRS_A_RANK_COUNT - 1U;
        } else if (IsP12OutputTreeAlgorithm(algorithm)) {
            slotCount = 1U;
        }

        uint64_t sliceElements = param.count;
        if (IsP12AllPairsOneSegmentAlgorithm(algorithm)) {
            sliceElements = param.count;
        } else if (IsP12AllPairs512MAlgorithm(algorithm)) {
            sliceElements = param.count / P12_ALLPAIRS_SEGMENT_COUNT;
        } else if (IsP12AllPairsAlgorithm(algorithm)) {
            const uint64_t firstSegmentElements = (200ULL * 1024ULL * 1024ULL) / FP32_BYTES;
            sliceElements = std::max(firstSegmentElements, param.count - firstSegmentElements);
        }
        const uint64_t base = sliceElements / sliceCount;
        const uint64_t remainder = sliceElements % sliceCount;
        const uint64_t maxSliceBytes = (base + (remainder == 0 ? 0U : 1U)) * FP32_BYTES;
        CHK_PRT_RET(maxSliceBytes >= MAX_DATA_SIZE,
            HCCL_ERROR("[ValidateFusedScratch] transfer %lu must be strictly less than 256 MiB", maxSliceBytes),
            HCCL_E_NOT_SUPPORT);
        const uint64_t scratchStride = (maxSliceBytes + SCRATCH_ALIGNMENT - 1) & ~(SCRATCH_ALIGNMENT - 1);
        CHK_PRT_RET(scratchStride > std::numeric_limits<uint64_t>::max() / slotCount,
            HCCL_ERROR("[ValidateFusedScratch] scratch size overflow"), HCCL_E_PARA);
        const uint64_t required = scratchStride * slotCount;
        CHK_PRT_RET(required > resource.localBuffer.size,
            HCCL_ERROR("[ValidateFusedScratch] algorithm=%s scratch requires %lu bytes (%lu slots), only %lu available",
                AlgorithmName(algorithm), required, slotCount, resource.localBuffer.size),
            HCCL_E_MEMORY);
        return HCCL_SUCCESS;
    }

    size_t PrimaryKernelIndex(const AlgResourceCtx &resource)
    {
        return resource.ccuKernels.size() == 1 || (resource.kernelMeta[0].roleFlags & KERNEL_ROLE_PRIMARY) != 0 ? 0U
                                                                                                                : 1U;
    }

    HcclResult ExecDirectRsag(
        const OpParam &param, const AlgResourceCtx &resource, Algorithm2x8 algorithm, std::array<uint64_t, 6> &taskArgs)
    {
        auto launchPhase = [&](size_t kernelIndex, DirectRsagPhase phase) -> HcclResult {
            taskArgs.back() = static_cast<uint64_t>(phase);
            return LaunchKernel(KernelThread(param, resource, kernelIndex), resource.ccuKernels[kernelIndex],
                taskArgs.data(), static_cast<uint32_t>(taskArgs.size()), AlgorithmName(algorithm), kernelIndex,
                static_cast<uint64_t>(phase));
        };
        if (resource.ccuKernels.size() == 1) {
            return launchPhase(0, DirectRsagPhase::SOURCE_PUSH);
        }

        const ThreadHandle threads[2] = {param.cpuThread, resource.extraThreads[0]};
        const size_t primaryIndex = PrimaryKernelIndex(resource);
        const size_t secondaryIndex = 1U - primaryIndex;
        CHK_RET(SyncThreads(threads[0], threads[1], 0));
        CHK_RET(launchPhase(0, DirectRsagPhase::SOURCE_PUSH));
        CHK_RET(launchPhase(1, DirectRsagPhase::SOURCE_PUSH));
        CHK_RET(SyncThreads(threads[secondaryIndex], threads[primaryIndex], 1));
        CHK_RET(launchPhase(primaryIndex, DirectRsagPhase::LOCAL_REDUCE));
        CHK_RET(SyncThreads(threads[primaryIndex], threads[secondaryIndex], 0));
        CHK_RET(launchPhase(0, DirectRsagPhase::BROADCAST));
        CHK_RET(launchPhase(1, DirectRsagPhase::BROADCAST));
        CHK_RET(SyncThreads(threads[1], threads[0], 1));
        return HCCL_SUCCESS;
    }

    HcclResult ExecP12AllPairs(const OpParam &param, const AlgResourceCtx &resource, Algorithm2x8 algorithm,
        uint64_t inputAddr, uint64_t outputAddr, uint64_t inputToken, uint64_t outputToken)
    {
        std::array<uint64_t, 7> taskArgs
            = {inputAddr, outputAddr, inputToken, outputToken, inputAddr == outputAddr ? 1U : 0U, 0U, 0U};
        auto launchPhase = [&](size_t kernelIndex, uint32_t segment, DirectRsagPhase phase) -> HcclResult {
            taskArgs[5] = static_cast<uint64_t>(phase);
            taskArgs[6] = segment;
            return LaunchKernel(KernelThread(param, resource, kernelIndex), resource.ccuKernels[kernelIndex],
                taskArgs.data(), static_cast<uint32_t>(taskArgs.size()), AlgorithmName(algorithm), kernelIndex,
                static_cast<uint64_t>(segment) * 10U + static_cast<uint64_t>(phase));
        };
        const uint32_t segmentCount
            = IsP12AllPairsOneSegmentAlgorithm(algorithm) ? 1U : P12_ALLPAIRS_SEGMENT_COUNT;

        if (resource.ccuKernels.size() == 1) {
            for (uint32_t segment = 0; segment < segmentCount; ++segment) {
                CHK_RET(launchPhase(0, segment, DirectRsagPhase::SOURCE_PUSH));
            }
            return HCCL_SUCCESS;
        }

        const ThreadHandle threads[2] = {param.cpuThread, resource.extraThreads[0]};
        const size_t primaryIndex = PrimaryKernelIndex(resource);
        const size_t secondaryIndex = 1U - primaryIndex;
        for (uint32_t segment = 0; segment < segmentCount; ++segment) {
            CHK_RET(SyncThreads(threads[0], threads[1], 0));
            CHK_RET(launchPhase(0, segment, DirectRsagPhase::SOURCE_PUSH));
            CHK_RET(launchPhase(1, segment, DirectRsagPhase::SOURCE_PUSH));
            CHK_RET(SyncThreads(threads[secondaryIndex], threads[primaryIndex], 1));
            CHK_RET(launchPhase(primaryIndex, segment,
                IsP12OutputTreeAlgorithm(algorithm) ? DirectRsagPhase::FINAL_MERGE
                                                    : DirectRsagPhase::LOCAL_REDUCE));
            CHK_RET(SyncThreads(threads[primaryIndex], threads[secondaryIndex], 0));
            CHK_RET(launchPhase(0, segment, DirectRsagPhase::BROADCAST));
            CHK_RET(launchPhase(1, segment, DirectRsagPhase::BROADCAST));
            CHK_RET(SyncThreads(threads[1], threads[0], 1));
        }
        return HCCL_SUCCESS;
    }

    HcclResult ExecP12Hier8(
        const OpParam &param, const AlgResourceCtx &resource, Algorithm2x8 algorithm, std::array<uint64_t, 6> &taskArgs)
    {
        auto launchPhase = [&](size_t kernelIndex, P12HierPhase phase) -> HcclResult {
            taskArgs.back() = static_cast<uint64_t>(phase);
            return LaunchKernel(KernelThread(param, resource, kernelIndex), resource.ccuKernels[kernelIndex],
                taskArgs.data(), static_cast<uint32_t>(taskArgs.size()), AlgorithmName(algorithm), kernelIndex,
                static_cast<uint64_t>(phase));
        };
        if (resource.ccuKernels.size() == 1) {
            for (uint64_t phase = 0; phase < HIER_LOCAL_TASK_PHASE_COUNT; ++phase) {
                CHK_RET(launchPhase(0, static_cast<P12HierPhase>(phase)));
            }
            CHK_RET(launchPhase(0, P12HierPhase::LOCAL_REDUCE));
            CHK_RET(launchPhase(0, P12HierPhase::CROSS_REDUCE));
            return launchPhase(0, P12HierPhase::LOCAL_ALLGATHER);
        }

        const ThreadHandle threads[2] = {param.cpuThread, resource.extraThreads[0]};
        const size_t primaryIndex = PrimaryKernelIndex(resource);
        const size_t secondaryIndex = 1U - primaryIndex;
        CHK_RET(SyncThreads(threads[0], threads[1], 0));
        for (uint64_t phase = 0; phase < HIER_LOCAL_TASK_PHASE_COUNT; ++phase) {
            if (phase != 0) {
                CHK_RET(SyncThreads(threads[secondaryIndex], threads[primaryIndex], 1));
                CHK_RET(SyncThreads(threads[primaryIndex], threads[secondaryIndex], 0));
            }
            const auto hierPhase = static_cast<P12HierPhase>(phase);
            CHK_RET(launchPhase(0, hierPhase));
            CHK_RET(launchPhase(1, hierPhase));
        }
        CHK_RET(SyncThreads(threads[secondaryIndex], threads[primaryIndex], 1));
        CHK_RET(launchPhase(primaryIndex, P12HierPhase::LOCAL_REDUCE));
        CHK_RET(SyncThreads(threads[primaryIndex], threads[secondaryIndex], 0));
        CHK_RET(launchPhase(0, P12HierPhase::CROSS_REDUCE));
        CHK_RET(launchPhase(1, P12HierPhase::CROSS_REDUCE));
        CHK_RET(SyncThreads(threads[secondaryIndex], threads[primaryIndex], 1));
        CHK_RET(SyncThreads(threads[primaryIndex], threads[secondaryIndex], 0));
        CHK_RET(launchPhase(0, P12HierPhase::LOCAL_ALLGATHER));
        CHK_RET(launchPhase(1, P12HierPhase::LOCAL_ALLGATHER));
        CHK_RET(SyncThreads(threads[1], threads[0], 1));
        return HCCL_SUCCESS;
    }

    HcclResult ExecFusedTopology(const OpParam &param, const AlgResourceCtx &resource, Algorithm2x8 algorithm,
        uint64_t inputAddr, uint64_t outputAddr, uint64_t inputToken, uint64_t outputToken)
    {
        CHK_RET(ValidateFusedScratch(param, resource, algorithm));
        std::array<uint64_t, 6> taskArgs
            = {inputAddr, outputAddr, inputToken, outputToken, inputAddr == outputAddr ? 1U : 0U, 0U};
        if (IsP12AllPairsAlgorithm(algorithm)) {
            return ExecP12AllPairs(param, resource, algorithm, inputAddr, outputAddr, inputToken, outputToken);
        }
        if (algorithm == Algorithm2x8::P4_DIRECT_RSAG_512K
            || algorithm == Algorithm2x8::P4_DIRECT_SLOT3_512M
            || algorithm == Algorithm2x8::P4_DIRECT_SLOT3_400M4B
            || IsP12DirectRsagAlgorithm(algorithm)) {
            return ExecDirectRsag(param, resource, algorithm, taskArgs);
        }
        if (algorithm == Algorithm2x8::P12_HIER8_512M || algorithm == Algorithm2x8::P12_HIER8_400M4B) {
            return ExecP12Hier8(param, resource, algorithm, taskArgs);
        }
        if (resource.ccuKernels.size() == 1) {
            return LaunchKernel(param.cpuThread, resource.ccuKernels[0], taskArgs.data(),
                static_cast<uint32_t>(taskArgs.size()), AlgorithmName(algorithm), 0, 0);
        }

        const ThreadHandle secondary = resource.extraThreads[0];
        CHK_RET(SyncThreads(param.cpuThread, secondary, 0));
        CHK_RET(LaunchKernel(param.cpuThread, resource.ccuKernels[0], taskArgs.data(),
            static_cast<uint32_t>(taskArgs.size()), AlgorithmName(algorithm), 0, 0));
        CHK_RET(LaunchKernel(secondary, resource.ccuKernels[1], taskArgs.data(), static_cast<uint32_t>(taskArgs.size()),
            AlgorithmName(algorithm), 1, 0));
        CHK_RET(SyncThreads(secondary, param.cpuThread, 1));
        return HCCL_SUCCESS;
    }

    HcclResult ExecCrossLane2x8(const OpParam &param, const AlgResourceCtx &resource, Algorithm2x8 algorithm,
        uint64_t inputAddr, uint64_t outputAddr, uint64_t inputToken, uint64_t outputToken, uint64_t scratchAddr,
        uint64_t scratchToken)
    {
        const uint64_t base = param.count / param.rankSize;
        const uint64_t remainder = param.count % param.rankSize;
        const uint64_t maxOwnerBytes = (base + (remainder == 0 ? 0U : 1U)) * FP32_BYTES;
        CHK_PRT_RET(maxOwnerBytes >= MAX_DATA_SIZE,
            HCCL_ERROR("[ExecCrossLane2x8] owner transfer %lu must be strictly less than 256 MiB", maxOwnerBytes),
            HCCL_E_NOT_SUPPORT);
        const uint64_t scratchStride = (maxOwnerBytes + SCRATCH_ALIGNMENT - 1) & ~(SCRATCH_ALIGNMENT - 1);
        const uint32_t laneCount = resource.kernelMeta[0].roleFlags >> KERNEL_ROLE_LANE_COUNT_SHIFT;
        const uint64_t slotCount
            = IsP16MeshLaneAlgorithm(algorithm) ? MESH_LANE_MAX_COUNT : CROSS_LANE_SAME_SLOT_COUNT + laneCount;
        CHK_PRT_RET(scratchStride > std::numeric_limits<uint64_t>::max() / slotCount
                        || scratchStride * slotCount > resource.localBuffer.size,
            HCCL_ERROR("[ExecCrossLane2x8] scratch requires %lu bytes (%lu slots), only %lu available",
                scratchStride * slotCount, slotCount, resource.localBuffer.size),
            HCCL_E_MEMORY);

        std::array<uint64_t, 7> taskArgs
            = {inputAddr, outputAddr, inputToken, outputToken, scratchAddr, scratchToken, 0};
        auto launchPhase = [&](size_t kernelIndex, CrossLanePhase phase) -> HcclResult {
            taskArgs.back() = static_cast<uint64_t>(phase);
            return LaunchKernel(KernelThread(param, resource, kernelIndex), resource.ccuKernels[kernelIndex],
                taskArgs.data(), static_cast<uint32_t>(taskArgs.size()), AlgorithmName(algorithm), kernelIndex,
                static_cast<uint64_t>(phase));
        };

        if (resource.ccuKernels.size() == 1) {
            if (IsP16MeshLaneAlgorithm(algorithm)) {
                for (uint64_t phase = 0; phase < CROSS_LANE_SHARD_COUNT; ++phase) {
                    CHK_RET(launchPhase(
                        0, static_cast<CrossLanePhase>(static_cast<uint64_t>(CrossLanePhase::MESH_LANE_0) + phase)));
                }
            } else {
                CHK_RET(launchPhase(0, CrossLanePhase::REDUCE_SCATTER));
            }
            CHK_RET(launchPhase(0, CrossLanePhase::LOCAL_REDUCE));
            CHK_RET(launchPhase(0, CrossLanePhase::BROADCAST));
            return HCCL_SUCCESS;
        }

        const ThreadHandle secondary = resource.extraThreads[0];
        if (IsP16MeshLaneAlgorithm(algorithm)) {
            for (uint64_t phase = 0; phase < CROSS_LANE_SHARD_COUNT; ++phase) {
                const auto meshPhase
                    = static_cast<CrossLanePhase>(static_cast<uint64_t>(CrossLanePhase::MESH_LANE_0) + phase);
                CHK_RET(SyncThreads(param.cpuThread, secondary, 0));
                CHK_RET(launchPhase(0, meshPhase));
                CHK_RET(launchPhase(1, meshPhase));
                CHK_RET(SyncThreads(secondary, param.cpuThread, 1));
            }
            CHK_RET(SyncThreads(param.cpuThread, secondary, 0));
            CHK_RET(launchPhase(0, CrossLanePhase::LOCAL_REDUCE));
            CHK_RET(launchPhase(1, CrossLanePhase::LOCAL_REDUCE));
            CHK_RET(SyncThreads(secondary, param.cpuThread, 1));
            CHK_RET(SyncThreads(param.cpuThread, secondary, 0));
            CHK_RET(launchPhase(0, CrossLanePhase::BROADCAST));
            CHK_RET(launchPhase(1, CrossLanePhase::BROADCAST));
            CHK_RET(SyncThreads(secondary, param.cpuThread, 1));
            return HCCL_SUCCESS;
        }

        CHK_RET(SyncThreads(param.cpuThread, secondary, 0));
        CHK_RET(launchPhase(0, CrossLanePhase::REDUCE_SCATTER));
        CHK_RET(launchPhase(1, CrossLanePhase::REDUCE_SCATTER));
        CHK_RET(SyncThreads(secondary, param.cpuThread, 1));

        CHK_RET(SyncThreads(param.cpuThread, secondary, 0));
        CHK_RET(launchPhase(0, CrossLanePhase::LOCAL_REDUCE));
        CHK_RET(launchPhase(1, CrossLanePhase::LOCAL_REDUCE));
        CHK_RET(SyncThreads(secondary, param.cpuThread, 1));

        CHK_RET(SyncThreads(param.cpuThread, secondary, 0));
        CHK_RET(launchPhase(0, CrossLanePhase::BROADCAST));
        CHK_RET(launchPhase(1, CrossLanePhase::BROADCAST));
        CHK_RET(SyncThreads(secondary, param.cpuThread, 1));
        return HCCL_SUCCESS;
    }
} // namespace

HcclResult ExecOp(const OpParam &param)
{
    CHK_PRT_RET(param.resCtx == nullptr || param.ctxSize == 0, HCCL_ERROR("[ExecOp] EngineCtx is empty"), HCCL_E_PTR);
    CHK_PRT_RET(param.ctxSize > AlgResourceCtx::MAX_SERIALIZED_SIZE,
        HCCL_ERROR("[ExecOp] EngineCtx is too large: %lu", param.ctxSize), HCCL_E_INTERNAL);

    char *ctx = static_cast<char *>(param.resCtx);
    std::vector<char> serialized(ctx, ctx + param.ctxSize);
    AlgResourceCtx resource;
    CHK_PRT_RET(
        !resource.DeSerialize(serialized), HCCL_ERROR("[ExecOp] invalid serialized EngineCtx"), HCCL_E_INTERNAL);

    constexpr uint64_t dataTypeSize = sizeof(float);
    CHK_PRT_RET(param.count > std::numeric_limits<uint64_t>::max() / dataTypeSize,
        HCCL_ERROR("[ExecOp] element count overflows byte size"), HCCL_E_PARA);
    const uint64_t totalBytes = param.count * dataTypeSize;
    const Algorithm2x8 algorithm = ExpectedAlgorithm(param, totalBytes);
    CHK_RET(ValidateResource(param, totalBytes, algorithm, resource));
    HCCL_INFO("[ExecDispatchHit] topology=%u ranks=%u bytes=%lu algorithm=%s version=%u kernels=%zu",
        static_cast<uint32_t>(resource.topology), param.rankSize, totalBytes, AlgorithmName(algorithm),
        resource.version, resource.ccuKernels.size());

    const uint64_t inputAddr = reinterpret_cast<uint64_t>(param.inputPtr);
    const uint64_t outputAddr = reinterpret_cast<uint64_t>(param.outputPtr);
    uint64_t inputToken = 0;
    uint64_t outputToken = 0;
    CHK_RET(GetMemToken(inputAddr, totalBytes, inputToken));
    if (outputAddr == inputAddr) {
        outputToken = inputToken;
    } else {
        CHK_RET(GetMemToken(outputAddr, totalBytes, outputToken));
    }

    if (IsV101SmallAlgorithm(algorithm)) {
        return ExecV101Small(param, resource, inputAddr, outputAddr, inputToken, outputToken);
    }
    if (algorithm == Algorithm2x8::BFLY16_CLOS_4STEP) {
        return ExecButterfly2x8(param, resource, totalBytes, inputAddr, outputAddr, inputToken, outputToken);
    }
    if (IsFusedTopologyAlgorithm(algorithm)) {
        return ExecFusedTopology(param, resource, algorithm, inputAddr, outputAddr, inputToken, outputToken);
    }

    const uint64_t baseOwnerElements = param.count / param.rankSize;
    const uint64_t remainder = param.count % param.rankSize;
    const uint64_t ownerElements = baseOwnerElements + (param.myRank < remainder ? 1U : 0U);
    const uint64_t ownerOffsetElements
        = static_cast<uint64_t>(param.myRank) * baseOwnerElements + std::min<uint64_t>(param.myRank, remainder);
    const uint64_t ownerOffsetBytes = ownerOffsetElements * dataTypeSize;
    const uint64_t ownerBytes = ownerElements * dataTypeSize;

    CHK_PRT_RET(ownerBytes >= MAX_DATA_SIZE,
        HCCL_ERROR("[ExecOp] owner transfer %lu must be strictly less than 256 MiB", ownerBytes), HCCL_E_NOT_SUPPORT);
    const uint64_t scratchStride = (ownerBytes + SCRATCH_ALIGNMENT - 1) & ~(SCRATCH_ALIGNMENT - 1);
    CHK_PRT_RET(scratchStride > std::numeric_limits<uint64_t>::max() / resource.ccuKernels.size(),
        HCCL_ERROR("[ExecOp] scratch size overflow"), HCCL_E_PARA);
    const uint64_t scratchRequired = scratchStride * resource.ccuKernels.size();
    CHK_PRT_RET(scratchRequired > resource.localBuffer.size,
        HCCL_ERROR(
            "[ExecOp] scratch requires %lu bytes, only %lu available", scratchRequired, resource.localBuffer.size),
        HCCL_E_MEMORY);

    const uint64_t scratchAddr = reinterpret_cast<uint64_t>(resource.localBuffer.addr);
    uint64_t scratchToken = 0;
    CHK_RET(GetMemToken(scratchAddr, resource.localBuffer.size, scratchToken));
    if (IsCrossLaneAlgorithm(algorithm) || IsP16MeshLaneAlgorithm(algorithm)) {
        return ExecCrossLane2x8(
            param, resource, algorithm, inputAddr, outputAddr, inputToken, outputToken, scratchAddr, scratchToken);
    }

    std::array<uint64_t, 10> taskArgs = {
        inputAddr,
        outputAddr,
        inputToken,
        outputToken,
        scratchAddr,
        scratchToken,
        ownerOffsetBytes,
        ownerBytes,
        scratchStride,
        0,
    };

    auto launchPhase = [&](size_t i, OwnerRsagPhase phase) -> HcclResult {
        taskArgs.back() = static_cast<uint64_t>(phase);
        const ThreadHandle thread = i == 0 ? param.cpuThread : resource.extraThreads[i - 1];
        const CcuResult launchRet = HcommCcuKernelLaunch(
            thread, resource.ccuKernels[i], taskArgs.data(), static_cast<uint32_t>(taskArgs.size()));
        if (launchRet != CCU_SUCCESS) {
            HCCL_ERROR(
                "[ExecOp] kernel %zu phase %lu launch failed, ret %d", i, static_cast<uint64_t>(phase), launchRet);
            return ConvertCcuToHccl(launchRet);
        }
        return HCCL_SUCCESS;
    };

    if (resource.ccuKernels.size() == 1) {
        CHK_RET(launchPhase(0, OwnerRsagPhase::PARTIAL_REDUCE));
        CHK_RET(launchPhase(0, OwnerRsagPhase::COMBINE));
        CHK_RET(launchPhase(0, OwnerRsagPhase::BROADCAST));
        return HCCL_SUCCESS;
    }

    const ThreadHandle threads[2] = {param.cpuThread, resource.extraThreads[0]};
    const size_t primaryIndex = (resource.kernelMeta[0].roleFlags & KERNEL_ROLE_PRIMARY) != 0 ? 0 : 1;
    const size_t secondaryIndex = 1 - primaryIndex;

    // 额外 thread 先接入用户 stream 的任务图，两个 die 再并发生成 partial。
    CHK_RET(SyncThreads(threads[0], threads[1], 0));
    CHK_RET(launchPhase(0, OwnerRsagPhase::PARTIAL_REDUCE));
    CHK_RET(launchPhase(1, OwnerRsagPhase::PARTIAL_REDUCE));

    // primary 在两个 partial 都完成后本地合并，不依赖 HVM 不能正确建模的跨 die LocalNotify。
    CHK_RET(SyncThreads(threads[secondaryIndex], threads[primaryIndex], 1));
    CHK_RET(launchPhase(primaryIndex, OwnerRsagPhase::COMBINE));

    // 将合并完成作为两个 broadcast graph 的共同起点，再在主 stream 收口。
    CHK_RET(SyncThreads(threads[primaryIndex], threads[secondaryIndex], 0));
    CHK_RET(launchPhase(0, OwnerRsagPhase::BROADCAST));
    CHK_RET(launchPhase(1, OwnerRsagPhase::BROADCAST));
    CHK_RET(SyncThreads(threads[1], threads[0], 1));
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
