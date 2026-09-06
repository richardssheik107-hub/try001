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
#include <cstdio>
#include <limits>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include <ccu/ccu_launch.h>
#include <ccu/ccu_res.h>
#include <hccl/hccl_ccu_res.h>
#include <hccl/hccl_diag.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_res_expt.h>

#include "ccu_kernel.h"
#include "common.h"
#include "custom.h"
#include "exec_op.h"
#include "hccl.h"
#include "log.h"

namespace {
constexpr uint32_t OWNER_CHANNEL_NOTIFY_NUM = 3;
constexpr uint32_t BFLY_CHANNEL_NOTIFY_NUM = 2;
constexpr uint32_t CROSS_LANE_CHANNEL_NOTIFY_NUM = 5;
constexpr uint32_t DIRECT_CHANNEL_NOTIFY_NUM = 3;
constexpr uint32_t P4_ROTATE_CHANNEL_NOTIFY_NUM = 4;
constexpr uint32_t P4_ROTATE_UNIQUE_NOTIFY_NUM = 5;
constexpr uint32_t P12_HIER_CHANNEL_NOTIFY_NUM = 6;
constexpr uint32_t P12_CLOS_CHANNEL_NOTIFY_NUM = 4;
constexpr uint32_t V101_CHANNEL_NOTIFY_NUM = 6;
constexpr uint32_t THREAD_NOTIFY_NUM = 2;
constexpr uint32_t V101_THREAD_NOTIFY_NUM = 1;
constexpr uint32_t MAX_DIE_NUM = 2;
constexpr uint64_t FP32_BYTES = sizeof(float);
constexpr uint64_t BYTES_512K = 524288ULL;
constexpr uint64_t BYTES_512M = 536870912ULL;
constexpr uint64_t BYTES_400M4B = 419430404ULL;
constexpr uint64_t SCRATCH_ALIGNMENT = 512ULL;
constexpr uint32_t MAX_LANE_SIZE = 4;

struct ChannelWithDie {
    HcclChannelDesc desc{};
    uint32_t peerRank{0};
    uint32_t actualDieId{0};
    uint32_t remoteDieId{0};
    uint32_t isLocal{0};
    uint32_t selectedLayer{0};
    EndpointAttrBwCoeff bwCoeff{0};
};

struct LanePlacement {
    uint32_t dieId{0};
    uint32_t lane{0};
    uint32_t position{0};
    uint32_t size{0};
};

struct ChannelPlan {
    ChannelHandle handle{};
    ChannelWithDie info{};
    uint32_t targetSlot{0};
    LanePlacement targetPlacement{};
    LanePlacement sourcePlacement{};
};

struct DieGroup {
    uint32_t actualDieId{0};
    uint32_t isPrimary{0};
    std::vector<ChannelPlan> peers;
};

struct TopologyView {
    FinalTopology topology{FinalTopology::TOPOLOGY_4X1};
    std::vector<uint32_t> layers;
    uint32_t localLayer{0};
    uint32_t fullClosLayer{0};
    std::vector<uint32_t> localRanks;
    std::vector<uint32_t> remoteRanks;
    uint32_t localIndex{0};
};

const char *TopologyName(FinalTopology topology)
{
    switch (topology) {
        case FinalTopology::TOPOLOGY_4X1:
            return "4x1";
        case FinalTopology::TOPOLOGY_8X4:
            return "8+4";
        case FinalTopology::TOPOLOGY_2X8:
            return "2x8";
        default:
            return "unknown";
    }
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

Algorithm2x8 SelectCrossLaneInplaceFallback(Algorithm2x8 algorithm, const void *input, const void *output)
{
    if (algorithm == Algorithm2x8::P4_OUTPUT_ACCUMULATE_WAVE2_512M && input == output) {
        return Algorithm2x8::P4_OUTPUT_ACCUMULATE_CHAIN_512M;
    }
    if (algorithm == Algorithm2x8::P4_OUTPUT_ACCUMULATE_WAVE2_400M4B && input == output) {
        return Algorithm2x8::P4_OUTPUT_ACCUMULATE_CHAIN_400M4B;
    }
    if (algorithm == Algorithm2x8::CROSS_LANE_DIRECT_OUTPUT_400M4B && input == output) {
        return Algorithm2x8::CROSS_LANE_400M4B;
    }
    if (algorithm == Algorithm2x8::P12_ALLPAIRS_1SEG_OUTPUT_TREE_512M && input == output) {
        return Algorithm2x8::P12_ALLPAIRS_EARLY_INIT_512M;
    }
    if (algorithm == Algorithm2x8::P12_ALLPAIRS_1SEG_OUTPUT_TREE_400M4B && input == output) {
        return Algorithm2x8::P12_ALLPAIRS_1SEG_400M4B;
    }
    return algorithm;
}

bool IsP12ClosAccumulateAlgorithm(Algorithm2x8 algorithm)
{
    return algorithm == Algorithm2x8::P12_CLOS_ACCUMULATE_512M || algorithm == Algorithm2x8::P12_CLOS_ACCUMULATE_400M4B;
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

Algorithm2x8 SelectAlgorithm(const TopologyView &view, uint64_t totalBytes)
{
    if (view.topology == FinalTopology::TOPOLOGY_4X1 && view.localRanks.size() == 1 && view.remoteRanks.size() == 3) {
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
    if (view.topology == FinalTopology::TOPOLOGY_8X4 && (view.localRanks.size() == 8 || view.localRanks.size() == 4)
        && view.localRanks.size() + view.remoteRanks.size() == 12) {
        if (totalBytes == BYTES_512K) {
            return Algorithm2x8::V101_DIRECT_MESH_512K;
        }
        if (totalBytes == BYTES_512M) {
            return Algorithm2x8::P12_ALLPAIRS_1SEG_OUTPUT_TREE_512M;
        }
        if (totalBytes == BYTES_400M4B) {
            return Algorithm2x8::P12_ALLPAIRS_1SEG_OUTPUT_TREE_400M4B;
        }
        return Algorithm2x8::OWNER_RSAG;
    }
    if (view.topology == FinalTopology::TOPOLOGY_2X8 && view.localRanks.size() == 8 && view.remoteRanks.size() == 8) {
        if (totalBytes == BYTES_512K) {
            return Algorithm2x8::V109_DIRECT_MESH_R16_512K;
        }
        if (totalBytes == BYTES_512M) {
            return Algorithm2x8::CROSS_LANE_512M;
        }
        if (totalBytes == BYTES_400M4B) {
            return Algorithm2x8::CROSS_LANE_DIRECT_OUTPUT_400M4B;
        }
    }
    return Algorithm2x8::OWNER_RSAG;
}

Algorithm2x8 SelectAlgorithm(uint32_t rankSize, uint64_t totalBytes)
{
    if (rankSize == 4) {
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
    if (rankSize == 12) {
        if (totalBytes == BYTES_512K) {
            return Algorithm2x8::V101_DIRECT_MESH_512K;
        }
        if (totalBytes == BYTES_512M) {
            return Algorithm2x8::P12_ALLPAIRS_1SEG_OUTPUT_TREE_512M;
        }
        if (totalBytes == BYTES_400M4B) {
            return Algorithm2x8::P12_ALLPAIRS_1SEG_OUTPUT_TREE_400M4B;
        }
        return Algorithm2x8::OWNER_RSAG;
    }
    if (rankSize == 16) {
        if (totalBytes == BYTES_512K) {
            return Algorithm2x8::V109_DIRECT_MESH_R16_512K;
        }
        if (totalBytes == BYTES_512M) {
            return Algorithm2x8::CROSS_LANE_512M;
        }
        if (totalBytes == BYTES_400M4B) {
            return Algorithm2x8::CROSS_LANE_DIRECT_OUTPUT_400M4B;
        }
    }
    return Algorithm2x8::OWNER_RSAG;
}

uint32_t ResourceVersion(Algorithm2x8 algorithm)
{
    if (IsV109R16SmallAlgorithm(algorithm)) {
        return AlgResourceCtx::V109_DIRECT_MESH_R16_512K_VERSION;
    }
    if (IsV101SmallAlgorithm(algorithm)) {
        return AlgResourceCtx::V101_DIRECT_MESH_512K_VERSION;
    }
    if (algorithm == Algorithm2x8::P12_ALLPAIRS_1SEG_OUTPUT_TREE_512M) {
        return AlgResourceCtx::P12_ALLPAIRS_1SEG_OUTPUT_TREE_512M_VERSION;
    }
    if (algorithm == Algorithm2x8::P12_ALLPAIRS_1SEG_OUTPUT_TREE_400M4B) {
        return AlgResourceCtx::P12_ALLPAIRS_1SEG_OUTPUT_TREE_400M4B_VERSION;
    }
    if (IsP4OutputAccumulateWave2Algorithm(algorithm)) {
        return algorithm == Algorithm2x8::P4_OUTPUT_ACCUMULATE_WAVE2_400M4B
                   ? AlgResourceCtx::P4_OUTPUT_ACCUMULATE_WAVE2_400M4B_VERSION
                   : AlgResourceCtx::P4_OUTPUT_ACCUMULATE_WAVE2_VERSION;
    }
    if (IsP4OutputAccumulateChainAlgorithm(algorithm)) {
        return AlgResourceCtx::P4_OUTPUT_ACCUMULATE_CHAIN_VERSION;
    }
    if (IsP4OutputAccumulateAlgorithm(algorithm)) {
        return AlgResourceCtx::P4_OUTPUT_ACCUMULATE_VERSION;
    }
    if (IsP16MeshLaneAlgorithm(algorithm)) {
        return AlgResourceCtx::P16_MESH_LANE_VERSION;
    }
    if (IsP12ClosAccumulateAlgorithm(algorithm)) {
        return AlgResourceCtx::P12_CLOS_ACCUMULATE_VERSION;
    }
    if (IsP4PullR3Algorithm(algorithm)) {
        return AlgResourceCtx::P4_PULL_R3_VERSION;
    }
    if (algorithm == Algorithm2x8::P12_DIRECT_RSAG_EARLY_INIT_512K) {
        return AlgResourceCtx::P12_DIRECT_RSAG_EARLY_INIT_512K_VERSION;
    }
    if (algorithm == Algorithm2x8::P12_ALLPAIRS_EARLY_INIT_512M) {
        return AlgResourceCtx::P12_ALLPAIRS_EARLY_INIT_512M_VERSION;
    }
    if (algorithm == Algorithm2x8::P12_ALLPAIRS_1SEG_400M4B) {
        return AlgResourceCtx::P12_ALLPAIRS_1SEG_400M4B_VERSION;
    }
    if (IsP12PreSyncOnceAlgorithm(algorithm)) {
        return AlgResourceCtx::P12_PRESYNC_ONCE_VERSION;
    }
    if (algorithm == Algorithm2x8::BFLY16_CLOS_4STEP) {
        return AlgResourceCtx::BFLY_FUSED_VERSION;
    }
    if (algorithm == Algorithm2x8::CROSS_LANE_512M) {
        return AlgResourceCtx::CROSS_LANE_V82_512M_VERSION;
    }
    if (IsCrossLaneAlgorithm(algorithm)) {
        return IsCrossLaneDirectOutputAlgorithm(algorithm) ? AlgResourceCtx::CROSS_LANE_DIRECT_OUTPUT_VERSION
                                                           : AlgResourceCtx::CROSS_LANE_VERSION;
    }
    if (algorithm == Algorithm2x8::P4_DIRECT_RSAG_512K || algorithm == Algorithm2x8::P12_DIRECT_RSAG_512K) {
        return AlgResourceCtx::DIRECT_RSAG_VERSION;
    }
    if (algorithm == Algorithm2x8::P4_ROTATE3_512M || algorithm == Algorithm2x8::P4_ROTATE3_400M4B) {
        return AlgResourceCtx::P4_ROTATE3_VERSION;
    }
    if (algorithm == Algorithm2x8::P4_DIRECT_SLOT3_512M
        || algorithm == Algorithm2x8::P4_DIRECT_SLOT3_400M4B) {
        return AlgResourceCtx::P4_DIRECT_SLOT3_VERSION;
    }
    if (algorithm == Algorithm2x8::P4_ROTATE3_UNIQUE_NOTIFY_512M
        || algorithm == Algorithm2x8::P4_ROTATE3_UNIQUE_NOTIFY_400M4B) {
        return AlgResourceCtx::P4_ROTATE3_UNIQUE_NOTIFY_VERSION;
    }
    if (algorithm == Algorithm2x8::P4_ROTATE3_DIRECT_OUTPUT_512M
        || algorithm == Algorithm2x8::P4_ROTATE3_DIRECT_OUTPUT_400M4B) {
        return AlgResourceCtx::P4_ROTATE3_DIRECT_OUTPUT_VERSION;
    }
    if (algorithm == Algorithm2x8::P12_HIER8_512M || algorithm == Algorithm2x8::P12_HIER8_400M4B) {
        return AlgResourceCtx::P12_HIER8_VERSION;
    }
    if (IsP12AllPairsBalancedAlgorithm(algorithm)) {
        return AlgResourceCtx::P12_ALLPAIRS_BALANCED_VERSION;
    }
    if (IsP12AllPairsAlgorithm(algorithm)) {
        return AlgResourceCtx::P12_ALLPAIRS_VERSION;
    }
    return AlgResourceCtx::OWNER_VERSION;
}

HcclResult GetExpectedTopology(uint32_t rankSize, FinalTopology &topology, std::vector<uint32_t> &expectedInstanceSizes)
{
    switch (rankSize) {
        case 4:
            topology = FinalTopology::TOPOLOGY_4X1;
            expectedInstanceSizes = {1, 1, 1, 1};
            return HCCL_SUCCESS;
        case 12:
            topology = FinalTopology::TOPOLOGY_8X4;
            expectedInstanceSizes = {4, 8};
            return HCCL_SUCCESS;
        case 16:
            topology = FinalTopology::TOPOLOGY_2X8;
            expectedInstanceSizes = {8, 8};
            return HCCL_SUCCESS;
        default:
            HCCL_ERROR("[GetExpectedTopology] unsupported final rank size %u", rankSize);
            return HCCL_E_NOT_SUPPORT;
    }
}

HcclResult GetRankGraphLayers(HcclComm comm, std::vector<uint32_t> &layers)
{
    uint32_t *layerData = nullptr;
    uint32_t layerCount = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layerData, &layerCount));
    CHK_PRT_RET(layerData == nullptr || layerCount == 0,
        HCCL_ERROR("[GetRankGraphLayers] rank graph contains no network layer"), HCCL_E_INTERNAL);
    layers.assign(layerData, layerData + layerCount);
    return HCCL_SUCCESS;
}

HcclResult GetInstanceSizes(HcclComm comm, uint32_t layer, std::vector<uint32_t> &instanceSizes)
{
    uint32_t *sizeData = nullptr;
    uint32_t sizeCount = 0;
    CHK_RET(HcclRankGraphGetInstSizeListByLayer(comm, layer, &sizeData, &sizeCount));
    CHK_PRT_RET(sizeData == nullptr || sizeCount == 0,
        HCCL_ERROR("[GetInstanceSizes] layer %u contains no topology instance", layer), HCCL_E_INTERNAL);
    instanceSizes.assign(sizeData, sizeData + sizeCount);
    std::sort(instanceSizes.begin(), instanceSizes.end());
    return HCCL_SUCCESS;
}

HcclResult GetRanksInLayer(HcclComm comm, uint32_t layer, uint32_t rankSize, std::vector<uint32_t> &ranks)
{
    uint32_t *rankData = nullptr;
    uint32_t rankCount = 0;
    CHK_RET(HcclRankGraphGetRanksByLayer(comm, layer, &rankData, &rankCount));
    CHK_PRT_RET(rankData == nullptr || rankCount == 0, HCCL_ERROR("[GetRanksInLayer] layer %u contains no rank", layer),
        HCCL_E_INTERNAL);
    ranks.assign(rankData, rankData + rankCount);
    std::sort(ranks.begin(), ranks.end());
    CHK_PRT_RET(std::adjacent_find(ranks.begin(), ranks.end()) != ranks.end(),
        HCCL_ERROR("[GetRanksInLayer] layer %u contains duplicate ranks", layer), HCCL_E_INTERNAL);
    CHK_PRT_RET(ranks.back() >= rankSize,
        HCCL_ERROR("[GetRanksInLayer] layer %u contains out-of-range rank %u", layer, ranks.back()), HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult DetectFinalTopology(HcclComm comm, const OpParam &param, TopologyView &view)
{
    std::vector<uint32_t> expectedInstanceSizes;
    CHK_RET(GetExpectedTopology(param.rankSize, view.topology, expectedInstanceSizes));
    CHK_RET(GetRankGraphLayers(comm, view.layers));

    bool foundLocalLayer = false;
    bool foundFullClosLayer = false;
    for (uint32_t layer : view.layers) {
        std::vector<uint32_t> instanceSizes;
        CHK_RET(GetInstanceSizes(comm, layer, instanceSizes));
        uint64_t instanceRankSum = 0;
        for (uint32_t instanceSize : instanceSizes) {
            CHK_PRT_RET(instanceSize == 0 || instanceSize > param.rankSize,
                HCCL_ERROR("[DetectFinalTopology] layer %u has invalid instance size %u", layer, instanceSize),
                HCCL_E_INTERNAL);
            instanceRankSum += instanceSize;
        }
        CHK_PRT_RET(instanceRankSum != param.rankSize,
            HCCL_ERROR("[DetectFinalTopology] layer %u instances cover %lu ranks, expected %u", layer, instanceRankSum,
                param.rankSize),
            HCCL_E_NOT_SUPPORT);

        std::vector<uint32_t> ranksInLayer;
        CHK_RET(GetRanksInLayer(comm, layer, param.rankSize, ranksInLayer));
        CHK_PRT_RET(!std::binary_search(ranksInLayer.begin(), ranksInLayer.end(), param.myRank),
            HCCL_ERROR("[DetectFinalTopology] layer %u does not contain rank %u", layer, param.myRank),
            HCCL_E_INTERNAL);

        CommTopo topoType = COMM_TOPO_RESERVED;
        CHK_RET(HcclRankGraphGetTopoTypeByLayer(comm, layer, &topoType));
        if (instanceSizes == expectedInstanceSizes) {
            CHK_PRT_RET(topoType != COMM_TOPO_CUSTOM && topoType != COMM_TOPO_1DMESH,
                HCCL_ERROR("[DetectFinalTopology] local layer %u has unexpected topology type %d", layer, topoType),
                HCCL_E_NOT_SUPPORT);
            CHK_PRT_RET(foundLocalLayer,
                HCCL_ERROR("[DetectFinalTopology] multiple layers match the final local topology"), HCCL_E_INTERNAL);
            view.localLayer = layer;
            view.localRanks = ranksInLayer;
            foundLocalLayer = true;
        }
        if (instanceSizes.size() == 1 && instanceSizes[0] == param.rankSize && topoType == COMM_TOPO_CLOS) {
            CHK_PRT_RET(ranksInLayer.size() != param.rankSize,
                HCCL_ERROR("[DetectFinalTopology] CLOS layer %u does not cover all ranks", layer), HCCL_E_NOT_SUPPORT);
            for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
                CHK_PRT_RET(ranksInLayer[rank] != rank,
                    HCCL_ERROR("[DetectFinalTopology] CLOS layer %u rank set is incomplete", layer),
                    HCCL_E_NOT_SUPPORT);
            }
            if (!foundFullClosLayer) {
                view.fullClosLayer = layer;
                foundFullClosLayer = true;
            }
        }
    }

    CHK_PRT_RET(!foundFullClosLayer,
        HCCL_ERROR("[DetectFinalTopology] rank graph has no full-rank CLOS layer for %u ranks", param.rankSize),
        HCCL_E_NOT_SUPPORT);

    // 4x1 的本地实例可能被简化掉；其他拓扑必须显式提供 8/4 或 8/8 实例。
    if (!foundLocalLayer) {
        CHK_PRT_RET(view.topology != FinalTopology::TOPOLOGY_4X1 || view.layers.size() != 1,
            HCCL_ERROR("[DetectFinalTopology] local instance shape does not match rank size %u", param.rankSize),
            HCCL_E_NOT_SUPPORT);
        view.localRanks = {param.myRank};
    } else {
        CHK_PRT_RET(std::find(expectedInstanceSizes.begin(), expectedInstanceSizes.end(), view.localRanks.size())
                        == expectedInstanceSizes.end(),
            HCCL_ERROR("[DetectFinalTopology] local instance size %zu is invalid for rank size %u",
                view.localRanks.size(), param.rankSize),
            HCCL_E_NOT_SUPPORT);
    }

    const auto localIt = std::lower_bound(view.localRanks.begin(), view.localRanks.end(), param.myRank);
    CHK_PRT_RET(localIt == view.localRanks.end() || *localIt != param.myRank,
        HCCL_ERROR("[DetectFinalTopology] rank %u is absent from its local instance", param.myRank), HCCL_E_INTERNAL);
    view.localIndex = static_cast<uint32_t>(std::distance(view.localRanks.begin(), localIt));
    view.remoteRanks.clear();
    for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
        if (!std::binary_search(view.localRanks.begin(), view.localRanks.end(), rank)) {
            view.remoteRanks.push_back(rank);
        }
    }

    HCCL_INFO("[TopologyHit] topology=%s rank=%u/%u localLayer=%u closLayer=%u localSize=%zu remoteSize=%zu "
              "localIndex=%u",
        TopologyName(view.topology), param.myRank, param.rankSize, view.localLayer, view.fullClosLayer,
        view.localRanks.size(), view.remoteRanks.size(), view.localIndex);
    return HCCL_SUCCESS;
}

HcclResult SelectLinkForPair(HcclComm comm, uint32_t sourceRank, uint32_t targetRank,
    const std::vector<uint32_t> &layers, bool requireCtp, CommLink &selectedLink, uint32_t &selectedLayer)
{
    constexpr std::array<CommProtocol, 2> protocolPriority = {
        CommProtocol::COMM_PROTOCOL_UBC_CTP,
        CommProtocol::COMM_PROTOCOL_UBC_TP,
    };
    const size_t protocolCount = requireCtp ? 1U : protocolPriority.size();
    for (size_t protocolIndex = 0; protocolIndex < protocolCount; ++protocolIndex) {
        for (uint32_t layer : layers) {
            CommLink *links = nullptr;
            uint32_t linkCount = 0;
            CHK_RET(HcclRankGraphGetLinks(comm, layer, sourceRank, targetRank, &links, &linkCount));
            for (uint32_t i = 0; i < linkCount; ++i) {
                if (links[i].linkAttr.linkProtocol == protocolPriority[protocolIndex]) {
                    selectedLink = links[i];
                    selectedLayer = layer;
                    return HCCL_SUCCESS;
                }
            }
        }
    }
    HCCL_ERROR("[SelectLinkForPair] no eligible CCU link from rank %u to rank %u (requireCtp=%u)", sourceRank,
        targetRank, requireCtp ? 1U : 0U);
    return HCCL_E_NOT_FOUND;
}

HcclResult GetEndpointDie(HcclComm comm, uint32_t rank, const EndpointDesc &endpoint, uint32_t &dieId, const char *role)
{
    EndpointAttrDieId endpointDie{};
    CHK_RET(
        HcclRankGraphGetEndpointInfo(comm, rank, &endpoint, ENDPOINT_ATTR_DIE_ID, sizeof(endpointDie), &endpointDie));
    CHK_PRT_RET(endpointDie >= MAX_DIE_NUM,
        HCCL_ERROR("[GetEndpointDie] invalid %s die %u for rank %u", role, endpointDie, rank), HCCL_E_INTERNAL);
    dieId = endpointDie;
    return HCCL_SUCCESS;
}

HcclResult GetEndpointBwCoeff(
    HcclComm comm, uint32_t rank, const EndpointDesc &endpoint, EndpointAttrBwCoeff &bwCoeff)
{
    return HcclRankGraphGetEndpointInfo(
        comm, rank, &endpoint, ENDPOINT_ATTR_BW_COEFF, sizeof(bwCoeff), &bwCoeff);
}

HcclResult SelectChannelForPeer(HcclComm comm, uint32_t myRank, uint32_t peerRank, const std::vector<uint32_t> &layers,
    bool requireCtp, uint32_t notifyNum, ChannelWithDie &channel)
{
    CommLink selectedLink{};
    CHK_RET(SelectLinkForPair(comm, myRank, peerRank, layers, requireCtp, selectedLink, channel.selectedLayer));

    CHK_RET(HcclChannelDescInit(&channel.desc, 1));
    channel.desc.remoteRank = peerRank;
    channel.desc.channelProtocol = selectedLink.linkAttr.linkProtocol;
    channel.desc.localEndpoint = selectedLink.srcEndpointDesc;
    channel.desc.remoteEndpoint = selectedLink.dstEndpointDesc;
    channel.desc.notifyNum = notifyNum;
    channel.peerRank = peerRank;
    CHK_RET(GetEndpointDie(comm, myRank, channel.desc.localEndpoint, channel.actualDieId, "local"));
    CHK_RET(GetEndpointBwCoeff(comm, myRank, channel.desc.localEndpoint, channel.bwCoeff));
    // HcclRankGraphGetEndpointInfo only guarantees attributes for the current
    // rank. Keep remote die unknown here; production paths must not infer the
    // owner-side actual die from this endpoint descriptor.
    channel.remoteDieId = MAX_DIE_NUM;
    return HCCL_SUCCESS;
}

struct LinkCandidate {
    CommLink link{};
    uint32_t layer{0};
    uint32_t actualDieId{0};
};

struct P4LinkCandidate {
    CommLink link{};
    uint32_t layer{0};
    uint32_t ordinal{0};
    uint32_t actualDieId{MAX_DIE_NUM};
    uint32_t remoteDieId{MAX_DIE_NUM};
};

struct P4EndpointAudit {
    std::vector<std::vector<P4LinkCandidate>> candidatesByPeer;
    std::vector<uint32_t> localDieMaskByPeer;
    std::vector<uint32_t> preferredProtocolByPeer;
    uint32_t commonLocalDieMask{0};
    uint32_t unionLocalDieMask{0};
};

HcclResult GetButterflyLinkCandidates(
    HcclComm comm, uint32_t myRank, uint32_t peerRank, uint32_t fullClosLayer, std::vector<LinkCandidate> &candidates)
{
    CommLink *links = nullptr;
    uint32_t linkCount = 0;
    CHK_RET(HcclRankGraphGetLinks(comm, fullClosLayer, myRank, peerRank, &links, &linkCount));
    for (uint32_t i = 0; i < linkCount; ++i) {
        if (links[i].linkAttr.linkProtocol != CommProtocol::COMM_PROTOCOL_UBC_CTP) {
            continue;
        }
        uint32_t dieId = 0;
        CHK_RET(GetEndpointDie(comm, myRank, links[i].srcEndpointDesc, dieId, "butterfly-local"));
        candidates.push_back(LinkCandidate{links[i], fullClosLayer, dieId});
    }
    CHK_PRT_RET(candidates.empty(),
        HCCL_ERROR("[GetButterflyLinkCandidates] no CTP link from rank %u to rank %u", myRank, peerRank),
        HCCL_E_NOT_FOUND);
    return HCCL_SUCCESS;
}

HcclResult BuildChannelFromCandidate(
    uint32_t peerRank, uint32_t notifyNum, const LinkCandidate &candidate, ChannelWithDie &channel)
{
    CHK_RET(HcclChannelDescInit(&channel.desc, 1));
    channel.desc.remoteRank = peerRank;
    channel.desc.channelProtocol = candidate.link.linkAttr.linkProtocol;
    channel.desc.localEndpoint = candidate.link.srcEndpointDesc;
    channel.desc.remoteEndpoint = candidate.link.dstEndpointDesc;
    channel.desc.notifyNum = notifyNum;
    channel.peerRank = peerRank;
    channel.actualDieId = candidate.actualDieId;
    channel.remoteDieId = MAX_DIE_NUM;
    channel.selectedLayer = candidate.layer;
    return HCCL_SUCCESS;
}

HcclResult AuditP4EndpointCandidates(HcclComm comm, const OpParam &param, const TopologyView &topologyView,
    const std::vector<uint32_t> &peerRanks, P4EndpointAudit &audit)
{
    CHK_PRT_RET(peerRanks.size() != 3,
        HCCL_ERROR("[P4EndpointAudit] rank %u expected three peers, got %zu", param.myRank, peerRanks.size()),
        HCCL_E_INTERNAL);
    constexpr uint32_t ALL_DIE_MASK = (1U << MAX_DIE_NUM) - 1U;
    audit.candidatesByPeer.clear();
    audit.candidatesByPeer.resize(peerRanks.size());
    audit.localDieMaskByPeer.assign(peerRanks.size(), 0);
    audit.preferredProtocolByPeer.assign(
        peerRanks.size(), static_cast<uint32_t>(CommProtocol::COMM_PROTOCOL_RESERVED));
    audit.commonLocalDieMask = ALL_DIE_MASK;
    audit.unionLocalDieMask = 0;

    for (size_t peerIndex = 0; peerIndex < peerRanks.size(); ++peerIndex) {
        const uint32_t peerRank = peerRanks[peerIndex];
        CommLink *links = nullptr;
        uint32_t linkCount = 0;
        CHK_RET(HcclRankGraphGetLinks(comm, topologyView.fullClosLayer, param.myRank, peerRank, &links, &linkCount));
        bool hasCtp = false;
        for (uint32_t i = 0; i < linkCount; ++i) {
            hasCtp = hasCtp || links[i].linkAttr.linkProtocol == CommProtocol::COMM_PROTOCOL_UBC_CTP;
        }
        const CommProtocol preferredProtocol
            = hasCtp ? CommProtocol::COMM_PROTOCOL_UBC_CTP : CommProtocol::COMM_PROTOCOL_UBC_TP;
        audit.preferredProtocolByPeer[peerIndex] = static_cast<uint32_t>(preferredProtocol);

        uint32_t eligibleCount = 0;
        for (uint32_t i = 0; i < linkCount; ++i) {
            const CommProtocol protocol = links[i].linkAttr.linkProtocol;
            if (protocol != CommProtocol::COMM_PROTOCOL_UBC_CTP && protocol != CommProtocol::COMM_PROTOCOL_UBC_TP) {
                continue;
            }
            uint32_t localDie = MAX_DIE_NUM;
            const HcclResult localDieRet
                = GetEndpointDie(comm, param.myRank, links[i].srcEndpointDesc, localDie, "p4-audit-local");
            // The rank-graph API only guarantees attribute lookup for the local
            // rank in this environment. CommLink already carries the complete
            // remote EndpointDesc, including its superDevId (Die id).
            uint32_t remoteDie = links[i].dstEndpointDesc.loc.locType == ENDPOINT_LOC_TYPE_DEVICE
                                     ? links[i].dstEndpointDesc.loc.device.superDevId
                                     : MAX_DIE_NUM;
            if (remoteDie >= MAX_DIE_NUM) {
                remoteDie = MAX_DIE_NUM;
            }
            std::printf("[P4_ENDPOINT_CANDIDATE] rank=%u peer=%u layer=%u ordinal=%u protocol=%d hop=%u "
                        "localDie=%u remoteDie=%u localDev=%u remoteDev=%u preferred=%u\n",
                param.myRank, peerRank, topologyView.fullClosLayer, i, static_cast<int>(protocol),
                static_cast<uint32_t>(links[i].linkAttr.hop), localDie, remoteDie,
                links[i].srcEndpointDesc.loc.device.devPhyId, links[i].dstEndpointDesc.loc.device.devPhyId,
                protocol == preferredProtocol ? 1U : 0U);
            ++eligibleCount;
            if (protocol != preferredProtocol || localDieRet != HCCL_SUCCESS) {
                continue;
            }
            P4LinkCandidate candidate{links[i], topologyView.fullClosLayer, i, localDie, remoteDie};
            audit.candidatesByPeer[peerIndex].push_back(candidate);
            audit.localDieMaskByPeer[peerIndex] |= 1U << localDie;
        }
        CHK_PRT_RET(audit.candidatesByPeer[peerIndex].empty(),
            HCCL_ERROR("[P4EndpointAudit] no preferred protocol %d candidate from rank %u to peer %u",
                static_cast<int>(preferredProtocol), param.myRank, peerRank),
            HCCL_E_NOT_FOUND);
        audit.commonLocalDieMask &= audit.localDieMaskByPeer[peerIndex];
        audit.unionLocalDieMask |= audit.localDieMaskByPeer[peerIndex];
        std::printf("[P4_ENDPOINT_PEER] rank=%u peer=%u links=%u eligible=%u preferredProtocol=%d "
                    "preferredCandidates=%zu localDieMask=0x%x\n",
            param.myRank, peerRank, linkCount, eligibleCount, static_cast<int>(preferredProtocol),
            audit.candidatesByPeer[peerIndex].size(), audit.localDieMaskByPeer[peerIndex]);
    }

    const uint32_t canSingleDie = audit.commonLocalDieMask != 0 ? 1U : 0U;
    const uint32_t canDualDie = audit.unionLocalDieMask == ALL_DIE_MASK ? 1U : 0U;
    std::printf("[P4_ENDPOINT_SUMMARY] rank=%u commonDieMask=0x%x unionDieMask=0x%x canSingleDie=%u "
                "canDualDie=%u selection=legacy-first-link\n",
        param.myRank, audit.commonLocalDieMask, audit.unionLocalDieMask, canSingleDie, canDualDie);
    return HCCL_SUCCESS;
}

HcclResult SelectButterflyChannels(HcclComm comm, const OpParam &param, const TopologyView &topologyView,
    const std::vector<uint32_t> &peerRanks, std::vector<ChannelPlan> &plans, std::vector<HcclChannelDesc> &descs)
{
    CHK_PRT_RET(peerRanks.size() != 4 || plans.size() != peerRanks.size() || descs.size() != peerRanks.size(),
        HCCL_ERROR("[SelectButterflyChannels] invalid four-phase resource shape"), HCCL_E_INTERNAL);
    constexpr uint32_t ALL_DIE_MASK = (1U << MAX_DIE_NUM) - 1U;
    std::array<std::vector<LinkCandidate>, 4> candidatesByPhase;
    std::array<uint32_t, 4> candidateDieMask{};
    uint32_t commonDieMask = ALL_DIE_MASK;
    for (size_t phase = 0; phase < peerRanks.size(); ++phase) {
        CHK_RET(GetButterflyLinkCandidates(
            comm, param.myRank, peerRanks[phase], topologyView.fullClosLayer, candidatesByPhase[phase]));
        for (const auto &candidate : candidatesByPhase[phase]) {
            candidateDieMask[phase] |= 1U << candidate.actualDieId;
        }
        commonDieMask &= candidateDieMask[phase];
    }

    std::array<uint32_t, 4> selectedDie{};
    if (commonDieMask != 0) {
        const uint32_t commonDie = (commonDieMask & 1U) != 0 ? 0U : 1U;
        selectedDie.fill(commonDie);
        HCCL_INFO("[ButterflyDiePlan] rank=%u mode=single-die die=%u", param.myRank, commonDie);
    } else {
        uint32_t bestAssignment = 0;
        uint32_t bestScore = std::numeric_limits<uint32_t>::max();
        for (uint32_t assignment = 0; assignment < (1U << peerRanks.size()); ++assignment) {
            bool valid = true;
            uint32_t usedDieMask = 0;
            uint32_t transitions = 0;
            for (size_t phase = 0; phase < peerRanks.size(); ++phase) {
                const uint32_t dieId = (assignment >> phase) & 1U;
                if ((candidateDieMask[phase] & (1U << dieId)) == 0) {
                    valid = false;
                    break;
                }
                usedDieMask |= 1U << dieId;
                if (phase > 0 && dieId != ((assignment >> (phase - 1)) & 1U)) {
                    ++transitions;
                }
            }
            if (!valid) {
                continue;
            }
            const uint32_t dieCount = usedDieMask == ALL_DIE_MASK ? 2U : 1U;
            const uint32_t score = dieCount * 16U + transitions;
            if (score < bestScore) {
                bestScore = score;
                bestAssignment = assignment;
            }
        }
        CHK_PRT_RET(bestScore == std::numeric_limits<uint32_t>::max(),
            HCCL_ERROR("[SelectButterflyChannels] no valid die assignment for rank %u", param.myRank),
            HCCL_E_NOT_FOUND);
        for (size_t phase = 0; phase < peerRanks.size(); ++phase) {
            selectedDie[phase] = (bestAssignment >> phase) & 1U;
        }
        HCCL_INFO("[ButterflyDiePlan] rank=%u mode=dual-die assignment=%u%u%u%u transitions=%u", param.myRank,
            selectedDie[0], selectedDie[1], selectedDie[2], selectedDie[3], bestScore - 32U);
    }

    for (size_t phase = 0; phase < peerRanks.size(); ++phase) {
        const auto candidateIt = std::find_if(
            candidatesByPhase[phase].begin(), candidatesByPhase[phase].end(), [&](const LinkCandidate &candidate) {
                return candidate.actualDieId == selectedDie[phase];
            });
        CHK_PRT_RET(candidateIt == candidatesByPhase[phase].end(),
            HCCL_ERROR(
                "[SelectButterflyChannels] selected die %u is unavailable for phase %zu", selectedDie[phase], phase),
            HCCL_E_INTERNAL);
        CHK_RET(BuildChannelFromCandidate(peerRanks[phase], BFLY_CHANNEL_NOTIFY_NUM, *candidateIt, plans[phase].info));
        plans[phase].info.isLocal = 0;
        plans[phase].targetSlot = static_cast<uint32_t>(phase);
        descs[phase] = plans[phase].info.desc;
    }
    return HCCL_SUCCESS;
}

uint32_t SameServerSourceSlot(const std::vector<uint32_t> &localRanks, uint32_t sourceRank, uint32_t ownerRank)
{
    uint32_t slot = 0;
    for (uint32_t rank : localRanks) {
        if (rank == ownerRank) {
            continue;
        }
        if (rank == sourceRank) {
            return slot;
        }
        ++slot;
    }
    return MAX_RANK_SIZE;
}

HcclResult BuildLaneTableForOwner(HcclComm comm, const TopologyView &view, uint32_t ownerRank,
    const std::vector<uint32_t> &sourceRanks, uint32_t laneSizeLimit,
    std::map<uint32_t, LanePlacement> &placements, uint32_t &laneCount)
{
    (void)comm;
    (void)view;
    std::vector<uint32_t> sources = sourceRanks;
    std::sort(sources.begin(), sources.end());
    CHK_PRT_RET(std::adjacent_find(sources.begin(), sources.end()) != sources.end(),
        HCCL_ERROR("[BuildLaneTableForOwner] owner %u has duplicate source ranks", ownerRank), HCCL_E_INTERNAL);
    laneCount = 0;
    CHK_PRT_RET(laneSizeLimit == 0 || laneSizeLimit > MAX_LANE_SIZE,
        HCCL_ERROR("[BuildLaneTableForOwner] invalid lane-size limit %u", laneSizeLimit), HCCL_E_PARA);
    for (size_t begin = 0; begin < sources.size(); begin += laneSizeLimit) {
        const uint32_t laneSize = static_cast<uint32_t>(std::min<size_t>(laneSizeLimit, sources.size() - begin));
        CHK_PRT_RET(laneCount >= CROSS_LANE_MAX_COUNT,
            HCCL_ERROR(
                "[BuildLaneTableForOwner] owner %u requires more than %u lanes", ownerRank, CROSS_LANE_MAX_COUNT),
            HCCL_E_NOT_SUPPORT);
        for (uint32_t position = 0; position < laneSize; ++position) {
            // Remote endpoint attributes are not queryable through the public
            // RankGraph API.  Lane/position are therefore derived solely from
            // the sorted source set, so sender and owner reconstruct the same
            // table independently.  dieId is intentionally left as unknown.
            placements[sources[begin + position]] = LanePlacement{MAX_DIE_NUM, laneCount, position, laneSize};
        }
        ++laneCount;
    }
    CHK_PRT_RET(placements.size() != sourceRanks.size() || laneCount < 2 || laneCount > CROSS_LANE_MAX_COUNT,
        HCCL_ERROR("[BuildLaneTableForOwner] owner %u has invalid placement count %zu / lane count %u", ownerRank,
            placements.size(), laneCount),
        HCCL_E_NOT_SUPPORT);
    return HCCL_SUCCESS;
}

HcclResult PopulateCrossLaneMetadata(HcclComm comm, const OpParam &param, const TopologyView &view,
    Algorithm2x8 algorithm, std::vector<ChannelPlan> &plans, uint32_t &localLaneCount)
{
    const uint32_t laneSizeLimit = algorithm == Algorithm2x8::CROSS_LANE_512M ? 2U : MAX_LANE_SIZE;
    std::map<uint32_t, LanePlacement> localPlacements;
    CHK_RET(BuildLaneTableForOwner(
        comm, view, param.myRank, view.remoteRanks, laneSizeLimit, localPlacements, localLaneCount));

    std::map<uint32_t, std::map<uint32_t, LanePlacement>> targetTables;
    for (auto &plan : plans) {
        if (plan.info.isLocal != 0) {
            plan.targetSlot = SameServerSourceSlot(view.localRanks, param.myRank, plan.info.peerRank);
            CHK_PRT_RET(plan.targetSlot >= CROSS_LANE_SAME_SLOT_COUNT,
                HCCL_ERROR("[PopulateCrossLaneMetadata] invalid same slot for rank %u -> %u", param.myRank,
                    plan.info.peerRank),
                HCCL_E_INTERNAL);
            continue;
        }

        auto targetIt = targetTables.find(plan.info.peerRank);
        if (targetIt == targetTables.end()) {
            std::map<uint32_t, LanePlacement> targetPlacements;
            uint32_t targetLaneCount = 0;
            CHK_RET(BuildLaneTableForOwner(comm, view, plan.info.peerRank, view.localRanks, laneSizeLimit,
                targetPlacements, targetLaneCount));
            targetIt = targetTables.emplace(plan.info.peerRank, std::move(targetPlacements)).first;
        }
        const auto placementIt = targetIt->second.find(param.myRank);
        const auto sourceIt = localPlacements.find(plan.info.peerRank);
        CHK_PRT_RET(placementIt == targetIt->second.end() || sourceIt == localPlacements.end(),
            HCCL_ERROR("[PopulateCrossLaneMetadata] missing lane mapping for peer %u", plan.info.peerRank),
            HCCL_E_INTERNAL);
        plan.targetPlacement = placementIt->second;
        plan.sourcePlacement = sourceIt->second;
    }
    return HCCL_SUCCESS;
}

std::vector<uint32_t> RanksWithoutOwner(const std::vector<uint32_t> &ranks, uint32_t ownerRank)
{
    std::vector<uint32_t> sources;
    sources.reserve(ranks.size());
    for (uint32_t rank : ranks) {
        if (rank != ownerRank) {
            sources.push_back(rank);
        }
    }
    return sources;
}

HcclResult GetMeshTargetDie(
    const std::vector<uint32_t> &localRanks, uint32_t sourceRank, uint32_t ownerRank, uint32_t &targetDie)
{
    // 950 2x8 的 8-rank Mesh 不是按 rank 奇偶选端点。下表来自该拓扑
    // RankGraph 的默认 UBC_TP 路由，行是 source local index，列是 target
    // local index。通道建立后还会用 CommLink 携带的实际端点做一致性校验，
    // 因而拓扑或选路改变时会 fail closed，而不会静默写入错误 scratch lane。
    constexpr uint32_t invalidDie = MAX_DIE_NUM;
    constexpr uint32_t meshTargetDie[8][8] = {
        {invalidDie, 1, 1, 1, 1, 1, 1, 1},
        {1, invalidDie, 0, 0, 0, 0, 0, 0},
        {0, 0, invalidDie, 1, 1, 1, 1, 1},
        {1, 1, 1, invalidDie, 0, 0, 0, 0},
        {0, 0, 0, 0, invalidDie, 1, 1, 1},
        {1, 1, 1, 0, 1, invalidDie, 0, 0},
        {0, 0, 0, 0, 1, 0, invalidDie, 1},
        {1, 1, 1, 1, 1, 1, 1, invalidDie},
    };
    CHK_PRT_RET(localRanks.size() != 8,
        HCCL_ERROR("[GetMeshTargetDie] expected eight local ranks, got %zu", localRanks.size()), HCCL_E_INTERNAL);
    const auto sourceIt = std::lower_bound(localRanks.begin(), localRanks.end(), sourceRank);
    const auto ownerIt = std::lower_bound(localRanks.begin(), localRanks.end(), ownerRank);
    CHK_PRT_RET(sourceIt == localRanks.end() || *sourceIt != sourceRank || ownerIt == localRanks.end()
                    || *ownerIt != ownerRank || sourceRank == ownerRank,
        HCCL_ERROR("[GetMeshTargetDie] invalid source %u / owner %u", sourceRank, ownerRank), HCCL_E_INTERNAL);
    const uint32_t sourceIndex = static_cast<uint32_t>(std::distance(localRanks.begin(), sourceIt));
    const uint32_t ownerIndex = static_cast<uint32_t>(std::distance(localRanks.begin(), ownerIt));
    targetDie = meshTargetDie[sourceIndex][ownerIndex];
    CHK_PRT_RET(targetDie >= MAX_DIE_NUM,
        HCCL_ERROR("[GetMeshTargetDie] no route for source %u / owner %u", sourceRank, ownerRank), HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult BuildMeshLaneTable(const std::vector<uint32_t> &localRanks, uint32_t ownerRank,
    const std::vector<uint32_t> &sourceRanks, std::map<uint32_t, LanePlacement> &placements, uint32_t &laneCount)
{
    std::array<std::vector<uint32_t>, MAX_DIE_NUM> sourcesByDie;
    for (uint32_t sourceRank : sourceRanks) {
        uint32_t targetDie = MAX_DIE_NUM;
        CHK_RET(GetMeshTargetDie(localRanks, sourceRank, ownerRank, targetDie));
        sourcesByDie[targetDie].push_back(sourceRank);
    }

    laneCount = 0;
    for (uint32_t targetDie = 0; targetDie < MAX_DIE_NUM; ++targetDie) {
        auto &sources = sourcesByDie[targetDie];
        std::sort(sources.begin(), sources.end());
        for (size_t begin = 0; begin < sources.size(); begin += MAX_LANE_SIZE) {
            const uint32_t laneSize = static_cast<uint32_t>(std::min<size_t>(MAX_LANE_SIZE, sources.size() - begin));
            for (uint32_t position = 0; position < laneSize; ++position) {
                placements[sources[begin + position]] = LanePlacement{targetDie, laneCount, position, laneSize};
            }
            ++laneCount;
        }
    }
    CHK_PRT_RET(placements.size() != sourceRanks.size(),
        HCCL_ERROR(
            "[BuildMeshLaneTable] owner %u mapped %zu/%zu sources", ownerRank, placements.size(), sourceRanks.size()),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(laneCount < MAX_DIE_NUM || laneCount > 3,
        HCCL_ERROR("[BuildMeshLaneTable] owner %u expected two or three physical lanes, got %u", ownerRank, laneCount),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

void OffsetLanePlacements(std::map<uint32_t, LanePlacement> &placements, uint32_t laneBase)
{
    for (auto &entry : placements) {
        entry.second.lane += laneBase;
    }
}

HcclResult PopulateMeshLaneMetadata(HcclComm comm, const OpParam &param, const TopologyView &view,
    std::vector<ChannelPlan> &plans, uint32_t &localLaneCount)
{
    std::map<uint32_t, LanePlacement> incomingMesh;
    std::map<uint32_t, LanePlacement> incomingClos;
    uint32_t meshLaneCount = 0;
    uint32_t closLaneCount = 0;
    const std::vector<uint32_t> meshSources = RanksWithoutOwner(view.localRanks, param.myRank);
    CHK_RET(BuildMeshLaneTable(view.localRanks, param.myRank, meshSources, incomingMesh, meshLaneCount));
    CHK_RET(BuildLaneTableForOwner(
        comm, view, param.myRank, view.remoteRanks, MAX_LANE_SIZE, incomingClos, closLaneCount));
    OffsetLanePlacements(incomingClos, meshLaneCount);
    localLaneCount = meshLaneCount + closLaneCount;
    CHK_PRT_RET(meshLaneCount < 2 || meshLaneCount > 3 || closLaneCount != 2 || localLaneCount > MESH_LANE_MAX_COUNT,
        HCCL_ERROR("[PopulateMeshLaneMetadata] invalid lane shape mesh=%u clos=%u", meshLaneCount, closLaneCount),
        HCCL_E_INTERNAL);

    std::map<uint32_t, std::map<uint32_t, LanePlacement>> targetTables;
    for (auto &plan : plans) {
        auto targetIt = targetTables.find(plan.info.peerRank);
        if (targetIt == targetTables.end()) {
            std::vector<uint32_t> targetSources;
            const bool isLocal = plan.info.isLocal != 0;
            if (plan.info.isLocal != 0) {
                targetSources = RanksWithoutOwner(view.localRanks, plan.info.peerRank);
            } else {
                targetSources = view.localRanks;
            }
            std::map<uint32_t, LanePlacement> targetPlacements;
            uint32_t targetLaneCount = 0;
            if (isLocal) {
                CHK_RET(BuildMeshLaneTable(
                    view.localRanks, plan.info.peerRank, targetSources, targetPlacements, targetLaneCount));
            } else {
                std::map<uint32_t, LanePlacement> targetMeshPlacements;
                uint32_t targetMeshLaneCount = 0;
                const std::vector<uint32_t> targetMeshSources = RanksWithoutOwner(view.remoteRanks, plan.info.peerRank);
                CHK_RET(BuildMeshLaneTable(view.remoteRanks, plan.info.peerRank, targetMeshSources,
                    targetMeshPlacements, targetMeshLaneCount));
                CHK_RET(BuildLaneTableForOwner(comm, view, plan.info.peerRank, targetSources, MAX_LANE_SIZE,
                    targetPlacements, targetLaneCount));
                OffsetLanePlacements(targetPlacements, targetMeshLaneCount);
            }
            CHK_PRT_RET((isLocal && (targetLaneCount < 2 || targetLaneCount > 3)) || (!isLocal && targetLaneCount != 2),
                HCCL_ERROR("[PopulateMeshLaneMetadata] peer %u has %u lanes", plan.info.peerRank, targetLaneCount),
                HCCL_E_INTERNAL);
            targetIt = targetTables.emplace(plan.info.peerRank, std::move(targetPlacements)).first;
        }
        const auto targetPlacement = targetIt->second.find(param.myRank);
        const auto &sourceTable = plan.info.isLocal != 0 ? incomingMesh : incomingClos;
        const auto sourcePlacement = sourceTable.find(plan.info.peerRank);
        CHK_PRT_RET(targetPlacement == targetIt->second.end() || sourcePlacement == sourceTable.end(),
            HCCL_ERROR("[PopulateMeshLaneMetadata] missing mapping for peer %u", plan.info.peerRank), HCCL_E_INTERNAL);
        plan.targetPlacement = targetPlacement->second;
        plan.sourcePlacement = sourcePlacement->second;
        if (plan.info.isLocal != 0) {
            CHK_PRT_RET(plan.info.actualDieId != plan.sourcePlacement.dieId,
                HCCL_ERROR("[PopulateMeshLaneMetadata] incoming peer %u expected die %u, channel uses die %u",
                    plan.info.peerRank, plan.sourcePlacement.dieId, plan.info.actualDieId),
                HCCL_E_NOT_SUPPORT);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AcquireV101SmallChannels(HcclComm comm, const OpParam &param, const TopologyView &topologyView,
    std::vector<DieGroup> &groups)
{
    std::vector<ChannelPlan> plans;
    std::vector<HcclChannelDesc> descs;
    plans.reserve(param.rankSize - 1U);
    descs.reserve(param.rankSize - 1U);
    for (uint32_t peerRank = 0; peerRank < param.rankSize; ++peerRank) {
        if (peerRank == param.myRank) {
            continue;
        }
        ChannelPlan plan;
        // v101 scans graph layers in their native order and selects the first
        // UBC_CTP link. SelectChannelForPeer with requireCtp=true preserves
        // that descriptor choice while also auditing the endpoint die.
        CHK_RET(SelectChannelForPeer(comm, param.myRank, peerRank, topologyView.layers, true,
            V101_CHANNEL_NOTIFY_NUM, plan.info));
        plan.info.isLocal = 0U;
        plans.push_back(plan);
        descs.push_back(plan.info.desc);
    }
    CHK_PRT_RET(plans.size() != param.rankSize - 1U,
        HCCL_ERROR("[AcquireV101SmallChannels] incomplete full-mesh peer set"), HCCL_E_INTERNAL);

    std::vector<ChannelHandle> handles(plans.size());
    CHK_RET(HcclChannelAcquire(
        comm, CommEngine::COMM_ENGINE_CCU, descs.data(), static_cast<uint32_t>(descs.size()), handles.data()));
    for (size_t i = 0; i < plans.size(); ++i) {
        plans[i].handle = handles[i];
    }

    if (param.rankSize == 4U) {
        DieGroup group;
        // The v101 R4 candidate is deliberately registered on die 0 and uses
        // a single kernel even when endpoint metadata exposes another die.
        group.actualDieId = 0U;
        group.isPrimary = 1U;
        group.peers = std::move(plans);
        groups.push_back(std::move(group));
        return HCCL_SUCCESS;
    }

    std::map<uint32_t, DieGroup> groupsByDie;
    for (auto &plan : plans) {
        DieGroup &group = groupsByDie[plan.info.actualDieId];
        group.actualDieId = plan.info.actualDieId;
        group.peers.push_back(std::move(plan));
    }
    CHK_PRT_RET(groupsByDie.size() != 2U,
        HCCL_ERROR("[AcquireV101SmallChannels] rank %u requires two populated CCU dies, got %zu", param.rankSize,
            groupsByDie.size()),
        HCCL_E_NOT_SUPPORT);
    for (auto &entry : groupsByDie) {
        groups.push_back(std::move(entry.second));
    }
    groups[0].isPrimary = 1U;
    return HCCL_SUCCESS;
}

HcclResult AcquireChannels(HcclComm comm, const OpParam &param, const TopologyView &topologyView,
    Algorithm2x8 algorithm, std::vector<DieGroup> &groups, uint32_t &localLaneCount)
{
    std::vector<uint32_t> peerRanks;
    std::vector<uint32_t> eligibleLayers = topologyView.layers;
    bool requireCtp = false;
    uint32_t notifyNum = OWNER_CHANNEL_NOTIFY_NUM;
    if (algorithm == Algorithm2x8::BFLY16_CLOS_4STEP) {
        constexpr std::array<uint32_t, 4> masks = {0U, 1U, 2U, 4U};
        eligibleLayers = {topologyView.fullClosLayer};
        requireCtp = true;
        notifyNum = BFLY_CHANNEL_NOTIFY_NUM;
        for (uint32_t mask : masks) {
            const uint32_t peerIndex = topologyView.localIndex ^ mask;
            CHK_PRT_RET(peerIndex >= topologyView.remoteRanks.size(),
                HCCL_ERROR("[AcquireChannels] invalid butterfly peer index %u", peerIndex), HCCL_E_INTERNAL);
            peerRanks.push_back(topologyView.remoteRanks[peerIndex]);
        }
    } else if (algorithm == Algorithm2x8::P12_HIER8_512M || algorithm == Algorithm2x8::P12_HIER8_400M4B) {
        peerRanks.reserve(topologyView.localRanks.size() + 1);
        for (uint32_t rank : topologyView.localRanks) {
            if (rank != param.myRank) {
                peerRanks.push_back(rank);
            }
        }
        if (topologyView.localRanks.size() == 8) {
            CHK_PRT_RET(topologyView.localIndex >= 8 || topologyView.remoteRanks.size() != 4,
                HCCL_ERROR("[AcquireChannels] invalid 8-side hierarchy shape"), HCCL_E_INTERNAL);
            peerRanks.push_back(topologyView.remoteRanks[topologyView.localIndex % 4]);
        } else {
            CHK_PRT_RET(topologyView.localRanks.size() != 4 || topologyView.remoteRanks.size() != 8
                            || topologyView.localIndex >= 4,
                HCCL_ERROR("[AcquireChannels] invalid 4-side hierarchy shape"), HCCL_E_INTERNAL);
            peerRanks.push_back(topologyView.remoteRanks[topologyView.localIndex]);
            peerRanks.push_back(topologyView.remoteRanks[topologyView.localIndex + 4]);
        }
        notifyNum = P12_HIER_CHANNEL_NOTIFY_NUM;
    } else {
        peerRanks.reserve(param.rankSize - 1);
        for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
            if (rank != param.myRank) {
                peerRanks.push_back(rank);
            }
        }
        if (IsCrossLaneAlgorithm(algorithm) || IsP16MeshLaneAlgorithm(algorithm)) {
            notifyNum = CROSS_LANE_CHANNEL_NOTIFY_NUM;
        } else if (IsP12ClosAccumulateAlgorithm(algorithm)) {
            notifyNum = P12_CLOS_CHANNEL_NOTIFY_NUM;
        } else if (algorithm == Algorithm2x8::P4_DIRECT_RSAG_512K || algorithm == Algorithm2x8::P4_DIRECT_SLOT3_512M
                   || algorithm == Algorithm2x8::P4_DIRECT_SLOT3_400M4B || IsP12DirectRsagAlgorithm(algorithm)
                   || IsP12AllPairsAlgorithm(algorithm)) {
            notifyNum = DIRECT_CHANNEL_NOTIFY_NUM;
        } else if (algorithm == Algorithm2x8::P4_ROTATE3_512M || algorithm == Algorithm2x8::P4_ROTATE3_400M4B) {
            notifyNum = P4_ROTATE_CHANNEL_NOTIFY_NUM;
        } else if (algorithm == Algorithm2x8::P4_ROTATE3_UNIQUE_NOTIFY_512M
                   || algorithm == Algorithm2x8::P4_ROTATE3_UNIQUE_NOTIFY_400M4B) {
            notifyNum = P4_ROTATE_UNIQUE_NOTIFY_NUM;
        } else if (algorithm == Algorithm2x8::P4_ROTATE3_DIRECT_OUTPUT_512M
                   || algorithm == Algorithm2x8::P4_ROTATE3_DIRECT_OUTPUT_400M4B) {
            notifyNum = P4_ROTATE_UNIQUE_NOTIFY_NUM;
        } else if (IsP4OutputAccumulateAlgorithm(algorithm)) {
            notifyNum = P4_ROTATE_UNIQUE_NOTIFY_NUM;
        } else if (IsP4PullR3Algorithm(algorithm)) {
            notifyNum = P4_ROTATE_UNIQUE_NOTIFY_NUM;
        }
    }

    const bool isP4Specialized
        = algorithm == Algorithm2x8::P4_DIRECT_RSAG_512K || algorithm == Algorithm2x8::P4_DIRECT_SLOT3_512M
          || algorithm == Algorithm2x8::P4_DIRECT_SLOT3_400M4B || algorithm == Algorithm2x8::P4_ROTATE3_512M
          || algorithm == Algorithm2x8::P4_ROTATE3_400M4B || algorithm == Algorithm2x8::P4_ROTATE3_UNIQUE_NOTIFY_512M
          || algorithm == Algorithm2x8::P4_ROTATE3_UNIQUE_NOTIFY_400M4B
          || algorithm == Algorithm2x8::P4_ROTATE3_DIRECT_OUTPUT_512M
          || algorithm == Algorithm2x8::P4_ROTATE3_DIRECT_OUTPUT_400M4B || IsP4OutputAccumulateAlgorithm(algorithm)
          || IsP4PullR3Algorithm(algorithm);
    P4EndpointAudit p4EndpointAudit;
    if (isP4Specialized) {
        CHK_RET(AuditP4EndpointCandidates(comm, param, topologyView, peerRanks, p4EndpointAudit));
    }

    std::vector<ChannelPlan> plans(peerRanks.size());
    std::vector<HcclChannelDesc> descs(peerRanks.size());
    if (algorithm == Algorithm2x8::BFLY16_CLOS_4STEP) {
        CHK_RET(SelectButterflyChannels(comm, param, topologyView, peerRanks, plans, descs));
    } else {
        for (size_t i = 0; i < peerRanks.size(); ++i) {
            plans[i].info.isLocal
                = std::binary_search(topologyView.localRanks.begin(), topologyView.localRanks.end(), peerRanks[i]) ? 1U
                                                                                                                   : 0U;
            const bool topologySpecific
                = algorithm == Algorithm2x8::P4_DIRECT_RSAG_512K || algorithm == Algorithm2x8::P4_DIRECT_SLOT3_512M
                  || algorithm == Algorithm2x8::P4_DIRECT_SLOT3_400M4B || algorithm == Algorithm2x8::P4_ROTATE3_512M
                  || algorithm == Algorithm2x8::P4_ROTATE3_400M4B
                  || algorithm == Algorithm2x8::P4_ROTATE3_UNIQUE_NOTIFY_512M
                  || algorithm == Algorithm2x8::P4_ROTATE3_UNIQUE_NOTIFY_400M4B
                  || algorithm == Algorithm2x8::P4_ROTATE3_DIRECT_OUTPUT_512M
                  || algorithm == Algorithm2x8::P4_ROTATE3_DIRECT_OUTPUT_400M4B
                  || IsP4OutputAccumulateAlgorithm(algorithm) || IsP16MeshLaneAlgorithm(algorithm)
                  || IsP4PullR3Algorithm(algorithm) || IsP12DirectRsagAlgorithm(algorithm)
                  || algorithm == Algorithm2x8::P12_HIER8_512M || algorithm == Algorithm2x8::P12_HIER8_400M4B
                  || IsP12AllPairsAlgorithm(algorithm);
            std::vector<uint32_t> peerLayers = eligibleLayers;
            if (topologySpecific) {
                peerLayers = plans[i].info.isLocal != 0 ? std::vector<uint32_t>{topologyView.localLayer}
                                                        : std::vector<uint32_t>{topologyView.fullClosLayer};
            }
            CHK_RET(SelectChannelForPeer(
                comm, param.myRank, peerRanks[i], peerLayers, requireCtp, notifyNum, plans[i].info));
            plans[i].info.isLocal
                = std::binary_search(topologyView.localRanks.begin(), topologyView.localRanks.end(), peerRanks[i]) ? 1U
                                                                                                                   : 0U;
            descs[i] = plans[i].info.desc;
        }
    }

    if (IsCrossLaneAlgorithm(algorithm)) {
        CHK_RET(PopulateCrossLaneMetadata(comm, param, topologyView, algorithm, plans, localLaneCount));
    } else if (IsP16MeshLaneAlgorithm(algorithm)) {
        CHK_RET(PopulateMeshLaneMetadata(comm, param, topologyView, plans, localLaneCount));
    }

    std::vector<ChannelHandle> handles(plans.size());
    CHK_RET(HcclChannelAcquire(
        comm, CommEngine::COMM_ENGINE_CCU, descs.data(), static_cast<uint32_t>(descs.size()), handles.data()));

    std::map<uint32_t, DieGroup> groupsByDie;
    for (size_t i = 0; i < plans.size(); ++i) {
        plans[i].handle = handles[i];
        DieGroup &group = groupsByDie[plans[i].info.actualDieId];
        group.actualDieId = plans[i].info.actualDieId;
        group.peers.push_back(plans[i]);
    }
    CHK_PRT_RET(groupsByDie.empty() || groupsByDie.size() > MAX_DIE_NUM,
        HCCL_ERROR("[AcquireChannels] invalid non-empty die group count %zu", groupsByDie.size()), HCCL_E_INTERNAL);
    for (auto &entry : groupsByDie) {
        groups.push_back(std::move(entry.second));
    }

    size_t primaryIndex = 0;
    if (groups.size() == 2 && groups[1].peers.size() < groups[0].peers.size()) {
        primaryIndex = 1;
    }
    if (IsP12OutputTreeAlgorithm(algorithm)) {
        CHK_PRT_RET(groups.size() != MAX_DIE_NUM,
            HCCL_ERROR("[AcquireChannels] output-tree P12 requires two Die groups"), HCCL_E_INTERNAL);
        const uint32_t directSourceRank = param.myRank == 0 ? 1U : 0U;
        size_t outputGroup = groups.size();
        for (size_t groupIndex = 0; groupIndex < groups.size(); ++groupIndex) {
            const auto directSource = std::find_if(groups[groupIndex].peers.begin(), groups[groupIndex].peers.end(),
                [directSourceRank](const ChannelPlan &peer) { return peer.info.peerRank == directSourceRank; });
            if (directSource != groups[groupIndex].peers.end()) {
                outputGroup = groupIndex;
                break;
            }
        }
        CHK_PRT_RET(outputGroup == groups.size(),
            HCCL_ERROR("[AcquireChannels] output-tree direct source has no Die group"), HCCL_E_INTERNAL);
        // v112's compact output tree requires the group that owns the direct
        // output source to remain secondary; the other group performs the
        // local-input accumulation and the final partial merge.  Production
        // endpoint layouts already satisfy this through the size heuristic,
        // while the local mock topology can assign the two equal roles in the
        // opposite order.
        primaryIndex = 1U - outputGroup;
    }
    groups[primaryIndex].isPrimary = 1;
    for (size_t groupIndex = 0; groupIndex < groups.size(); ++groupIndex) {
        for (const auto &peer : groups[groupIndex].peers) {
            HCCL_INFO("[ResourcePeer] algorithm=%s rank=%u peer=%u local=%u localDie=%u remoteDie=%u layer=%u "
                      "protocol=%u bwCoeff=%u phaseOrSlot=%u targetLane=%u/%u/%u sourceLane=%u/%u/%u",
                AlgorithmName(algorithm), param.myRank, peer.info.peerRank, peer.info.isLocal, peer.info.actualDieId,
                peer.info.remoteDieId, peer.info.selectedLayer, static_cast<uint32_t>(peer.info.desc.channelProtocol),
                peer.info.bwCoeff, peer.targetSlot, peer.targetPlacement.lane, peer.targetPlacement.position,
                peer.targetPlacement.size, peer.sourcePlacement.lane, peer.sourcePlacement.position,
                peer.sourcePlacement.size);
        }
    }
    return HCCL_SUCCESS;
}

void GetOwnerSlice(uint64_t count, uint32_t rankSize, uint32_t ownerRank, uint64_t &offsetBytes, uint64_t &ownerBytes)
{
    const uint64_t base = count / rankSize;
    const uint64_t remainder = count % rankSize;
    const uint64_t ownerElements = base + (ownerRank < remainder ? 1U : 0U);
    const uint64_t ownerOffsetElements
        = static_cast<uint64_t>(ownerRank) * base + std::min<uint64_t>(ownerRank, remainder);
    offsetBytes = ownerOffsetElements * FP32_BYTES;
    ownerBytes = ownerElements * FP32_BYTES;
}

HcclResult ConfigureV101SmallArg(const OpParam &param, const std::vector<DieGroup> &groups, size_t groupIndex,
    CcuKernelArgV101Small &arg)
{
    constexpr uint64_t tileBytes = 4096ULL;
    constexpr uint32_t r12HeavyRankCount = 8U;
    constexpr uint32_t r12HeavyTiles = 11U;
    constexpr uint32_t r12LightTiles = 10U;
    const bool r12Aligned = param.rankSize == 12U;
    const auto sliceBytesForRank = [&](uint32_t rank) -> uint64_t {
        if (r12Aligned) {
            return static_cast<uint64_t>(rank < r12HeavyRankCount ? r12HeavyTiles : r12LightTiles) * tileBytes;
        }
        const uint64_t base = BYTES_512K / param.rankSize;
        const uint64_t remainder = BYTES_512K % param.rankSize;
        return base + (rank < remainder ? 1U : 0U);
    };
    const auto sliceOffsetForRank = [&](uint32_t rank) -> uint64_t {
        if (r12Aligned) {
            if (rank < r12HeavyRankCount) {
                return static_cast<uint64_t>(rank) * r12HeavyTiles * tileBytes;
            }
            return (static_cast<uint64_t>(r12HeavyRankCount) * r12HeavyTiles
                       + static_cast<uint64_t>(rank - r12HeavyRankCount) * r12LightTiles)
                   * tileBytes;
        }
        const uint64_t base = BYTES_512K / param.rankSize;
        const uint64_t remainder = BYTES_512K % param.rankSize;
        return base * rank + std::min<uint64_t>(rank, remainder);
    };

    arg.rankSize = param.rankSize;
    arg.rankId = param.myRank;
    arg.dataType = param.dataType;
    arg.reduceType = param.reduceType;
    arg.directSliceOffset = sliceOffsetForRank(param.myRank);
    arg.directSliceBytes = sliceBytesForRank(param.myRank);
    arg.directTileBytes = arg.directSliceBytes;
    arg.directTailBytes = arg.directSliceBytes;
    arg.directTileCount = 1U;
    arg.directUseCompactArgs = 1U;
    std::fill(std::begin(arg.channelIndexByRank), std::end(arg.channelIndexByRank),
        V101_INVALID_CHANNEL_INDEX);
    const DieGroup &group = groups[groupIndex];
    arg.channelCount = static_cast<uint32_t>(group.peers.size());
    for (size_t channelIndex = 0; channelIndex < group.peers.size(); ++channelIndex) {
        arg.channels[channelIndex] = group.peers[channelIndex].handle;
        arg.channelIndexByRank[group.peers[channelIndex].info.peerRank] = static_cast<uint32_t>(channelIndex);
    }

    if (groups.size() == 1U) {
        CHK_PRT_RET(param.rankSize != 4U || arg.channelCount != 3U || arg.directSliceBytes != 128U * 1024U,
            HCCL_ERROR("[ConfigureV101SmallArg] invalid R4 resource shape"), HCCL_E_INTERNAL);
        arg.directUseMsGroup = 1U;
        arg.directUseR4MsDirect = 1U;
        arg.directMsParallel = 16U;
        return HCCL_SUCCESS;
    }

    CHK_PRT_RET(groups.size() != 2U || groupIndex >= 2U || (param.rankSize != 12U && param.rankSize != 16U),
        HCCL_ERROR("[ConfigureV101SmallArg] invalid two-die resource shape"), HCCL_E_INTERNAL);
    CHK_PRT_RET(groups[0].actualDieId != 0U || groups[1].actualDieId != 1U,
        HCCL_ERROR("[ConfigureV101SmallArg] expected ordered die groups 0/1"), HCCL_E_INTERNAL);
    const bool die0IncludesLocalInput = groups[0].peers.size() <= groups[1].peers.size();
    const bool includeLocalInput = groupIndex == 0U ? die0IncludesLocalInput : !die0IncludesLocalInput;
    const uint32_t groupInputs = arg.channelCount + (includeLocalInput ? 1U : 0U);
    const uint32_t die0Inputs
        = static_cast<uint32_t>(groups[0].peers.size()) + (die0IncludesLocalInput ? 1U : 0U);
    CHK_PRT_RET(groupInputs == 0U || groupInputs > param.rankSize,
        HCCL_ERROR("[ConfigureV101SmallArg] invalid input count %u", groupInputs), HCCL_E_INTERNAL);
    arg.directTwoDie = 1U;
    arg.directDieId = group.actualDieId;
    arg.directIncludeLocalInput = includeLocalInput ? 1U : 0U;
    arg.directUseMsGroup = groupInputs <= 8U ? 1U : 0U;
    // Both merge kernels consume die0's partial.  They must therefore agree
    // on whether that partial lives in output (MS) or scratch (serial), even
    // when the two endpoint groups use different partial implementations.
    arg.directPartialZeroInOutput = param.rankSize == 12U && die0Inputs <= 8U ? 1U : 0U;
    arg.directReplicatedMerge = param.rankSize == 16U && param.inputPtr != param.outputPtr ? 1U : 0U;
    arg.directMsParallel = arg.directUseMsGroup == 0U
                               ? 0U
                               : (param.rankSize == 12U
                                         ? (arg.directSliceBytes == r12HeavyTiles * tileBytes ? r12HeavyTiles
                                                                                              : r12LightTiles)
                                         : 8U);
    arg.directScratchOffset = group.actualDieId == 0U ? 0U : arg.directTileBytes;
    const uint64_t firstMergeBytes = (arg.directSliceBytes / FP32_BYTES / 2U) * FP32_BYTES;
    arg.directMergeOffset = group.actualDieId == 0U ? 0U : firstMergeBytes;
    arg.directMergeBytes
        = group.actualDieId == 0U ? firstMergeBytes : arg.directSliceBytes - firstMergeBytes;
    return HCCL_SUCCESS;
}

uint32_t SourceSlot(uint32_t sourceRank, uint32_t ownerRank)
{
    return sourceRank < ownerRank ? sourceRank : sourceRank - 1U;
}

uint32_t P12TempOutputRank(uint32_t ownerRank)
{
    return ownerRank < P12_ALLPAIRS_A_RANK_COUNT ? (ownerRank + 1U) % P12_ALLPAIRS_A_RANK_COUNT
                                                 : ownerRank - P12_ALLPAIRS_A_RANK_COUNT;
}

void GetP12AllPairsSegments(
    Algorithm2x8 algorithm, uint64_t count, uint64_t (&offsetElements)[P12_ALLPAIRS_SEGMENT_COUNT],
    uint64_t (&segmentElements)[P12_ALLPAIRS_SEGMENT_COUNT], uint32_t &segmentCount)
{
    std::fill(std::begin(offsetElements), std::end(offsetElements), 0);
    std::fill(std::begin(segmentElements), std::end(segmentElements), 0);
    offsetElements[0] = 0;
    if (IsP12AllPairsOneSegmentAlgorithm(algorithm)) {
        segmentCount = 1;
        segmentElements[0] = count;
        return;
    }
    segmentCount = P12_ALLPAIRS_SEGMENT_COUNT;
    if (IsP12AllPairs512MAlgorithm(algorithm)) {
        segmentElements[0] = count / P12_ALLPAIRS_SEGMENT_COUNT; // 256 MiB
    } else {
        segmentElements[0] = (200ULL * 1024ULL * 1024ULL) / FP32_BYTES;
    }
    offsetElements[1] = segmentElements[0];
    segmentElements[1] = count - segmentElements[0];
}

HcclResult GetLocalLanePlacement(
    const TopologyView &view, uint32_t sourceRank, uint32_t ownerRank, LanePlacement &placement)
{
    CHK_PRT_RET(view.localRanks.size() != 4 && view.localRanks.size() != 8,
        HCCL_ERROR("[GetLocalLanePlacement] invalid local size %zu", view.localRanks.size()), HCCL_E_INTERNAL);
    uint32_t position = 0;
    bool found = false;
    for (uint32_t rank : view.localRanks) {
        if (rank == ownerRank) {
            continue;
        }
        if (rank == sourceRank) {
            found = true;
            break;
        }
        ++position;
    }
    CHK_PRT_RET(!found, HCCL_ERROR("[GetLocalLanePlacement] source %u is not local to owner %u", sourceRank, ownerRank),
        HCCL_E_INTERNAL);
    if (view.localRanks.size() == 8 && position >= 4) {
        placement.lane = 1;
        placement.position = position - 4;
        placement.size = 3;
    } else {
        placement.lane = 0;
        placement.position = position;
        placement.size = view.localRanks.size() == 8 ? 4U : 3U;
    }
    return HCCL_SUCCESS;
}

void BuildGlobalHierShards(
    uint64_t count, uint64_t (&offsets)[HIER_GLOBAL_SHARD_COUNT], uint64_t (&sizes)[HIER_GLOBAL_SHARD_COUNT])
{
    const uint64_t base = count / HIER_GLOBAL_SHARD_COUNT;
    const uint64_t remainder = count % HIER_GLOBAL_SHARD_COUNT;
    uint64_t offsetElements = 0;
    for (uint32_t shard = 0; shard < HIER_GLOBAL_SHARD_COUNT; ++shard) {
        offsets[shard] = offsetElements * FP32_BYTES;
        const uint64_t shardElements = base + (shard < remainder ? 1U : 0U);
        sizes[shard] = shardElements * FP32_BYTES;
        offsetElements += shardElements;
    }
}

void BuildLocalShards(Algorithm2x8 algorithm, uint64_t ownerBytes, uint64_t (&offsets)[CROSS_LANE_SHARD_COUNT],
    uint64_t (&sizes)[CROSS_LANE_SHARD_COUNT])
{
    if ((algorithm == Algorithm2x8::CROSS_LANE_512M || algorithm == Algorithm2x8::P16_MESH_LANE_512M)
        && ownerBytes == 32ULL * 1024 * 1024) {
        sizes[0] = 8ULL * 1024 * 1024;
        sizes[1] = 12ULL * 1024 * 1024;
        sizes[2] = 8ULL * 1024 * 1024;
        sizes[3] = 4ULL * 1024 * 1024;
    } else {
        const uint64_t elements = ownerBytes / FP32_BYTES;
        const uint64_t base = elements / CROSS_LANE_SHARD_COUNT;
        const uint64_t remainder = elements % CROSS_LANE_SHARD_COUNT;
        for (uint32_t shard = 0; shard < CROSS_LANE_SHARD_COUNT; ++shard) {
            sizes[shard] = (base + (shard < remainder ? 1U : 0U)) * FP32_BYTES;
        }
    }
    offsets[0] = 0;
    for (uint32_t shard = 1; shard < CROSS_LANE_SHARD_COUNT; ++shard) {
        offsets[shard] = offsets[shard - 1] + sizes[shard - 1];
    }
}

HcclResult GetP12ClosPhase(const TopologyView &view, uint32_t peerRank, uint32_t &phase)
{
    const auto peerIt = std::lower_bound(view.remoteRanks.begin(), view.remoteRanks.end(), peerRank);
    CHK_PRT_RET(peerIt == view.remoteRanks.end() || *peerIt != peerRank,
        HCCL_ERROR("[GetP12ClosPhase] peer %u is not remote", peerRank), HCCL_E_INTERNAL);
    const uint32_t peerIndex = static_cast<uint32_t>(std::distance(view.remoteRanks.begin(), peerIt));
    uint32_t aIndex = 0;
    uint32_t bIndex = 0;
    if (view.localRanks.size() == P12_ALLPAIRS_A_RANK_COUNT) {
        aIndex = view.localIndex;
        bIndex = peerIndex;
    } else {
        CHK_PRT_RET(view.localRanks.size() != P12_ALLPAIRS_B_RANK_COUNT,
            HCCL_ERROR("[GetP12ClosPhase] invalid local group size %zu", view.localRanks.size()), HCCL_E_INTERNAL);
        aIndex = peerIndex;
        bIndex = view.localIndex;
    }
    CHK_PRT_RET(aIndex >= P12_ALLPAIRS_A_RANK_COUNT || bIndex >= P12_ALLPAIRS_B_RANK_COUNT,
        HCCL_ERROR("[GetP12ClosPhase] invalid A/B index %u/%u", aIndex, bIndex), HCCL_E_INTERNAL);
    phase = (aIndex + P12_ALLPAIRS_A_RANK_COUNT - bIndex) % P12_ALLPAIRS_A_RANK_COUNT;
    return HCCL_SUCCESS;
}

HcclResult RegisterKernels(HcclComm comm, const OpParam &param, const TopologyView &topologyView,
    Algorithm2x8 algorithm, const std::vector<DieGroup> &groups, uint32_t localLaneCount, AlgResourceCtx &resource)
{
    CcuInsHandle insHandle{};
    uint32_t insCount = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insCount));
    CHK_PRT_RET(insCount != 1, HCCL_ERROR("[RegisterKernels] expected one CCU instruction instance, got %u", insCount),
        HCCL_E_INTERNAL);

    std::array<uint32_t, 4> butterflyPhaseOwner{};
    uint64_t butterflyScratchToken = 0;
    if (algorithm == Algorithm2x8::BFLY16_CLOS_4STEP) {
        butterflyPhaseOwner.fill(MAX_DIE_NUM);
        for (size_t groupIndex = 0; groupIndex < groups.size(); ++groupIndex) {
            for (const auto &peer : groups[groupIndex].peers) {
                CHK_PRT_RET(peer.targetSlot >= butterflyPhaseOwner.size()
                                || butterflyPhaseOwner[peer.targetSlot] != MAX_DIE_NUM,
                    HCCL_ERROR("[RegisterKernels] invalid butterfly phase ownership for phase %u", peer.targetSlot),
                    HCCL_E_INTERNAL);
                butterflyPhaseOwner[peer.targetSlot] = static_cast<uint32_t>(groupIndex);
            }
        }
        CHK_PRT_RET(
            std::find(butterflyPhaseOwner.begin(), butterflyPhaseOwner.end(), MAX_DIE_NUM) != butterflyPhaseOwner.end(),
            HCCL_ERROR("[RegisterKernels] incomplete butterfly phase ownership"), HCCL_E_INTERNAL);
        const CcuResult tokenRet = HcommCcuGetMemToken(
            reinterpret_cast<uint64_t>(resource.localBuffer.addr), resource.localBuffer.size, &butterflyScratchToken);
        if (tokenRet != CCU_SUCCESS) {
            HCCL_ERROR("[RegisterKernels] failed to cache butterfly scratch token, ret %d", tokenRet);
            return ConvertCcuToHccl(tokenRet);
        }
    }

    const bool staticScratchAlgorithm
        = algorithm == Algorithm2x8::P4_DIRECT_RSAG_512K || algorithm == Algorithm2x8::P4_DIRECT_SLOT3_512M
          || algorithm == Algorithm2x8::P4_DIRECT_SLOT3_400M4B || algorithm == Algorithm2x8::P4_ROTATE3_512M
          || algorithm == Algorithm2x8::P4_ROTATE3_400M4B || algorithm == Algorithm2x8::P4_ROTATE3_UNIQUE_NOTIFY_512M
          || algorithm == Algorithm2x8::P4_ROTATE3_UNIQUE_NOTIFY_400M4B
          || algorithm == Algorithm2x8::P4_ROTATE3_DIRECT_OUTPUT_512M
          || algorithm == Algorithm2x8::P4_ROTATE3_DIRECT_OUTPUT_400M4B || IsP4OutputAccumulateAlgorithm(algorithm)
          || IsP4PullR3Algorithm(algorithm) || IsP12DirectRsagAlgorithm(algorithm)
          || algorithm == Algorithm2x8::P12_HIER8_512M || algorithm == Algorithm2x8::P12_HIER8_400M4B
          || IsP12AllPairsAlgorithm(algorithm);
    uint64_t staticScratchToken = 0;
    if (staticScratchAlgorithm) {
        const CcuResult tokenRet = HcommCcuGetMemToken(
            reinterpret_cast<uint64_t>(resource.localBuffer.addr), resource.localBuffer.size, &staticScratchToken);
        if (tokenRet != CCU_SUCCESS) {
            HCCL_ERROR("[RegisterKernels] failed to cache special scratch token, ret %d", tokenRet);
            return ConvertCcuToHccl(tokenRet);
        }
    }

    CcuResult ccuRet = HcommCcuKernelRegisterStart(insHandle);
    if (ccuRet != CCU_SUCCESS) {
        HCCL_ERROR("[RegisterKernels] register start failed, ret %d", ccuRet);
        return ConvertCcuToHccl(ccuRet);
    }

    std::vector<std::shared_ptr<CcuKernelArgBase>> kernelArgs;
    kernelArgs.reserve(groups.size());
    resource.ccuKernels.resize(groups.size());
    resource.kernelMeta.resize(groups.size());

    size_t p12ClosOwnerGroup = groups.size();
    if (IsP12ClosAccumulateAlgorithm(algorithm)) {
        for (size_t groupIndex = 0; groupIndex < groups.size(); ++groupIndex) {
            const size_t remoteCount = static_cast<size_t>(std::count_if(
                groups[groupIndex].peers.begin(), groups[groupIndex].peers.end(), [](const ChannelPlan &peer) {
                    return peer.info.isLocal == 0;
                }));
            if (remoteCount == 0) {
                continue;
            }
            CHK_PRT_RET(p12ClosOwnerGroup != groups.size() || remoteCount != topologyView.remoteRanks.size(),
                HCCL_ERROR("[RegisterKernels] P12 CLOS channels are split across actual dies"), HCCL_E_NOT_SUPPORT);
            p12ClosOwnerGroup = groupIndex;
        }
        CHK_PRT_RET(p12ClosOwnerGroup == groups.size(), HCCL_ERROR("[RegisterKernels] P12 CLOS owner group is missing"),
            HCCL_E_INTERNAL);
    }

    uint32_t p12OutputTreeGroupRoots[MAX_DIE_NUM]{};
    uint32_t p12OutputTreeOutputGroup = MAX_DIE_NUM;
    if (IsP12OutputTreeAlgorithm(algorithm)) {
        CHK_PRT_RET(groups.size() != MAX_DIE_NUM,
            HCCL_ERROR("[RegisterKernels] output-tree P12 requires two Die groups"), HCCL_E_INTERNAL);
        const uint32_t directSourceRank = param.myRank == 0 ? 1U : 0U;
        for (size_t groupIndex = 0; groupIndex < groups.size(); ++groupIndex) {
            CHK_PRT_RET(groups[groupIndex].peers.empty(),
                HCCL_ERROR("[RegisterKernels] empty output-tree P12 Die group"), HCCL_E_INTERNAL);
            p12OutputTreeGroupRoots[groupIndex] = groups[groupIndex].peers[0].info.peerRank;
            for (const auto &peer : groups[groupIndex].peers) {
                if (peer.info.peerRank == directSourceRank) {
                    p12OutputTreeOutputGroup = static_cast<uint32_t>(groupIndex);
                    p12OutputTreeGroupRoots[groupIndex] = directSourceRank;
                    break;
                }
            }
        }
        CHK_PRT_RET(p12OutputTreeOutputGroup == MAX_DIE_NUM,
            HCCL_ERROR("[RegisterKernels] output-tree direct source is not assigned to a Die group"),
            HCCL_E_INTERNAL);
        CHK_PRT_RET(groups[p12OutputTreeOutputGroup].isPrimary != 0,
            HCCL_ERROR("[RegisterKernels] output-tree source group must be secondary"), HCCL_E_INTERNAL);
    }

    for (size_t i = 0; i < groups.size(); ++i) {
        std::shared_ptr<CcuKernelArgBase> kernelArg;
        const char *kernelName = nullptr;
        void *kernelFunc = nullptr;
        uint32_t roleFlags = groups[i].isPrimary != 0 ? KERNEL_ROLE_PRIMARY : 0U;
        if (IsV101SmallAlgorithm(algorithm)) {
            auto v101Arg = std::make_shared<CcuKernelArgV101Small>();
            CHK_RET(ConfigureV101SmallArg(param, groups, i, *v101Arg));
            kernelArg = v101Arg;
            kernelName = groups.size() == 1U ? "FinalCcuAllReduce" :
                                               (groups[i].actualDieId == 0U ? "FinalCcuAllReduceD0" :
                                                                             "FinalCcuAllReduceD1");
            kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuV101SmallKernel);
        } else if (algorithm == Algorithm2x8::OWNER_RSAG) {
            auto ownerArg = std::make_shared<CcuKernelArgOwnerRsag>();
            ownerArg->rankId = param.myRank;
            ownerArg->rankSize = param.rankSize;
            ownerArg->kernelIndex = static_cast<uint32_t>(i);
            ownerArg->kernelCount = static_cast<uint32_t>(groups.size());
            ownerArg->actualDieId = groups[i].actualDieId;
            ownerArg->isPrimary = groups[i].isPrimary;
            ownerArg->dataType = param.dataType;
            ownerArg->reduceType = param.reduceType;
            ownerArg->channelCount = static_cast<uint32_t>(groups[i].peers.size());
            for (size_t channelIndex = 0; channelIndex < groups[i].peers.size(); ++channelIndex) {
                ownerArg->channels[channelIndex] = groups[i].peers[channelIndex].handle;
                ownerArg->peerRanks[channelIndex] = groups[i].peers[channelIndex].info.peerRank;
                ownerArg->peerIsLocal[channelIndex] = groups[i].peers[channelIndex].info.isLocal;
            }
            kernelArg = ownerArg;
            kernelName = "CcuOwnerRsagKernel";
            kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuKernel);
        } else if (algorithm == Algorithm2x8::BFLY16_CLOS_4STEP) {
            auto bflyArg = std::make_shared<CcuKernelArgButterfly2x8>();
            bflyArg->rankId = param.myRank;
            bflyArg->kernelIndex = static_cast<uint32_t>(i);
            bflyArg->kernelCount = static_cast<uint32_t>(groups.size());
            bflyArg->actualDieId = groups[i].actualDieId;
            bflyArg->totalBytes = param.count * FP32_BYTES;
            bflyArg->scratchStride = (bflyArg->totalBytes + SCRATCH_ALIGNMENT - 1) & ~(SCRATCH_ALIGNMENT - 1);
            bflyArg->scratchAddr = reinterpret_cast<uint64_t>(resource.localBuffer.addr);
            bflyArg->scratchToken = butterflyScratchToken;
            bflyArg->dataType = param.dataType;
            bflyArg->reduceType = param.reduceType;
            bflyArg->channelCount = static_cast<uint32_t>(groups[i].peers.size());
            std::copy(butterflyPhaseOwner.begin(), butterflyPhaseOwner.end(), bflyArg->phaseOwnerKernel);
            uint32_t phaseMask = 0;
            for (size_t channelIndex = 0; channelIndex < groups[i].peers.size(); ++channelIndex) {
                const uint32_t phase = groups[i].peers[channelIndex].targetSlot;
                bflyArg->channels[channelIndex] = groups[i].peers[channelIndex].handle;
                bflyArg->phaseByChannel[channelIndex] = phase;
                phaseMask |= 1U << phase;
            }
            roleFlags |= phaseMask << KERNEL_ROLE_BFLY_PHASE_SHIFT;
            kernelArg = bflyArg;
            kernelName = "CcuButterfly2x8Kernel";
            kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuButterfly2x8Kernel);
        } else if (IsCrossLaneAlgorithm(algorithm) || IsP16MeshLaneAlgorithm(algorithm)) {
            auto crossArg = std::make_shared<CcuKernelArgCrossLane2x8>();
            crossArg->rankId = param.myRank;
            crossArg->kernelIndex = static_cast<uint32_t>(i);
            crossArg->kernelCount = static_cast<uint32_t>(groups.size());
            crossArg->actualDieId = groups[i].actualDieId;
            crossArg->localShardMask = groups.size() == 1 ? 0xFU : (i == 0 ? 0x5U : 0xAU);
            crossArg->localLaneCount = localLaneCount;
            crossArg->meshLaneMode = IsP16MeshLaneAlgorithm(algorithm) ? 1U : 0U;
            crossArg->directOutputAccumulator = IsCrossLaneDirectOutputAlgorithm(algorithm) ? 1U : 0U;
            crossArg->mode = algorithm == Algorithm2x8::CROSS_LANE_512M ? CROSS_LANE_MODE_PARALLEL_LOCAL : 0U;
            crossArg->dataType = param.dataType;
            crossArg->reduceType = param.reduceType;
            crossArg->channelCount = static_cast<uint32_t>(groups[i].peers.size());
            uint64_t maxOwnerBytes = 0;
            uint64_t unusedOffset = 0;
            for (uint32_t ownerRank = 0; ownerRank < param.rankSize; ++ownerRank) {
                uint64_t candidateBytes = 0;
                GetOwnerSlice(param.count, param.rankSize, ownerRank, unusedOffset, candidateBytes);
                maxOwnerBytes = std::max(maxOwnerBytes, candidateBytes);
            }
            crossArg->scratchStride = (maxOwnerBytes + SCRATCH_ALIGNMENT - 1) & ~(SCRATCH_ALIGNMENT - 1);
            GetOwnerSlice(param.count, param.rankSize, param.myRank, crossArg->ownerOffset, crossArg->ownerBytes);
            BuildLocalShards(algorithm, crossArg->ownerBytes, crossArg->ownerShardOffset, crossArg->ownerShardBytes);
            for (size_t channelIndex = 0; channelIndex < groups[i].peers.size(); ++channelIndex) {
                const ChannelPlan &peer = groups[i].peers[channelIndex];
                crossArg->channels[channelIndex] = peer.handle;
                crossArg->peerRanks[channelIndex] = peer.info.peerRank;
                crossArg->peerIsLocal[channelIndex] = peer.info.isLocal;
                crossArg->targetSlot[channelIndex] = peer.targetSlot;
                crossArg->targetLane[channelIndex] = peer.targetPlacement.lane;
                crossArg->targetLanePosition[channelIndex] = peer.targetPlacement.position;
                crossArg->targetLaneSize[channelIndex] = peer.targetPlacement.size;
                crossArg->sourceLane[channelIndex] = peer.sourcePlacement.lane;
                crossArg->sourceLanePosition[channelIndex] = peer.sourcePlacement.position;
                crossArg->sourceLaneSize[channelIndex] = peer.sourcePlacement.size;
                GetOwnerSlice(param.count, param.rankSize, peer.info.peerRank,
                    crossArg->targetOwnerOffset[channelIndex], crossArg->targetOwnerBytes[channelIndex]);
                BuildLocalShards(algorithm, crossArg->targetOwnerBytes[channelIndex],
                    crossArg->targetShardOffset[channelIndex], crossArg->targetShardBytes[channelIndex]);
            }
            roleFlags |= localLaneCount << KERNEL_ROLE_LANE_COUNT_SHIFT;
            kernelArg = crossArg;
            kernelName = "CcuCrossLane2x8Kernel";
            kernelFunc = algorithm == Algorithm2x8::CROSS_LANE_512M
                             ? reinterpret_cast<void *>(ops_hccl::CcuCrossLaneFourLane512MKernel)
                             : reinterpret_cast<void *>(ops_hccl::CcuCrossLane2x8Kernel);
        } else if (algorithm == Algorithm2x8::P4_DIRECT_RSAG_512K || algorithm == Algorithm2x8::P4_DIRECT_SLOT3_512M
                   || algorithm == Algorithm2x8::P4_DIRECT_SLOT3_400M4B || IsP12DirectRsagAlgorithm(algorithm)) {
            auto directArg = std::make_shared<CcuKernelArgDirectRsag>();
            directArg->rankId = param.myRank;
            directArg->rankSize = param.rankSize;
            directArg->kernelIndex = static_cast<uint32_t>(i);
            directArg->kernelCount = static_cast<uint32_t>(groups.size());
            directArg->actualDieId = groups[i].actualDieId;
            directArg->isPrimary = groups[i].isPrimary;
            directArg->preSyncOnce = IsP12PreSyncOnceAlgorithm(algorithm) ? 1U : 0U;
            directArg->earlyInit
                = algorithm == Algorithm2x8::P12_DIRECT_RSAG_EARLY_INIT_512K ? 1U : 0U;
            directArg->scratchAddr = reinterpret_cast<uint64_t>(resource.localBuffer.addr);
            directArg->scratchToken = staticScratchToken;
            directArg->dataType = param.dataType;
            directArg->reduceType = param.reduceType;
            uint64_t maxOwnerBytes = 0;
            uint64_t unusedOffset = 0;
            for (uint32_t owner = 0; owner < param.rankSize; ++owner) {
                uint64_t candidateBytes = 0;
                GetOwnerSlice(param.count, param.rankSize, owner, unusedOffset, candidateBytes);
                maxOwnerBytes = std::max(maxOwnerBytes, candidateBytes);
            }
            directArg->scratchStride = (maxOwnerBytes + SCRATCH_ALIGNMENT - 1) & ~(SCRATCH_ALIGNMENT - 1);
            GetOwnerSlice(param.count, param.rankSize, param.myRank, directArg->ownerOffset, directArg->ownerBytes);
            directArg->channelCount = static_cast<uint32_t>(groups[i].peers.size());
            for (size_t channelIndex = 0; channelIndex < groups[i].peers.size(); ++channelIndex) {
                const ChannelPlan &peer = groups[i].peers[channelIndex];
                directArg->channels[channelIndex] = peer.handle;
                directArg->peerRanks[channelIndex] = peer.info.peerRank;
                directArg->peerIsLocal[channelIndex] = peer.info.isLocal;
                directArg->targetSlot[channelIndex] = SourceSlot(param.myRank, peer.info.peerRank);
                GetOwnerSlice(param.count, param.rankSize, peer.info.peerRank,
                    directArg->targetOwnerOffset[channelIndex], directArg->targetOwnerBytes[channelIndex]);
            }
            kernelArg = directArg;
            kernelName = "CcuDirectRsagKernel";
            kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuDirectRsagKernel);
        } else if (IsP12OutputTreeAlgorithm(algorithm)) {
            auto outputTreeArg = std::make_shared<CcuKernelArgP12OutputTree>();
            outputTreeArg->rankId = param.myRank;
            outputTreeArg->rankSize = param.rankSize;
            outputTreeArg->kernelIndex = static_cast<uint32_t>(i);
            outputTreeArg->kernelCount = static_cast<uint32_t>(groups.size());
            outputTreeArg->actualDieId = groups[i].actualDieId;
            outputTreeArg->isPrimary = groups[i].isPrimary;
            outputTreeArg->balancedReduce = 1U;
            outputTreeArg->preSyncOnce = 1U;
            outputTreeArg->compactOutputTree = 1U;
            outputTreeArg->dualGroupReduce = 1U;
            outputTreeArg->rotatingReadReduce = 1U;
            outputTreeArg->groupOutputRoot = i == p12OutputTreeOutputGroup ? 1U : 0U;
            outputTreeArg->groupRootRank = p12OutputTreeGroupRoots[i];
            outputTreeArg->directOutputSourceRank = param.myRank == 0 ? 1U : 0U;
            outputTreeArg->tempOutputSourceRank = param.myRank <= 1 ? 2U : 1U;
            outputTreeArg->scratchAddr = reinterpret_cast<uint64_t>(resource.localBuffer.addr);
            outputTreeArg->scratchToken = staticScratchToken;
            outputTreeArg->dataType = param.dataType;
            outputTreeArg->reduceType = param.reduceType;

            const std::vector<uint32_t> &aRanks
                = topologyView.localRanks.size() == P12_ALLPAIRS_A_RANK_COUNT ? topologyView.localRanks
                                                                             : topologyView.remoteRanks;
            const std::vector<uint32_t> &bRanks
                = topologyView.localRanks.size() == P12_ALLPAIRS_B_RANK_COUNT ? topologyView.localRanks
                                                                             : topologyView.remoteRanks;
            CHK_PRT_RET(aRanks.size() != P12_ALLPAIRS_A_RANK_COUNT
                            || bRanks.size() != P12_ALLPAIRS_B_RANK_COUNT,
                HCCL_ERROR("[RegisterKernels] invalid output-tree P12 shape A=%zu B=%zu", aRanks.size(),
                    bRanks.size()),
                HCCL_E_INTERNAL);
            std::copy(aRanks.begin(), aRanks.end(), outputTreeArg->treeRanks);
            std::copy(bRanks.begin(), bRanks.end(), outputTreeArg->treeRanks + P12_ALLPAIRS_A_RANK_COUNT);

            uint64_t segmentOffsetElements[P12_ALLPAIRS_SEGMENT_COUNT]{};
            uint64_t segmentElements[P12_ALLPAIRS_SEGMENT_COUNT]{};
            GetP12AllPairsSegments(
                algorithm, param.count, segmentOffsetElements, segmentElements, outputTreeArg->segmentCount);
            uint64_t maxOwnerBytes = 0;
            for (uint32_t segment = 0; segment < outputTreeArg->segmentCount; ++segment) {
                outputTreeArg->segmentBytes[segment] = segmentElements[segment] * FP32_BYTES;
                uint64_t relativeOffset = 0;
                GetOwnerSlice(segmentElements[segment], param.rankSize, param.myRank, relativeOffset,
                    outputTreeArg->ownerBytes[segment]);
                outputTreeArg->ownerOffset[segment] = segmentOffsetElements[segment] * FP32_BYTES + relativeOffset;
                maxOwnerBytes = std::max(maxOwnerBytes, outputTreeArg->ownerBytes[segment]);
            }
            outputTreeArg->scratchStride = (maxOwnerBytes + SCRATCH_ALIGNMENT - 1) & ~(SCRATCH_ALIGNMENT - 1);
            uint64_t unusedTempBytes = 0;
            GetOwnerSlice(param.count, param.rankSize, P12TempOutputRank(param.myRank),
                outputTreeArg->tempOutputOffset, unusedTempBytes);

            outputTreeArg->channelCount = static_cast<uint32_t>(groups[i].peers.size());
            std::array<size_t, MAX_RANK_SIZE> peerOrder{};
            for (size_t channelIndex = 0; channelIndex < groups[i].peers.size(); ++channelIndex) {
                peerOrder[channelIndex] = channelIndex;
            }
            std::sort(peerOrder.begin(), peerOrder.begin() + groups[i].peers.size(),
                [&groups, i](size_t lhs, size_t rhs) {
                    const auto &left = groups[i].peers[lhs].info;
                    const auto &right = groups[i].peers[rhs].info;
                    return left.bwCoeff != right.bwCoeff ? left.bwCoeff < right.bwCoeff
                                                        : left.peerRank < right.peerRank;
                });
            for (size_t channelIndex = 0; channelIndex < groups[i].peers.size(); ++channelIndex) {
                const ChannelPlan &peer = groups[i].peers[peerOrder[channelIndex]];
                outputTreeArg->channels[channelIndex] = peer.handle;
                outputTreeArg->peerRanks[channelIndex] = peer.info.peerRank;
                outputTreeArg->targetSlot[channelIndex] = SourceSlot(param.myRank, peer.info.peerRank);
                uint64_t unusedTargetTempBytes = 0;
                GetOwnerSlice(param.count, param.rankSize, P12TempOutputRank(peer.info.peerRank),
                    outputTreeArg->targetTempOutputOffset[channelIndex], unusedTargetTempBytes);
                for (uint32_t segment = 0; segment < outputTreeArg->segmentCount; ++segment) {
                    uint64_t relativeOffset = 0;
                    GetOwnerSlice(segmentElements[segment], param.rankSize, peer.info.peerRank, relativeOffset,
                        outputTreeArg->targetOwnerBytes[segment][channelIndex]);
                    outputTreeArg->targetOwnerOffset[segment][channelIndex]
                        = segmentOffsetElements[segment] * FP32_BYTES + relativeOffset;
                }
            }
            kernelArg = outputTreeArg;
            kernelName = "CcuP12AllPairsKernel";
            kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuP12OutputTreeKernel);
        } else if (IsP12AllPairsAlgorithm(algorithm)) {
            auto allPairsArg = std::make_shared<CcuKernelArgP12AllPairs>();
            allPairsArg->rankId = param.myRank;
            allPairsArg->rankSize = param.rankSize;
            allPairsArg->kernelIndex = static_cast<uint32_t>(i);
            allPairsArg->kernelCount = static_cast<uint32_t>(groups.size());
            allPairsArg->actualDieId = groups[i].actualDieId;
            allPairsArg->isPrimary = groups[i].isPrimary;
            allPairsArg->balancedReduce = IsP12AllPairsBalancedAlgorithm(algorithm) ? 1U : 0U;
            allPairsArg->preSyncOnce = IsP12PreSyncOnceAlgorithm(algorithm) ? 1U : 0U;
            allPairsArg->earlyInit
                = algorithm == Algorithm2x8::P12_ALLPAIRS_EARLY_INIT_512M ? 1U : 0U;
            allPairsArg->closAccumulate = IsP12ClosAccumulateAlgorithm(algorithm) ? 1U : 0U;
            allPairsArg->isClosOwner = IsP12ClosAccumulateAlgorithm(algorithm) && i == p12ClosOwnerGroup ? 1U : 0U;
            allPairsArg->scratchAddr = reinterpret_cast<uint64_t>(resource.localBuffer.addr);
            allPairsArg->scratchToken = staticScratchToken;
            allPairsArg->dataType = param.dataType;
            allPairsArg->reduceType = param.reduceType;

            const std::vector<uint32_t> &aRanks
                = topologyView.localRanks.size() == P12_ALLPAIRS_A_RANK_COUNT ? topologyView.localRanks
                                                                             : topologyView.remoteRanks;
            const std::vector<uint32_t> &bRanks
                = topologyView.localRanks.size() == P12_ALLPAIRS_B_RANK_COUNT ? topologyView.localRanks
                                                                             : topologyView.remoteRanks;
            CHK_PRT_RET(aRanks.size() != P12_ALLPAIRS_A_RANK_COUNT
                            || bRanks.size() != P12_ALLPAIRS_B_RANK_COUNT,
                HCCL_ERROR("[RegisterKernels] invalid P12 balanced tree shape A=%zu B=%zu", aRanks.size(),
                    bRanks.size()),
                HCCL_E_INTERNAL);
            std::copy(aRanks.begin(), aRanks.end(), allPairsArg->treeRanks);
            std::copy(bRanks.begin(), bRanks.end(), allPairsArg->treeRanks + P12_ALLPAIRS_A_RANK_COUNT);

            uint64_t segmentOffsetElements[P12_ALLPAIRS_SEGMENT_COUNT]{};
            uint64_t segmentElements[P12_ALLPAIRS_SEGMENT_COUNT]{};
            GetP12AllPairsSegments(
                algorithm, param.count, segmentOffsetElements, segmentElements, allPairsArg->segmentCount);
            uint64_t maxOwnerBytes = 0;
            for (uint32_t segment = 0; segment < allPairsArg->segmentCount; ++segment) {
                allPairsArg->segmentBytes[segment] = segmentElements[segment] * FP32_BYTES;
                uint64_t relativeOffset = 0;
                GetOwnerSlice(segmentElements[segment], param.rankSize, param.myRank, relativeOffset,
                    allPairsArg->ownerBytes[segment]);
                allPairsArg->ownerOffset[segment] = segmentOffsetElements[segment] * FP32_BYTES + relativeOffset;
                maxOwnerBytes = std::max(maxOwnerBytes, allPairsArg->ownerBytes[segment]);
            }
            allPairsArg->scratchStride = (maxOwnerBytes + SCRATCH_ALIGNMENT - 1) & ~(SCRATCH_ALIGNMENT - 1);
            allPairsArg->channelCount = static_cast<uint32_t>(groups[i].peers.size());
            for (size_t channelIndex = 0; channelIndex < groups[i].peers.size(); ++channelIndex) {
                const ChannelPlan &peer = groups[i].peers[channelIndex];
                allPairsArg->channels[channelIndex] = peer.handle;
                allPairsArg->peerRanks[channelIndex] = peer.info.peerRank;
                allPairsArg->peerIsLocal[channelIndex] = peer.info.isLocal;
                allPairsArg->targetSlot[channelIndex]
                    = allPairsArg->closAccumulate != 0 && peer.info.isLocal != 0
                          ? SameServerSourceSlot(topologyView.localRanks, param.myRank, peer.info.peerRank)
                          : SourceSlot(param.myRank, peer.info.peerRank);
                if (allPairsArg->closAccumulate != 0 && peer.info.isLocal == 0) {
                    CHK_RET(
                        GetP12ClosPhase(topologyView, peer.info.peerRank, allPairsArg->peerClosPhase[channelIndex]));
                }
                for (uint32_t segment = 0; segment < allPairsArg->segmentCount; ++segment) {
                    uint64_t relativeOffset = 0;
                    GetOwnerSlice(segmentElements[segment], param.rankSize, peer.info.peerRank, relativeOffset,
                        allPairsArg->targetOwnerBytes[segment][channelIndex]);
                    allPairsArg->targetOwnerOffset[segment][channelIndex]
                        = segmentOffsetElements[segment] * FP32_BYTES + relativeOffset;
                }
            }
            kernelArg = allPairsArg;
            kernelName = "CcuP12AllPairsKernel";
            kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuP12AllPairsKernel);
        } else if (algorithm == Algorithm2x8::P4_ROTATE3_512M || algorithm == Algorithm2x8::P4_ROTATE3_400M4B
                   || algorithm == Algorithm2x8::P4_ROTATE3_UNIQUE_NOTIFY_512M
                   || algorithm == Algorithm2x8::P4_ROTATE3_UNIQUE_NOTIFY_400M4B
                   || algorithm == Algorithm2x8::P4_ROTATE3_DIRECT_OUTPUT_512M
                   || algorithm == Algorithm2x8::P4_ROTATE3_DIRECT_OUTPUT_400M4B
                   || IsP4OutputAccumulateAlgorithm(algorithm) || IsP4PullR3Algorithm(algorithm)) {
            auto rotateArg = std::make_shared<CcuKernelArgP4Rotate3>();
            rotateArg->rankId = param.myRank;
            rotateArg->kernelIndex = static_cast<uint32_t>(i);
            rotateArg->kernelCount = static_cast<uint32_t>(groups.size());
            rotateArg->actualDieId = groups[i].actualDieId;
            rotateArg->isPrimary = groups[i].isPrimary;
            rotateArg->uniquePhaseNotify = algorithm == Algorithm2x8::P4_ROTATE3_UNIQUE_NOTIFY_512M
                                                   || algorithm == Algorithm2x8::P4_ROTATE3_UNIQUE_NOTIFY_400M4B
                                                   || algorithm == Algorithm2x8::P4_ROTATE3_DIRECT_OUTPUT_512M
                                                   || algorithm == Algorithm2x8::P4_ROTATE3_DIRECT_OUTPUT_400M4B
                                                   || IsP4OutputAccumulateAlgorithm(algorithm)
                                                   || IsP4PullR3Algorithm(algorithm)
                                               ? 1U
                                               : 0U;
            rotateArg->directOutputAccumulator = algorithm == Algorithm2x8::P4_ROTATE3_DIRECT_OUTPUT_512M
                                                         || algorithm == Algorithm2x8::P4_ROTATE3_DIRECT_OUTPUT_400M4B
                                                         || IsP4OutputAccumulateAlgorithm(algorithm)
                                                         || IsP4PullR3Algorithm(algorithm)
                                                     ? 1U
                                                     : 0U;
            rotateArg->ownerPull = IsP4PullR3Algorithm(algorithm) ? 1U : 0U;
            rotateArg->fullSliceAccumulate = IsP4OutputAccumulateAlgorithm(algorithm) ? 1U : 0U;
            rotateArg->phaseChain = IsP4OutputAccumulateChainAlgorithm(algorithm) ? 1U : 0U;
            rotateArg->wave2Accumulate = IsP4OutputAccumulateWave2Algorithm(algorithm) ? 1U : 0U;
            rotateArg->scratchAddr = reinterpret_cast<uint64_t>(resource.localBuffer.addr);
            rotateArg->scratchToken = staticScratchToken;
            rotateArg->dataType = param.dataType;
            rotateArg->reduceType = param.reduceType;
            uint64_t maxOwnerBytes = 0;
            uint64_t unusedOffset = 0;
            for (uint32_t owner = 0; owner < param.rankSize; ++owner) {
                uint64_t candidateBytes = 0;
                GetOwnerSlice(param.count, param.rankSize, owner, unusedOffset, candidateBytes);
                maxOwnerBytes = std::max(maxOwnerBytes, candidateBytes);
            }
            rotateArg->scratchStride = (maxOwnerBytes + SCRATCH_ALIGNMENT - 1) & ~(SCRATCH_ALIGNMENT - 1);
            GetOwnerSlice(param.count, param.rankSize, param.myRank, rotateArg->ownerOffset, rotateArg->ownerBytes);
            rotateArg->channelCount = static_cast<uint32_t>(groups[i].peers.size());
            for (size_t channelIndex = 0; channelIndex < groups[i].peers.size(); ++channelIndex) {
                const ChannelPlan &peer = groups[i].peers[channelIndex];
                rotateArg->channels[channelIndex] = peer.handle;
                rotateArg->peerRanks[channelIndex] = peer.info.peerRank;
                rotateArg->targetPosition[channelIndex]
                    = rotateArg->ownerPull != 0 ? SourceSlot(peer.info.peerRank, param.myRank)
                                                : SourceSlot(param.myRank, peer.info.peerRank);
                GetOwnerSlice(param.count, param.rankSize, peer.info.peerRank,
                    rotateArg->targetOwnerOffset[channelIndex], rotateArg->targetOwnerBytes[channelIndex]);
            }
            kernelArg = rotateArg;
            kernelName = "CcuP4Rotate3Kernel";
            kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuP4Rotate3Kernel);
        } else {
            CHK_PRT_RET(algorithm != Algorithm2x8::P12_HIER8_512M && algorithm != Algorithm2x8::P12_HIER8_400M4B,
                HCCL_ERROR("[RegisterKernels] unsupported algorithm %u", static_cast<uint32_t>(algorithm)),
                HCCL_E_INTERNAL);
            auto hierArg = std::make_shared<CcuKernelArgP12Hier8>();
            hierArg->rankId = param.myRank;
            hierArg->kernelIndex = static_cast<uint32_t>(i);
            hierArg->kernelCount = static_cast<uint32_t>(groups.size());
            hierArg->actualDieId = groups[i].actualDieId;
            hierArg->isPrimary = groups[i].isPrimary;
            hierArg->localSize = static_cast<uint32_t>(topologyView.localRanks.size());
            hierArg->localIndex = topologyView.localIndex;
            hierArg->localLaneCount = hierArg->localSize == 8 ? 2U : 1U;
            hierArg->ownedShardCount = hierArg->localSize == 8 ? 1U : 2U;
            hierArg->ownedShardId[0] = topologyView.localIndex;
            if (hierArg->ownedShardCount == 2) {
                hierArg->ownedShardId[1] = topologyView.localIndex + 4U;
            }
            BuildGlobalHierShards(param.count, hierArg->shardOffset, hierArg->shardBytes);
            uint64_t maxShardBytes = 0;
            for (uint32_t shard = 0; shard < HIER_GLOBAL_SHARD_COUNT; ++shard) {
                maxShardBytes = std::max(maxShardBytes, hierArg->shardBytes[shard]);
            }
            hierArg->scratchStride = (maxShardBytes + SCRATCH_ALIGNMENT - 1) & ~(SCRATCH_ALIGNMENT - 1);
            hierArg->scratchAddr = reinterpret_cast<uint64_t>(resource.localBuffer.addr);
            hierArg->scratchToken = staticScratchToken;
            hierArg->dataType = param.dataType;
            hierArg->reduceType = param.reduceType;
            hierArg->channelCount = static_cast<uint32_t>(groups[i].peers.size());
            for (size_t channelIndex = 0; channelIndex < groups[i].peers.size(); ++channelIndex) {
                const ChannelPlan &peer = groups[i].peers[channelIndex];
                hierArg->channels[channelIndex] = peer.handle;
                hierArg->peerRanks[channelIndex] = peer.info.peerRank;
                hierArg->peerIsLocal[channelIndex] = peer.info.isLocal;
                const std::vector<uint32_t> &peerGroup
                    = peer.info.isLocal != 0 ? topologyView.localRanks : topologyView.remoteRanks;
                const auto peerIt = std::lower_bound(peerGroup.begin(), peerGroup.end(), peer.info.peerRank);
                CHK_PRT_RET(peerIt == peerGroup.end() || *peerIt != peer.info.peerRank,
                    HCCL_ERROR("[RegisterKernels] hierarchy peer %u missing from group", peer.info.peerRank),
                    HCCL_E_INTERNAL);
                hierArg->peerLocalIndex[channelIndex] = static_cast<uint32_t>(std::distance(peerGroup.begin(), peerIt));
                if (peer.info.isLocal != 0) {
                    LanePlacement targetPlacement;
                    LanePlacement sourcePlacement;
                    CHK_RET(GetLocalLanePlacement(topologyView, param.myRank, peer.info.peerRank, targetPlacement));
                    CHK_RET(GetLocalLanePlacement(topologyView, peer.info.peerRank, param.myRank, sourcePlacement));
                    hierArg->targetLane[channelIndex] = targetPlacement.lane;
                    hierArg->targetLanePosition[channelIndex] = targetPlacement.position;
                    hierArg->targetLaneSize[channelIndex] = targetPlacement.size;
                    hierArg->sourceLane[channelIndex] = sourcePlacement.lane;
                    hierArg->sourceLaneSize[channelIndex] = sourcePlacement.size;
                } else if (hierArg->localSize == 4) {
                    hierArg->crossOwnedIndex[channelIndex] = hierArg->peerLocalIndex[channelIndex] >= 4 ? 1U : 0U;
                }
            }
            kernelArg = hierArg;
            kernelName = "CcuP12Hier8Kernel";
            kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuP12Hier8Kernel);
        }

        const void *registrationArgs[] = {kernelArg.get()};
        const uint32_t reservedDieId = IsV101SmallAlgorithm(algorithm) ? groups[i].actualDieId : 0U;
        ccuRet = HcommCcuKernelRegister(
            insHandle, reservedDieId, kernelName, kernelFunc, registrationArgs, 1, &resource.ccuKernels[i]);
        if (ccuRet != CCU_SUCCESS) {
            HCCL_ERROR("[RegisterKernels] kernel %zu register failed, ret %d", i, ccuRet);
            (void)HcommCcuKernelRegisterEnd(insHandle);
            return ConvertCcuToHccl(ccuRet);
        }

        resource.kernelMeta[i]
            = KernelLaunchMeta{groups[i].actualDieId, static_cast<uint32_t>(groups[i].peers.size()), roleFlags};
        kernelArgs.push_back(std::move(kernelArg));
    }

    ccuRet = HcommCcuKernelRegisterEnd(insHandle);
    if (ccuRet != CCU_SUCCESS) {
        HCCL_ERROR("[RegisterKernels] register end failed, ret %d", ccuRet);
        return ConvertCcuToHccl(ccuRet);
    }
    return HCCL_SUCCESS;
}

HcclResult CreateResources(HcclComm comm, const OpParam &param, const TopologyView &topologyView,
    Algorithm2x8 algorithm, AlgResourceCtx &resource)
{
    resource.rankSize = param.rankSize;
    resource.topology = topologyView.topology;
    resource.version = ResourceVersion(algorithm);

    void *cclBufferAddr = nullptr;
    uint64_t cclBufferSize = 0;
    CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
    CHK_PRT_RET(cclBufferAddr == nullptr || cclBufferSize == 0,
        HCCL_ERROR("[CreateResources] invalid local HCCL buffer"), HCCL_E_INTERNAL);
    resource.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};

    std::vector<DieGroup> groups;
    uint32_t localLaneCount = 0;
    if (IsV101SmallAlgorithm(algorithm)) {
        CHK_RET(AcquireV101SmallChannels(comm, param, topologyView, groups));
    } else {
        CHK_RET(AcquireChannels(comm, param, topologyView, algorithm, groups, localLaneCount));
    }
    if (IsV101SmallAlgorithm(algorithm)) {
        const uint64_t maxSliceBytes = param.rankSize == 12U ? 11U * 4096U : BYTES_512K / param.rankSize;
        const uint64_t scratchSlotCount
            = IsV109R16SmallAlgorithm(algorithm) && param.inputPtr != param.outputPtr ? 3U : 2U;
        CHK_PRT_RET(maxSliceBytes > resource.localBuffer.size / scratchSlotCount,
            HCCL_ERROR("[CreateResources] small-message scratch requires %lu bytes, only %lu available",
                scratchSlotCount * maxSliceBytes, resource.localBuffer.size),
            HCCL_E_MEMORY);
    } else if (algorithm == Algorithm2x8::BFLY16_CLOS_4STEP) {
        const uint64_t stride = (param.count * FP32_BYTES + SCRATCH_ALIGNMENT - 1) & ~(SCRATCH_ALIGNMENT - 1);
        CHK_PRT_RET(stride > std::numeric_limits<uint64_t>::max() / 3 || stride * 3 > resource.localBuffer.size,
            HCCL_ERROR("[CreateResources] butterfly scratch requires %lu bytes, only %lu available", stride * 3,
                resource.localBuffer.size),
            HCCL_E_MEMORY);
    } else if (IsCrossLaneAlgorithm(algorithm) || IsP16MeshLaneAlgorithm(algorithm)) {
        uint64_t maxOwnerBytes = 0;
        uint64_t ownerOffset = 0;
        for (uint32_t owner = 0; owner < param.rankSize; ++owner) {
            uint64_t ownerBytes = 0;
            GetOwnerSlice(param.count, param.rankSize, owner, ownerOffset, ownerBytes);
            maxOwnerBytes = std::max(maxOwnerBytes, ownerBytes);
        }
        const uint64_t stride = (maxOwnerBytes + SCRATCH_ALIGNMENT - 1) & ~(SCRATCH_ALIGNMENT - 1);
        const uint64_t slotCount
            = IsP16MeshLaneAlgorithm(algorithm) ? MESH_LANE_MAX_COUNT : CROSS_LANE_SAME_SLOT_COUNT + localLaneCount;
        CHK_PRT_RET(
            stride > std::numeric_limits<uint64_t>::max() / slotCount || stride * slotCount > resource.localBuffer.size,
            HCCL_ERROR("[CreateResources] cross-lane scratch requires %lu bytes (%lu slots), only %lu available",
                stride * slotCount, slotCount, resource.localBuffer.size),
            HCCL_E_MEMORY);
    } else if (algorithm == Algorithm2x8::P4_DIRECT_RSAG_512K || algorithm == Algorithm2x8::P4_DIRECT_SLOT3_512M
               || algorithm == Algorithm2x8::P4_DIRECT_SLOT3_400M4B || IsP12DirectRsagAlgorithm(algorithm)) {
        uint64_t maxOwnerBytes = 0;
        uint64_t ownerOffset = 0;
        for (uint32_t owner = 0; owner < param.rankSize; ++owner) {
            uint64_t ownerBytes = 0;
            GetOwnerSlice(param.count, param.rankSize, owner, ownerOffset, ownerBytes);
            maxOwnerBytes = std::max(maxOwnerBytes, ownerBytes);
        }
        const uint64_t stride = (maxOwnerBytes + SCRATCH_ALIGNMENT - 1) & ~(SCRATCH_ALIGNMENT - 1);
        const uint64_t slotCount = param.rankSize - 1U;
        CHK_PRT_RET(
            stride > std::numeric_limits<uint64_t>::max() / slotCount || stride * slotCount > resource.localBuffer.size,
            HCCL_ERROR("[CreateResources] direct-RSAG scratch requires %lu bytes (%lu slots), only %lu available",
                stride * slotCount, slotCount, resource.localBuffer.size),
            HCCL_E_MEMORY);
    } else if (IsP12AllPairsAlgorithm(algorithm)) {
        uint64_t segmentOffsetElements[P12_ALLPAIRS_SEGMENT_COUNT]{};
        uint64_t segmentElements[P12_ALLPAIRS_SEGMENT_COUNT]{};
        uint32_t segmentCount = 0;
        GetP12AllPairsSegments(algorithm, param.count, segmentOffsetElements, segmentElements, segmentCount);
        uint64_t maxOwnerBytes = 0;
        for (uint32_t segment = 0; segment < segmentCount; ++segment) {
            CHK_PRT_RET(segmentElements[segment] == 0,
                HCCL_ERROR("[CreateResources] P12 All-Pairs segment %u is empty", segment), HCCL_E_INTERNAL);
            for (uint32_t owner = 0; owner < param.rankSize; ++owner) {
                uint64_t ownerOffset = 0;
                uint64_t ownerBytes = 0;
                GetOwnerSlice(segmentElements[segment], param.rankSize, owner, ownerOffset, ownerBytes);
                CHK_PRT_RET(ownerBytes == 0 || ownerBytes >= MAX_DATA_SIZE,
                    HCCL_ERROR("[CreateResources] P12 All-Pairs owner transfer %lu is outside (0, 256 MiB)",
                        ownerBytes),
                    HCCL_E_NOT_SUPPORT);
                maxOwnerBytes = std::max(maxOwnerBytes, ownerBytes);
            }
        }
        const uint64_t stride = (maxOwnerBytes + SCRATCH_ALIGNMENT - 1) & ~(SCRATCH_ALIGNMENT - 1);
        const uint64_t slotCount = IsP12OutputTreeAlgorithm(algorithm)
                                       ? 1U
                                       : (IsP12ClosAccumulateAlgorithm(algorithm)
                                                 ? P12_ALLPAIRS_A_RANK_COUNT - 1U
                                                 : param.rankSize - 1U);
        CHK_PRT_RET(
            stride > std::numeric_limits<uint64_t>::max() / slotCount || stride * slotCount > resource.localBuffer.size,
            HCCL_ERROR("[CreateResources] P12 All-Pairs scratch requires %lu bytes (%lu slots), only %lu available",
                stride * slotCount, slotCount, resource.localBuffer.size),
            HCCL_E_MEMORY);
    } else if (algorithm == Algorithm2x8::P4_ROTATE3_512M || algorithm == Algorithm2x8::P4_ROTATE3_400M4B
               || algorithm == Algorithm2x8::P4_ROTATE3_UNIQUE_NOTIFY_512M
               || algorithm == Algorithm2x8::P4_ROTATE3_UNIQUE_NOTIFY_400M4B
               || algorithm == Algorithm2x8::P4_ROTATE3_DIRECT_OUTPUT_512M
               || algorithm == Algorithm2x8::P4_ROTATE3_DIRECT_OUTPUT_400M4B || IsP4OutputAccumulateAlgorithm(algorithm)
               || IsP4PullR3Algorithm(algorithm)) {
        uint64_t maxOwnerBytes = 0;
        uint64_t ownerOffset = 0;
        for (uint32_t owner = 0; owner < param.rankSize; ++owner) {
            uint64_t ownerBytes = 0;
            GetOwnerSlice(param.count, param.rankSize, owner, ownerOffset, ownerBytes);
            maxOwnerBytes = std::max(maxOwnerBytes, ownerBytes);
        }
        const uint64_t stride = (maxOwnerBytes + SCRATCH_ALIGNMENT - 1) & ~(SCRATCH_ALIGNMENT - 1);
        CHK_PRT_RET(stride > resource.localBuffer.size,
            HCCL_ERROR("[CreateResources] P4 rotate scratch requires %lu bytes, only %lu available", stride,
                resource.localBuffer.size),
            HCCL_E_MEMORY);
    } else if (algorithm == Algorithm2x8::P12_HIER8_512M || algorithm == Algorithm2x8::P12_HIER8_400M4B) {
        uint64_t shardOffsets[HIER_GLOBAL_SHARD_COUNT]{};
        uint64_t shardBytes[HIER_GLOBAL_SHARD_COUNT]{};
        BuildGlobalHierShards(param.count, shardOffsets, shardBytes);
        uint64_t maxShardBytes = 0;
        for (uint32_t shard = 0; shard < HIER_GLOBAL_SHARD_COUNT; ++shard) {
            maxShardBytes = std::max(maxShardBytes, shardBytes[shard]);
        }
        const uint64_t stride = (maxShardBytes + SCRATCH_ALIGNMENT - 1) & ~(SCRATCH_ALIGNMENT - 1);
        CHK_PRT_RET(stride > std::numeric_limits<uint64_t>::max() / HIER_TOTAL_SLOT_COUNT
                        || stride * HIER_TOTAL_SLOT_COUNT > resource.localBuffer.size,
            HCCL_ERROR("[CreateResources] P12 hierarchy scratch requires %lu bytes (%u slots), only %lu available",
                stride * HIER_TOTAL_SLOT_COUNT, HIER_TOTAL_SLOT_COUNT, resource.localBuffer.size),
            HCCL_E_MEMORY);
    }
    if (groups.size() > 1) {
        resource.extraThreads.resize(groups.size() - 1);
        CHK_RET(HcclThreadAcquire(comm, CommEngine::COMM_ENGINE_CCU,
            static_cast<uint32_t>(resource.extraThreads.size()),
            IsV101SmallAlgorithm(algorithm) ? V101_THREAD_NOTIFY_NUM : THREAD_NOTIFY_NUM,
            resource.extraThreads.data()));
    }
    CHK_RET(RegisterKernels(comm, param, topologyView, algorithm, groups, localLaneCount, resource));
    HCCL_INFO("[ResourceShape] topology=%s bytes=%lu algorithm=%s version=%u kernels=%zu lanes=%u",
        TopologyName(topologyView.topology), param.count * FP32_BYTES, AlgorithmName(algorithm), resource.version,
        resource.ccuKernels.size(), localLaneCount);
    return HCCL_SUCCESS;
}
} // namespace

HcclResult HcclAllReduce(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op,
    HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);
    CHK_PRT_RET(
        dataType != HCCL_DATA_TYPE_FP32, HCCL_ERROR("[HcclAllReduce] only FP32 is supported"), HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(op != HCCL_REDUCE_SUM, HCCL_ERROR("[HcclAllReduce] only SUM is supported"), HCCL_E_NOT_SUPPORT);
    if (count == 0) {
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(count > std::numeric_limits<uint64_t>::max() / FP32_BYTES,
        HCCL_ERROR("[HcclAllReduce] element count overflows byte size"), HCCL_E_PARA);
    const uint64_t totalBytes = count * FP32_BYTES;

    OpParam param;
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = count;
    param.dataType = dataType;
    param.reduceType = op;
    param.opType = HcclCMDType::HCCL_CMD_ALLREDUCE;
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));

    FinalTopology expectedTopology;
    std::vector<uint32_t> expectedInstanceSizes;
    CHK_RET(GetExpectedTopology(param.rankSize, expectedTopology, expectedInstanceSizes));
    const Algorithm2x8 algorithm
        = SelectCrossLaneInplaceFallback(SelectAlgorithm(param.rankSize, totalBytes), sendBuf, recvBuf);
    const char *tagFormat = "ar_ccu_p%u_owner_rsag_v6";
    if (IsV101SmallAlgorithm(algorithm)) {
        tagFormat = "hccl_final_ccu_v101_r%u_512k_k1";
    } else if (algorithm == Algorithm2x8::BFLY16_CLOS_4STEP) {
        tagFormat = "ar_ccu_p16_bfly512k_v8";
    } else if (algorithm == Algorithm2x8::CROSS_LANE_512M) {
        tagFormat = "ar_ccu_p16_crosslane512m_fourlane_v24";
    } else if (algorithm == Algorithm2x8::CROSS_LANE_400M4B) {
        tagFormat = "ar_ccu_p16_crosslane400m4b_v7";
    } else if (algorithm == Algorithm2x8::CROSS_LANE_DIRECT_OUTPUT_400M4B) {
        tagFormat = "ar_ccu_p16_crosslane_output400m4b_v23";
    } else if (algorithm == Algorithm2x8::P4_DIRECT_RSAG_512K) {
        tagFormat = "ar_ccu_p4_direct512k_v9";
    } else if (algorithm == Algorithm2x8::P4_DIRECT_SLOT3_512M) {
        tagFormat = "ar_ccu_p4_directslot3_512m_v16";
    } else if (algorithm == Algorithm2x8::P4_DIRECT_SLOT3_400M4B) {
        tagFormat = "ar_ccu_p4_directslot3_400m4b_v16";
    } else if (algorithm == Algorithm2x8::P4_ROTATE3_UNIQUE_NOTIFY_512M) {
        tagFormat = "ar_ccu_p4_rotate3_unotify_512m_v17";
    } else if (algorithm == Algorithm2x8::P4_ROTATE3_UNIQUE_NOTIFY_400M4B) {
        tagFormat = "ar_ccu_p4_rotate3_unotify_400m4b_v17";
    } else if (algorithm == Algorithm2x8::P4_ROTATE3_DIRECT_OUTPUT_512M) {
        tagFormat = "ar_ccu_p4_rotate3_output_512m_v21";
    } else if (algorithm == Algorithm2x8::P4_ROTATE3_DIRECT_OUTPUT_400M4B) {
        tagFormat = "ar_ccu_p4_rotate3_output_400m4b_v21";
    } else if (algorithm == Algorithm2x8::P4_PULL_R3_512M) {
        tagFormat = "ar_ccu_p4_pull_r3_512m_v26";
    } else if (algorithm == Algorithm2x8::P4_PULL_R3_400M4B) {
        tagFormat = "ar_ccu_p4_pull_r3_400m4b_v26";
    } else if (algorithm == Algorithm2x8::P4_ROTATE3_512M) {
        tagFormat = "ar_ccu_p4_rotate512m_v10";
    } else if (algorithm == Algorithm2x8::P4_ROTATE3_400M4B) {
        tagFormat = "ar_ccu_p4_rotate400m4b_v10";
    } else if (algorithm == Algorithm2x8::P12_DIRECT_RSAG_512K) {
        tagFormat = "ar_ccu_p12_direct512k_v9";
    } else if (algorithm == Algorithm2x8::P12_HIER8_512M) {
        tagFormat = "ar_ccu_p12_hier8_512m_v11";
    } else if (algorithm == Algorithm2x8::P12_HIER8_400M4B) {
        tagFormat = "ar_ccu_p12_hier8_400m4b_v11";
    } else if (algorithm == Algorithm2x8::P12_ALLPAIRS_512M) {
        tagFormat = "ar_ccu_p12_allpairs512m_v15";
    } else if (algorithm == Algorithm2x8::P12_ALLPAIRS_400M4B) {
        tagFormat = "ar_ccu_p12_allpairs400m4b_v15";
    } else if (algorithm == Algorithm2x8::P12_ALLPAIRS_BALANCED_512M) {
        tagFormat = "ar_ccu_p12_allpairs_balanced512m_v18";
    } else if (algorithm == Algorithm2x8::P12_ALLPAIRS_BALANCED_400M4B) {
        tagFormat = "ar_ccu_p12_allpairs_balanced400m4b_v18";
    } else if (algorithm == Algorithm2x8::P12_DIRECT_RSAG_PRESYNC_ONCE_512K) {
        tagFormat = "ar_ccu_p12_direct_presync512k_v25";
    } else if (algorithm == Algorithm2x8::P12_ALLPAIRS_BALANCED_PRESYNC_ONCE_512M) {
        tagFormat = "ar_ccu_p12_allpairs_presync512m_v25";
    } else if (algorithm == Algorithm2x8::P12_ALLPAIRS_BALANCED_PRESYNC_ONCE_400M4B) {
        tagFormat = "ar_ccu_p12_allpairs_presync400m4b_v25";
    } else if (algorithm == Algorithm2x8::P12_DIRECT_RSAG_EARLY_INIT_512K) {
        tagFormat = "ar_ccu_p12_direct_early512k_v31";
    } else if (algorithm == Algorithm2x8::P12_ALLPAIRS_EARLY_INIT_512M) {
        tagFormat = "ar_ccu_p12_allpairs_early512m_v28";
    } else if (algorithm == Algorithm2x8::P12_ALLPAIRS_1SEG_400M4B) {
        tagFormat = "ar_ccu_p12_allpairs_1seg_400m4b_v27";
    } else if (algorithm == Algorithm2x8::P12_ALLPAIRS_1SEG_OUTPUT_TREE_512M) {
        tagFormat = "ar_ccu_p12_zero_increment_pipeline_512m_v41";
    } else if (algorithm == Algorithm2x8::P12_ALLPAIRS_1SEG_OUTPUT_TREE_400M4B) {
        tagFormat = "ar_ccu_p12_zero_increment_pipeline_400m4b_v42";
    } else if (algorithm == Algorithm2x8::P16_MESH_LANE_512M) {
        tagFormat = "ar_ccu_p16_meshlane_512m_v33";
    } else if (algorithm == Algorithm2x8::P16_MESH_LANE_400M4B) {
        tagFormat = "ar_ccu_p16_meshlane_400m4b_v33";
    } else if (algorithm == Algorithm2x8::P4_OUTPUT_ACCUMULATE_512M) {
        tagFormat = "ar_ccu_p4_output_accum_512m_v32";
    } else if (algorithm == Algorithm2x8::P4_OUTPUT_ACCUMULATE_400M4B) {
        tagFormat = "ar_ccu_p4_output_accum_400m4b_v32";
    } else if (algorithm == Algorithm2x8::P12_CLOS_ACCUMULATE_512M) {
        tagFormat = "ar_ccu_p12_clos_accum_512m_v34";
    } else if (algorithm == Algorithm2x8::P12_CLOS_ACCUMULATE_400M4B) {
        tagFormat = "ar_ccu_p12_clos_accum_400m4b_v34";
    } else if (algorithm == Algorithm2x8::P4_OUTPUT_ACCUMULATE_CHAIN_512M) {
        tagFormat = "ar_ccu_p4_output_chain_512m_v35";
    } else if (algorithm == Algorithm2x8::P4_OUTPUT_ACCUMULATE_CHAIN_400M4B) {
        tagFormat = "ar_ccu_p4_output_chain_400m4b_v35";
    } else if (algorithm == Algorithm2x8::P4_OUTPUT_ACCUMULATE_WAVE2_512M) {
        tagFormat = "ar_ccu_p4_output_wave2_512m_v37";
    } else if (algorithm == Algorithm2x8::P4_OUTPUT_ACCUMULATE_WAVE2_400M4B) {
        tagFormat = "ar_ccu_p4_output_wave2_400m4b_v38";
    }
    int tagLength = 0;
    if (IsV109R16SmallAlgorithm(algorithm)) {
        tagLength = std::snprintf(param.tag, sizeof(param.tag),
            "hccl_final_ccu_v109_r%u_512k_k1_i%u", param.rankSize, sendBuf == recvBuf ? 1U : 0U);
    } else if (IsV101SmallAlgorithm(algorithm)) {
        tagLength = std::snprintf(param.tag, sizeof(param.tag),
            "hccl_final_ccu_v101_r%u_512k_k1_i%u", param.rankSize, sendBuf == recvBuf ? 1U : 0U);
    } else {
        tagLength = std::snprintf(param.tag, sizeof(param.tag), tagFormat, param.rankSize);
    }
    CHK_PRT_RET(tagLength < 0 || static_cast<size_t>(tagLength) >= sizeof(param.tag),
        HCCL_ERROR("[HcclAllReduce] failed to build EngineCtx tag"), HCCL_E_INTERNAL);
    HCCL_INFO("[DispatchHit] topology=%s ranks=%u bytes=%lu count=%lu algorithm=%s tag=%s",
        TopologyName(expectedTopology), param.rankSize, totalBytes, count, AlgorithmName(algorithm), param.tag);

    HcclDfxOpInfo dfxInfo{};
    char commName[COMM_INDENTIFIER_MAX_LENGTH]{};
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    constexpr CommEngine ccuEngine = CommEngine::COMM_ENGINE_CCU;
    // 主 thread 与本次用户 stream 绑定，不能缓存首个调用的 thread。
    CHK_RET(HcclThreadAcquireWithStream(comm, ccuEngine, stream,
        IsV101SmallAlgorithm(algorithm) ? V101_THREAD_NOTIFY_NUM : THREAD_NOTIFY_NUM, &param.cpuThread));

    void *ctx = nullptr;
    uint64_t ctxSize = 0;
    if (HcclEngineCtxGet(comm, param.tag, ccuEngine, &ctx, &ctxSize) == HCCL_SUCCESS) {
        param.resCtx = ctx;
        param.ctxSize = ctxSize;
    } else {
        TopologyView topologyView;
        CHK_RET(DetectFinalTopology(comm, param, topologyView));
        const Algorithm2x8 detectedAlgorithm
            = SelectCrossLaneInplaceFallback(SelectAlgorithm(topologyView, totalBytes), sendBuf, recvBuf);
        CHK_PRT_RET(detectedAlgorithm != algorithm,
            HCCL_ERROR("[HcclAllReduce] rank-size dispatch %s disagrees with detected topology dispatch %s",
                AlgorithmName(algorithm), AlgorithmName(detectedAlgorithm)),
            HCCL_E_INTERNAL);
        AlgResourceCtx resource;
        CHK_RET(CreateResources(comm, param, topologyView, algorithm, resource));
        // 每个 EngineCtx/tag 仅创建一次；rank 0 的审计行用于证明精确拓扑和字节数确实命中目标路径。
        if (param.myRank == 0) {
            std::printf("[DISPATCH_AUDIT] topology=%s ranks=%u bytes=%lu algorithm=%s tag=%s version=%u\n",
                TopologyName(topologyView.topology), param.rankSize, totalBytes, AlgorithmName(algorithm), param.tag,
                resource.version);
        }

        std::vector<char> serialized = resource.Serialize();
        param.ctxSize = serialized.size();
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, ccuEngine, param.ctxSize, &param.resCtx));
        const HcclResult copyRet
            = HcclEngineCtxCopy(comm, ccuEngine, param.tag, serialized.data(), serialized.size(), 0);
        if (copyRet != HCCL_SUCCESS) {
            (void)HcclEngineCtxDestroy(comm, param.tag, ccuEngine);
            return copyRet;
        }
    }

    CHK_RET(ops_hccl::ExecOp(param));
    return HCCL_SUCCESS;
}
