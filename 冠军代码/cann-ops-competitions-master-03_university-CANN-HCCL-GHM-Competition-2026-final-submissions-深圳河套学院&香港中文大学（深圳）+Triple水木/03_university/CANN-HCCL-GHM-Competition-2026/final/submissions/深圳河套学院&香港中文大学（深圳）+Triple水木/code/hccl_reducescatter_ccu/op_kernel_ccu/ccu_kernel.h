/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_CCU_KERNEL_H
#define OPS_HCCL_CCU_KERNEL_H

#include <ccu/ccu_types.h>

#include "custom.h"

namespace ops_hccl {

enum class CommKernelRole : uint32_t {
    GENERAL = 0,
    MESH = 2,
    CLOS = 3,
};

enum class ReduceKernelRole : uint32_t {
    GENERAL = 0,
    MESH = 2,
    CLOS = 3,
};

struct ReduceScatterKernelArg : public CcuKernelArgBase {
    uint32_t rankId = 0;
    uint32_t peerRanks[MAX_RANK_SIZE]{};
    uint32_t directMemberCount = 0;
    uint32_t directMemberRanks[MAX_RANK_SIZE]{};
    uint32_t fusedMemberCount = 0;
    uint32_t fusedMemberRanks[MAX_RANK_SIZE]{};
    uint32_t localMemberCount = 0;
    uint32_t remoteLocalMemberCount = 0;
    uint32_t remoteLocalMemberRanks[MAX_RANK_SIZE]{};
    uint32_t publishTargetCount = 0;
    uint32_t publishTargetRanks[MAX_RANK_SIZE]{};
    uint32_t aggregateSourceCount = 0;
    uint32_t aggregateSourceRanks[MAX_RANK_SIZE]{};
    uint64_t aggregateSourceRemoteOffsets[MAX_RANK_SIZE]{};
    uint64_t aggregateSourceLocalOffsets[MAX_RANK_SIZE]{};
    uint64_t aggregateSourceBytes[MAX_RANK_SIZE]{};
    uint32_t smallBufferCapacityBlocksPerSource = 0;
    bool smallBufferFuseSync = true;
    CommKernelRole commRole = CommKernelRole::GENERAL;
    ReduceKernelRole reduceRole = ReduceKernelRole::GENERAL;
};

// MixedRoute large-data communication kernel. Runtime mode selects address
// exchange, the topology-specific remote HBM fetch, or the final barrier.
CcuResult CcuReduceScatterKernel_Comm_General(CcuKernelArg arg);

// 4x1 recursive-halving communication kernel. It completes all eight rank^2
// reads, waits for the outgoing in-place reduction and performs one group-level
// partner handshake before starting the nine-segment exchange with rank^1.
CcuResult CcuReduceScatterKernel_Comm_4X1(CcuKernelArg arg);

// Small-data variant: only address exchange and fused A2A are emitted.  Keeping
// it as a separate registered kernel removes the large-path mode branches.
CcuResult CcuReduceScatterKernel_Comm_Small(CcuKernelArg arg);

// 2x8 small-data variant: fixed six-argument ABI, streaming peer reads and
// output-copy/post-barrier overlap.
CcuResult CcuReduceScatterKernel_Comm_Small_2X8(CcuKernelArg arg);

// Small dual-die route combines the two already reduced die partials.
CcuResult CcuReduceScatterKernel_Reduce_Small(CcuKernelArg arg);

// 2x8 small-data combine queued directly on the result communication Thread.
CcuResult CcuReduceScatterKernel_Reduce_Small_2X8(CcuKernelArg arg);

// 4x1 recursive-halving reduction Mission. It builds the outgoing partial in
// scratch and the retained partial directly in output; round 2 is fused into
// communication-side ReadReduce.
CcuResult CcuReduceScatterKernel_Reduce_4X1(CcuKernelArg arg);

// S9 large-data local HBM kernel used by both MixedRoute IO dies.
CcuResult CcuReduceScatterKernel_Reduce_General(CcuKernelArg arg);
} // namespace ops_hccl

#endif // OPS_HCCL_CCU_KERNEL_H
