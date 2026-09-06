/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_CUSTOM_H
#define OPS_HCCL_CUSTOM_H

#include <cstdint>
#include <memory>
#include <type_traits>

#include <hccl/hccl_res.h>
#include <hccl/hccl_types.h>

#include "common.h"

constexpr uint64_t CUSTOM_SMALL_MESSAGE_LIMIT = 512ULL * 1024ULL;

enum class TopologyKind : uint32_t {
    TOPO_2X8 = 0,
    TOPO_4X1 = 1,
    TOPO_8P4 = 2,
    TOPO_UNKNOWN = 3,
};

enum class TopologyRole : uint32_t {
    ROLE_COMMON = 0,
    ROLE_8P4_FULL_SERVER = 1,
    ROLE_8P4_PARTIAL_SERVER = 2,
};

enum KernelMask : uint32_t {
    KERNEL_NETWORK_DIRECT = 1U << 0,
    KERNEL_LOCAL_DIRECT = 1U << 1,
    KERNEL_LOCAL_RELAY = 1U << 2,
    KERNEL_NETWORK_SEED = 1U << 3,
    KERNEL_NETWORK_SMALL = 1U << 4,
    KERNEL_LOCAL_SMALL = 1U << 5,
};

struct CcuKernelArgBase {
    ChannelHandle channels[MAX_RANK_SIZE]{};
    uint32_t channelCount = 0;
};

struct CcuKernelInfo {
    char kernelFuncName[64]{};
    void *kernelFunc = nullptr;
    void *kernelArg = nullptr;

private:
    std::shared_ptr<CcuKernelArgBase> kernelArgSmartPtr;

public:
    template <typename T> void SetKernelArg(const std::shared_ptr<T> &arg)
    {
        kernelArgSmartPtr = std::static_pointer_cast<CcuKernelArgBase>(arg);
        kernelArg = static_cast<void *>(arg.get());
    }
};

struct AlgResourceCtx {
    static constexpr uint32_t MAGIC = 0x43414752U; // "CAGR"
    static constexpr uint32_t VERSION = 8;

    uint32_t magic = MAGIC;
    uint32_t version = VERSION;
    TopologyKind topology = TopologyKind::TOPO_UNKNOWN;
    TopologyRole role = TopologyRole::ROLE_COMMON;
    uint32_t kernelMask = 0;
    uint32_t threadCount = 0;
    uint32_t relayCount = 0;
    uint32_t relayRanks[2]{};
    uint32_t networkPeerCount = 0;
    uint32_t networkPeerRanks[MAX_RANK_SIZE]{};
    uint32_t selfCopyOnNetwork = 0;
    ThreadHandle mainThread{};
    ThreadHandle localThread{};
    CcuKernelHandle networkDirectKernel{};
    CcuKernelHandle networkSeedKernel{};
    CcuKernelHandle localDirectKernel{};
    CcuKernelHandle localRelayKernel{};
    CcuKernelHandle networkSmallKernel{};
    CcuKernelHandle localSmallKernel{};
};

static_assert(std::is_trivially_copyable<AlgResourceCtx>::value,
    "AlgResourceCtx must be safe for EngineCtx byte copies");

#endif // OPS_HCCL_CUSTOM_H
