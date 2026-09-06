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

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <ccu/ccu_types.h>

#include "common.h"
#include "custom.h"

namespace ops_hccl {

namespace ccu = ::AscendC::ccu;

constexpr uint64_t CCU_MS_INTERLEAVE = 8;
constexpr uint64_t CCU_MS_SIZE = 4096;
constexpr uint32_t CCU_LOCAL_COPY_MS_PER_LOOP = 8;
constexpr uint32_t CCU_MS_LOCAL_COPY_LOOP_COUNT = 8;

struct LoopGroupConfig {
    uint32_t msInterleave = 0;
    uint32_t loopCount = 0;
    uint64_t memSlice = 0;
};

struct LoopGroupResource {
    ccu::Array<ccu::Event> completedEvent{0};
    ccu::Array<ccu::CcuBuffer> ccuBuf{0};
    uint32_t eventCount = 0;
    uint32_t bufCount = 0;
};

struct GroupOpSizeVars {
    ccu::Variable addrOffset;
    ccu::Variable loopParam;
    ccu::Variable parallelParam;
    ccu::Variable residual;
};

struct CcuLoopEntity {
    std::unique_ptr<ccu::Func> body[2];
    std::unique_ptr<ccu::Loop> loops[2];
    ccu::Variable loopParam[2];
};

struct CcuKernelArgDirect : CcuKernelArgBase {
    uint32_t doLocalCopy = 0;
    uint32_t writeMask = 0;
    uint32_t publishMask = 0;
    uint32_t readyWaitMask = 0;
    uint32_t syncMask = 0;
    uint32_t syncWaitMask = 0;
    uint32_t waitWriteCq = 1;
};

struct CcuKernelArgSmallDirect : CcuKernelArgBase {
    uint32_t doLocalCopy = 0;
    uint32_t waitWriteCq = 1;
};

struct CcuKernelArgRelay : CcuKernelArgBase {
    uint32_t relayCount = 0;
    uint32_t reuseDirectAddresses = 0;
    uint32_t postSync = 0;
};

struct CcuKernelArg8p4Network : CcuKernelArgBase {
    uint32_t localIndex = 0;
    uint32_t isPartialSide = 0;
    uint32_t waitSeedData = 0;
    uint32_t seedPeerCount = 0;
    uint32_t seedPeerIndices[2]{};
};

struct CcuDirectContext {
    CcuKernelArgDirect *arg = nullptr;
    ccu::Variable input;
    ccu::Variable output;
    ccu::Variable token;
    ccu::Variable outputOffset;
    ccu::Variable firstSize;
    ccu::Variable secondSize;
    ccu::Variable copyFlag;
    ccu::Variable skipDone;
    std::vector<ccu::Variable> remoteOutputs;
    std::vector<ccu::Variable> remoteTokens;
    ccu::Event firstEvent;
    ccu::Event secondEvent;
    GroupOpSizeVars goSize;
    LoopGroupConfig copyConfig;
    LoopGroupResource copyResource;
    bool copyResourceAllocated = false;
    std::map<std::string, CcuLoopEntity> loopMap;

    void CreateLoopEntity(const std::string &name)
    {
        loopMap.emplace(name, CcuLoopEntity());
    }

    bool IsLoopEntityRegistered(const std::string &name) const
    {
        return loopMap.count(name) != 0;
    }
};

struct CcuSmallDirectContext {
    CcuKernelArgSmallDirect *arg = nullptr;
    ccu::Variable input;
    ccu::Variable output;
    ccu::Variable token;
    ccu::Variable outputOffset;
    ccu::Variable size;
    std::vector<ccu::Variable> remoteOutputs;
    std::vector<ccu::Variable> remoteTokens;
    ccu::Event event;
    GroupOpSizeVars goSize;
    LoopGroupConfig copyConfig;
    LoopGroupResource copyResource;
    bool copyResourceAllocated = false;
    std::map<std::string, CcuLoopEntity> loopMap;

    void CreateLoopEntity(const std::string &name)
    {
        loopMap.emplace(name, CcuLoopEntity());
    }

    bool IsLoopEntityRegistered(const std::string &name) const
    {
        return loopMap.count(name) != 0;
    }
};

struct CcuRelayContext {
    CcuKernelArgRelay *arg = nullptr;
    ccu::Variable input;
    ccu::Variable output;
    ccu::Variable token;
    ccu::Variable ownOffset;
    ccu::Variable firstSize;
    ccu::Variable secondSize;
    ccu::Variable sendOwn;
    ccu::Variable relayOffsets[2];
    std::vector<ccu::Variable> remoteOutputs;
    std::vector<ccu::Variable> remoteTokens;
    ccu::Event ownFirstEvent;
    ccu::Event ownSecondEvent;
    ccu::Event relayFirstEvents[2];
    ccu::Event relaySecondEvents[2];
};

struct Ccu8p4SeedContext {
    CcuKernelArg8p4Network *arg = nullptr;
    ccu::Variable input;
    ccu::Variable output;
    ccu::Variable token;
    ccu::Variable fullSeedSize;
    ccu::Variable partialSeedSize;
    ccu::Variable ownOutputOffset;
    ccu::Variable remoteOutputOffsets[8];
    std::vector<ccu::Variable> remoteInputs;
    std::vector<ccu::Variable> remoteOutputs;
    std::vector<ccu::Variable> remoteTokens;
    ccu::Event readEvent;
    ccu::Event writeEvent;
};

struct Ccu8p4SuffixContext {
    CcuKernelArg8p4Network *arg = nullptr;
    ccu::Variable input;
    ccu::Variable output;
    ccu::Variable token;
    ccu::Variable fullSeedSize;
    ccu::Variable partialSeedSize;
    ccu::Variable fullSuffixFirstSize;
    ccu::Variable fullSuffixSecondSize;
    ccu::Variable partialSuffixFirstSize;
    ccu::Variable partialSuffixSecondSize;
    ccu::Variable ownOutputOffset;
    ccu::Variable remoteOutputOffsets[8];
    std::vector<ccu::Variable> remoteInputs;
    std::vector<ccu::Variable> remoteOutputs;
    std::vector<ccu::Variable> remoteTokens;
    ccu::Event readFirstEvent;
    ccu::Event readSecondEvent;
    ccu::Event writeFirstEvent;
    ccu::Event writeSecondEvent;
};

CcuResult CcuDirectKernel(CcuKernelArg arg);
CcuResult CcuLargeDirectKernel(CcuKernelArg arg);
CcuResult Ccu4x1DirectKernel(CcuKernelArg arg);
CcuResult Ccu4x1SmallDirectKernel(CcuKernelArg arg);
CcuResult CcuSmallDirectKernel(CcuKernelArg arg);
CcuResult CcuRelayKernel(CcuKernelArg arg);
CcuResult Ccu8p4SeedKernel(CcuKernelArg arg);
CcuResult Ccu8p4SuffixKernel(CcuKernelArg arg);

} // namespace ops_hccl

#endif // OPS_HCCL_CCU_KERNEL_H
