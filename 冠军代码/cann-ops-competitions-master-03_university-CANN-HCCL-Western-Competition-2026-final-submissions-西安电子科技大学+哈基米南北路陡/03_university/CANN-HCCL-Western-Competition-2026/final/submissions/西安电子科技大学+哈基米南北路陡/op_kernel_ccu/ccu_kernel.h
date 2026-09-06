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

// 按实际 die 拆分的 owner reduce-scatter + all-gather kernel。
CcuResult CcuKernel(CcuKernelArg arg);

// 2x8 / 512 KiB：单/双 die 一次 launch 的四步融合 CLOS butterfly。
CcuResult CcuButterfly2x8Kernel(CcuKernelArg arg);

// 2x8 / 大消息：die-native cross-lane reduce-scatter、本地归约和 owner broadcast。
CcuResult CcuCrossLane2x8Kernel(CcuKernelArg arg);

// 2x8 / 512 MiB：四条跨机 lane，并行提交本地 shard 归约。
CcuResult CcuCrossLaneFourLane512MKernel(CcuKernelArg arg);

// 4x1 / 8+4 的 512 KiB：全 peer 独立 slot、单/双 die 融合 Direct-RSAG。
CcuResult CcuDirectRsagKernel(CcuKernelArg arg);

// 4x1 / 大消息：三 source、三 shard 的旋转 WriteReduce。
CcuResult CcuP4Rotate3Kernel(CcuKernelArg arg);

// 8+4 / 大消息：Server 内 lane Reduce、8->4 聚合、结果回传和本地 AllGather。
CcuResult CcuP12Hier8Kernel(CcuKernelArg arg);

// 8+4 / 大消息：两 segment 的 12-owner All-Pairs Direct-RSAG。
CcuResult CcuP12AllPairsKernel(CcuKernelArg arg);

// v112 / 8+4 大消息：双 die 旋转 ReadReduce + 主 die 合并。
CcuResult CcuP12OutputTreeKernel(CcuKernelArg arg);

// v101 / 512 KiB：R4 MS direct 或 R12/R16 双 die MS-group 三阶段实现。
CcuResult CcuV101SmallKernel(CcuKernelArg arg);
} // namespace ops_hccl

#endif // OPS_HCCL_CCU_KERNEL_H
