/** Copyright (c) 2026 Huawei Technologies Co., Ltd. */
#ifndef OPS_HCCL_CCU_KERNEL_H
#define OPS_HCCL_CCU_KERNEL_H
#include <ccu/ccu_types.h>
#include <hccl/hccl_types.h>
#include "custom.h"
namespace ccu = ::AscendC::ccu;
namespace ops_hccl {
CcuResult Ccu2x8SmallRootSpecializedIntraKernel(CcuKernelArg arg);
CcuResult Ccu2x8SmallRootSpecializedInterKernel(CcuKernelArg arg);
CcuResult Ccu2x8OwnerPipeline2CompactIntraKernel(CcuKernelArg arg);
CcuResult Ccu2x8OwnerPipeline2CompactInterKernel(CcuKernelArg arg);
CcuResult Ccu4x1SmallRootSpecializedKernel(CcuKernelArg arg);
CcuResult Ccu4x1LargePipeline15Kernel(CcuKernelArg arg);
CcuResult Ccu8p4SmallRootSpecializedIntraKernel(CcuKernelArg arg);
CcuResult Ccu8p4SmallRootSpecializedInterKernel(CcuKernelArg arg);
CcuResult Ccu8p4OwnerPipeline2CompactIntraKernel(CcuKernelArg arg);
CcuResult Ccu8p4OwnerPipeline2CompactInterKernel(CcuKernelArg arg);
}
#endif
