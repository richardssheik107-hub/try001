/** Copyright (c) 2026 Huawei Technologies Co., Ltd. */
#ifndef OPS_HCCL_EXEC_OP_H
#define OPS_HCCL_EXEC_OP_H
#include "common.h"
namespace ops_hccl {
HcclResult ExecOp2x8SmallRootSpecializedBalanced(const OpParam &param);
HcclResult ExecOp2x8Large15OwnerPipeline2Compact(const OpParam &param);
HcclResult ExecOp4x1SmallRootSpecialized(const OpParam &param);
HcclResult ExecOp4x1LargePipeline15(const OpParam &param);
HcclResult ExecOp8p4SmallRootSpecializedBalanced(const OpParam &param);
HcclResult ExecOp8p4Large11OwnerPipeline2Compact(const OpParam &param);
}
#endif
