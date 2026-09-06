#include <cstdint>

#include "register/op_def_registry.h"

#include "../op_kernel/sparse_softmax_tiling.h"
#include "../op_kernel/tiling_key_sparse_softmax.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    const gert::Tensor *src = context->GetRequiredInputTensor(0);
    const gert::Tensor *index = context->GetOptionalInputTensor(1);
    const gert::Tensor *ptr = context->GetOptionalInputTensor(2);
    if (src == nullptr || (index == nullptr && ptr == nullptr)) {
        return ge::GRAPH_FAILED;
    }

    auto shape = src->GetOriginShape();
    const int64_t rank = static_cast<int64_t>(shape.GetDimNum());
    if (rank <= 0) {
        return ge::GRAPH_FAILED;
    }

    int64_t dim = 0;
    float eps = 1e-16f;
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const int64_t *attrDim = attrs->GetInt(0);
        const float *attrEps = attrs->GetFloat(1);
        if (attrDim != nullptr) {
            dim = *attrDim;
        }
        if (attrEps != nullptr) {
            eps = *attrEps;
        }
    }

    if (dim < 0) {
        dim += rank;
    }
    if (dim < 0 || dim >= rank) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t totalLength = static_cast<uint64_t>(shape.GetShapeSize());
    const uint64_t dimSize = static_cast<uint64_t>(shape.GetDim(dim));

    uint64_t innerSize = 1;
    for (int64_t i = dim + 1; i < rank; ++i) {
        innerSize *= static_cast<uint64_t>(shape.GetDim(i));
    }

    uint64_t outerSize = 0;
    if (dimSize != 0 && innerSize != 0) {
        outerSize = totalLength / (dimSize * innerSize);
    }

    const uint64_t indexLength = index == nullptr ? 0 : static_cast<uint64_t>(index->GetShapeSize());
    const uint64_t ptrLength = ptr == nullptr ? 0 : static_cast<uint64_t>(ptr->GetShapeSize());

    const uint32_t mode = ptr == nullptr ? 0U : 1U;
    if (mode == 0U && indexLength == 0U && totalLength != 0U) {
        return ge::GRAPH_FAILED;
    }
    if (mode == 1U && ptrLength < 2U && dimSize != 0U) {
        return ge::GRAPH_FAILED;
    }

    uint32_t DT_MODE = SPARSE_SOFTMAX_FP32;
    const ge::DataType dtypeSrc = src->GetDataType();
    if (dtypeSrc == ge::DT_FLOAT) {
        DT_MODE = SPARSE_SOFTMAX_FP32;
    } else if (dtypeSrc == ge::DT_FLOAT16) {
        DT_MODE = SPARSE_SOFTMAX_FP16;
    } else if (dtypeSrc == ge::DT_BF16) {
        DT_MODE = SPARSE_SOFTMAX_BF16;
    } else {
        return ge::GRAPH_FAILED;
    }
    ASCENDC_TPL_SEL_PARAM(context, DT_MODE);

    SparseSoftmaxTilingData *tiling = context->GetTilingData<SparseSoftmaxTilingData>();
    tiling->totalLength = totalLength;
    tiling->outerSize = outerSize;
    tiling->dimSize = dimSize;
    tiling->innerSize = innerSize;
    tiling->indexLength = indexLength;
    tiling->ptrLength = ptrLength;
    tiling->mode = mode;
    tiling->eps = eps;

    context->SetBlockDim(1);

    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    const gert::Shape *srcShape = context->GetInputShape(0);
    gert::Shape *outShape = context->GetOutputShape(0);
    if (srcShape == nullptr || outShape == nullptr) {
        return GRAPH_FAILED;
    }
    *outShape = *srcShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class SparseSoftmax : public OpDef {
public:
    explicit SparseSoftmax(const char *name) : OpDef(name) {
        this->Input("src")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("index")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("ptr")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("dim").AttrType(OPTIONAL).Int(0);
        this->Attr("eps").AttrType(OPTIONAL).Float(1e-16);
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};
OP_ADD(SparseSoftmax);
}  // namespace ops
