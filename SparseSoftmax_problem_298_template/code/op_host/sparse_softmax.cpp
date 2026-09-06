// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/sparse_softmax_tiling.h"
#include "../op_kernel/tiling_key_sparse_softmax.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // 示例: 获取平台信息
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        uint64_t ub_size;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
        // 示例: 获取算子输入数组信息
        const gert::Tensor *tensor_src = context->GetRequiredInputTensor(0);
        const gert::Tensor *tensor_index = context->GetOptionalInputTensor(1);
        if(tensor_index) {
            // 可选输入数组存在
        }
        const gert::Tensor *tensor_ptr = context->GetOptionalInputTensor(2);
        if(tensor_ptr) {
            // 可选输入数组存在
        }
        ge::DataType dtype_src = tensor_src->GetDataType(); // 获取数据类型
        int dtype_size_src = ge::GetSizeByDataType(dtype_src); // 获取数据类型的字长
        uint32_t length_src = tensor_src->GetShapeSize(); // 获取元素个数
        uint32_t size_src = tensor_src->GetSize(); // 获取内存大小
        // 示例: 获取算子输入属性
        const gert::RuntimeAttrs *attrs = context->GetAttrs();
        const int64_t *attr_dim = attrs->GetInt(0);
        const float *attr_eps = attrs->GetFloat(1);
        // 示例: 配置tiling key, 从而实现kernel侧不同数据类型/算法的区分
        uint32_t DT_SRC = static_cast<uint32_t>(dtype_src);
        ASCENDC_TPL_SEL_PARAM(context, DT_SRC);
        // 示例: 计算tiling方案并填充tiling结构体
        SparseSoftmaxTilingData *tiling = context->GetTilingData<SparseSoftmaxTilingData>();
        tiling->length = length_src;
        // 配置启动核数
        context->SetBlockDim(num_cores_aiv);
        // 配置workspace大小
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        return ge::GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    class SparseSoftmax : public OpDef {
    public:
        explicit SparseSoftmax(const char *name) : OpDef(name) {
            this->Input("src")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("index")
                .ParamType(OPTIONAL)
                .DataType({ge::DT_INT32, ge::DT_INT32})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("ptr")
                .ParamType(OPTIONAL)
                .DataType({ge::DT_INT32, ge::DT_INT32})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("out")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Attr("dim").AttrType(OPTIONAL).Int();
            this->Attr("eps").AttrType(OPTIONAL).Float(1e-16);
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(SparseSoftmax);
}  // namespace ops
