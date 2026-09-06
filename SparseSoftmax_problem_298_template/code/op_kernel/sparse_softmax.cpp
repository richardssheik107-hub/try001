// Kernel侧核函数实现
#include "kernel_operator.h"

#include "sparse_softmax_tiling.h"
#include "tiling_key_sparse_softmax.h"

template <class DT_SRC>
class KernelSparseSoftmax {
public:
    __aicore__ inline KernelSparseSoftmax() {}
    __aicore__ inline void Init(GM_ADDR src, GM_ADDR index, GM_ADDR ptr, GM_ADDR out, uint32_t length) {

    }
    __aicore__ inline void Process() {

    }
private:

};

template <typename DT_SRC>
 __global__ __aicore__ void sparse_softmax(GM_ADDR src, GM_ADDR index, GM_ADDR ptr, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(SparseSoftmaxTilingData);
    GET_TILING_DATA_WITH_STRUCT(SparseSoftmaxTilingData, tiling_data, tiling);
    KernelSparseSoftmax<DT_SRC> op;
    op.Init(src, index, ptr, out, tiling_data.length);
    op.Process();
}
