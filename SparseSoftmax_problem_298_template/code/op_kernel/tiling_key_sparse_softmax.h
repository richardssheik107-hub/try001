#pragma once

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(SparseSoftmax,
    ASCENDC_TPL_DATATYPE_DECL(DT_SRC, C_DT_FLOAT, C_DT_FLOAT16, C_DT_BF16),
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_SRC, C_DT_FLOAT, C_DT_FLOAT16, C_DT_BF16),
    ),
);
