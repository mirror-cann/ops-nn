/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file group_norm_silu_quant_infershape.cpp
 * \brief GroupNormSiluQuant graph infershape / inferdatatype.
 *   yOut  : same shape as x, dtype INT8 (quantized output).
 *   meanOut / rstdOut : shape (N, num_groups), dtype same as x.
 */
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {
static constexpr size_t GNSQ_IDX_IN_X = 0;
static constexpr size_t GNSQ_IDX_OUT_Y = 0;
static constexpr size_t GNSQ_IDX_OUT_MEAN = 1;
static constexpr size_t GNSQ_IDX_OUT_RSTD = 2;
static constexpr size_t GNSQ_NUMGROUPS_IDX = 0;
static constexpr size_t GNSQ_N_IDX = 0;

static ge::graphStatus GroupNormSiluQuantInferShape(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do GroupNormSiluQuantInferShape");

    const gert::Shape* x_shape = context->GetInputShape(GNSQ_IDX_IN_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);

    gert::Shape* y_shape = context->GetOutputShape(GNSQ_IDX_OUT_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, y_shape);
    gert::Shape* mean_shape = context->GetOutputShape(GNSQ_IDX_OUT_MEAN);
    OP_CHECK_NULL_WITH_CONTEXT(context, mean_shape);
    gert::Shape* rstd_shape = context->GetOutputShape(GNSQ_IDX_OUT_RSTD);
    OP_CHECK_NULL_WITH_CONTEXT(context, rstd_shape);

    // y 与 x 同形
    *y_shape = *x_shape;

    // mean/rstd = (N, num_groups)
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* num_groups = attrs->GetAttrPointer<int64_t>(GNSQ_NUMGROUPS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, num_groups);

    const int64_t n_dim = x_shape->GetDim(GNSQ_N_IDX);
    mean_shape->SetDimNum(0);
    mean_shape->AppendDim(n_dim);
    mean_shape->AppendDim(*num_groups);
    *rstd_shape = *mean_shape;

    OP_LOGD(context, "End to do GroupNormSiluQuantInferShape");
    return ge::GRAPH_SUCCESS;
}

static graphStatus GroupNormSiluQuantInferDataType(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "Begin to do GroupNormSiluQuantInferDataType");
    auto xDtype = context->GetInputDataType(GNSQ_IDX_IN_X);
    // yOut 恒为量化 INT8; meanOut/rstdOut 与 x 同 dtype(gamma/beta 已约束与 x 一致)。
    context->SetOutputDataType(GNSQ_IDX_OUT_Y, ge::DT_INT8);
    context->SetOutputDataType(GNSQ_IDX_OUT_MEAN, xDtype);
    context->SetOutputDataType(GNSQ_IDX_OUT_RSTD, xDtype);
    OP_LOGD(context, "End to do GroupNormSiluQuantInferDataType");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GroupNormSiluQuant)
    .InferShape(GroupNormSiluQuantInferShape)
    .InferDataType(GroupNormSiluQuantInferDataType);
} // namespace ops
