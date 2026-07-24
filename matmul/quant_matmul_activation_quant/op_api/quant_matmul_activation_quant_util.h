/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_QUANT_MATMUL_ACTIVATION_QUANT_UTIL_H
#define OP_API_INC_QUANT_MATMUL_ACTIVATION_QUANT_UTIL_H
#include "opdev/common_types.h"

namespace QBMMActivationQuant {
using namespace op;
struct QuantMatmulActivationQuantWeightNzParams {
    const aclTensor* x1 = nullptr;
    const aclTensor* x2 = nullptr;
    const aclTensor* x1Scale = nullptr;
    const aclTensor* x2Scale = nullptr;
    const aclTensor* bias = nullptr;
    aclTensor* y = nullptr;
    aclTensor* yScale = nullptr;

    bool transposeX1;
    bool transposeX2;
    int64_t groupSize;
    const char* activationType;
    int64_t y_dtype;
    const char* quantMode;
    const char* roundMode;
    int64_t scaleAlg;
    double dstTypeMax;
};

static const std::initializer_list<op::DataType> X1_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT8_E4M3FN,
                                                                          DataType::DT_FLOAT8_E5M2};
static const std::initializer_list<op::DataType> X2_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT8_E4M3FN};
static const std::initializer_list<op::DataType> Y_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT8_E4M3FN,
                                                                         DataType::DT_FLOAT8_E5M2};

constexpr uint32_t MX_X1_DIM = 2U;
constexpr uint32_t MX_X1_DIM_MIN = 2U;
constexpr uint32_t MX_X1_DIM_MAX = 6L;
constexpr uint32_t MX_X2_DIM = 2U;
constexpr uint32_t MX_X2_DIM_MIN = 4U;
constexpr uint32_t MX_X2_DIM_MAX = 8L;
constexpr uint32_t MX_X1_SCALE_DIM = 3U;
constexpr uint32_t MX_X2_SCALE_DIM = 3U;
constexpr uint32_t PERTENSOR_SCALE_DIM = 1U;
constexpr uint32_t Y_INPUT_DIM = 2U;
constexpr uint32_t Y_OUTPUT_DIM = 2U;
constexpr uint32_t MX_X1_PER_TOKEN_SCALE_DIM = 3U;
constexpr size_t LAST_FIRST_DIM_INDEX = 1;
constexpr size_t LAST_SECOND_DIM_INDEX = 2;
constexpr size_t LAST_THIRD_DIM_INDEX = 3;
constexpr int64_t MXFP_MULTI_BASE_SIZE = 2L;
constexpr int64_t SPLIT_SIZE = 64L;
static constexpr int PENULTIMATE_DIM = 2;

static const int32_t GROUP_M_OFFSET = 32;
static const int32_t GROUP_N_OFFSET = 16;
static const uint64_t GROUP_MNK_BIT_SIZE = 0xFFFF;
static const int64_t PERGROUP_GROUP_SIZE = 32L;
static const size_t MX_SCALE_MAX_DIM = 3;
static constexpr int64_t OUTPUT_INFER_FAIL = -1L;

static inline bool isA8W4Float(const aclTensor* x1, const aclTensor* x2)
{
    if (x1 == nullptr || x2 == nullptr) {
        return false;
    }
    return x1->GetDataType() == op::DataType::DT_FLOAT8_E4M3FN &&
           (x2->GetDataType() == op::DataType::DT_FLOAT || x2->GetDataType() == op::DataType::DT_FLOAT4_E2M1);
}
} // namespace QBMMActivationQuant
#endif
