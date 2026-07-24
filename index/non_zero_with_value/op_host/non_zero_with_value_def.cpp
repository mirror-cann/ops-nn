/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file non_zero_with_value_def.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
// A5(arch35) 对齐 A2 实机:仅 float32、严格 2D、transpose=true、静态 max-size 输出。
static const std::vector<ge::DataType> xDataType = {ge::DT_FLOAT};
static const std::vector<ge::DataType> valueDataType = {ge::DT_FLOAT};
static const std::vector<ge::DataType> indexDataType = {ge::DT_INT32};
static const std::vector<ge::DataType> countDataType = {ge::DT_INT32};
static const std::vector<ge::Format> nzvFormat = {ge::FORMAT_ND};

class NonZeroWithValue : public OpDef {
public:
    explicit NonZeroWithValue(const char* name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType(xDataType).Format(nzvFormat).UnknownShapeFormat(nzvFormat);
        // 静态 max-size 输出:value=[numel]/index=[2*numel]/count=[1](由 infershape 静态推导);
        // 非数据依赖 shape,故不加 OutputShapeDependOnCompute。
        this->Output("value")
            .ParamType(REQUIRED)
            .DataType(valueDataType)
            .Format(nzvFormat)
            .UnknownShapeFormat(nzvFormat);
        this->Output("index")
            .ParamType(REQUIRED)
            .DataType(indexDataType)
            .Format(nzvFormat)
            .UnknownShapeFormat(nzvFormat);
        this->Output("count")
            .ParamType(REQUIRED)
            .DataType(countDataType)
            .Format(nzvFormat)
            .UnknownShapeFormat(nzvFormat);
        this->Attr("transpose").AttrType(OPTIONAL).Bool(false);
        this->Attr("dtype").AttrType(OPTIONAL).Int(ge::DT_INT32);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .ExtendCfgInfo("opFile.value", "non_zero_with_value");
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

OP_ADD(NonZeroWithValue);
} // namespace ops
