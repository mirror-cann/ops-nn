/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */
#include <cstring>
#include "log/log.h"
#include "graph/utils/type_utils.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/multilabel_margin_loss_tiling_data.h"
#include "../op_kernel/multilabel_margin_loss_tiling_key.h"

namespace optiling {

struct MultilabelMarginLossCompileInfo {};

constexpr uint32_t BATCH_MODE = 1;
constexpr int32_t REDUCTION_INVALID = -1;

// The reduction attr is a String ("none"/"mean"/"sum"), matching the aclnn launcher and every other
// loss op in this repo. Map it to the int code (0/1/2) the kernel expects. (Previously declared/read
// as Int, which silently mis-decoded the string bytes and made the kernel always take the sum/scalar
// path -> none & mean ST cases failed.) Returns REDUCTION_INVALID(-1) for an unknown value so the
// tiling entry can reject it defensively (aclnn already validates, but tiling is an independent entry).
static int32_t ParseReductionAttr(gert::TilingContext* context)
{
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return REDUCTION_INVALID;
    }
    const char* reductionStr = attrs->GetAttrPointer<char>(0);
    if (reductionStr == nullptr) {
        return REDUCTION_INVALID;
    }
    if (strcmp(reductionStr, "none") == 0) {
        return 0;
    }
    if (strcmp(reductionStr, "mean") == 0) {
        return 1;
    }
    if (strcmp(reductionStr, "sum") == 0) {
        return 2;
    }
    return REDUCTION_INVALID;
}

// Tiling 是独立入口:单算子直测不经过 aclnn/GE 的 CheckDtypeValid,dtype 守护必须在此独立再做一遍。
// 契约(对齐 def 6 combo):x/y ∈ {FLOAT,FLOAT16,BF16} 且 y==x;target=INT32;
// is_target 双约定 = {INT32(GE 图路径) 或 ==x(aclnn 路径)}——其余(如 x=fp32、is_target=fp16)一律拒。
static ge::graphStatus CheckDtypeValid(gert::TilingContext* context)
{
    auto xDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    auto targetDesc = context->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, targetDesc);
    auto yDesc = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yDesc);
    auto isTargetDesc = context->GetOutputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, isTargetDesc);

    ge::DataType xDtype = xDesc->GetDataType();
    if (xDtype != ge::DT_FLOAT && xDtype != ge::DT_FLOAT16 && xDtype != ge::DT_BF16) {
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "x", ge::TypeUtils::DataTypeToSerialString(xDtype).c_str(),
                                  "FLOAT, FLOAT16 or BF16");
        return ge::GRAPH_FAILED;
    }
    if (yDesc->GetDataType() != xDtype) {
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "y",
                                  ge::TypeUtils::DataTypeToSerialString(yDesc->GetDataType()).c_str(),
                                  "equal to x dtype");
        return ge::GRAPH_FAILED;
    }
    if (targetDesc->GetDataType() != ge::DT_INT32) {
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "target",
                                  ge::TypeUtils::DataTypeToSerialString(targetDesc->GetDataType()).c_str(), "INT32");
        return ge::GRAPH_FAILED;
    }
    // is_target dtype 守护按 soc 隔离(tiling 是独立入口, 必须自己守护):
    //   非 regbase(ascend910b/ascend910_93) 原始逻辑 = 仅 INT32;
    //   regbase(ascend950) 扩展 = INT32(GE 图) 或 ==x(aclnn 跟随 self)。
    // 不隔离会把 float is_target 在 ascend910b 上放通, 违反其原始逻辑。
    ge::DataType isTgtDtype = isTargetDesc->GetDataType();
    bool isRegBase = (platform_ascendc::PlatformAscendC(context->GetPlatformInfo()).GetSocVersion() ==
                      platform_ascendc::SocVersion::ASCEND950);
    if (isRegBase) {
        if (isTgtDtype != ge::DT_INT32 && isTgtDtype != xDtype) {
            OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "is_target",
                                      ge::TypeUtils::DataTypeToSerialString(isTgtDtype).c_str(),
                                      "INT32 (GE graph) or equal to x dtype (aclnn)");
            return ge::GRAPH_FAILED;
        }
    } else {
        if (isTgtDtype != ge::DT_INT32) {
            OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "is_target",
                                      ge::TypeUtils::DataTypeToSerialString(isTgtDtype).c_str(), "INT32");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MultilabelMarginLossTilingFunc(gert::TilingContext* context)
{
    MultilabelMarginLossTilingData tiling;

    // 红线:A2(ascend910b/ascend910_93)tiling 走基线原路径,不加任何新校验。
    // dtype 守护仅 regbase(ascend950)才做——A2 依赖上游 GE/aclnn 校验,与基线一致。
    bool isRegBase = (platform_ascendc::PlatformAscendC(context->GetPlatformInfo()).GetSocVersion() ==
                      platform_ascendc::SocVersion::ASCEND950);
    if (isRegBase && CheckDtypeValid(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    const gert::StorageShape* x1_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1_shape);
    uint32_t N = 1;
    uint32_t C = 1;
    int dimNum = x1_shape->GetStorageShape().GetDimNum();
    if (dimNum >= 2) {
        N = static_cast<uint32_t>(x1_shape->GetStorageShape().GetDim(0));
        C = static_cast<uint32_t>(x1_shape->GetStorageShape().GetDim(1));
    } else if (dimNum == 1) {
        N = 1;
        C = static_cast<uint32_t>(x1_shape->GetStorageShape().GetDim(0));
    }

    int32_t reduction = ParseReductionAttr(context);
    if (reduction == REDUCTION_INVALID) {
        OP_LOGE(context->GetNodeName(), "The reduction attribute must be 'none', 'mean', or 'sum'.");
        return ge::GRAPH_FAILED;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
    if (coreNum == 0) {
        coreNum = 1;
    }

    // Empty batch (N == 0): keep N == 0 so the kernel processes zero rows; use a single
    // block for a valid grid. Do NOT fake N = 1, which would read an unallocated phantom row.
    uint32_t usedCoreNum = (N == 0) ? 1u : ((N < coreNum) ? N : coreNum);
    context->SetBlockDim(usedCoreNum);
    // The kernel uses SyncAll for the cross-core workspace reduction; batch mode makes all cores
    // start together, otherwise SyncAll can deadlock probabilistically.
    context->SetScheduleMode(BATCH_MODE);

    uint32_t basePerCore = N / usedCoreNum;
    uint32_t pivot = N % usedCoreNum;

    tiling.N = N;
    tiling.C = C;
    tiling.basePerCore = basePerCore;
    tiling.pivot = pivot;
    tiling.usedCoreNum = usedCoreNum;
    tiling.reduction = reduction;

    auto* tilingData = context->GetTilingData<MultilabelMarginLossTilingData>();
    *tilingData = tiling;

    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    // Float accumulation workspace: N slots for reduction=none (per-row loss), else 1 scalar.
    // Rounded to 32B. The kernel atomic-adds row losses here, then core 0 casts+writes y.
    uint32_t wsElems = (reduction == 0) ? ((N == 0u) ? 1u : N) : 1u;
    size_t accBytes = ((static_cast<size_t>(wsElems) * sizeof(float) + 31u) / 32u) * 32u;
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = accBytes + sysWorkspaceSize;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForMultilabelMarginLoss([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MultilabelMarginLoss)
    .Tiling(MultilabelMarginLossTilingFunc)
    .TilingParse<MultilabelMarginLossCompileInfo>(TilingParseForMultilabelMarginLoss);
} // namespace optiling
