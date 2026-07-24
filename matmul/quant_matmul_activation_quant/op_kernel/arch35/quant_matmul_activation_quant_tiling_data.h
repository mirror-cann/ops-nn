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
 * \file quant_matmul_activation_quant_tiling_data.h
 * \brief
 */
#pragma once
#include "kernel_tiling/kernel_tiling.h"
#if defined(__CCE_AICORE__)
#include "../../quant_batch_matmul_v3/arch35/quant_batch_matmul_v3_tiling_data.h"
#else
#include "matmul/quant_batch_matmul_v3/op_kernel/arch35/quant_batch_matmul_v3_tiling_data.h"
#endif
#ifndef __CCE_AICORE__
#include <cstdint>
#endif

// QuantMatmulActivationQuant tiling_data
namespace QMMAQ {
enum class BasicQuantMode : uint32_t {
    DEFAULT = 0x0U,
    PERTENSOR_MODE = 0x1U,
    PERCHANNEL_MODE = 0x1U << 1,
    PERTOKEN_MODE = 0x1U << 2,
    MX_PERGROUP_MODE = 0x1U << 3,
    PERBLOCK_MODE = 0x1U << 4,
    PERGROUP_MODE = 0x1U << 5,
};

enum class QuantAlg : uint8_t {
    OCP = 0,
    BLAS = 1,
};

enum class GeluAlg : uint8_t {
    TANH = 0,
    ERF = 1,
};

enum class MX_QUANT_ROUND_MODE : uint8_t {
    RINT = 0,
    FLOOR = 1,
    ROUND = 2,
};
#pragma pack(push, 8)
struct QuantBatchMatmulV3BasicAPIDataParams {
    uint32_t batchA = 0;
    uint32_t batchB = 0;
    uint32_t batchC = 0;
    uint32_t batchA1 = 0;
    uint32_t batchA2 = 0;
    uint32_t batchA3 = 0;
    uint32_t batchA4 = 0;
    uint32_t batchB1 = 0;
    uint32_t batchB2 = 0;
    uint32_t batchB3 = 0;
    uint32_t batchB4 = 0;
    uint32_t batchC1 = 0;
    uint32_t batchC2 = 0;
    uint32_t batchC3 = 0;
    uint32_t batchC4 = 0;
    uint32_t x1QuantMode = 0;
    uint32_t x2QuantMode = 0;
    uint32_t biasThreeDim = 0;
    uint32_t biasDtype = 0;
    uint32_t groupSizeM = 0;
    uint32_t groupSizeN = 0;
    uint32_t groupSizeK = 0;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct BasicAPICubeTiling {
    uint32_t m = 0;
    uint32_t n = 0;
    uint32_t k = 0;
    uint32_t baseM = 0;
    uint32_t baseN = 0;
    uint32_t baseK = 0;
    uint32_t scaleKL1 = 0;
    uint16_t stepKa = 0;
    uint16_t stepKb = 0;
    uint16_t scaleFactorA = 0;
    uint16_t scaleFactorB = 0;
    uint8_t nBufferNum = 0;
    uint8_t isBias = 0;
    uint8_t dbL0C = 0;
    uint8_t reserved = 0;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct SlidingWindowParams {
    uint32_t mTailTile = 0;
    uint32_t nTailTile = 0;
    uint32_t mBaseTailSplitCnt = 1;
    uint32_t nBaseTailSplitCnt = 1;
    uint32_t mTailMain = 0;
    uint32_t nTailMain = 0;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct QuantMatmulActivationQuantMMTilingData {
    QuantBatchMatmulV3BasicAPIDataParams params;
    BasicAPICubeTiling matmulTiling;
    SlidingWindowParams adaptiveSlidingWin;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct QuantMatmulActivationQuantTilingData {
    DequantBmm::QuantBatchMatmulV3BasicAPITilingData mmTilingData;
    GeluAlg activationType = GeluAlg::TANH;
    QuantAlg scaleAlg = QuantAlg::OCP;
    MX_QUANT_ROUND_MODE roundMode = MX_QUANT_ROUND_MODE::RINT;
    float dstTypeMax = 0.0;
};
#pragma pack(pop)
} // namespace QMMAQ
