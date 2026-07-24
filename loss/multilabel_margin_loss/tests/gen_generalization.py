#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
# 生成 MultilabelMarginLoss 全泛化用例 CSV（覆盖 dtype×reduction×1D/2D×密度×对齐×边界）。
# 输出/输入 shape 规则严格对齐 infershape：none+2D->[N], none+1D 或 mean/sum -> 标量[1]；is_target=x.shape。
import random

random.seed(20260719)

DTYPES = {
    "float32": (0.0001, 0.0001),
    "float16": (0.001, 0.001),
    "bfloat16": (0.004, 0.004),
}
REDUCTIONS = ["none", "mean", "sum"]
HDR = (
    "testcase_name,op_name,input_shapes,input_dtypes,input_formats,output_shapes,output_dtypes,"
    "output_formats,attributes,input_data_ranges,precision_tolerances,absolute_precision"
)


def y_shape(red, ndim, n):
    return f"({n},)" if (red == "none" and ndim == 2) else "(1,)"


def row(name, dt, red, ndim, n, c):
    tol = DTYPES[dt]
    if ndim == 2:
        xs = f"({n}, {c})"
    else:
        xs = f"({c},)"
    ish = f'"({xs},{xs})"'
    idt = f"\"('{dt}','int32')\""
    fmt = "\"('ND','ND')\""
    osh = f'"({y_shape(red, ndim, n)},{xs})"'
    odt = f"\"('{dt}','int32')\""
    attr = f"\"{{'reduction': '{red}'}}\""
    rng = f'"((-2, 2),(-1, {c}))"'
    ptol = f'"(({tol[0]}, {tol[1]}),(0, 0))"'
    return f"{name},multilabel_margin_loss,{ish},{idt},{fmt},{osh},{odt},{fmt},{attr},{rng},{ptol},0"


lines = [HDR]
idx = 0

# --- 随机主体 ~900：dtype×reduction×1D/2D，N/C 覆盖单核~多核、对齐/非对齐 ---
for _ in range(900):
    dt = random.choice(list(DTYPES))
    red = random.choice(REDUCTIONS)
    ndim = random.choice([1, 2, 2, 2])  # 2D 权重更高
    c = random.choice(
        [
            1,
            2,
            3,
            7,
            8,
            15,
            16,
            17,
            31,
            32,
            33,
            63,
            64,
            100,
            127,
            128,
            200,
            255,
            256,
            257,
            384,
            512,
        ]
    )
    n = (
        1
        if ndim == 1
        else random.choice([1, 2, 3, 4, 8, 16, 32, 48, 56, 64, 65, 96, 128])
    )
    idx += 1
    lines.append(
        row(f"mll_rand_{idx:04d}_{dt[:3]}_{red}_{ndim}d_n{n}c{c}", dt, red, ndim, n, c)
    )

# --- 边界/特例 ~100：极小 C、单类、大多核、对齐成对、各 dtype×reduction 网格 ---
BOUNDARY = []
for dt in DTYPES:
    for red in REDUCTIONS:
        BOUNDARY += [
            (dt, red, 2, 1, 1),  # 最小 2D
            (dt, red, 1, 1, 1),  # 最小 1D 单元素
            (dt, red, 2, 1, 8),  # 单行
            (dt, red, 2, 64, 256),  # 多核 blockdim 大
            (dt, red, 2, 128, 512),  # 更大多核+大 C
            (dt, red, 2, 4, 17),  # 非对齐 C
            (dt, red, 2, 4, 16),  # 对齐 C 成对
            (dt, red, 1, 1, 1024),  # 1D 长向量
            (dt, red, 2, 33, 65),  # 非对齐 N+C
        ]
for dt, red, ndim, n, c in BOUNDARY:
    idx += 1
    lines.append(
        row(f"mll_bd_{idx:04d}_{dt[:3]}_{red}_{ndim}d_n{n}c{c}", dt, red, ndim, n, c)
    )

with open("mll_generalization.csv", "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"generated {len(lines) - 1} cases -> mll_generalization.csv")
