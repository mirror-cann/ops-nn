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
"""
TTK golden plugin for non_zero_with_value (kernel mode, arch35 / Ascend950).

Semantics (transpose=true 坐标主序):
    mask  = x != 0        # nan 与 +-inf 计非零；-0.0 视为零，遍历行主序
    N     = mask 非零元素个数
    count = [N]
    value = 静态 max-size [numel]，前 N 段 = x[mask]（行主序），其余为无效预留区(0 填充)
    index = 静态 max-size [2*numel]，坐标主序展平：
                index[0:N]           = 非零元素行号(行主序)
                index[numel:numel+N] = 非零元素列号(行主序)
            其余为无效预留区(0 填充)

三输出均为静态 max-size；有效长度由 count 给出。TTK 比对整块 buffer，无效预留区
在 NPU 上未定义，故本 golden 对尾部补 0；padded 尾部的 PASS/FAIL 非权威判据
（权威判据 = 有效前缀 [0:N] 逐位比对）。

Canonical IO order (non_zero_with_value_def.cpp / CSV output_dtypes):
    input : x(float32, 严格 2D)
    output: value(float32), index(int32), count(int32)
"""

import numpy as np


def __golden_non_zero_with_value(x, **kwargs):
    xf = np.asarray(x).astype(np.float32)
    col = xf.shape[1]
    numel = xf.size
    flat = xf.reshape(-1)  # 行主序
    nz = np.flatnonzero(flat != 0.0)  # nan != 0 -> True，行主序保持
    n = nz.size

    value = np.zeros((numel,), dtype=np.float32)
    value[:n] = flat[nz]

    index = np.zeros((2 * numel,), dtype=np.int32)
    if col > 0:
        index[:n] = (nz // col).astype(np.int32)  # 行号
        index[numel : numel + n] = (nz % col).astype(np.int32)  # 列号

    count = np.array([n], dtype=np.int32)
    return [value, index, count]


__golden__ = {"kernel": {"non_zero_with_value": "__golden_non_zero_with_value"}}
