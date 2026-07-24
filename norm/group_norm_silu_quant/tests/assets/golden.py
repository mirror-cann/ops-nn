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
TTK golden plugin for group_norm_silu_quant (kernel mode, arch35/Ascend950).

Transcribed from the authoritative reference executor
    ops-nn/norm/group_norm_silu_quant/tests/st/aclnnGroupNormSiluQuant/executor_aclnnGroupNormSiluQuant.py
and the requirement/design docs (changwei_deliverables/GroupNormSiluQuant/).

Compute (per group of elemNum = (C/num_groups)*HW elements):
    mean = mean(x)                          # over each group
    var  = var(x, ddof=0)                   # population variance
    rstd = 1/sqrt(var + eps)
    x_norm = (x - mean) * rstd
    y_f    = x_norm * gamma[c] + beta[c]     # per-channel affine
    silu   = activate_silu ? y_f*sigmoid(y_f) : y_f
    y_i8   = clamp(round(silu / quantScale), -128, 127)   # per-tensor(len 1) or per-channel(len C)
    meanOut/rstdOut: per group, shape (N, num_groups), cast back to x dtype.

Canonical IO order (group_norm_silu_quant_def.cpp):
    inputs : x, gamma, beta, quantScale      (x/gamma/beta same dtype bf16|fp16; quantScale fp32)
    outputs: yOut(int8), meanOut, rstdOut
    attrs  : num_groups(REQUIRED int), eps(OPTIONAL float=1e-5), activate_silu(OPTIONAL bool=True)

TTK passes input arrays positionally in CSV input_shapes order; attributes + output_dtypes via **kwargs.
"""

import numpy as np

try:
    from ml_dtypes import bfloat16 as _bf16
except ImportError:
    _bf16 = None


def _f32(a):
    return np.asarray(a).astype(np.float32)


def _cast_back(arr_f32, target):
    if target == "bfloat16":
        return (
            arr_f32.astype(_bf16) if _bf16 is not None else arr_f32.astype(np.float32)
        )
    return arr_f32.astype(target)


def _attr(kwargs, name, default):
    v = kwargs.get(name)
    if v is None:
        attrs = kwargs.get("attributes")
        if isinstance(attrs, dict):
            v = attrs.get(name)
    return default if v is None else v


def __golden_group_norm_silu_quant(x, gamma, beta, quant_scale, **kwargs):
    num_groups = int(_attr(kwargs, "num_groups", 1))
    eps = float(_attr(kwargs, "eps", 1e-5))
    silu = _attr(kwargs, "activate_silu", True)
    if isinstance(silu, str):
        silu = silu.strip().lower() in ("1", "true", "yes")
    silu = bool(silu)

    output_dtypes = kwargs.get("output_dtypes")

    def _od(i, default):
        if output_dtypes and i < len(output_dtypes):
            od = output_dtypes[i]
            return od[0] if isinstance(od, (tuple, list)) else str(od)
        return default

    x_dt = str(np.asarray(x).dtype)
    mean_dt = _od(1, x_dt)
    rstd_dt = _od(2, x_dt)

    xf = _f32(x)
    N, C = xf.shape[0], xf.shape[1]
    HW = int(np.prod(xf.shape[2:])) if xf.ndim > 2 else 1
    G = num_groups

    gf = _f32(gamma).reshape(1, C, 1)
    bf = _f32(beta).reshape(1, C, 1)

    xg = xf.reshape(N, G, -1)  # (N, G, elemNum)
    mean = xg.mean(axis=-1, keepdims=True)  # (N, G, 1)
    var = xg.var(axis=-1, ddof=0, keepdims=True)  # population variance
    rstd = 1.0 / np.sqrt(var + eps)  # (N, G, 1)

    gn = ((xg - mean) * rstd).reshape(N, C, HW)  # normalized
    out = gn * gf + bf  # affine (N, C, HW)
    if silu:
        out = out * (1.0 / (1.0 + np.exp(-out)))

    qs = _f32(quant_scale).reshape(-1)
    if qs.size == C:
        scale = qs.reshape(1, C, 1)  # per-channel
    else:
        scale = np.float32(qs[0])  # per-tensor
    q = np.clip(np.round(out / scale), -128.0, 127.0)
    y_i8 = q.astype(np.int8).reshape(np.asarray(x).shape)

    mean_out = _cast_back(mean.reshape(N, G), mean_dt)
    rstd_out = _cast_back(rstd.reshape(N, G), rstd_dt)
    return [y_i8, mean_out, rstd_out]


__golden__ = {"kernel": {"group_norm_silu_quant": "__golden_group_norm_silu_quant"}}
