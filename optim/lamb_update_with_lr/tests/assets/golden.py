#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import numpy as np


__golden__ = {"kernel": {"lamb_update_with_lr": "lamb_update_with_lr_golden"}}


def _scalars(*xs):
    return tuple(float(np.asarray(x, "float32").reshape(-1)[0]) for x in xs)


def lamb_update_with_lr_golden(
    input_greater1,
    input_greater_realdiv,
    input_realdiv,
    input_mul0,
    input_mul1,
    input_sub,
    greater_y,
    select_e,
    minimum_y,
    **kwargs,
):
    """Golden for LambUpdateWithLr. Params follow lamb_update_with_lr_def.cpp (without outputs). All inputs are numpy.ndarray."""
    dt = input_mul1.dtype
    g1, grd, rd, lr, gy, se, miny = _scalars(
        input_greater1,
        input_greater_realdiv,
        input_realdiv,
        input_mul0,
        greater_y,
        select_e,
        minimum_y,
    )
    upd, param = input_mul1.astype("float32"), input_sub.astype("float32")
    # Match kernel Vec::Div<float> (arch35 DivAlgo::INTRINSIC): IEEE-754 fp32 division.
    # x/0 -> +-inf, 0/0 -> nan (never raises); clipped downstream by min/max like the kernel.
    with np.errstate(divide="ignore", invalid="ignore"):
        realdiv0 = float(np.float32(grd) / np.float32(rd))
    select0 = realdiv0 if g1 > gy else se
    select1 = select0 if grd > gy else se
    clip = max(min(select1, miny), gy)
    return [(param - clip * lr * upd).astype(dt)]
