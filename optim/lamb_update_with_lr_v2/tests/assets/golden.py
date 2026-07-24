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


__golden__ = {"kernel": {"lamb_update_with_lr_v2": "lamb_update_with_lr_v2_golden"}}


def _scalars(*xs):
    return tuple(float(np.asarray(x, "float32").reshape(-1)[0]) for x in xs)


def lamb_update_with_lr_v2_golden(x1, x2, x3, x4, x5, greater_y, select_e, **kwargs):
    """Golden for LambUpdateWithLrV2. Params follow lamb_update_with_lr_v2_def.cpp (without outputs). All inputs are numpy.ndarray."""
    dt = x4.dtype
    a, b, lr, gy, se = _scalars(x1, x2, x3, greater_y, select_e)
    upd, param = x4.astype("float32"), x5.astype("float32")
    # Match kernel Vec::Div<float> (arch35 DivAlgo::INTRINSIC): IEEE-754 fp32 division.
    # b<=gy stays on select_e (kernel Select); when b>gy and b==0 the kernel outputs inf
    # (v2 has NO clip), so mirror that instead of raising ZeroDivisionError.
    with np.errstate(divide="ignore", invalid="ignore"):
        inner = float(np.float32(a) / np.float32(b)) if b > gy else se
    ratio = inner if a > gy else se
    return [(param - lr * ratio * upd).astype(dt)]
