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

import numpy as np

__golden__ = {"kernel": {"multilabel_margin_loss": "multilabel_margin_loss_golden"}}


def multilabel_margin_loss_golden(x, target, *, reduction="mean", **kwargs):
    """
    Golden for MultilabelMarginLoss. Parameter names/order follow multilabel_margin_loss_def.cpp
    (inputs x, target; attr reduction) without the outputs. Inputs are numpy.ndarray.

    Semantics (PyTorch MultiLabelMarginLoss, authoritative A2 golden = aten forward):
        loss[n] = (1/C) * sum_{j: target label} sum_{i not a label} max(0, 1 - (x[n][j] - x[n][i]))
    with the target row read up to the first -1 sentinel; is_target[n][i] = 1 iff i is a label.
    fp16/bf16 accumulate in fp32 (aten promotes), the loss is cast back to x dtype; is_target is int32.

    Returns:
        (y, is_target)
    """
    import torch

    ori_dtype = x.dtype
    low_prec = ("float16" in str(ori_dtype)) or ("bfloat16" in str(ori_dtype))
    xf = x.astype(np.float32) if low_prec else x

    x_t = torch.from_numpy(xf)
    tgt_t = torch.from_numpy(target.astype(np.int64))
    reduction_int = {"none": 0, "mean": 1, "sum": 2}[reduction]

    out, is_target = torch.ops.aten.multilabel_margin_loss_forward(
        x_t, tgt_t, reduction_int
    )

    y = out.numpy().astype(ori_dtype)
    is_target_np = is_target.numpy().astype(np.int32)
    return y, is_target_np
