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

__golden__ = {"kernel": {"ge_glu_v2": "ge_glu_v2_golden"}}


def do_gelu(x, approximate):
    import torch

    if approximate == 0:
        temp_x = x / np.sqrt(2)
        gelu_y = 0.5 * x * (1.0 + torch.erf(temp_x))
    else:
        alpha = torch.sqrt(torch.tensor(2.0 / torch.pi))
        beta = 0.044715
        temp_x = alpha * (x + beta * x * x * x)
        gelu_y = 0.5 * x * (1.0 + torch.tanh(temp_x))
    return gelu_y


def process(input_x, split_dim, activateLeft, approximate):
    import torch

    tensor_x = torch.from_numpy(input_x)
    if activateLeft:
        gate, x = tensor_x.chunk(2, dim=split_dim)
    else:
        x, gate = tensor_x.chunk(2, dim=split_dim)
    y_gelu = do_gelu(gate, approximate)
    y = x * y_gelu
    return y, y_gelu


def ge_glu_v2_golden(x, *, dim=-1, approximate=1, activate_left=False, **kwargs):
    """
    Golden function for ge_glu_v2.
    All the parameters (names and order) follow @ge_glu_v2_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """
    dtype = x.dtype
    if dtype != np.float64:
        x = x.astype(np.float32)

    y, y_gelu = process(x, dim, activate_left, approximate)
    return y.numpy().astype(dtype), y_gelu.numpy().astype(dtype)
