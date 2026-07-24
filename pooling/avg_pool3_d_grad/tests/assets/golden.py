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

__golden__ = {"kernel": {"avg_pool3_d_grad": "avg_pool3_d_grad_golden"}}


def avg_pool3_d_grad_golden(
    orig_input_shape,
    grads,
    *,
    ksize,
    strides,
    pads,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=0,
    data_format="NDHWC",
    **kwargs,
):
    """
    Golden function for avg_pool3_d_grad.
    All the parameters (names and order) follow @avg_pool3_d_grad_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """
    import tensorflow as tf

    tf.compat.v1.disable_eager_execution()

    if len(grads.shape) != 6:
        raise RuntimeError(
            "avgpool3dgrad testcase golden function supports NDC1HWC0 input only!"
        )
    # Collect shape info
    n_index = data_format.index("N")
    d_index = data_format.index("D")
    h_index = data_format.index("H")
    w_index = data_format.index("W")
    c_index = data_format.index("C")
    stride_h, stride_w, stride_d = strides[h_index], strides[w_index], strides[d_index]
    filter_h, filter_w, filter_d = ksize[h_index], ksize[w_index], ksize[d_index]
    GN, GD, GC, GH, GW, C0 = grads.shape
    IN, ID, IH, IW, IC = (
        orig_input_shape[n_index],
        orig_input_shape[d_index],
        orig_input_shape[h_index],
        orig_input_shape[w_index],
        orig_input_shape[c_index],
    )
    IC = (IC + 15) // 16 * 16
    if all(i == 0 for i in pads):
        padding = "VALID"
    else:
        padding = "SAME"

    # grads to NDHWC
    output_backprop = grads.transpose(0, 1, 3, 4, 2, 5).reshape(GN, GD, GH, GW, GC * C0)
    grads_tensor = tf.compat.v1.placeholder(grads.dtype, shape=output_backprop.shape)
    grads_tensor = tf.compat.v1.cast(grads_tensor, tf.float32)
    res = tf.compat.v1.raw_ops.AvgPool3DGrad(
        orig_input_shape=[GN, ID, IH, IW, GC * C0],
        grad=grads_tensor,
        ksize=[1, filter_d, filter_h, filter_w, 1],
        strides=[1, stride_d, stride_h, stride_w, 1],
        padding=padding,
        data_format="NDHWC",
        name="avg_pool3d_grad",
    )
    res = tf.compat.v1.cast(res, tf.float16)
    feed_dict = {grads_tensor: output_backprop}
    init_op = tf.compat.v1.global_variables_initializer()

    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        out = sess.run(res, feed_dict=feed_dict)

    res = (
        out.reshape((IN, ID, IH, IW, IC // C0, C0))
        .transpose(0, 1, 4, 2, 3, 5)
        .copy()
        .astype(np.float16)
    )
    return res
