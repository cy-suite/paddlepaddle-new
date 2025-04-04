# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import unittest

import numpy

os.environ['FLAGS_prim_all'] = 'true'
os.environ['FLAGS_prim_enable_dynamic'] = 'true'
os.environ['FLAGS_use_cinn'] = '1'

import paddle


def init():
    var_1 = paddle.rand([32, 1])
    var_1 = paddle.cast(var_1, 'int32')
    var_0 = paddle.rand([32, 80])
    return var_1, var_0


def func(var_1, var_0):
    var_15 = paddle.full(shape=[1], dtype='float32', fill_value=1.0)
    var_16 = var_0 * -1.0 + 0.0
    var_17 = paddle.exp(var_16)
    var_18 = var_15 + var_17
    var_19 = var_15 / var_18
    var_20 = var_19 * -1.0 + 1.0
    var_21 = var_20 * 1.0 + 1e-12
    var_22 = paddle.log(var_21)
    var_23 = var_22 * -1.0 + 0.0
    var_24 = var_23 * 0.75 + 0.0
    var_25 = var_19 * var_19
    var_26 = var_24 * var_25
    var_27 = var_19 * 1.0 + 1e-12
    var_28 = paddle.log(var_27)
    var_29 = var_28 * -1.0 + 0.0
    var_30 = var_29 * 0.25 + 0.0
    var_31 = var_19 * -1.0 + 1.0
    var_32 = var_31 * var_31
    var_33 = var_30 * var_32
    var_34 = paddle.squeeze(var_1, [-1])
    var_35 = paddle.transpose(var_33, perm=[1, 0])
    var_36 = paddle.unsqueeze(var_34, [-1])
    var_37 = paddle.gather_nd(var_35, var_36)
    var_38 = paddle.transpose(var_37, perm=[1, 0])
    var_39 = paddle.squeeze(var_1, [-1])
    var_40 = paddle.transpose(var_26, perm=[1, 0])
    var_41 = paddle.unsqueeze(var_39, [-1])
    var_42 = paddle.gather_nd(var_40, var_41)
    var_43 = paddle.transpose(var_42, perm=[1, 0])
    var_44 = var_38 - var_43
    var_45 = var_44 * 2.0 + 0.0
    return (
        var_24,
        var_25,
        var_30,
        var_32,
        var_35,
        var_36,
        var_40,
        var_41,
        var_45,
    )


def input_spec():
    return [
        paddle.static.InputSpec(shape=[None, 1], dtype='int32', name='var_1'),
        paddle.static.InputSpec(
            shape=[None, 80], dtype='float32', name='var_0'
        ),
    ]


class TestCase(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def compare_result(self, dy_compute, data_init):
        static_compute = paddle.jit.to_static(
            full_graph=True,
            backend="CINN",
            input_spec=input_spec(),
        )(dy_compute)
        inputs = data_init()
        dy_out = dy_compute(*inputs)
        st_out = static_compute(*inputs)
        if isinstance(dy_out, paddle.Tensor):
            numpy.testing.assert_allclose(dy_out, st_out, atol=1e-5, rtol=1e-6)
            return
        for d, s in zip(dy_out, st_out):
            numpy.testing.assert_allclose(d, s, atol=1e-5, rtol=1e-6)

    def test_case(self):
        self.compare_result(func, init)


if __name__ == "__main__":
    unittest.main()
