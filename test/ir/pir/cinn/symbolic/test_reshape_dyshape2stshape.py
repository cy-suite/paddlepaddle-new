# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

import numpy as np

import paddle
from paddle.static import InputSpec


class ReshapeCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_0,
    ):
        var_0 = var_0 + 1.0
        var_1 = paddle.sqrt(var_0)
        var_2 = paddle.reshape(var_1, shape=[1, 32, 64])
        var_3 = var_2 * 2
        var_4 = var_0 / 2
        return var_3, var_4


class TestSigmoid(unittest.TestCase):
    def setUp(self):
        self.inputs = (paddle.rand(shape=[1, 32, 16, 4], dtype=paddle.float32),)
        self.net = ReshapeCase()

    def train(self, net, to_static, with_cinn=False):
        if to_static:
            if with_cinn:
                input_spec = [
                    InputSpec(shape=[None, 32, 16, 4], dtype='float32')
                ]
                net = paddle.jit.to_static(
                    net,
                    backend="CINN",
                    input_spec=input_spec,
                    full_graph=True,
                )
            else:
                net = paddle.jit.to_static(
                    net,
                    backend=None,
                    full_graph=True,
                )
        paddle.seed(123)
        outs = net(*self.inputs)
        return outs

    def test_ast_prim_cinn(self):
        st_out = self.train(self.net, to_static=True)
        cinn_out = self.train(self.net, to_static=True, with_cinn=True)
        for st, cinn in zip(
            paddle.utils.flatten(st_out), paddle.utils.flatten(cinn_out)
        ):
            np.testing.assert_allclose(st.numpy(), cinn.numpy(), atol=1e-8)


if __name__ == '__main__':
    unittest.main()
