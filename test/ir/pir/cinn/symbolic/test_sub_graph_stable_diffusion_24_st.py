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

# repo: diffusers_sub_graph
# model: stable_diffusion
# api:paddle.nn.functional.common.interpolate||api:paddle.nn.functional.conv.conv2d
import unittest

import numpy as np

import paddle


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
            shape=[640, 640, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_1 = self.create_parameter(
            shape=[640],
            dtype=paddle.float32,
        )
        self.size = [3, 8]

    def forward(
        self,
        var_0,  # (shape: [1, 640, 1, 1], dtype: paddle.float32, stop_gradient: False)
    ):
        var_2 = paddle.nn.functional.common.interpolate(
            var_0, size=self.size, mode='nearest'
        )
        var_3 = paddle.nn.functional.conv.conv2d(
            var_2, self.parameter_0, self.parameter_1, [1, 1], 1, [1, 1], 1
        )
        return var_3


def create_paddle_inputs():
    inputs = (paddle.rand(shape=[1, 640, 1, 1], dtype=paddle.float32),)
    return inputs


class TestLayer(unittest.TestCase):
    def setUp(self):
        self.inputs = create_paddle_inputs()
        self.net = LayerCase()

    def train(self, net, to_static, with_prim=False, with_cinn=False):
        if to_static:
            paddle.base.core._set_prim_all_enabled(with_prim)
            if with_cinn:
                assert (
                    with_prim
                ), "with_cinn=True but with_prim=False is unsupported"
                net = paddle.jit.to_static(net, backend="CINN", full_graph=True)
            else:
                net = paddle.jit.to_static(net, backend=None, full_graph=True)
        paddle.seed(123)
        outs = net(*self.inputs)
        return outs

    def test_ast_prim_cinn(self):
        st_out = self.train(self.net, to_static=True)
        cinn_out = self.train(
            self.net, to_static=True, with_prim=True, with_cinn=False
        )
        for st, cinn in zip(
            paddle.utils.flatten(st_out), paddle.utils.flatten(cinn_out)
        ):
            np.testing.assert_allclose(st.numpy(), cinn.numpy(), atol=1e-8)


if __name__ == '__main__':
    unittest.main()
