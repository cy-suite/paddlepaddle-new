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

# repo: llm_sub_graphs
# model: chatglm2
# api:paddle.nn.functional.common.linear||method:transpose||method:reshape||method:astype||method:reshape||method:__ne__||method:astype||api:paddle.nn.functional.loss.cross_entropy||method:reshape||method:cast||method:reshape||method:cast||method:__mul__||api:paddle.tensor.math.sum||method:sum||method:__truediv__||method:astype||method:astype
import unittest

import numpy as np

import paddle


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
            shape=[32, 64896],
            dtype=paddle.float32,
        )

    def forward(
        self,
        var_0,  # (shape: [1024, 4, 32], dtype: paddle.float32, stop_gradient: False)
        var_1,  # (shape: [4, 1024], dtype: paddle.int64, stop_gradient: True)
    ):
        var_2 = paddle.nn.functional.common.linear(
            x=var_0, weight=self.parameter_0, bias=None, name=None
        )
        var_3 = var_2.transpose([1, 0, 2])
        var_4 = var_3.reshape([-1, 64896])
        var_5 = var_4.astype('float32')
        var_6 = var_1.reshape([-1])
        var_7 = var_1 != -100
        var_8 = var_7.astype('float32')
        var_9 = paddle.nn.functional.loss.cross_entropy(
            var_5,
            var_6,
            weight=None,
            ignore_index=-100,
            reduction='none',
            soft_label=False,
            axis=-1,
            use_softmax=True,
            label_smoothing=0.0,
            name=None,
        )
        var_10 = var_9.reshape([-1])
        var_11 = var_10.cast('float32')
        var_12 = var_8.reshape([-1])
        var_13 = var_12.cast('float32')
        var_14 = var_11 * var_13
        var_15 = paddle.tensor.math.sum(var_14)
        var_16 = var_8.sum()
        var_17 = var_15 / var_16
        var_18 = var_3.astype('float32')
        var_19 = var_17.astype('float32')
        return var_19, var_18


def create_paddle_inputs():
    inputs = (
        paddle.rand(shape=[1024, 4, 32], dtype=paddle.float32),
        paddle.randint(low=0, high=10, shape=[4, 1024], dtype=paddle.int64),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[1024, 4, 32]).astype('float32'),
        np.random.randint(low=0, high=10, size=[4, 1024], dtype='int64'),
    )
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
        # TODO: open this test case
        return
        for st, cinn in zip(
            paddle.utils.flatten(st_out), paddle.utils.flatten(cinn_out)
        ):
            np.testing.assert_allclose(st.numpy(), cinn.numpy(), atol=1e-8)


if __name__ == '__main__':
    unittest.main()
