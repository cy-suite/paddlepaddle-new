# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import unittest

import numpy as np

import paddle

sys.path.insert(0, '.')


class TestFwdADContext(unittest.TestCase):
    def test_enter_exit_dual_level(self):
        paddle.autograd.enter_dual_level()
        assert (
            paddle.autograd.forward_mode._current_level == 0
        ), "The first enter dual level should be 0."
        paddle.autograd.exit_dual_level()
        assert (
            paddle.autograd.forward_mode._current_level == -1
        ), "The current dual level should be -1."

    def test_dual_tensor_deconstruct(self):
        x = paddle.randn([10, 10])
        with paddle.autograd.dual_level():
            x_t = paddle.randn(x.shape)
            x_dual = paddle.autograd.make_dual(x, x_t)
            x_p, x_t = paddle.autograd.unpack_dual(x_dual)
            self.assertIsInstance(x_p, paddle.Tensor)
            self.assertIsInstance(x_t, paddle.Tensor)
        x_p, x_t = paddle.autograd.unpack_dual(x_dual)
        self.assertIsInstance(x_p, paddle.Tensor)
        self.assertTrue(x_t is None)

    def test_reverse_on_forward(self):
        x = paddle.randn([100, 100])
        x.stop_gradient = False

        def func(x):
            return paddle.tanh(x)

        with paddle.autograd.dual_level():
            x_primal = x
            x_tangent = paddle.randn(x.shape)
            x_dual = paddle.autograd.forward_mode.make_dual(x_primal, x_tangent)

            y_dual = func(x_dual)

            y_primal, y_tangent = paddle.autograd.forward_mode.unpack_dual(
                y_dual
            )

        y = func(x)
        dy_dx_bwd = paddle.grad(y, x, x_tangent, create_graph=True)[0]
        np.testing.assert_allclose(
            dy_dx_bwd.numpy(), y_tangent.numpy(), 1e-6, 1e-6
        )

        vv = paddle.randn(y_tangent.shape)
        ddy_ddx_bwdfwd = paddle.grad(
            y_tangent, x_primal, vv, create_graph=False
        )[0]
        ddy_ddx_bwdbwd = paddle.grad(dy_dx_bwd, x, vv, create_graph=False)[0]

        np.testing.assert_allclose(
            ddy_ddx_bwdfwd.numpy(), ddy_ddx_bwdbwd.numpy(), 1e-6, 1e-6
        )


class TestFwdAD_eager_ops(unittest.TestCase):
    def test_concat_jvp(self):
        xs = [paddle.randn([2, 2, 1]) for _ in range(4)]
        vs = [paddle.randn(xs[i].shape) for i in range(4)]
        for i in range(4):
            xs[i].stop_gradient = False
            vs[i].stop_gradient = False

        axis = 1
        with paddle.autograd.dual_level():
            xs_dual = [paddle.autograd.make_dual(x, v) for x, v in zip(xs, vs)]
            # test concat with one vanilla Tensor
            xs_dual[2] = xs[i]
            y_dual = paddle.concat(xs_dual, axis)
            y_primal, y_tangent = paddle.autograd.unpack_dual(y_dual)
            # print(y_tangent)

        np.testing.assert_allclose(
            y_primal.numpy(),
            paddle.concat(xs, axis).numpy(),
            1e-6,
            1e-6,
        )
        np.testing.assert_allclose(
            y_tangent.numpy(),
            paddle.concat(vs, axis).numpy(),
            1e-6,
            1e-6,
        )


if __name__ == "__main__":
    unittest.main()
