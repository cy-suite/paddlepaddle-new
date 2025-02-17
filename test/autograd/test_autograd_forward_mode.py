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
from parameterized import parameterized

import paddle

sys.path.insert(0, '.')


class TestDualContext(unittest.TestCase):
    def test_dual_level(self):
        paddle.autograd.enter_dual_level()
        assert (
            paddle.autograd.forward_mode._current_level == 0
        ), "The first enter dual level should be 0."
        paddle.autograd.exit_dual_level()
        assert (
            paddle.autograd.forward_mode._current_level == -1
        ), "The current dual level should be -1."

    @parameterized.expand(
        [
            (True, True),
            (False, True),
            (True, False),
            (False, False),
        ]
    )
    def test_make_dual(self, x_p_st, x_t_st):
        # test stop_gradient should be same with primal
        x_p = paddle.randn([10, 10])
        x_p.stop_gradient = x_p_st
        with paddle.autograd.dual_level():
            x_t = paddle.randn(x_p.shape)
            x_t.stop_gradient = x_t_st
            x_dual = paddle.autograd.make_dual(x_p, x_t)
            self.assertIsInstance(x_dual, paddle.Tensor)
            self.assertEqual(x_dual.stop_gradient, x_p_st)

    @parameterized.expand(
        [
            (True, True),
            (False, True),
            (True, False),
            (False, False),
        ]
    )
    def test_unpack_dual(self, x_p_st, x_t_st):
        # test stop_gradient unchanged through make/unpack dual
        # and gradient chain should be built correctly between
        # (primal, UnpackedDualTensor.primal) and (tangent, UnpackedDualTensor.tangent)
        x_p = paddle.randn([10, 10])
        x_p.stop_gradient = x_p_st
        with paddle.autograd.dual_level():
            x_t = paddle.randn(x_p.shape)
            x_t.stop_gradient = x_t_st
            x_dual = paddle.autograd.make_dual(x_p, x_t)
            x_p_identity, x_t_view = paddle.autograd.unpack_dual(x_dual)
            self.assertEqual(x_p.data_ptr(), x_p_identity.data_ptr())
            self.assertIsInstance(x_p_identity, paddle.Tensor)
            self.assertIsInstance(x_t_view, paddle.Tensor)
            self.assertEqual(x_p_identity.stop_gradient, x_p_st)
            self.assertEqual(x_t_view.stop_gradient, x_t_st)

            if not x_p_st:
                v = paddle.randn(x_p.shape)
                dx_p = paddle.grad(x_p_identity, x_p, v)[0]
                np.testing.assert_allclose(v.numpy(), dx_p.numpy())

            if not x_t_st:
                v = paddle.randn(x_t.shape)
                dx_t = paddle.grad(x_t_view, x_t, v)[0]
                np.testing.assert_allclose(v.numpy(), dx_t.numpy())


class TestFwdAD_eager_ops(unittest.TestCase):
    def test_auto_elementwise_jvp(self):
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

    def test_auto_linear_jvp(self):
        x = paddle.randn([4, 4])
        x.stop_gradient = False

        def func(x, num_or_sections=4, axis=-1):
            return paddle.split(x, num_or_sections, axis)

        with paddle.autograd.dual_level():
            x_primal = x
            x_tangent = paddle.randn(x.shape)
            x_dual = paddle.autograd.forward_mode.make_dual(x_primal, x_tangent)

            y_dual = func(x_dual)  # List[Tensor]
            # print(y_dual)

            tmp = [
                paddle.autograd.forward_mode.unpack_dual(y_dual_)
                for y_dual_ in y_dual
            ]
            y_primal = [t[0] for t in tmp]
            y_tangent = [t[1] for t in tmp]

        y = func(x)

        for out1, out2 in zip(y, y_primal):
            np.testing.assert_allclose(out1.numpy(), out2.numpy(), 1e-6, 1e-6)

        size = x.shape[-1] // 4
        for i in range(4):
            grad1 = y_tangent[i]
            grad2 = x_tangent[:, i * size : (i + 1) * size]
            np.testing.assert_allclose(grad1.numpy(), grad2.numpy(), 1e-6, 1e-6)


if __name__ == "__main__":
    unittest.main()
