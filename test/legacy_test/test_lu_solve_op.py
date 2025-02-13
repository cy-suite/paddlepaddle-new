#   Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest

import paddle
from paddle import base
from paddle.base import core


def _transpose_last_2dim(x):
    """transpose the last 2 dimension of a tensor"""
    x_new_dims = list(range(len(x.shape)))
    x_new_dims[-1], x_new_dims[-2] = x_new_dims[-2], x_new_dims[-1]
    x = paddle.transpose(x, x_new_dims)
    return x


def get_inandout(A, b, trans="N", dtype="float64"):
    paddle.disable_static(base.CPUPlace())
    A = paddle.randn(A, dtype=dtype)
    b = paddle.randn(b, dtype=dtype)
    lu, pivots = paddle.linalg.lu(A)
    if trans == "N":  # Ax = b
        out = paddle.linalg.solve(A, b)
    elif trans == "T":  # A^Tx = b
        A = _transpose_last_2dim(A)
        out = paddle.linalg.solve(A, b)
    lu = lu.numpy().astype(dtype)
    pivots = pivots.numpy().astype("int32")
    b = b.numpy().astype(dtype)
    out = out.numpy().astype(dtype)
    paddle.enable_static()
    return lu, pivots, b, out


class TestLuSolveOp(OpTest):
    def setUp(self):

        self.python_api = paddle.linalg.lu_solve
        self.op_type = "lu_solve"
        self.init_value()
        self.LU, self.pivots, self.b, self.out = get_inandout(
            self.A_shape, self.b_shape, self.trans, self.dtype
        )
        self.inputs = {
            'X': self.b,
            'Lu': self.LU,
            'Pivots': self.pivots,
        }
        self.attrs = {'trans': self.trans}
        self.outputs = {'Out': self.out}

    def init_value(self):
        self.A_shape = [15, 15]
        self.b_shape = [15, 10]
        self.trans = "N"
        self.dtype = "float64"

    def test_check_output(self):
        paddle.enable_static()
        self.check_output(check_pir=True)
        paddle.disable_static()

    def test_check_grad(self):
        paddle.enable_static()
        self.check_grad(['X', 'Lu'], ['Out'], check_pir=True)
        paddle.disable_static()


class TestLuSolveOpAPI(unittest.TestCase):
    def setUp(self):
        self.init_value()
        self.LU, self.pivots, self.b, self.out = get_inandout(
            self.A_shape, self.b_shape, self.trans, self.dtype
        )
        self.place = []
        self.place.append(base.CPUPlace())
        if core.is_compiled_with_cuda():
            self.place.append(base.CUDAPlace(0))

    def init_value(self):
        # Ax = b
        self.A_shape = [15, 15]
        self.b_shape = [15, 10]
        self.trans = "N"
        self.dtype = "float64"

    def test_dygraph(self):
        def run(place):
            paddle.disable_static(place)
            LU = paddle.to_tensor(self.LU, self.dtype)
            pivots = paddle.to_tensor(self.pivots, self.dtype)
            b = paddle.to_tensor(self.b, self.dtype)
            lu_solve_x = paddle.linalg.lu_solve(b, LU, pivots, self.trans)
            np.testing.assert_allclose(lu_solve_x.numpy(), self.out, rtol=1e-05)
            paddle.enable_static()

        for place in self.place:
            run(place)

    def test_static(self):
        def run(place):
            paddle.enable_static()
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                b = paddle.static.data(
                    name='X', shape=self.b_shape, dtype=self.dtype
                )
                LU = paddle.static.data(
                    name='Lu', shape=self.LU.shape, dtype=self.dtype
                )
                pivots = paddle.static.data(
                    name='Pivots', shape=self.pivots.shape, dtype=self.dtype
                )
                lu_solve_x = paddle.linalg.lu_solve(b, LU, pivots, self.trans)
                exe = base.Executor(place)
                fetches = exe.run(
                    paddle.static.default_main_program(),
                    feed={
                        'X': self.b,
                        'Lu': self.LU,
                        'Pivots': self.pivots,
                    },
                    fetch_list=[lu_solve_x],
                )
                np.testing.assert_allclose(fetches[0], self.out, rtol=1e-05)
            paddle.disable_static()

        for place in self.place:
            run(place)


class TestLuSolveOpAPI2(TestLuSolveOpAPI):
    def init_value(self):
        # Ax = b
        self.A_shape = [2, 15, 15]
        self.b_shape = [1, 15, 10]
        self.trans = "N"
        self.dtype = "float64"


class TestLuSolveOpAPI3(TestLuSolveOpAPI):
    def init_value(self):
        # A^Tx = b
        self.A_shape = [15, 15]
        self.b_shape = [15, 10]
        self.trans = "T"
        self.dtype = "float64"


class TestLuSolveOpAPI4(TestLuSolveOpAPI):
    def init_value(self):
        # A^Tx = b
        self.A_shape = [2, 15, 15]
        self.b_shape = [1, 15, 10]
        self.trans = "T"
        self.dtype = "float64"


class TestLuSolveOpAPI5(TestLuSolveOpAPI):
    def init_value(self):
        # Ax = b
        self.A_shape = [15, 15]
        self.b_shape = [15, 10]
        self.trans = "N"
        self.dtype = "float32"


class TestLuSolveOpAPI6(TestLuSolveOpAPI):
    def init_value(self):
        # Ax = b
        self.A_shape = [2, 15, 15]
        self.b_shape = [1, 15, 10]
        self.trans = "N"
        self.dtype = "float32"


class TestLuSolveOpAPI7(TestLuSolveOpAPI):
    def init_value(self):
        # A^Tx = b
        self.A_shape = [15, 15]
        self.b_shape = [15, 10]
        self.trans = "T"
        self.dtype = "float32"


class TestLuSolveOpAPI8(TestLuSolveOpAPI):
    def init_value(self):
        # A^Tx = b
        self.A_shape = [2, 15, 15]
        self.b_shape = [1, 15, 10]
        self.trans = "T"
        self.dtype = "float32"


class TestLSolveError(unittest.TestCase):
    def test_errors(self):
        with paddle.base.dygraph.guard():
            # The size of b should gather than 2.
            def test_b_size():
                b = paddle.randn([3])
                lu = paddle.randn([3, 3])
                pivots = paddle.randn([3])
                paddle.linalg.lu_solve(b, lu, pivots)

            self.assertRaises(ValueError, test_b_size)

            # The size of lu should gather than 2.
            def test_lu_size():
                b = paddle.randn([3, 1])
                lu = paddle.randn([3])
                pivots = paddle.randn([3])
                paddle.linalg.lu_solve(b, lu, pivots)

            self.assertRaises(ValueError, test_lu_size)

            # The size of pivots should gather than 1.
            def test_pivots_size():
                b = paddle.randn([3, 1])
                lu = paddle.randn([3, 3])
                pivots = paddle.randn([])
                paddle.linalg.lu_solve(b, lu, pivots)

            self.assertRaises(ValueError, test_pivots_size)

            # b.shape[-2] should equal to lu.shape[-2].
            def test_b_lu_shape():
                b = paddle.randn([1, 3])
                lu = paddle.randn([3, 3])
                pivots = paddle.randn([3])
                paddle.linalg.lu_solve(b, lu, pivots)

            self.assertRaises(ValueError, test_b_lu_shape)

            # lu.shape[-1] should equal to pivots.shape[-1].
            def test_b_pivots_shape():
                b = paddle.randn([3, 1])
                lu = paddle.randn([3, 3])
                pivots = paddle.randn([2])
                paddle.linalg.lu_solve(b, lu, pivots)

            self.assertRaises(ValueError, test_b_pivots_shape)

            # lu.shape[-2] should equal to lu.shape[-1].
            def test_lu_shape():
                b = paddle.randn([3, 1])
                lu = paddle.randn([3, 2])
                pivots = paddle.randn([3])
                paddle.linalg.lu_solve(b, lu, pivots)

            self.assertRaises(ValueError, test_lu_shape)


if __name__ == "__main__":
    paddle.seed(2025)
    unittest.main()
