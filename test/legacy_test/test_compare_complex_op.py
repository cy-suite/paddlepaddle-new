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

import numpy
import numpy as np
from utils import dygraph_guard, static_guard

import paddle
from paddle import base
from paddle.base import core


class TestEqualComplex64Api(unittest.TestCase):
    def setUp(self):
        self.op_type = 'equal'
        self.typename = ("float32", "complex64")
        self.dtype = "complex64"
        self.place = base.CPUPlace()
        if core.is_compiled_with_cuda():
            self.place = paddle.CUDAPlace(0)
        self.a_real_np = numpy.random.random(size=(6, 5, 4, 3)).astype(
            self.typename[0]
        )
        self.a_imag_np = numpy.random.random(size=(6, 5, 4, 3)).astype(
            self.typename[0]
        )
        self.b_real_np = numpy.random.random(size=(6, 5, 4, 3)).astype(
            self.typename[0]
        )
        self.b_imag_np = numpy.random.random(size=(6, 5, 4, 3)).astype(
            self.typename[0]
        )

    def test_dynamic_api(self):
        with dygraph_guard():
            a_complex_np = self.a_real_np + 1j * self.a_imag_np
            b_complex_np = self.b_real_np + 1j * self.b_imag_np
            a_complex = paddle.to_tensor(a_complex_np, dtype=self.dtype)
            b_complex = paddle.to_tensor(b_complex_np, dtype=self.dtype)
            c_np = a_complex_np == b_complex_np
            c = a_complex.equal(b_complex)
            np.testing.assert_allclose(c.numpy(), c_np)

    def test_static_api(self):
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                a_complex = paddle.static.data(
                    name='a', shape=[6, 5, 4, 3], dtype=self.dtype
                )
                b_complex = paddle.static.data(
                    name='b', shape=[6, 5, 4, 3], dtype=self.dtype
                )
                op = eval(f"paddle.{self.op_type}")
                c = op(a_complex, b_complex)
                exe = paddle.static.Executor(self.place)
                c_np = (
                    self.a_real_np + 1j * self.a_imag_np
                    == self.b_real_np + 1j * self.b_imag_np
                )

                c_out = exe.run(
                    feed={
                        'a': self.a_real_np + 1j * self.a_imag_np,
                        'b': self.b_real_np + 1j * self.b_imag_np,
                    },
                    fetch_list=[c],
                )
                np.testing.assert_allclose(c_out[0], c_np)

    def test_dygraph_special_case(self):
        with dygraph_guard():
            a_np = np.array(1 + 1j, dtype=self.dtype)
            a = paddle.to_tensor(1 + 1j, dtype=self.dtype)
            b = complex(1, 1)
            c_np = a_np == b
            c = a.equal(b)
            np.testing.assert_allclose(c.numpy(), c_np)


class TestEqualComplex128Api(unittest.TestCase):
    def setUp(self):
        self.op_type = 'equal'
        self.typename = ("float64", "complex128")
        self.dtype = "complex128"
        self.place = base.CPUPlace()
        if core.is_compiled_with_cuda():
            self.place = paddle.CUDAPlace(0)
        self.a_real_np = numpy.random.random(size=(6, 5, 4, 3)).astype(
            self.typename[0]
        )
        self.a_imag_np = numpy.random.random(size=(6, 5, 4, 3)).astype(
            self.typename[0]
        )
        self.b_real_np = numpy.random.random(size=(6, 5, 4, 3)).astype(
            self.typename[0]
        )
        self.b_imag_np = numpy.random.random(size=(6, 5, 4, 3)).astype(
            self.typename[0]
        )

    def test_dynamic_api(self):
        with dygraph_guard():
            a_complex_np = self.a_real_np + 1j * self.a_imag_np
            b_complex_np = self.b_real_np + 1j * self.b_imag_np
            a_complex = paddle.to_tensor(a_complex_np, dtype=self.dtype)
            b_complex = paddle.to_tensor(b_complex_np, dtype=self.dtype)
            c_np = a_complex_np == b_complex_np
            c = a_complex.equal(b_complex)
            np.testing.assert_allclose(c.numpy(), c_np)

    def test_static_api(self):
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                a_complex = paddle.static.data(
                    name='a', shape=[6, 5, 4, 3], dtype=self.dtype
                )
                b_complex = paddle.static.data(
                    name='b', shape=[6, 5, 4, 3], dtype=self.dtype
                )
                op = eval(f"paddle.{self.op_type}")
                c = op(a_complex, b_complex)
                exe = paddle.static.Executor(self.place)
                c_np = (
                    self.a_real_np + 1j * self.a_imag_np
                    == self.b_real_np + 1j * self.b_imag_np
                )

                c_out = exe.run(
                    feed={
                        'a': self.a_real_np + 1j * self.a_imag_np,
                        'b': self.b_real_np + 1j * self.b_imag_np,
                    },
                    fetch_list=[c],
                )
                np.testing.assert_allclose(c_out[0], c_np)

    def test_dygraph_special_case(self):
        with dygraph_guard():
            a_np = np.array(1 + 1j, dtype=self.dtype)
            a = paddle.to_tensor(1 + 1j, dtype=self.dtype)
            b = complex(1, 1)
            c_np = a == b
            c = a.equal(b)
            np.testing.assert_allclose(c.numpy(), c_np)


class TestNotEqualComplex64Api(unittest.TestCase):
    def setUp(self):
        self.op_type = 'not_equal'
        self.typename = ("float32", "complex64")
        self.dtype = "complex64"
        self.place = base.CPUPlace()
        if core.is_compiled_with_cuda():
            self.place = paddle.CUDAPlace(0)
        self.a_real_np = numpy.random.random(size=(6, 5, 4, 3)).astype(
            self.typename[0]
        )
        self.a_imag_np = numpy.random.random(size=(6, 5, 4, 3)).astype(
            self.typename[0]
        )
        self.b_real_np = numpy.random.random(size=(6, 5, 4, 3)).astype(
            self.typename[0]
        )
        self.b_imag_np = numpy.random.random(size=(6, 5, 4, 3)).astype(
            self.typename[0]
        )

    def test_dynamic_api(self):
        with dygraph_guard():
            a_complex_np = self.a_real_np + 1j * self.a_imag_np
            b_complex_np = self.b_real_np + 1j * self.b_imag_np
            a_complex = paddle.to_tensor(a_complex_np, dtype=self.dtype)
            b_complex = paddle.to_tensor(b_complex_np, dtype=self.dtype)
            c_np = a_complex_np != b_complex_np
            c = a_complex.not_equal(b_complex)
            np.testing.assert_allclose(c.numpy(), c_np)

    def test_static_api(self):
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                a_complex = paddle.static.data(
                    name='a', shape=[6, 5, 4, 3], dtype=self.dtype
                )
                b_complex = paddle.static.data(
                    name='b', shape=[6, 5, 4, 3], dtype=self.dtype
                )
                op = eval(f"paddle.{self.op_type}")
                c = op(a_complex, b_complex)
                exe = paddle.static.Executor(self.place)
                c_np = (
                    self.a_real_np + 1j * self.a_imag_np
                    != self.b_real_np + 1j * self.b_imag_np
                )

                c_out = exe.run(
                    feed={
                        'a': self.a_real_np + 1j * self.a_imag_np,
                        'b': self.b_real_np + 1j * self.b_imag_np,
                    },
                    fetch_list=[c],
                )
                np.testing.assert_allclose(c_out[0], c_np)


class TestNotEqualComplex128Api(unittest.TestCase):
    def setUp(self):
        self.op_type = 'not_equal'
        self.typename = ("float64", "complex128")
        self.dtype = "complex128"
        self.place = base.CPUPlace()
        if core.is_compiled_with_cuda():
            self.place = paddle.CUDAPlace(0)
        self.a_real_np = numpy.random.random(size=(6, 5, 4, 3)).astype(
            self.typename[0]
        )
        self.a_imag_np = numpy.random.random(size=(6, 5, 4, 3)).astype(
            self.typename[0]
        )
        self.b_real_np = numpy.random.random(size=(6, 5, 4, 3)).astype(
            self.typename[0]
        )
        self.b_imag_np = numpy.random.random(size=(6, 5, 4, 3)).astype(
            self.typename[0]
        )

    def test_dynamic_api(self):
        with dygraph_guard():
            a_complex_np = self.a_real_np + 1j * self.a_imag_np
            b_complex_np = self.b_real_np + 1j * self.b_imag_np
            a_complex = paddle.to_tensor(a_complex_np, dtype=self.dtype)
            b_complex = paddle.to_tensor(b_complex_np, dtype=self.dtype)
            c_np = a_complex_np != b_complex_np
            c = a_complex.not_equal(b_complex)
            np.testing.assert_allclose(c.numpy(), c_np)

    def test_static_api(self):
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                a_complex = paddle.static.data(
                    name='a', shape=[6, 5, 4, 3], dtype=self.dtype
                )
                b_complex = paddle.static.data(
                    name='b', shape=[6, 5, 4, 3], dtype=self.dtype
                )
                op = eval(f"paddle.{self.op_type}")
                c = op(a_complex, b_complex)
                exe = paddle.static.Executor(self.place)
                c_np = (
                    self.a_real_np + 1j * self.a_imag_np
                    != self.b_real_np + 1j * self.b_imag_np
                )

                c_out = exe.run(
                    feed={
                        'a': self.a_real_np + 1j * self.a_imag_np,
                        'b': self.b_real_np + 1j * self.b_imag_np,
                    },
                    fetch_list=[c],
                )
                np.testing.assert_allclose(c_out[0], c_np)


if __name__ == '__main__':
    unittest.main()
