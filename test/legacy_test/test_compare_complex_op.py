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
        self.callback = lambda _a, _b: _a == _b
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
        self.real_np_2d = numpy.random.random(size=(4, 3)).astype(
            self.typename[0]
        )
        self.imag_np_2d = numpy.random.random(size=(4, 3)).astype(
            self.typename[0]
        )
        self.a_inf = np.array([1, np.inf, -np.inf], self.dtype)
        self.b_inf = np.array([1, -np.inf, np.inf], self.dtype)
        self.a_nan = np.array([1, np.nan, -np.nan], self.dtype)
        self.b_nan = np.array([1, -np.nan, -np.nan], self.dtype)

    def test_dynamic_api(self):
        with dygraph_guard():
            a_complex_np = self.a_real_np + 1j * self.a_imag_np
            b_complex_np = self.b_real_np + 1j * self.b_imag_np
            a_complex = paddle.to_tensor(a_complex_np, dtype=self.dtype)
            b_complex = paddle.to_tensor(b_complex_np, dtype=self.dtype)
            c_np = self.callback(a_complex_np, b_complex_np)
            c = self.callback(a_complex, b_complex)
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
                c_np = self.callback(
                    self.a_real_np + 1j * self.a_imag_np,
                    self.b_real_np + 1j * self.b_imag_np,
                )
                c_out = exe.run(
                    feed={
                        'a': self.a_real_np + 1j * self.a_imag_np,
                        'b': self.b_real_np + 1j * self.b_imag_np,
                    },
                    fetch_list=[c],
                )
                np.testing.assert_allclose(c_out[0], c_np)

    def test_dygraph_nan_inf_special_case(self):
        with dygraph_guard():
            # nan + inf, return false directly.
            a_inf_complex_np = (self.a_nan + self.a_real_np) + 1j * (
                self.a_inf + self.a_imag_np
            )
            b_inf_complex_np = (self.b_nan + self.b_real_np) + 1j * (
                self.b_inf + self.b_imag_np
            )
            a_inf_complex = paddle.to_tensor(a_inf_complex_np, dtype=self.dtype)
            b_inf_complex = paddle.to_tensor(b_inf_complex_np, dtype=self.dtype)
            c_np = self.callback(a_inf_complex_np, b_inf_complex_np)
            c = self.callback(a_inf_complex, b_inf_complex)
            np.testing.assert_allclose(c.numpy(), c_np)

    def test_dygraph_inf_special_case(self):
        with dygraph_guard():
            a_inf_complex_np = (
                self.a_inf + self.a_real_np
            ) + self.a_imag_np * 1j
            b_inf_complex_np = (
                self.b_inf + self.b_real_np
            ) + self.b_imag_np * 1j
            a_inf_complex = paddle.to_tensor(a_inf_complex_np, dtype=self.dtype)
            b_inf_complex = paddle.to_tensor(b_inf_complex_np, dtype=self.dtype)
            c_np = self.callback(a_inf_complex_np, b_inf_complex_np)
            c = self.callback(a_inf_complex, b_inf_complex)
            np.testing.assert_allclose(c.numpy(), c_np)

    def test_dygraph_nan_special_case(self):
        with dygraph_guard():
            a_nan_complex_np = self.a_nan + self.a_real_np + 1j * self.a_imag_np
            b_nan_complex_np = self.b_nan + self.b_real_np + 1j * self.b_imag_np
            a_nan_complex = paddle.to_tensor(a_nan_complex_np, dtype=self.dtype)
            b_nan_complex = paddle.to_tensor(b_nan_complex_np, dtype=self.dtype)
            c_np = self.callback(a_nan_complex_np, b_nan_complex_np)
            c = self.callback(a_nan_complex, b_nan_complex)
            np.testing.assert_allclose(c.numpy(), c_np)

    def test_dygraph_broadcast_api(self):
        with dygraph_guard():
            a_complex_np = self.a_real_np + 1j * self.a_imag_np
            b_complex_np = self.real_np_2d + 1j * self.imag_np_2d
            a_complex = paddle.to_tensor(a_complex_np, dtype=self.dtype)
            b_complex = paddle.to_tensor(b_complex_np, dtype=self.dtype)
            c_np = self.callback(a_complex_np, b_complex_np)
            c = self.callback(a_complex, b_complex)
            np.testing.assert_allclose(c.numpy(), c_np)

    def test_dygraph_special_case(self):
        with dygraph_guard():
            a_np = np.array(1 + 1j, dtype=self.dtype)
            a = paddle.to_tensor(1 + 1j, dtype=self.dtype)
            b = complex(1, 1)
            c = self.callback(a, b)
            c_np = self.callback(a, b)
            np.testing.assert_allclose(c.numpy(), c_np)

    def test_static_nan_inf_special_case(self):
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                a_inf_complex = paddle.static.data(
                    name='a', shape=[6, 5, 4, 3], dtype=self.dtype
                )
                b_inf_complex = paddle.static.data(
                    name='b', shape=[6, 5, 4, 3], dtype=self.dtype
                )
                op = eval(f"paddle.{self.op_type}")
                c = op(a_inf_complex, b_inf_complex)
                exe = paddle.static.Executor(self.place)
                c_np = self.callback(
                    (self.a_nan + self.a_real_np)
                    + 1j * (self.a_inf + self.a_imag_np),
                    (self.b_nan + self.b_real_np)
                    + 1j * (self.b_inf + self.b_imag_np),
                )
                c_out = exe.run(
                    feed={
                        'a': (self.a_nan + self.a_real_np)
                        + 1j * (self.a_inf + self.a_imag_np),
                        'b': (self.b_nan + self.b_real_np)
                        + 1j * (self.b_inf + self.b_imag_np),
                    },
                    fetch_list=[c],
                )
                np.testing.assert_allclose(c_out[0], c_np)

    def test_static_inf_special_case(self):
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                a_inf_complex = paddle.static.data(
                    name='a', shape=[6, 5, 4, 3], dtype=self.dtype
                )
                b_inf_complex = paddle.static.data(
                    name='b', shape=[6, 5, 4, 3], dtype=self.dtype
                )
                op = eval(f"paddle.{self.op_type}")
                c = op(a_inf_complex, b_inf_complex)
                exe = paddle.static.Executor(self.place)
                c_np = self.callback(
                    (self.a_inf + self.a_real_np) + 1j * self.a_imag_np,
                    (self.b_inf + self.b_real_np) + 1j * self.b_imag_np,
                )
                c_out = exe.run(
                    feed={
                        'a': (self.a_inf + self.a_real_np)
                        + 1j * self.a_imag_np,
                        'b': (self.b_inf + self.b_real_np)
                        + 1j * self.b_imag_np,
                    },
                    fetch_list=[c],
                )
                np.testing.assert_allclose(c_out[0], c_np)

    def test_static_nan_special_case(self):
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                a_nan_complex = paddle.static.data(
                    name='a', shape=[6, 5, 4, 3], dtype=self.dtype
                )
                b_nan_complex = paddle.static.data(
                    name='b', shape=[6, 5, 4, 3], dtype=self.dtype
                )
                op = eval(f"paddle.{self.op_type}")
                c = op(a_nan_complex, b_nan_complex)
                exe = paddle.static.Executor(self.place)
                c_np = self.callback(
                    (self.a_nan + self.a_real_np) + 1j * self.a_imag_np,
                    (self.b_nan + self.b_real_np) + 1j * self.b_imag_np,
                )
                c_out = exe.run(
                    feed={
                        'a': (self.a_nan + self.a_real_np)
                        + 1j * self.a_imag_np,
                        'b': (self.b_nan + self.b_real_np)
                        + 1j * self.b_imag_np,
                    },
                    fetch_list=[c],
                )
                np.testing.assert_allclose(c_out[0], c_np)

    def test_static_broadcast_api(self):
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                a_complex = paddle.static.data(
                    name='a', shape=[6, 5, 4, 3], dtype=self.dtype
                )
                b_complex = paddle.static.data(
                    name='b', shape=[4, 3], dtype=self.dtype
                )
                op = eval(f"paddle.{self.op_type}")
                c = op(a_complex, b_complex)
                exe = paddle.static.Executor(self.place)
                c_np = self.callback(
                    self.a_real_np + 1j * self.a_imag_np,
                    self.real_np_2d + 1j * self.imag_np_2d,
                )
                c_out = exe.run(
                    feed={
                        'a': self.a_real_np + 1j * self.a_imag_np,
                        'b': self.real_np_2d + 1j * self.imag_np_2d,
                    },
                    fetch_list=[c],
                )
                np.testing.assert_allclose(c_out[0], c_np)


class TestEqualComplex128Api(unittest.TestCase):
    def setUp(self):
        self.callback = lambda _a, _b: _a == _b
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
        self.real_np_2d = numpy.random.random(size=(4, 3)).astype(
            self.typename[0]
        )
        self.imag_np_2d = numpy.random.random(size=(4, 3)).astype(
            self.typename[0]
        )
        self.a_inf = np.array([1, np.inf, -np.inf], self.dtype)
        self.b_inf = np.array([1, -np.inf, np.inf], self.dtype)
        self.a_nan = np.array([1, np.nan, -np.nan], self.dtype)
        self.b_nan = np.array([1, -np.nan, -np.nan], self.dtype)

    def test_dynamic_api(self):
        with dygraph_guard():
            a_complex_np = self.a_real_np + 1j * self.a_imag_np
            b_complex_np = self.b_real_np + 1j * self.b_imag_np
            a_complex = paddle.to_tensor(a_complex_np, dtype=self.dtype)
            b_complex = paddle.to_tensor(b_complex_np, dtype=self.dtype)
            c_np = self.callback(a_complex_np, b_complex_np)
            c = self.callback(a_complex, b_complex)
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
                c_np = self.callback(
                    self.a_real_np + 1j * self.a_imag_np,
                    self.b_real_np + 1j * self.b_imag_np,
                )
                c_out = exe.run(
                    feed={
                        'a': self.a_real_np + 1j * self.a_imag_np,
                        'b': self.b_real_np + 1j * self.b_imag_np,
                    },
                    fetch_list=[c],
                )
                np.testing.assert_allclose(c_out[0], c_np)

    def test_dygraph_nan_inf_special_case(self):
        with dygraph_guard():
            # nan + inf, return false directly.
            a_inf_complex_np = (self.a_nan + self.a_real_np) + 1j * (
                self.a_inf + self.a_imag_np
            )
            b_inf_complex_np = (self.b_nan + self.b_real_np) + 1j * (
                self.b_inf + self.b_imag_np
            )
            a_inf_complex = paddle.to_tensor(a_inf_complex_np, dtype=self.dtype)
            b_inf_complex = paddle.to_tensor(b_inf_complex_np, dtype=self.dtype)
            c_np = self.callback(a_inf_complex_np, b_inf_complex_np)
            c = self.callback(a_inf_complex, b_inf_complex)
            np.testing.assert_allclose(c.numpy(), c_np)

    def test_dygraph_inf_special_case(self):
        with dygraph_guard():
            a_inf_complex_np = (
                self.a_inf + self.a_real_np
            ) + self.a_imag_np * 1j
            b_inf_complex_np = (
                self.b_inf + self.b_real_np
            ) + self.b_imag_np * 1j
            a_inf_complex = paddle.to_tensor(a_inf_complex_np, dtype=self.dtype)
            b_inf_complex = paddle.to_tensor(b_inf_complex_np, dtype=self.dtype)
            c_np = self.callback(a_inf_complex_np, b_inf_complex_np)
            c = self.callback(a_inf_complex, b_inf_complex)
            np.testing.assert_allclose(c.numpy(), c_np)

    def test_dygraph_nan_special_case(self):
        with dygraph_guard():
            a_nan_complex_np = self.a_nan + self.a_real_np + 1j * self.a_imag_np
            b_nan_complex_np = self.b_nan + self.b_real_np + 1j * self.b_imag_np
            a_nan_complex = paddle.to_tensor(a_nan_complex_np, dtype=self.dtype)
            b_nan_complex = paddle.to_tensor(b_nan_complex_np, dtype=self.dtype)
            c_np = self.callback(a_nan_complex_np, b_nan_complex_np)
            c = self.callback(a_nan_complex, b_nan_complex)
            np.testing.assert_allclose(c.numpy(), c_np)

    def test_dygraph_broadcast_api(self):
        with dygraph_guard():
            a_complex_np = self.a_real_np + 1j * self.a_imag_np
            b_complex_np = self.real_np_2d + 1j * self.imag_np_2d
            a_complex = paddle.to_tensor(a_complex_np, dtype=self.dtype)
            b_complex = paddle.to_tensor(b_complex_np, dtype=self.dtype)
            c_np = self.callback(a_complex_np, b_complex_np)
            c = self.callback(a_complex, b_complex)
            np.testing.assert_allclose(c.numpy(), c_np)

    def test_dygraph_special_case(self):
        with dygraph_guard():
            a_np = np.array(1 + 1j, dtype=self.dtype)
            a = paddle.to_tensor(1 + 1j, dtype=self.dtype)
            b = complex(1, 1)
            c = self.callback(a, b)
            c_np = self.callback(a, b)
            np.testing.assert_allclose(c.numpy(), c_np)

    def test_static_nan_inf_special_case(self):
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                a_inf_complex = paddle.static.data(
                    name='a', shape=[6, 5, 4, 3], dtype=self.dtype
                )
                b_inf_complex = paddle.static.data(
                    name='b', shape=[6, 5, 4, 3], dtype=self.dtype
                )
                op = eval(f"paddle.{self.op_type}")
                c = op(a_inf_complex, b_inf_complex)
                exe = paddle.static.Executor(self.place)
                c_np = self.callback(
                    (self.a_nan + self.a_real_np)
                    + 1j * (self.a_inf + self.a_imag_np),
                    (self.b_nan + self.b_real_np)
                    + 1j * (self.b_inf + self.b_imag_np),
                )
                c_out = exe.run(
                    feed={
                        'a': (self.a_nan + self.a_real_np)
                        + 1j * (self.a_inf + self.a_imag_np),
                        'b': (self.b_nan + self.b_real_np)
                        + 1j * (self.b_inf + self.b_imag_np),
                    },
                    fetch_list=[c],
                )
                np.testing.assert_allclose(c_out[0], c_np)

    def test_static_inf_special_case(self):
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                a_inf_complex = paddle.static.data(
                    name='a', shape=[6, 5, 4, 3], dtype=self.dtype
                )
                b_inf_complex = paddle.static.data(
                    name='b', shape=[6, 5, 4, 3], dtype=self.dtype
                )
                op = eval(f"paddle.{self.op_type}")
                c = op(a_inf_complex, b_inf_complex)
                exe = paddle.static.Executor(self.place)
                c_np = self.callback(
                    (self.a_inf + self.a_real_np) + 1j * self.a_imag_np,
                    (self.b_inf + self.b_real_np) + 1j * self.b_imag_np,
                )
                c_out = exe.run(
                    feed={
                        'a': (self.a_inf + self.a_real_np)
                        + 1j * self.a_imag_np,
                        'b': (self.b_inf + self.b_real_np)
                        + 1j * self.b_imag_np,
                    },
                    fetch_list=[c],
                )
                np.testing.assert_allclose(c_out[0], c_np)

    def test_static_nan_special_case(self):
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                a_nan_complex = paddle.static.data(
                    name='a', shape=[6, 5, 4, 3], dtype=self.dtype
                )
                b_nan_complex = paddle.static.data(
                    name='b', shape=[6, 5, 4, 3], dtype=self.dtype
                )
                op = eval(f"paddle.{self.op_type}")
                c = op(a_nan_complex, b_nan_complex)
                exe = paddle.static.Executor(self.place)
                c_np = self.callback(
                    (self.a_nan + self.a_real_np) + 1j * self.a_imag_np,
                    (self.b_nan + self.b_real_np) + 1j * self.b_imag_np,
                )
                c_out = exe.run(
                    feed={
                        'a': (self.a_nan + self.a_real_np)
                        + 1j * self.a_imag_np,
                        'b': (self.b_nan + self.b_real_np)
                        + 1j * self.b_imag_np,
                    },
                    fetch_list=[c],
                )
                np.testing.assert_allclose(c_out[0], c_np)

    def test_static_broadcast_api(self):
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                a_complex = paddle.static.data(
                    name='a', shape=[6, 5, 4, 3], dtype=self.dtype
                )
                b_complex = paddle.static.data(
                    name='b', shape=[4, 3], dtype=self.dtype
                )
                op = eval(f"paddle.{self.op_type}")
                c = op(a_complex, b_complex)
                exe = paddle.static.Executor(self.place)
                c_np = self.callback(
                    self.a_real_np + 1j * self.a_imag_np,
                    self.real_np_2d + 1j * self.imag_np_2d,
                )
                c_out = exe.run(
                    feed={
                        'a': self.a_real_np + 1j * self.a_imag_np,
                        'b': self.real_np_2d + 1j * self.imag_np_2d,
                    },
                    fetch_list=[c],
                )
                np.testing.assert_allclose(c_out[0], c_np)


class TestNotEqualComplex64Api(unittest.TestCase):
    def setUp(self):
        self.callback = lambda _a, _b: _a != _b
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
        self.real_np_2d = numpy.random.random(size=(4, 3)).astype(
            self.typename[0]
        )
        self.imag_np_2d = numpy.random.random(size=(4, 3)).astype(
            self.typename[0]
        )
        self.a_inf = np.array([1, np.inf, -np.inf], self.dtype)
        self.b_inf = np.array([1, -np.inf, np.inf], self.dtype)
        self.a_nan = np.array([1, np.nan, -np.nan], self.dtype)
        self.b_nan = np.array([1, -np.nan, -np.nan], self.dtype)

    def test_dynamic_api(self):
        with dygraph_guard():
            a_complex_np = self.a_real_np + 1j * self.a_imag_np
            b_complex_np = self.b_real_np + 1j * self.b_imag_np
            a_complex = paddle.to_tensor(a_complex_np, dtype=self.dtype)
            b_complex = paddle.to_tensor(b_complex_np, dtype=self.dtype)
            c_np = self.callback(a_complex_np, b_complex_np)
            c = self.callback(a_complex, b_complex)
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
                c_np = self.callback(
                    self.a_real_np + 1j * self.a_imag_np,
                    self.b_real_np + 1j * self.b_imag_np,
                )
                c_out = exe.run(
                    feed={
                        'a': self.a_real_np + 1j * self.a_imag_np,
                        'b': self.b_real_np + 1j * self.b_imag_np,
                    },
                    fetch_list=[c],
                )
                np.testing.assert_allclose(c_out[0], c_np)

    def test_dygraph_nan_inf_special_case(self):
        with dygraph_guard():
            # nan + inf, return false directly.
            a_inf_complex_np = (self.a_nan + self.a_real_np) + 1j * (
                self.a_inf + self.a_imag_np
            )
            b_inf_complex_np = (self.b_nan + self.b_real_np) + 1j * (
                self.b_inf + self.b_imag_np
            )
            a_inf_complex = paddle.to_tensor(a_inf_complex_np, dtype=self.dtype)
            b_inf_complex = paddle.to_tensor(b_inf_complex_np, dtype=self.dtype)
            c_np = self.callback(a_inf_complex_np, b_inf_complex_np)
            c = self.callback(a_inf_complex, b_inf_complex)
            np.testing.assert_allclose(c.numpy(), c_np)

    def test_dygraph_inf_special_case(self):
        with dygraph_guard():
            a_inf_complex_np = (
                self.a_inf + self.a_real_np
            ) + self.a_imag_np * 1j
            b_inf_complex_np = (
                self.b_inf + self.b_real_np
            ) + self.b_imag_np * 1j
            a_inf_complex = paddle.to_tensor(a_inf_complex_np, dtype=self.dtype)
            b_inf_complex = paddle.to_tensor(b_inf_complex_np, dtype=self.dtype)
            c_np = self.callback(a_inf_complex_np, b_inf_complex_np)
            c = self.callback(a_inf_complex, b_inf_complex)
            np.testing.assert_allclose(c.numpy(), c_np)

    def test_dygraph_nan_special_case(self):
        with dygraph_guard():
            a_nan_complex_np = self.a_nan + self.a_real_np + 1j * self.a_imag_np
            b_nan_complex_np = self.b_nan + self.b_real_np + 1j * self.b_imag_np
            a_nan_complex = paddle.to_tensor(a_nan_complex_np, dtype=self.dtype)
            b_nan_complex = paddle.to_tensor(b_nan_complex_np, dtype=self.dtype)
            c_np = self.callback(a_nan_complex_np, b_nan_complex_np)
            c = self.callback(a_nan_complex, b_nan_complex)
            np.testing.assert_allclose(c.numpy(), c_np)

    def test_dygraph_broadcast_api(self):
        with dygraph_guard():
            a_complex_np = self.a_real_np + 1j * self.a_imag_np
            b_complex_np = self.real_np_2d + 1j * self.imag_np_2d
            a_complex = paddle.to_tensor(a_complex_np, dtype=self.dtype)
            b_complex = paddle.to_tensor(b_complex_np, dtype=self.dtype)
            c_np = self.callback(a_complex_np, b_complex_np)
            c = self.callback(a_complex, b_complex)
            np.testing.assert_allclose(c.numpy(), c_np)

    def test_dygraph_special_case(self):
        with dygraph_guard():
            a_np = np.array(1 + 1j, dtype=self.dtype)
            a = paddle.to_tensor(1 + 1j, dtype=self.dtype)
            b = complex(1, 1)
            c = self.callback(a, b)
            c_np = self.callback(a, b)
            np.testing.assert_allclose(c.numpy(), c_np)

    def test_static_nan_inf_special_case(self):
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                a_inf_complex = paddle.static.data(
                    name='a', shape=[6, 5, 4, 3], dtype=self.dtype
                )
                b_inf_complex = paddle.static.data(
                    name='b', shape=[6, 5, 4, 3], dtype=self.dtype
                )
                op = eval(f"paddle.{self.op_type}")
                c = op(a_inf_complex, b_inf_complex)
                exe = paddle.static.Executor(self.place)
                c_np = self.callback(
                    (self.a_nan + self.a_real_np)
                    + 1j * (self.a_inf + self.a_imag_np),
                    (self.b_nan + self.b_real_np)
                    + 1j * (self.b_inf + self.b_imag_np),
                )
                c_out = exe.run(
                    feed={
                        'a': (self.a_nan + self.a_real_np)
                        + 1j * (self.a_inf + self.a_imag_np),
                        'b': (self.b_nan + self.b_real_np)
                        + 1j * (self.b_inf + self.b_imag_np),
                    },
                    fetch_list=[c],
                )
                np.testing.assert_allclose(c_out[0], c_np)

    def test_static_inf_special_case(self):
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                a_inf_complex = paddle.static.data(
                    name='a', shape=[6, 5, 4, 3], dtype=self.dtype
                )
                b_inf_complex = paddle.static.data(
                    name='b', shape=[6, 5, 4, 3], dtype=self.dtype
                )
                op = eval(f"paddle.{self.op_type}")
                c = op(a_inf_complex, b_inf_complex)
                exe = paddle.static.Executor(self.place)
                c_np = self.callback(
                    (self.a_inf + self.a_real_np) + 1j * self.a_imag_np,
                    (self.b_inf + self.b_real_np) + 1j * self.b_imag_np,
                )
                c_out = exe.run(
                    feed={
                        'a': (self.a_inf + self.a_real_np)
                        + 1j * self.a_imag_np,
                        'b': (self.b_inf + self.b_real_np)
                        + 1j * self.b_imag_np,
                    },
                    fetch_list=[c],
                )
                np.testing.assert_allclose(c_out[0], c_np)

    def test_static_nan_special_case(self):
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                a_nan_complex = paddle.static.data(
                    name='a', shape=[6, 5, 4, 3], dtype=self.dtype
                )
                b_nan_complex = paddle.static.data(
                    name='b', shape=[6, 5, 4, 3], dtype=self.dtype
                )
                op = eval(f"paddle.{self.op_type}")
                c = op(a_nan_complex, b_nan_complex)
                exe = paddle.static.Executor(self.place)
                c_np = self.callback(
                    (self.a_nan + self.a_real_np) + 1j * self.a_imag_np,
                    (self.b_nan + self.b_real_np) + 1j * self.b_imag_np,
                )
                c_out = exe.run(
                    feed={
                        'a': (self.a_nan + self.a_real_np)
                        + 1j * self.a_imag_np,
                        'b': (self.b_nan + self.b_real_np)
                        + 1j * self.b_imag_np,
                    },
                    fetch_list=[c],
                )
                np.testing.assert_allclose(c_out[0], c_np)

    def test_static_broadcast_api(self):
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                a_complex = paddle.static.data(
                    name='a', shape=[6, 5, 4, 3], dtype=self.dtype
                )
                b_complex = paddle.static.data(
                    name='b', shape=[4, 3], dtype=self.dtype
                )
                op = eval(f"paddle.{self.op_type}")
                c = op(a_complex, b_complex)
                exe = paddle.static.Executor(self.place)
                c_np = self.callback(
                    self.a_real_np + 1j * self.a_imag_np,
                    self.real_np_2d + 1j * self.imag_np_2d,
                )
                c_out = exe.run(
                    feed={
                        'a': self.a_real_np + 1j * self.a_imag_np,
                        'b': self.real_np_2d + 1j * self.imag_np_2d,
                    },
                    fetch_list=[c],
                )
                np.testing.assert_allclose(c_out[0], c_np)


class TestNotEqualComplex128Api(unittest.TestCase):
    def setUp(self):
        self.callback = lambda _a, _b: _a != _b
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
        self.real_np_2d = numpy.random.random(size=(4, 3)).astype(
            self.typename[0]
        )
        self.imag_np_2d = numpy.random.random(size=(4, 3)).astype(
            self.typename[0]
        )
        self.a_inf = np.array([1, np.inf, -np.inf], self.dtype)
        self.b_inf = np.array([1, -np.inf, np.inf], self.dtype)
        self.a_nan = np.array([1, np.nan, -np.nan], self.dtype)
        self.b_nan = np.array([1, -np.nan, -np.nan], self.dtype)

    def test_dynamic_api(self):
        with dygraph_guard():
            a_complex_np = self.a_real_np + 1j * self.a_imag_np
            b_complex_np = self.b_real_np + 1j * self.b_imag_np
            a_complex = paddle.to_tensor(a_complex_np, dtype=self.dtype)
            b_complex = paddle.to_tensor(b_complex_np, dtype=self.dtype)
            c_np = self.callback(a_complex_np, b_complex_np)
            c = self.callback(a_complex, b_complex)
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
                c_np = self.callback(
                    self.a_real_np + 1j * self.a_imag_np,
                    self.b_real_np + 1j * self.b_imag_np,
                )
                c_out = exe.run(
                    feed={
                        'a': self.a_real_np + 1j * self.a_imag_np,
                        'b': self.b_real_np + 1j * self.b_imag_np,
                    },
                    fetch_list=[c],
                )
                np.testing.assert_allclose(c_out[0], c_np)

    def test_dygraph_nan_inf_special_case(self):
        with dygraph_guard():
            # nan + inf, return false directly.
            a_inf_complex_np = (self.a_nan + self.a_real_np) + 1j * (
                self.a_inf + self.a_imag_np
            )
            b_inf_complex_np = (self.b_nan + self.b_real_np) + 1j * (
                self.b_inf + self.b_imag_np
            )
            a_inf_complex = paddle.to_tensor(a_inf_complex_np, dtype=self.dtype)
            b_inf_complex = paddle.to_tensor(b_inf_complex_np, dtype=self.dtype)
            c_np = self.callback(a_inf_complex_np, b_inf_complex_np)
            c = self.callback(a_inf_complex, b_inf_complex)
            np.testing.assert_allclose(c.numpy(), c_np)

    def test_dygraph_inf_special_case(self):
        with dygraph_guard():
            a_inf_complex_np = (
                self.a_inf + self.a_real_np
            ) + self.a_imag_np * 1j
            b_inf_complex_np = (
                self.b_inf + self.b_real_np
            ) + self.b_imag_np * 1j
            a_inf_complex = paddle.to_tensor(a_inf_complex_np, dtype=self.dtype)
            b_inf_complex = paddle.to_tensor(b_inf_complex_np, dtype=self.dtype)
            c_np = self.callback(a_inf_complex_np, b_inf_complex_np)
            c = self.callback(a_inf_complex, b_inf_complex)
            np.testing.assert_allclose(c.numpy(), c_np)

    def test_dygraph_nan_special_case(self):
        with dygraph_guard():
            a_nan_complex_np = self.a_nan + self.a_real_np + 1j * self.a_imag_np
            b_nan_complex_np = self.b_nan + self.b_real_np + 1j * self.b_imag_np
            a_nan_complex = paddle.to_tensor(a_nan_complex_np, dtype=self.dtype)
            b_nan_complex = paddle.to_tensor(b_nan_complex_np, dtype=self.dtype)
            c_np = self.callback(a_nan_complex_np, b_nan_complex_np)
            c = self.callback(a_nan_complex, b_nan_complex)
            np.testing.assert_allclose(c.numpy(), c_np)

    def test_dygraph_broadcast_api(self):
        with dygraph_guard():
            a_complex_np = self.a_real_np + 1j * self.a_imag_np
            b_complex_np = self.real_np_2d + 1j * self.imag_np_2d
            a_complex = paddle.to_tensor(a_complex_np, dtype=self.dtype)
            b_complex = paddle.to_tensor(b_complex_np, dtype=self.dtype)
            c_np = self.callback(a_complex_np, b_complex_np)
            c = self.callback(a_complex, b_complex)
            np.testing.assert_allclose(c.numpy(), c_np)

    def test_dygraph_special_case(self):
        with dygraph_guard():
            a_np = np.array(1 + 1j, dtype=self.dtype)
            a = paddle.to_tensor(1 + 1j, dtype=self.dtype)
            b = complex(1, 1)
            c = self.callback(a, b)
            c_np = self.callback(a, b)
            np.testing.assert_allclose(c.numpy(), c_np)

    def test_static_nan_inf_special_case(self):
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                a_inf_complex = paddle.static.data(
                    name='a', shape=[6, 5, 4, 3], dtype=self.dtype
                )
                b_inf_complex = paddle.static.data(
                    name='b', shape=[6, 5, 4, 3], dtype=self.dtype
                )
                op = eval(f"paddle.{self.op_type}")
                c = op(a_inf_complex, b_inf_complex)
                exe = paddle.static.Executor(self.place)
                c_np = self.callback(
                    (self.a_nan + self.a_real_np)
                    + 1j * (self.a_inf + self.a_imag_np),
                    (self.b_nan + self.b_real_np)
                    + 1j * (self.b_inf + self.b_imag_np),
                )
                c_out = exe.run(
                    feed={
                        'a': (self.a_nan + self.a_real_np)
                        + 1j * (self.a_inf + self.a_imag_np),
                        'b': (self.b_nan + self.b_real_np)
                        + 1j * (self.b_inf + self.b_imag_np),
                    },
                    fetch_list=[c],
                )
                np.testing.assert_allclose(c_out[0], c_np)

    def test_static_inf_special_case(self):
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                a_inf_complex = paddle.static.data(
                    name='a', shape=[6, 5, 4, 3], dtype=self.dtype
                )
                b_inf_complex = paddle.static.data(
                    name='b', shape=[6, 5, 4, 3], dtype=self.dtype
                )
                op = eval(f"paddle.{self.op_type}")
                c = op(a_inf_complex, b_inf_complex)
                exe = paddle.static.Executor(self.place)
                c_np = self.callback(
                    (self.a_inf + self.a_real_np) + 1j * self.a_imag_np,
                    (self.b_inf + self.b_real_np) + 1j * self.b_imag_np,
                )
                c_out = exe.run(
                    feed={
                        'a': (self.a_inf + self.a_real_np)
                        + 1j * self.a_imag_np,
                        'b': (self.b_inf + self.b_real_np)
                        + 1j * self.b_imag_np,
                    },
                    fetch_list=[c],
                )
                np.testing.assert_allclose(c_out[0], c_np)

    def test_static_nan_special_case(self):
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                a_nan_complex = paddle.static.data(
                    name='a', shape=[6, 5, 4, 3], dtype=self.dtype
                )
                b_nan_complex = paddle.static.data(
                    name='b', shape=[6, 5, 4, 3], dtype=self.dtype
                )
                op = eval(f"paddle.{self.op_type}")
                c = op(a_nan_complex, b_nan_complex)
                exe = paddle.static.Executor(self.place)
                c_np = self.callback(
                    (self.a_nan + self.a_real_np) + 1j * self.a_imag_np,
                    (self.b_nan + self.b_real_np) + 1j * self.b_imag_np,
                )
                c_out = exe.run(
                    feed={
                        'a': (self.a_nan + self.a_real_np)
                        + 1j * self.a_imag_np,
                        'b': (self.b_nan + self.b_real_np)
                        + 1j * self.b_imag_np,
                    },
                    fetch_list=[c],
                )
                np.testing.assert_allclose(c_out[0], c_np)

    def test_static_broadcast_api(self):
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                a_complex = paddle.static.data(
                    name='a', shape=[6, 5, 4, 3], dtype=self.dtype
                )
                b_complex = paddle.static.data(
                    name='b', shape=[4, 3], dtype=self.dtype
                )
                op = eval(f"paddle.{self.op_type}")
                c = op(a_complex, b_complex)
                exe = paddle.static.Executor(self.place)
                c_np = self.callback(
                    self.a_real_np + 1j * self.a_imag_np,
                    self.real_np_2d + 1j * self.imag_np_2d,
                )
                c_out = exe.run(
                    feed={
                        'a': self.a_real_np + 1j * self.a_imag_np,
                        'b': self.real_np_2d + 1j * self.imag_np_2d,
                    },
                    fetch_list=[c],
                )
                np.testing.assert_allclose(c_out[0], c_np)
