# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import tempfile
import unittest

import numpy as np
from dygraph_to_static_utils import (
    Dy2StTestBase,
    enable_to_static_guard,
    static_guard,
    test_ast_only,
)

import paddle
from paddle.static import InputSpec

SEED = 2020
np.random.seed(SEED)


def test_slice_without_control_flow(x):
    # Python slice will not be transformed.
    x = paddle.to_tensor(x)
    a = [x]
    a[0] = paddle.full(shape=[2], fill_value=2, dtype="float32")
    return a[0]


def test_slice_in_if(x):
    x = paddle.to_tensor(x)
    a = []
    if x.numpy()[0] > 0:
        a.append(x)
    else:
        a.append(paddle.full(shape=[1, 2], fill_value=9, dtype="float32"))

    if x.numpy()[0] > 0:
        a[0] = x

    a[0] = x + 1
    out = a[0]
    return out


def test_slice_in_while_loop(x, iter_num=3):
    x = paddle.to_tensor(x)
    iter_num_var = paddle.full(shape=[1], fill_value=iter_num, dtype="int32")
    a = []
    i = 0

    while i < iter_num_var:
        a.append(x)
        i += 1

    i = 0
    while i < iter_num_var.numpy()[0]:
        a[i] = paddle.full(shape=[2], fill_value=2, dtype="float32")
        i += 1
    out = a[0:iter_num]
    return out[0]


def test_slice_in_for_loop(x, iter_num=3):
    x = paddle.to_tensor(x)
    a = []
    # Use `paddle.full` so that static analysis can analyze the type of iter_num is Tensor
    iter_num = paddle.full(
        shape=[1], fill_value=iter_num, dtype="int32"
    )  # TODO(liym27): Delete it if the type of parameter iter_num can be resolved

    for i in range(iter_num):
        a.append(x)

    for i in range(iter_num):
        a[i] = x
    out = a[2]
    return out


def test_set_value(x):
    x = paddle.to_tensor(x)
    x[0] = paddle.full(shape=[1], fill_value=2, dtype="float32")
    x[1:2, 0:1] = 10
    return x


class LayerWithSetValue(paddle.nn.Layer):
    def __init__(self, input_dim, hidden):
        super().__init__()
        self.linear = paddle.nn.Linear(input_dim, hidden)

    def forward(self, x):
        x = self.linear(x)
        x[0] = 1
        return x


class TestSliceBase(Dy2StTestBase):
    def setUp(self):
        self.init_input()
        self.dygraph_func = None

    def init_input(self):
        self.input = np.random.random(3).astype('float32')

    def init_dygraph_func(self):
        raise NotImplementedError(
            "For Enumerate test should implement set_test_func"
        )

    def run_dygraph_mode(self):
        return self._run(to_static=False)

    def _run(self, to_static):
        func = (
            paddle.jit.to_static(self.dygraph_func)
            if to_static
            else self.dygraph_func
        )
        res = func(self.input)
        return res.numpy()

    def run_static_mode(self):
        return self._run(to_static=True)


class TestSliceWithoutControlFlow(TestSliceBase):
    def init_dygraph_func(self):
        self.dygraph_func = test_slice_without_control_flow

    def test_transformed_static_result(self):
        self.init_dygraph_func()
        static_res = self.run_static_mode()
        dygraph_res = self.run_dygraph_mode()
        np.testing.assert_allclose(dygraph_res, static_res, rtol=1e-05)


class TestSliceInIf(TestSliceBase):
    def init_dygraph_func(self):
        self.dygraph_func = test_slice_in_if

    def test_transformed_static_result(self):
        self.init_dygraph_func()
        static_res = self.run_static_mode()
        dygraph_res = self.run_dygraph_mode()
        np.testing.assert_allclose(dygraph_res, static_res, rtol=1e-05)


class TestSliceInWhileLoop(TestSliceInIf):
    def init_dygraph_func(self):
        self.dygraph_func = test_slice_in_while_loop


class TestSliceInForLoop(TestSliceInIf):
    def init_dygraph_func(self):
        self.dygraph_func = test_slice_in_for_loop


class TestSetValue(TestSliceInIf):
    def init_input(self):
        self.input = np.full([3, 4, 5], 5).astype('float32')

    def init_dygraph_func(self):
        self.dygraph_func = test_set_value


class TestSetValueWithLayerAndSave(Dy2StTestBase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_path = os.path.join(
            self.temp_dir.name, "layer_use_set_value"
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    @test_ast_only
    def test_set_value_with_save(self):
        with enable_to_static_guard(True):
            model = paddle.jit.to_static(
                LayerWithSetValue(input_dim=10, hidden=1)
            )
            x = paddle.full(shape=[5, 10], fill_value=5.0, dtype="float32")
            paddle.jit.save(
                layer=model,
                path=self.model_path,
                input_spec=[x],
                output_spec=None,
            )


class TestSliceSupplementSpecialCase(Dy2StTestBase):
    # unittest for slice index which abs(step)>0. eg: x[::2]
    def test_static_slice_step(self):
        with static_guard(), paddle.static.program_guard(
            paddle.static.Program()
        ):
            array = np.arange(4**3).reshape((4, 4, 4)).astype('int64')

            x = paddle.static.data(name='x', shape=[4, 4, 4], dtype='int64')
            z1 = x[::2]
            z2 = x[::-2]

            place = paddle.CPUPlace()
            prog = paddle.static.default_main_program()
            exe = paddle.static.Executor(place)
            exe.run(paddle.static.default_startup_program())

            out = exe.run(prog, feed={'x': array}, fetch_list=[z1, z2])

        np.testing.assert_array_equal(out[0], array[::2])
        np.testing.assert_array_equal(out[1], array[::-2])

    def test_static_slice_step_dygraph2static(self):
        array = np.arange(4**2 * 5).reshape((5, 4, 4)).astype('int64')
        inps = paddle.to_tensor(array)

        def func(inps):
            return inps[::2], inps[::-2]

        origin_result = func(inps)
        sfunc = paddle.jit.to_static(
            func, input_spec=[InputSpec(shape=[None, 4, 4])]
        )
        static_result = sfunc(inps)

        np.testing.assert_array_equal(
            origin_result[0].numpy(), static_result[0].numpy()
        )
        np.testing.assert_array_equal(
            origin_result[1].numpy(), static_result[1].numpy()
        )


class TestPaddleStridedSlice(Dy2StTestBase):
    def test_compare_paddle_strided_slice_with_numpy(self):
        array = np.arange(5)
        pt = paddle.to_tensor(array)

        s1 = 3
        e1 = 1
        stride1 = -2
        sl = paddle.strided_slice(
            pt,
            axes=[0],
            starts=[s1],
            ends=[e1],
            strides=[stride1],
        )

        self.assertTrue(array[s1:e1:stride1], sl)

        array = np.arange(6 * 6).reshape((6, 6))
        pt = paddle.to_tensor(array)
        s2 = [8, -1]
        e2 = [1, -5]
        stride2 = [-2, -3]
        sl = paddle.strided_slice(
            pt, axes=[0, 1], starts=s2, ends=e2, strides=stride2
        )

        np.testing.assert_array_equal(
            sl.numpy(),
            array[s2[0] : e2[0] : stride2[0], s2[1] : e2[1] : stride2[1]],
        )

        array = np.arange(6 * 7 * 8).reshape((6, 7, 8))
        pt = paddle.to_tensor(array)
        s2 = [7, -1]
        e2 = [2, -5]
        stride2 = [-2, -3]
        sl = paddle.strided_slice(
            pt, axes=[0, 2], starts=s2, ends=e2, strides=stride2
        )

        array_slice = array[
            s2[0] : e2[0] : stride2[0], ::, s2[1] : e2[1] : stride2[1]
        ]
        np.testing.assert_array_equal(sl.numpy(), array_slice)


def slice_zero_shape_tensor(x):
    y = x[1:2]
    return y


class TestSliceZeroShapeTensor(Dy2StTestBase):
    def test_slice(self):
        x = paddle.ones([0, 0, 0, 0])
        y = slice_zero_shape_tensor(x)
        np.testing.assert_equal(y.shape, [0, 0, 0, 0])

        static_func = paddle.jit.to_static(slice_zero_shape_tensor)
        y = static_func(x)
        np.testing.assert_equal(y.shape, [0, 0, 0, 0])


if __name__ == '__main__':
    unittest.main()
