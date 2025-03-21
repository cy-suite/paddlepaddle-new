#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import paddle_static_guard

import paddle


def ref_var(x, axis=None, unbiased=True, keepdim=False):
    ddof = 1 if unbiased else 0
    if isinstance(axis, int):
        axis = (axis,)
    if axis is not None:
        axis = tuple(axis)
    return np.var(x, axis=axis, ddof=ddof, keepdims=keepdim)


class TestVarAPI(unittest.TestCase):
    def setUp(self):
        self.dtype = 'float64'
        self.shape = [1, 3, 4, 10]
        self.axis = [1, 3]
        self.keepdim = False
        self.unbiased = True
        self.set_attrs()
        self.x = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.base.core.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def set_attrs(self):
        pass

    def static(self):
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('X', self.shape, self.dtype)
            out = paddle.var(x, self.axis, self.unbiased, self.keepdim)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'X': self.x}, fetch_list=[out])
        return res[0]

    def dygraph(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.x)
        out = paddle.var(x, self.axis, self.unbiased, self.keepdim)
        paddle.enable_static()
        return out.numpy()

    def test_api(self):
        out_ref = ref_var(self.x, self.axis, self.unbiased, self.keepdim)
        out_dygraph = self.dygraph()

        np.testing.assert_allclose(out_ref, out_dygraph, rtol=1e-05)
        self.assertTrue(np.equal(out_ref.shape, out_dygraph.shape).all())

        def test_static_or_pir_mode():
            out_static = self.static()
            np.testing.assert_allclose(out_ref, out_static, rtol=1e-05)
            self.assertTrue(np.equal(out_ref.shape, out_static.shape).all())

        test_static_or_pir_mode()


class TestVarAPI_dtype(TestVarAPI):
    def set_attrs(self):
        self.dtype = 'float32'


class TestVarAPI_axis_int(TestVarAPI):
    def set_attrs(self):
        self.axis = 2


class TestVarAPI_axis_list(TestVarAPI):
    def set_attrs(self):
        self.axis = [1, 2]


class TestVarAPI_axis_tuple(TestVarAPI):
    def set_attrs(self):
        self.axis = (1, 3)


class TestVarAPI_keepdim(TestVarAPI):
    def set_attrs(self):
        self.keepdim = False


class TestVarAPI_unbiased(TestVarAPI):
    def set_attrs(self):
        self.unbiased = False


class TestVarAPI_alias(unittest.TestCase):
    def test_alias(self):
        paddle.disable_static()
        x = paddle.to_tensor(np.array([10, 12], 'float32'))
        out1 = paddle.var(x).numpy()
        out2 = paddle.tensor.var(x).numpy()
        out3 = paddle.tensor.stat.var(x).numpy()
        np.testing.assert_allclose(out1, out2, rtol=1e-05)
        np.testing.assert_allclose(out1, out3, rtol=1e-05)
        paddle.enable_static()


class TestVarError(unittest.TestCase):
    def test_error(self):
        with paddle_static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                x = paddle.static.data('X', [2, 3, 4], 'int32')
                self.assertRaises(TypeError, paddle.var, x)


class TestVarAPIZeroSize(unittest.TestCase):
    def setUp(self):
        self.dtype = 'float64'
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.base.core.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )
        self.zero_size_cases = {
            'empty': np.array([], dtype=self.dtype),
            'shape_0': np.array([], dtype=self.dtype).reshape([0]),
            'shape_0x3': np.array([], dtype=self.dtype).reshape([0, 3]),
            'shape_2x0x4': np.array([], dtype=self.dtype).reshape([2, 0, 4]),
            'shape_3x0x2': np.array([], dtype=self.dtype).reshape([3, 0, 2]),
        }
        self.set_dtype()

    def set_dtype(self):
        pass

    def _run_var_test(self, x, axis=None, unbiased=True, keepdim=False):
        # Dynamic graph test
        paddle.disable_static()
        x_tensor = paddle.to_tensor(x)
        out_dy = paddle.var(
            x_tensor, axis=axis, unbiased=unbiased, keepdim=keepdim
        )
        paddle.enable_static()

        # Static graph test
        with paddle_static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                x_static = paddle.static.data('X', x.shape, self.dtype)
                out_static = paddle.var(
                    x_static, axis=axis, unbiased=unbiased, keepdim=keepdim
                )
                exe = paddle.static.Executor(self.place)
                res = exe.run(feed={'X': x}, fetch_list=[out_static])

        # Reference result (expecting nan for zero-size inputs)
        out_ref = ref_var(x, axis, unbiased, keepdim)

        # Verify results
        np.testing.assert_allclose(out_dy.numpy(), out_ref, equal_nan=True)
        np.testing.assert_allclose(res[0], out_ref, equal_nan=True)

    def test_all(self):
        params = {
            'axis': [None, 0, -1, [0, 1]],
            'unbiased': [True, False],
            'keepdim': [True, False],
        }

        for case_name, zero_input in self.zero_size_cases.items():
            for axis in params['axis']:
                for unbiased in params['unbiased']:
                    for keepdim in params['keepdim']:
                        # Normalize axis and check validity
                        if axis is not None:
                            axis_list = (
                                [axis] if isinstance(axis, int) else axis
                            )
                            normalized_axis = [
                                a + zero_input.ndim if a < 0 else a
                                for a in axis_list
                            ]
                            if zero_input.ndim <= max(
                                normalized_axis, default=-1
                            ):
                                continue
                        with self.subTest(
                            case=case_name,
                            axis=axis,
                            unbiased=unbiased,
                            keepdim=keepdim,
                        ):
                            self._run_var_test(
                                zero_input,
                                axis=axis,
                                unbiased=unbiased,
                                keepdim=keepdim,
                            )


class TestVarAPIZeroSize_Float32(TestVarAPIZeroSize):
    def set_dtype(self):
        self.dtype = 'float32'


class TestVarAPIZeroSize_Float64(TestVarAPIZeroSize):
    def set_dtype(self):
        self.dtype = 'float64'


class TestVarAPIZeroSize_Int32(TestVarAPIZeroSize):
    def set_dtype(self):
        self.dtype = 'int32'


class TestVarAPIZeroSize_Int64(TestVarAPIZeroSize):
    def set_dtype(self):
        self.dtype = 'int64'


class TestVarAPIZeroSize_Complex64(TestVarAPIZeroSize):
    def set_dtype(self):
        self.dtype = 'complex64'


class TestVarAPIZeroSize_Complex128(TestVarAPIZeroSize):
    def set_dtype(self):
        self.dtype = 'complex128'


if __name__ == '__main__':
    unittest.main()
