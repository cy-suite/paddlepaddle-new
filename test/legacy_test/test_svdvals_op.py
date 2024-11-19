#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import unittest

import numpy as np
from op_test import OpTest, skip_check_grad_ci

import paddle
from paddle import base


class TestSvdvalsOp(OpTest):
    def setUp(self):
        paddle.enable_static()
        self.python_api = paddle.linalg.svdvals
        self.generate_input()
        self.generate_output()
        self.op_type = "svdvals"
        assert hasattr(self, "_output_data")
        self.inputs = {"X": self._input_data}
        self.outputs = {"S": self._output_data}

    def generate_input(self):
        """return a input_data and input_shape"""
        self._input_shape = (100, 1)
        self._input_data = np.random.random(self._input_shape).astype("float64")

    def generate_output(self):
        assert hasattr(self, "_input_data")
        self._output_data = np.linalg.svdvals(self._input_data)

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_svdvals_forward(self):
        """Check singular values calculation."""
        single_input = self._input_data.reshape(
            [-1, self._input_shape[-2], self._input_shape[-1]]
        )[0]
        paddle.disable_static()
        dy_x = paddle.to_tensor(single_input)
        dy_s = paddle.linalg.svdvals(dy_x)
        np_s = np.linalg.svd(single_input, compute_uv=False)
        np.testing.assert_allclose(dy_s.numpy(), np_s, rtol=1e-6)
        paddle.enable_static()


class TestSvdValsBatched(TestSvdvalsOp):
    def generate_input(self):
        """Generate batched input matrix."""
        self._input_shape = (10, 6, 3)
        base_matrix = np.array(
            [
                [1.0, 2.0, 3.0],
                [0.0, 1.0, 5.0],
                [0.0, 0.0, 6.0],
                [2.0, 4.0, 9.0],
                [3.0, 6.0, 8.0],
                [3.0, 1.0, 0.0],
            ]
        ).astype("float64")
        self._input_data = np.stack([base_matrix] * 10, axis=0)

    def test_svdvals_forward(self):
        """Check singular values calculation for batched input."""
        paddle.disable_static()
        dy_x = paddle.to_tensor(self._input_data)
        dy_s = paddle.linalg.svdvals(dy_x)
        np_s = np.array(
            [
                np.linalg.svd(matrix, compute_uv=False)
                for matrix in self._input_data
            ]
        )
        np.testing.assert_allclose(dy_s.numpy(), np_s, rtol=1e-6)
        paddle.enable_static()


@skip_check_grad_ci(
    reason="'check_grad' on singular values is not required for svdvals."
)
class TestSvdValsBigMatrix(TestSvdvalsOp):
    def generate_input(self):
        """Generate large input matrix."""
        self._input_shape = (200, 300)
        self._input_data = np.random.random(self._input_shape).astype("float64")

    def test_check_grad(self):
        pass


class TestSvdValsAPI(unittest.TestCase):
    def test_dygraph(self):
        paddle.disable_static()
        a = np.random.rand(5, 5)
        x = paddle.to_tensor(a)
        s = paddle.linalg.svdvals(x)
        gt_s = np.linalg.svd(a, compute_uv=False)
        np.testing.assert_allclose(s.numpy(), gt_s, rtol=1e-5)

    def test_static(self):
        paddle.enable_static()
        places = []
        if os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower() in [
            '1',
            'true',
            'yes',
        ]:
            places = [paddle.CPUPlace(), paddle.CUDAPlace(0)]
        else:
            places = [paddle.CPUPlace()]

        for place in places:
            with base.program_guard(base.Program(), base.Program()):
                x = paddle.static.data(name="x", shape=[5, 5], dtype="float64")
                s = paddle.linalg.svdvals(x)
                exe = base.Executor(place)
                a = np.random.rand(5, 5).astype("float64")
                out = exe.run(feed={"x": a}, fetch_list=[s])[0]
                gt_s = np.linalg.svd(a, compute_uv=False)
                np.testing.assert_allclose(out, gt_s, rtol=1e-5)


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
