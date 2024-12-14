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

import numpy as np
from tensorrt_test_base import TensorRTBaseTest

import paddle


class TestEluTRTPatternCase1(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.nn.functional.elu
        self.api_args = {
            "x": np.random.randn(3).astype("float32"),
            "alpha": 1.0,
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1]}
        self.max_shape = {"x": [5]}

    def test_trt_result(self):
        self.check_trt_result()


class TestEluTRTPatternCase2(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.nn.functional.elu
        self.api_args = {
            "x": np.random.randn(3).astype("float16"),
            "alpha": 1.0,
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1]}
        self.max_shape = {"x": [5]}

    def test_trt_result(self):
        self.check_trt_result()


class TestHardSigmoidTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.nn.functional.hardsigmoid
        self.api_args = {
            "x": np.random.randn(2, 3).astype("float32"),
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3], "y": [1, 3]}
        self.max_shape = {"x": [5, 3], "y": [5, 3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestHardSwishTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.nn.functional.hardswish
        self.api_args = {
            "x": np.random.randn(2, 3).astype("float32"),
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3], "y": [1, 3]}
        self.max_shape = {"x": [5, 3], "y": [5, 3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestReluTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.nn.functional.relu
        self.api_args = {"x": np.random.randn(3).astype("float32")}
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1]}
        self.max_shape = {"x": [5]}

    def test_trt_result(self):
        self.check_trt_result()


class TestTanhTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.tanh
        self.api_args = {"x": np.random.randn(3).astype("float32")}
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1]}
        self.max_shape = {"x": [5]}

    def test_trt_result(self):
        self.check_trt_result()


class TestSigmoidTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.nn.functional.sigmoid
        self.api_args = {
            "x": np.random.randn(2, 3).astype("float32"),
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3]}
        self.max_shape = {"x": [5, 3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestSoftplusTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.nn.Softplus()
        self.api_args = {
            "x": np.random.randn(2, 3).astype("float32"),
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3]}
        self.max_shape = {"x": [5, 3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestSiluFloatTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.nn.functional.silu
        self.api_args = {
            "x": np.random.randn(2, 3).astype("float32"),
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3]}
        self.max_shape = {"x": [5, 3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestSwishFloatTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.nn.functional.swish
        self.api_args = {
            "x": np.random.randn(2, 3).astype("float32"),
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3]}
        self.max_shape = {"x": [5, 3]}

    def test_trt_result(self):
        self.check_trt_result()


def prelu_wrapper1(x, num_parameters, data_format="NCHW"):
    prelu = paddle.nn.PReLU(num_parameters, 0.25, data_format=data_format)
    return prelu(x)


def prelu_wrapper2(x, num_parameters, data_format="NCHW"):
    # Now 0-D weigh parameter isn't supported.
    prelu = paddle.nn.PReLU(num_parameters, 0.25, data_format=data_format)
    prelu._weight = paddle.reshape(prelu._weight, [])
    return prelu(x)


class TestPreluCase1TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = prelu_wrapper1
        self.api_args = {
            "x": np.arange(24).reshape([2, 2, 2, 3]).astype("float32"),
            "num_parameters": 2,
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 2, 2, 3]}
        self.max_shape = {"x": [5, 2, 2, 3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestPreluCase2TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = prelu_wrapper1
        self.api_args = {
            "x": np.arange(24).reshape([2, 2, 2, 3]).astype("float32"),
            "num_parameters": 3,
            "data_format": "NHWC",
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 2, 2, 3]}
        self.max_shape = {"x": [5, 2, 2, 3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestPreluCase3TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = prelu_wrapper1
        self.api_args = {
            "x": np.arange(24).reshape([2, 2, 2, 3]).astype("float32"),
            "num_parameters": 1,  # test mode = all
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 2, 2, 3]}
        self.max_shape = {"x": [5, 2, 2, 3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestPreluCase4TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = prelu_wrapper1
        self.api_args = {
            "x": np.arange(12).reshape([2, 2, 3]).astype("float32"),
            "num_parameters": 2,
            "data_format": "NCL",
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 2, 3]}
        self.max_shape = {"x": [5, 2, 3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestPreluCase5TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.nn.functional.prelu
        self.api_args = {
            "x": np.arange(24).reshape([2, 2, 2, 3]).astype("float32"),
            "weight": np.array(1.0).astype("float32"),
        }
        self.program_config = {"feed_list": ["x", "weight"]}

    def test_trt_result(self):
        self.check_marker(expected_result=False)


class TestPreluCase6TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = prelu_wrapper2
        self.api_args = {
            "x": np.arange(24).reshape([2, 2, 2, 3]).astype("float32"),
            "num_parameters": 1,
        }
        self.program_config = {"feed_list": ["x"]}

    def test_trt_result(self):
        self.check_trt_result(expected_result=False)


class TestStanhFloatTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.stanh
        self.api_args = {
            "x": np.random.randn(2, 3).astype("float32"),
            "scale_a": 0.67,
            "scale_b": 1.7159,
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3]}
        self.max_shape = {"x": [5, 3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestCeluTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.nn.functional.celu
        self.api_args = {
            "x": np.random.randn(2, 3).astype("float32"),
            "alpha": 1.0,
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3]}
        self.max_shape = {"x": [5, 3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestThresholdedReluTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.nn.functional.thresholded_relu
        self.api_args = {
            "x": np.random.randn(2, 3).astype("float32"),
            "threshold": 1.0,
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3]}
        self.max_shape = {"x": [5, 3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestMishCase1TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.nn.functional.mish
        self.api_args = {
            "x": np.random.randn(2).astype("float32"),
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1]}
        self.max_shape = {"x": [5]}

    def test_trt_result(self):
        self.check_trt_result()


class TestMishCase2TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.nn.functional.mish
        self.api_args = {
            "x": np.random.randn(2, 3).astype("float32"),
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3]}
        self.max_shape = {"x": [5, 3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestMishCase3TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.nn.functional.mish
        self.api_args = {
            "x": np.random.randn(2, 3, 4).astype("float32"),
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3, 4]}
        self.max_shape = {"x": [5, 3, 4]}

    def test_trt_result(self):
        self.check_trt_result()


class TestMishCase4TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.nn.functional.mish
        self.api_args = {
            "x": np.random.randn(2, 3, 4, 2).astype("float32"),
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3, 4, 2]}
        self.max_shape = {"x": [5, 3, 4, 2]}

    def test_trt_result(self):
        self.check_trt_result()


if __name__ == '__main__':
    unittest.main()
