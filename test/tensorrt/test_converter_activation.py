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


class TestEluTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.nn.functional.elu
        self.api_args = {
            "x": np.random.randn(3).astype("float32"),
            "alpha": 1.0,
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1]}
        self.opt_shape = {"x": [1]}
        self.max_shape = {"x": [5]}

    def test_trt_result(self):
        self.check_trt_result()

    def test_trt_result_fp16(self):
        self.check_trt_result(rtol=1e-3, atol=1e-3, precision_mode="fp16")


def softmax_wrapper(x, axis=-1):
    softmax = paddle.nn.Softmax(axis=axis)
    return softmax(x)


class TestSoftmaxCase1TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = softmax_wrapper
        self.api_args = {
            "x": np.random.randn(2, 3, 3).astype("float32"),
            "axis": -1,
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3, 3]}
        self.opt_shape = {"x": [2, 3, 3]}
        self.max_shape = {"x": [5, 3, 3]}

    def test_trt_result_fp16(self):
        self.check_trt_result(precision_mode="fp16")

    def test_trt_result_fp32(self):
        self.check_trt_result()


class TestSoftmaxCase2TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = softmax_wrapper
        self.api_args = {
            "x": np.random.randn(2).astype("float32"),
            "axis": -1,
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1]}
        self.opt_shape = {"x": [2]}
        self.max_shape = {"x": [5]}

    def test_trt_result_fp16(self):
        self.check_trt_result(precision_mode="fp16")

    def test_trt_result_fp32(self):
        self.check_trt_result()


class TestSoftmaxCase3TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = softmax_wrapper
        self.api_args = {
            "x": np.random.randn(2, 3, 3).astype("float32"),
            "axis": 1,
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3, 3]}
        self.opt_shape = {"x": [2, 3, 3]}
        self.max_shape = {"x": [5, 3, 3]}

    def test_trt_result_fp16(self):
        self.check_trt_result(precision_mode="fp16")

    def test_trt_result_fp32(self):
        self.check_trt_result()


class TestHardSigmoidTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.nn.functional.hardsigmoid
        self.api_args = {
            "x": np.random.randn(2, 3).astype("float32"),
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3], "y": [1, 3]}
        self.opt_shape = {"x": [1, 3], "y": [1, 3]}
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
        self.opt_shape = {"x": [1, 3], "y": [1, 3]}
        self.max_shape = {"x": [5, 3], "y": [5, 3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestReluTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.nn.functional.relu
        self.api_args = {"x": np.random.randn(3).astype("float32")}
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1]}
        self.opt_shape = {"x": [1]}
        self.max_shape = {"x": [5]}

    def test_trt_result(self):
        self.check_trt_result()


class TestRelu6TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.nn.functional.relu6
        self.api_args = {"x": np.random.randn(3).astype("float32")}
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1]}
        self.opt_shape = {"x": [2]}
        self.max_shape = {"x": [5]}

    def test_trt_result(self):
        self.check_trt_result()

    def test_trt_result_fp16(self):
        self.check_trt_result(precision_mode="fp16")


class TestTanhTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.tanh
        self.api_args = {"x": np.random.randn(3).astype("float32")}
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1]}
        self.opt_shape = {"x": [1]}
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
        self.opt_shape = {"x": [1, 3]}
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
        self.opt_shape = {"x": [1, 3]}
        self.max_shape = {"x": [5, 3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestGeluTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.nn.GELU()
        self.api_args = {
            "x": np.random.randn(2, 3).astype("float32"),
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3]}
        self.opt_shape = {"x": [1, 3]}
        self.max_shape = {"x": [5, 3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestGeluCase1TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.nn.GELU(True)
        self.api_args = {
            "x": np.random.randn(2, 3).astype("float32"),
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3]}
        self.opt_shape = {"x": [1, 3]}
        self.max_shape = {"x": [5, 3]}

    def test_trt_result(self):
        self.check_trt_result()

    def test_trt_result_fp16(self):
        self.check_trt_result(precision_mode="fp16")


class TestSiluFloatTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.nn.functional.silu
        self.api_args = {
            "x": np.random.randn(2, 3).astype("float32"),
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3]}
        self.opt_shape = {"x": [1, 3]}
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
        self.opt_shape = {"x": [1, 3]}
        self.max_shape = {"x": [5, 3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestTanhShrinkOpFloatTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle._C_ops.tanh_shrink
        self.api_args = {
            "x": np.random.randn(2, 3).astype("float32"),
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3]}
        self.opt_shape = {"x": [2, 3]}
        self.max_shape = {"x": [5, 3]}

    def test_trt_result_fp16(self):
        self.check_trt_result(precision_mode="fp16")

    def test_trt_result_fp32(self):
        self.check_trt_result()


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
        self.opt_shape = {"x": [1, 3]}
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
        self.opt_shape = {"x": [1, 3]}
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
        self.opt_shape = {"x": [2, 3]}
        self.max_shape = {"x": [5, 3]}

    def test_trt_result(self):
        self.check_trt_result(rtol=1e-3, atol=1e-3)

    def test_trt_result_fp16(self):
        self.check_trt_result(rtol=1e-3, atol=1e-3, precision_mode="fp16")


class TestMishCase1TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.nn.functional.mish
        self.api_args = {
            "x": np.random.randn(2).astype("float32"),
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1]}
        self.opt_shape = {"x": [2]}
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
        self.opt_shape = {"x": [2, 3]}
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
        self.opt_shape = {"x": [2, 3, 4]}
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
        self.opt_shape = {"x": [2, 3, 4, 2]}
        self.max_shape = {"x": [5, 3, 4, 2]}

    def test_trt_result(self):
        self.check_trt_result()


class TestLogSigmoidTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.nn.functional.log_sigmoid
        x = np.random.random([1, 3, 32, 32]).astype(np.float32)
        self.api_args = {
            "x": x,
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {
            "x": [1, 3, 32, 32],
        }
        self.opt_shape = {"x": [4, 3, 32, 32]}
        self.max_shape = {"x": [4, 3, 32, 32]}

    def test_trt_result(self):
        self.check_trt_result()

    def test_trt_result_fp16(self):
        self.check_trt_result(precision_mode="fp16")


class TestSeluTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.nn.functional.selu
        self.api_args = {
            "x": np.random.randn(2, 3).astype("float32"),
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3]}
        self.opt_shape = {"x": [2, 3]}
        self.max_shape = {"x": [5, 3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestLeakyReluCas1TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.nn.functional.leaky_relu
        self.api_args = {
            "x": np.random.randn(2, 3).astype("float32"),
            "negative_slope": 0.5,
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3]}
        self.opt_shape = {"x": [2, 3]}
        self.max_shape = {"x": [5, 3]}

    def test_trt_result_fp16(self):
        self.check_trt_result(precision_mode="fp16")

    def test_trt_result_fp32(self):
        self.check_trt_result()


class TestLeakyReluCase2TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.nn.functional.leaky_relu
        self.api_args = {
            "x": np.random.randn(2, 3).astype("float32"),
            "negative_slope": -0.5,
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3]}
        self.opt_shape = {"x": [2, 3]}
        self.max_shape = {"x": [5, 3]}

    def test_trt_result_fp16(self):
        self.check_trt_result(precision_mode="fp16")

    def test_trt_result_fp32(self):
        self.check_trt_result()


class TestLeakyRelu_Cas1TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.nn.functional.leaky_relu_
        self.api_args = {
            "x": np.random.randn(2, 3).astype("float32"),
            "negative_slope": 0.5,
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3]}
        self.opt_shape = {"x": [2, 3]}
        self.max_shape = {"x": [5, 3]}

    def test_trt_result_fp16(self):
        self.check_trt_result(precision_mode="fp16")

    def test_trt_result_fp32(self):
        self.check_trt_result()


class TestLeakyRelu_Case2TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.nn.functional.leaky_relu_
        self.api_args = {
            "x": np.random.randn(2, 3).astype("float32"),
            "negative_slope": -0.5,
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3]}
        self.opt_shape = {"x": [2, 3]}
        self.max_shape = {"x": [5, 3]}

    def test_trt_result_fp16(self):
        self.check_trt_result(precision_mode="fp16")

    def test_trt_result_fp32(self):
        self.check_trt_result()


def prelu_wrapper(x, alpha_shape, data_format='NCHW'):
    alpha = paddle.create_parameter(
        shape=alpha_shape, dtype='float32', name="alpha"
    )
    return paddle.nn.functional.prelu(x, alpha, data_format)


class TestPReluCase1TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = prelu_wrapper
        self.api_args = {
            "x": np.random.randn(2, 3).astype("float32"),
            "alpha_shape": [3],
            "data_format": "NCHW",
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3]}
        self.opt_shape = {"x": [2, 3]}
        self.max_shape = {"x": [5, 3]}

    def test_trt_result_fp16(self):
        self.check_trt_result(precision_mode="fp16")

    def test_trt_result_fp32(self):
        self.check_trt_result()


class TestPReluCase2TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = prelu_wrapper
        self.api_args = {
            "x": np.random.randn(2, 3).astype("float32"),
            "alpha_shape": [3],
            "data_format": "NHWC",
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3]}
        self.opt_shape = {"x": [2, 3]}
        self.max_shape = {"x": [5, 3]}

    def test_trt_result_fp16(self):
        self.check_trt_result(precision_mode="fp16")

    def test_trt_result_fp32(self):
        self.check_trt_result()


class TestPReluCase3TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = prelu_wrapper
        self.api_args = {
            "x": np.random.randn(2, 3, 3).astype("float32"),
            "alpha_shape": [3],
            "data_format": "NCHW",
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3, 3]}
        self.opt_shape = {"x": [2, 3, 3]}
        self.max_shape = {"x": [5, 3, 3]}

    def test_trt_result_fp16(self):
        self.check_trt_result(precision_mode="fp16")

    def test_trt_result_fp32(self):
        self.check_trt_result()


class TestPReluCase4TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = prelu_wrapper
        self.api_args = {
            "x": np.random.randn(2, 3, 3).astype("float32"),
            "alpha_shape": [3],
            "data_format": "NHWC",
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3, 3]}
        self.opt_shape = {"x": [2, 3, 3]}
        self.max_shape = {"x": [5, 3, 3]}

    def test_trt_result_fp16(self):
        self.check_trt_result(precision_mode="fp16")

    def test_trt_result_fp32(self):
        self.check_trt_result()


if __name__ == '__main__':
    unittest.main()
