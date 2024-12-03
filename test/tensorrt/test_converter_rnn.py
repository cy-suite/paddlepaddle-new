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
from paddle import nn


def rnn_wrapper(
    Input,
    PreState_H,
    PreState_C,
    num_layers,
    hidden_size,
    input_size,
    is_bidirec,
):

    class SimpleRNN(nn.Layer):
        def __init__(self, num_layers, input_size, hidden_size, is_bidirec):
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                direction='bidirect' if is_bidirec else 'forward',
            )

        def forward(self, x, pre_state):
            x = paddle.transpose(x, [1, 0, 2])
            out, state = self.rnn(x, pre_state)
            return out

    rnn = SimpleRNN(num_layers, input_size, hidden_size, is_bidirec)
    x = (
        paddle.to_tensor(Input, dtype='float32')
        if not isinstance(Input, paddle.Tensor)
        else Input
    )
    h0 = (
        paddle.to_tensor(PreState_H, dtype='float32')
        if not isinstance(PreState_H, paddle.Tensor)
        else PreState_H
    )
    c0 = (
        paddle.to_tensor(PreState_C, dtype='float32')
        if not isinstance(PreState_C, paddle.Tensor)
        else PreState_C
    )

    out = rnn(x, (h0, c0))

    return out


class TestRnnTRTPattern(TensorRTBaseTest):
    def setUp(self):
        super().setUp()
        self.python_api = rnn_wrapper
        self.api_args = {
            "Input": np.random.random([5, 4, 10]).astype(
                "float32"
            ),  # [seq_len, batch, input_size]
            "PreState_H": np.random.random([1, 4, 20]).astype(
                "float32"
            ),  # [num_layers * num_directions, batch, hidden_size]
            "PreState_C": np.random.random([1, 4, 20]).astype(
                "float32"
            ),  # [num_layers * num_directions, batch, hidden_size]
            "num_layers": 1,
            "hidden_size": 20,
            "input_size": 10,
            "is_bidirec": False,
        }
        self.program_config = {
            "feed_list": ["Input", "PreState_H", "PreState_C"],
            "fetch_list": ["Out"],
        }
        self.min_shape = {
            "Input": [1, 4, 10],
            "PreState_H": [1, 4, 20],
            "PreState_C": [1, 4, 20],
        }
        self.max_shape = {
            "Input": [10, 4, 20],
            "PreState_H": [10, 4, 20],
            "PreState_C": [10, 4, 20],
        }

    def test_trt_result(self):
        self.check_trt_result()


if __name__ == "__main__":
    unittest.main()
