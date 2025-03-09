# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import paddle


class TestAdamWAmp(unittest.TestCase):
    def setUp(self):
        self.amsgrad = False

    def test_adamw_op_dygraph_amp(self):
        paddle.disable_static()
        value = np.arange(26).reshape(2, 13).astype("float32")
        a = paddle.to_tensor(value)
        linear = paddle.nn.Linear(13, 5)
        adam = paddle.optimizer.AdamW(
            learning_rate=0.01,
            parameters=linear.parameters(),
            apply_decay_param_fun=lambda name: True,
            weight_decay=0.01,
            amsgrad=self.amsgrad,
        )
        output1 = linear(a)
        model, optimizer = paddle.amp.decorate(
            models=linear, optimizers=adam, level='O2', dtype="bfloat16"
        )
        with paddle.amp.auto_cast(
            dtype="bfloat16",
            enable=True,
            custom_white_list=None,
            custom_black_list=None,
            level="O2",
        ):
            output2 = model(a)
        np.testing.assert_equal(output1.numpy(), output2.numpy())


if __name__ == "__main__":
    unittest.main()
