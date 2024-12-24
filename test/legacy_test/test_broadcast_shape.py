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
from utils import dygraph_guard

import paddle


class TestBroadcastShape(unittest.TestCase):
    def test_result(self):
        shape = paddle.broadcast_shape([2, 1, 3], [1, 3, 1])
        self.assertEqual(shape, [2, 3, 3])

        shape = paddle.broadcast_shape(
            [-1, 1, 3], [1, 3, 1]
        )  # support compile time infershape
        self.assertEqual(shape, [-1, 3, 3])

    def test_error(self):
        self.assertRaises(
            ValueError, paddle.broadcast_shape, [2, 1, 3], [3, 3, 1]
        )


class TestBroadcastShapeEmptyTensorInput(unittest.TestCase):
    def _get_places(self):
        places = [paddle.base.CPUPlace()]
        if paddle.is_compiled_with_cuda():
            places.append(paddle.base.CUDAPlace(0))
        return places

    def _generate_inputs_outputs(self, shape1, shape2):
        tensor1 = np.random.random(shape1)
        tensor2 = np.random.random(shape2)
        out_ref = np.broadcast(tensor1, tensor2)
        return (tensor1, tensor2), out_ref

    def _test_broadcast_shape_with_shapes(self, shape1, shape2):
        (tensor1, tensor2), out_ref = self._generate_inputs_outputs(
            shape1, shape2
        )
        tensor1_paddle = paddle.to_tensor(tensor1)
        tensor2_paddle = paddle.to_tensor(tensor2)
        result_shape = paddle.broadcast_shape(
            tensor1_paddle.shape, tensor2_paddle.shape
        )

        self.assertEqual(list(out_ref.shape), result_shape)

    def test_broadcast_shape_with_dygraph_empty_tensor_input(self):
        with dygraph_guard():
            self._test_broadcast_shape_with_shapes((0,), (0,))
            self._test_broadcast_shape_with_shapes((5, 0), (0,))
            self._test_broadcast_shape_with_shapes((5, 0, 10), (10,))
            self._test_broadcast_shape_with_shapes((7, 11, 0), (11, 0))
            self._test_broadcast_shape_with_shapes((0, 11, 22), (11, 22))


if __name__ == "__main__":
    unittest.main()
