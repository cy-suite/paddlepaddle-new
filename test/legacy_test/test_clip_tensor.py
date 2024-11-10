#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import paddle


class TestClipTenosr(unittest.TestCase):

    def test_shape_error(self):
        paddle.disable_static()

        def test_min_error():
            x = paddle.randn([3, 5, 8, 10], dtype='float16')
            min = paddle.randn([8, 3], dtype='float16')
            paddle.clip(x, min)

        self.assertRaises(ValueError, test_min_error)

        def test_max_error():
            x = paddle.randn([3, 5, 8, 10], dtype='float32')
            max = paddle.randn([8, 3], dtype='float32')
            paddle.clip(x, -5.0, max)

        self.assertRaises(ValueError, test_max_error)


class TestInplaceClipTensorAPI(unittest.TestCase):
    def test_shape_error(self):
        paddle.disable_static()

        def test_min_error():
            x = paddle.randn([3, 5, 8, 10], dtype='float16')
            min = paddle.randn([8, 3], dtype='float16')
            paddle.clip_(x, min)

        self.assertRaises(ValueError, test_min_error)

        def test_max_error():
            x = paddle.randn([3, 5, 8, 10], dtype='float32')
            max = paddle.randn([8, 3], dtype='float32')
            paddle.clip_(x, -5.0, max)

        self.assertRaises(ValueError, test_max_error)


if __name__ == '__main__':
    unittest.main()
