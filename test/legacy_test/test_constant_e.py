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

import math
import unittest

import paddle

paddle.enable_static()


class TestPaddleE(unittest.TestCase):
    def setUp(self):
        # Verifying paddle.e against math.e
        self.expected_value = math.e
        self.e_value = paddle.e

    def test_check_value(self):
        self.assertAlmostEqual(
            self.e_value,
            self.expected_value,
            places=6,
            msg="paddle.e does not match math.e",
        )

    def test_check_type(self):
        self.assertIsInstance(
            self.e_value, float, msg="paddle.e is not of type float"
        )

    def test_check_value_with_precision(self):
        self.assertTrue(
            abs(self.e_value - self.expected_value) < 1e-6,
            msg="paddle.e value mismatch with math.e",
        )


if __name__ == "__main__":
    unittest.main()
