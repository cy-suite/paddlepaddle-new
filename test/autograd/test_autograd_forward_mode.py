# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import unittest

import paddle

sys.path.insert(0, '.')


class TestFWDContext(unittest.TestCase):
    def test_enter_exit_dual_level(self):
        paddle.autograd.enter_dual_level()
        assert (
            paddle.autograd.forward_mode._current_level == 0
        ), "The first enter dual level should be 0."
        paddle.autograd.exit_dual_level()
        assert (
            paddle.autograd.forward_mode._current_level == -1
        ), "The current dual level should be -1."


if __name__ == "__main__":
    unittest.main()
