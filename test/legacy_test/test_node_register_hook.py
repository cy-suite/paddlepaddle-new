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

import unittest

import paddle


class TestNodeRegisterHook(unittest.TestCase):
    def test_node_post_hook(self):
        def hook(outputs, inputs):
            ret = []
            for out in outputs:
                ret.append(out + out)
            return ret

        a = paddle.rand([4, 4])
        b = paddle.rand([4, 4])
        a.stop_gradient = False
        b.stop_gradient = False

        c = a + b
        d = c + c

        handle = d.grad_fn._register_post_hook(hook)

        e = d + d
        d.sum().backward()
        check = paddle.ones([4, 4]) * 4
        self.assertTrue(paddle.equal_all(check, a.grad))


if __name__ == '__main__':
    unittest.main()
