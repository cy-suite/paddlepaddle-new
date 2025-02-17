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
from multiprocessing import Process


def run_test_with_env(env_name):
    import os

    os.environ[env_name] = ""
    import paddle

    a = paddle.zeros([2, 2])
    b = paddle.ones([2, 2])
    c = a + b
    assert c.place.is_cpu_place()


class TestEmptyVisibleDevices(unittest.TestCase):
    def test_xpu_env(self):
        p = Process(target=run_test_with_env, args=("XPU_VISIBLE_DEVICES",))
        p.start()
        p.join()
        self.assertEqual(p.exitcode, 0)

    def test_cuda_env(self):
        p = Process(target=run_test_with_env, args=("CUDA_VISIBLE_DEVICES",))
        p.start()
        p.join()
        self.assertEqual(p.exitcode, 0)


if __name__ == "__main__":
    unittest.main()
