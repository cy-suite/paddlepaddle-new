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

import paddle
from paddle.base import core
from paddle.device.cuda import (
    device_count,
    max_memory_allocated,
    memory_allocated,
    max_memory_reserved,
    memory_reserved,
    memory_stats,
)


class TestMemoryStats(unittest.TestCase):
    def func_test_memory_stats(self, device=None):
        if core.is_compiled_with_cuda():
            alloc_time = 100
            max_alloc_size = 10000
            for i in range(alloc_time):
                # first alloc
                shape = paddle.randint(low=max_alloc_size, high=max_alloc_size*2)
                tensor = paddle.zeros(shape)
                del shape
                del tensor

                # second alloc
                shape = paddle.randint(low=0, high=max_alloc_size)
                tensor = paddle.zeros(shape)

                memory_stats = paddle.device.cuda.memory_stats(device)

                self.assertEqual(memory_stats["memory.allocated.current"], memory_allocated(device))
                self.assertEqual(memory_stats["memory.allocated.peak"], max_memory_allocated(device))

                self.assertEqual(memory_stats["memory.reserved.current"], memory_reserved(device))
                self.assertEqual(memory_stats["memory.reserved.peak"], max_memory_reserved(device))

                del shape
                del tensor

    def test_memory_stats_for_all_places(self):
        if core.is_compiled_with_cuda():
            gpu_num = device_count()
            for i in range(gpu_num):
                paddle.device.set_device("gpu:" + str(i))
                self.func_test_memory_stats(core.CUDAPlace(i))
                self.func_test_memory_stats(i)
                self.func_test_memory_stats("gpu:" + str(i))

    def test_memory_stats_exception(self):
        if core.is_compiled_with_cuda():
            wrong_device = [
                core.CPUPlace(),
                device_count() + 1,
                -2,
                0.5,
                "gpu1",
            ]
            for device in wrong_device:
                with self.assertRaises(BaseException):  # noqa: B017
                    memory_stats(device)
        else:
            with self.assertRaises(ValueError):
                memory_stats()


if __name__ == "__main__":
    unittest.main()
