# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import parameterized as param

import paddle
from paddle.base import core


@param.parameterized_class(
    ('arr', 'indices', 'axis', 'cotangent', 'dtype'),
    [
        (
            np.random.rand(4, 3, 2),  # arr
            np.array([[0, 1], [2, 1]], dtype=np.int64),  # indices
            1,  # axis
            np.random.rand(4, 2, 2),  # cotangent
            np.float32,  # dtype
        ),
        # 可以添加更多测试用例
    ],
)
class TestTakeAlongAxisTanhDoubleGrad(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        core.set_prim_eager_enabled(True)
        cls.arr = cls.arr.astype(cls.dtype)
        if cls.cotangent is not None:
            cls.cotangent = cls.cotangent.astype(cls.dtype)

    @classmethod
    def tearDownClass(cls):
        core.set_prim_eager_enabled(False)

    def test_take_along_axis_tanh_double_grad(self):
        def actual(arr, indices, axis, cotangent):
            arr = paddle.to_tensor(arr)
            arr.stop_gradient = False
            indices = paddle.to_tensor(indices)

            # 复合函数：take_along_axis后接tanh
            def composite_func(x):
                taken = paddle.take_along_axis(x, indices, axis)
                return paddle.tanh(taken)

            # 计算一阶导数
            first_grad = paddle.grad(
                composite_func(arr), arr, paddle.to_tensor(cotangent)
            )[0]
            first_grad.stop_gradient = False

            # 计算二阶导数
            second_grad = paddle.grad(
                first_grad, arr, paddle.to_tensor(cotangent)
            )[0]

            return second_grad

        def desired(arr, indices, axis, cotangent):
            # 使用numpy计算参考结果
            def numpy_composite_grad(x):
                # 先计算take_along_axis
                taken = np.take_along_axis(x, indices, axis)
                # tanh的二阶导数是 -2 * tanh(x) * (1 - tanh(x)^2)
                tanh_x = np.tanh(taken)
                second_derivative = -2 * tanh_x * (1 - tanh_x**2)

                # 创建与输入相同形状的零数组
                result = np.zeros_like(arr)

                # 将二阶导数放回原始位置
                # 这里需要考虑cotangent的影响
                if axis == 1:
                    for i in range(indices.shape[0]):
                        for j in range(indices.shape[1]):
                            result[:, indices[i, j], :] += (
                                second_derivative[:, i, :] * cotangent[:, i, :]
                            )

                return result

            return numpy_composite_grad(arr)

        np.testing.assert_allclose(
            actual=actual(self.arr, self.indices, self.axis, self.cotangent),
            desired=desired(self.arr, self.indices, self.axis, self.cotangent),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_stop_gradients(self):
        with self.assertRaises(ValueError):
            arr = paddle.to_tensor(self.arr)
            arr.stop_gradient = True
            indices = paddle.to_tensor(self.indices)
            return paddle.grad(
                paddle.tanh(paddle.take_along_axis(arr, indices, self.axis)),
                arr,
                paddle.to_tensor(self.cotangent),
            )


if __name__ == '__main__':
    unittest.main()
