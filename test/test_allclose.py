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

import numpy as np

import paddle
from paddle.jit import to_static
from paddle.static import InputSpec

x_spcec = InputSpec([1])
y_spcec = InputSpec([1])
rtol_spcec = InputSpec([])
atol_spcec = InputSpec([])


x_np = np.array([10.1]).astype("float64")
y_np = np.array([10]).astype("float64")
z_np = np.allclose(x_np, y_np, 0.01, 0.0)
print(f"z_np: {z_np}")


def func(x, y):
    return paddle.allclose(x, y, rtol=0.01, atol=0.0)


static_func = to_static(func, input_spec=[x_spcec, y_spcec], full_graph=True)

x = paddle.to_tensor(x_np)
y = paddle.to_tensor(y_np)
_rtol = paddle.to_tensor(np.array([0.01]).astype("float64"))
_atol = paddle.to_tensor(np.array([0.0]).astype("float64"))

z = static_func(x, y)
z.backward()

print(z)
