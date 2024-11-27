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
from test_prim_sub_graph_backward_dynamic_shape import (
    TestPrimBaseWithGrad,
)

import paddle


def amax_net1(x):
    return paddle.amax(x, keepdim=True)


def amax_net2(x):
    return paddle.amax(x, keepdim=False)


def amax_net3(x):
    return paddle.amax(x, axis=[0, 1], keepdim=False)


def amax_net4(x):
    return paddle.amax(x, axis=[-1, -2], keepdim=False)


def amax_net5(x):
    return paddle.amax(x, axis=[-1, 0], keepdim=False)


def amin_net1(x):
    return paddle.amin(x, keepdim=True)


def amin_net2(x):
    return paddle.amin(x, keepdim=False)


def amin_net3(x):
    return paddle.amin(x, axis=[0, 1], keepdim=False)


def amin_net4(x):
    return paddle.amin(x, axis=[-1, -2], keepdim=False)


def amin_net5(x):
    return paddle.amin(x, axis=[-1, 0], keepdim=False)


class TestPrimAmaxWithGrad1(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.amax_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [30, 200, 40]
        self.x = np.ones(self.x_shape).astype(self.dtype)
        self.net = amax_net1
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimAmaxWithGrad2(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.amax_grad"
        self.dtype = "float32"
        self.x_shape = [30]
        self.init_x_shape = [30]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = amax_net1
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimAmaxWithGrad3(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.amax_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [30, 200, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = amax_net2
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimAmaxWithGrad4(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.amax_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [30, 200, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = amax_net3
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimAmaxWithGrad5(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.amax_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [30, 200, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = amax_net4
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimAmaxWithGrad6(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.amax_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [30, 200, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = amax_net5
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimAminWithGrad1(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.amin_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [30, 200, 40]
        self.x = np.ones(self.x_shape).astype(self.dtype)
        self.net = amin_net1
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimAminWithGrad2(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.amin_grad"
        self.dtype = "float32"
        self.x_shape = [30]
        self.init_x_shape = [30]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = amin_net1
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimAminWithGrad3(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.amin_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [30, 200, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = amin_net2
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimAminWithGrad4(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.amin_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [30, 200, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = amin_net3
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimAminWithGrad5(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.amin_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [30, 200, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = amin_net4
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimAminWithGrad6(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.amin_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [30, 200, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = amin_net5
        self.enable_cinn = False
        self.tol = 1e-6


if __name__ == "__main__":
    unittest.main()
