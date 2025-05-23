#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)
from op_test_xpu import XPUOpTest

import paddle
from paddle.base import core

paddle.enable_static()


def AffineGrid(theta, grid_shape):
    n = grid_shape[0]
    h = grid_shape[1]
    w = grid_shape[2]
    h_idx = np.repeat(np.linspace(-1, 1, h)[np.newaxis, :], w, axis=0).T[
        :, :, np.newaxis
    ]
    w_idx = np.repeat(np.linspace(-1, 1, w)[np.newaxis, :], h, axis=0)[
        :, :, np.newaxis
    ]
    grid = np.concatenate(
        [w_idx, h_idx, np.ones([h, w, 1])], axis=2
    )  # h * w * 3
    grid = np.repeat(grid[np.newaxis, :], n, axis=0)  # n * h * w *3

    ret = np.zeros([n, h * w, 2])
    theta = theta.transpose([0, 2, 1])
    for i in range(len(theta)):
        ret[i] = np.dot(grid[i].reshape([h * w, 3]), theta[i])

    return ret.reshape([n, h, w, 2]).astype("float64")


def getGridPointValue(data, x, y):
    data_shape = data.shape
    N = data_shape[0]
    C = data_shape[1]
    in_H = data_shape[2]
    in_W = data_shape[3]
    out_H = x.shape[1]
    out_W = x.shape[2]

    # out = np.zeros(data_shape, dtype='float64')
    out = np.zeros([N, C, out_H, out_W], dtype='float64')
    for i in range(N):
        for j in range(out_H):
            for k in range(out_W):
                if (
                    y[i, j, k] < 0
                    or y[i, j, k] > in_H - 1
                    or x[i, j, k] < 0
                    or x[i, j, k] > in_W - 1
                ):
                    out[i, :, j, k] = 0
                else:
                    out[i, :, j, k] = data[i, :, y[i, j, k], x[i, j, k]]

    return out


def AffineGrid3D(theta, grid_shape):
    n = grid_shape[0]
    d = grid_shape[1]
    h = grid_shape[2]
    w = grid_shape[3]
    d_idx = np.repeat(
        np.repeat(np.linspace(-1, 1, d)[:, np.newaxis, np.newaxis], h, axis=1),
        w,
        axis=2,
    )[:, :, :, np.newaxis]
    h_idx = np.repeat(
        np.repeat(np.linspace(-1, 1, h)[np.newaxis, :, np.newaxis], w, axis=2),
        d,
        axis=0,
    )[:, :, :, np.newaxis]
    w_idx = np.repeat(
        np.repeat(np.linspace(-1, 1, w)[np.newaxis, np.newaxis, :], h, axis=1),
        d,
        axis=0,
    )[:, :, :, np.newaxis]
    grid = np.concatenate(
        [w_idx, h_idx, d_idx, np.ones([d, h, w, 1])], axis=3
    )  # d * h * w * 4
    grid = np.repeat(grid[np.newaxis, :], n, axis=0)  # n * d * h * w *4
    ret = np.zeros([n, d * h * w, 3])
    theta = theta.transpose([0, 2, 1])
    for i in range(len(theta)):
        ret[i] = np.dot(grid[i].reshape([d * h * w, 4]), theta[i])

    return ret.reshape([n, d, h, w, 3]).astype("float64")


def getGridPointValue3D(data, x, y, z):
    data_shape = data.shape
    N = data_shape[0]
    C = data_shape[1]
    in_D = data_shape[2]
    in_H = data_shape[3]
    in_W = data_shape[4]
    out_D = x.shape[1]
    out_H = x.shape[2]
    out_W = x.shape[3]

    out = np.zeros([N, C, out_D, out_H, out_W], dtype='float64')
    for i in range(N):
        for j in range(out_D):
            for k in range(out_H):
                for l in range(out_W):
                    if (
                        y[i, j, k, l] < 0
                        or y[i, j, k, l] > in_H - 1
                        or x[i, j, k, l] < 0
                        or x[i, j, k, l] > in_W - 1
                        or z[i, j, k, l] < 0
                        or z[i, j, k, l] > in_D - 1
                    ):
                        out[i, :, j, k, l] = 0
                    else:
                        out[i, :, j, k, l] = data[
                            i, :, z[i, j, k, l], y[i, j, k, l], x[i, j, k, l]
                        ]

    return out


def clip(x, min_n, max_n):
    return np.maximum(np.minimum(x, max_n), min_n)


def unnormalizeAndClip(grid_slice, max_val, align_corners, padding_mode):
    if align_corners:
        grid_slice = 0.5 * ((grid_slice.astype('float64') + 1.0) * max_val)
    else:
        grid_slice = (
            0.5 * ((grid_slice.astype('float64') + 1.0) * (max_val + 1)) - 0.5
        )

    if padding_mode == "border":
        grid_slice = clip(grid_slice, 0, max_val)
    elif padding_mode == "reflection":
        double_range = 2 * max_val if align_corners else (max_val + 1) * 2
        grid_abs = (
            np.abs(grid_slice) if align_corners else np.abs(grid_slice + 0.5)
        )
        extra = grid_abs - np.floor(grid_abs / double_range) * double_range
        grid_slice = np.minimum(extra, double_range - extra)
        grid_slice = (
            grid_slice if align_corners else clip(grid_slice - 0.5, 0, max_val)
        )
    return grid_slice


def GridSampler(
    data, grid, align_corners=True, mode="bilinear", padding_mode="zeros"
):
    dims = data.shape
    N = dims[0]
    in_C = dims[1]
    in_H = dims[2]
    in_W = dims[3]

    out_H = grid.shape[1]
    out_W = grid.shape[2]

    x = grid[:, :, :, 0]
    y = grid[:, :, :, 1]
    y_max = in_H - 1
    x_max = in_W - 1

    x = unnormalizeAndClip(x, x_max, align_corners, padding_mode)
    y = unnormalizeAndClip(y, y_max, align_corners, padding_mode)

    if mode == "bilinear":
        x0 = np.floor(x).astype('int32')
        x1 = x0 + 1
        y0 = np.floor(y).astype('int32')
        y1 = y0 + 1

        wa = np.tile(
            ((x1 - x) * (y1 - y)).reshape((N, 1, out_H, out_W)), (1, in_C, 1, 1)
        )
        wb = np.tile(
            ((x1 - x) * (y - y0)).reshape((N, 1, out_H, out_W)), (1, in_C, 1, 1)
        )
        wc = np.tile(
            ((x - x0) * (y1 - y)).reshape((N, 1, out_H, out_W)), (1, in_C, 1, 1)
        )
        wd = np.tile(
            ((x - x0) * (y - y0)).reshape((N, 1, out_H, out_W)), (1, in_C, 1, 1)
        )

        va = getGridPointValue(data, x0, y0)
        vb = getGridPointValue(data, x0, y1)
        vc = getGridPointValue(data, x1, y0)
        vd = getGridPointValue(data, x1, y1)

        out = (wa * va + wb * vb + wc * vc + wd * vd).astype('float64')
    elif mode == "nearest":
        x = np.round(x).astype('int32')
        y = np.round(y).astype('int32')
        out = getGridPointValue(data, x, y)
    return out


def GridSampler3D(
    data, grid, align_corners=True, mode="bilinear", padding_mode="zeros"
):
    dims = data.shape
    N = dims[0]
    in_C = dims[1]
    in_D = dims[2]
    in_H = dims[3]
    in_W = dims[4]

    out_D = grid.shape[1]
    out_H = grid.shape[2]
    out_W = grid.shape[3]

    x = grid[:, :, :, :, 0]
    y = grid[:, :, :, :, 1]
    z = grid[:, :, :, :, 2]

    z_max = in_D - 1
    y_max = in_H - 1
    x_max = in_W - 1

    x = unnormalizeAndClip(x, x_max, align_corners, padding_mode)
    y = unnormalizeAndClip(y, y_max, align_corners, padding_mode)
    z = unnormalizeAndClip(z, z_max, align_corners, padding_mode)

    if mode == "bilinear":
        x0 = np.floor(x).astype('int32')
        x1 = x0 + 1
        y0 = np.floor(y).astype('int32')
        y1 = y0 + 1
        z0 = np.floor(z).astype('int32')
        z1 = z0 + 1

        w_tnw = np.tile(
            ((x1 - x) * (y1 - y) * (z1 - z)).reshape(
                (N, 1, out_D, out_H, out_W)
            ),
            (1, in_C, 1, 1, 1),
        )
        w_tne = np.tile(
            ((x - x0) * (y1 - y) * (z1 - z)).reshape(
                (N, 1, out_D, out_H, out_W)
            ),
            (1, in_C, 1, 1, 1),
        )
        w_tsw = np.tile(
            ((x1 - x) * (y - y0) * (z1 - z)).reshape(
                (N, 1, out_D, out_H, out_W)
            ),
            (1, in_C, 1, 1, 1),
        )
        w_tse = np.tile(
            ((x - x0) * (y - y0) * (z1 - z)).reshape(
                (N, 1, out_D, out_H, out_W)
            ),
            (1, in_C, 1, 1, 1),
        )
        w_bnw = np.tile(
            ((x1 - x) * (y1 - y) * (z - z0)).reshape(
                (N, 1, out_D, out_H, out_W)
            ),
            (1, in_C, 1, 1, 1),
        )
        w_bne = np.tile(
            ((x - x0) * (y1 - y) * (z - z0)).reshape(
                (N, 1, out_D, out_H, out_W)
            ),
            (1, in_C, 1, 1, 1),
        )
        w_bsw = np.tile(
            ((x1 - x) * (y - y0) * (z - z0)).reshape(
                (N, 1, out_D, out_H, out_W)
            ),
            (1, in_C, 1, 1, 1),
        )
        w_bse = np.tile(
            ((x - x0) * (y - y0) * (z - z0)).reshape(
                (N, 1, out_D, out_H, out_W)
            ),
            (1, in_C, 1, 1, 1),
        )

        v_tnw = getGridPointValue3D(data, x0, y0, z0)
        v_tne = getGridPointValue3D(data, x1, y0, z0)
        v_tsw = getGridPointValue3D(data, x0, y1, z0)
        v_tse = getGridPointValue3D(data, x1, y1, z0)
        v_bnw = getGridPointValue3D(data, x0, y0, z1)
        v_bne = getGridPointValue3D(data, x1, y0, z1)
        v_bsw = getGridPointValue3D(data, x0, y1, z1)
        v_bse = getGridPointValue3D(data, x1, y1, z1)

        out = (
            w_tnw * v_tnw
            + w_tne * v_tne
            + w_tsw * v_tsw
            + w_tse * v_tse
            + w_bnw * v_bnw
            + w_bne * v_bne
            + w_bsw * v_bsw
            + w_bse * v_bse
        ).astype('float64')

    elif mode == "nearest":
        x = np.round(x).astype('int32')
        y = np.round(y).astype('int32')
        z = np.round(z).astype('int32')
        out = getGridPointValue3D(data, x, y, z)
    return out


class XPUTestGridSamplerOP(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'grid_sampler'
        self.use_dynamic_create_class = False

    class TestXPUGridSamplerOp(XPUOpTest):
        def setUp(self):
            self.place = paddle.XPUPlace(0)
            self.init_dtype()
            self.op_type = 'grid_sampler'
            self.epsilon_xpu2xpu = 0.000001

            self.use_cudnn = False
            self.align_corners = True
            self.padding_mode = "zeros"
            self.mode = "bilinear"

            self.initTestCase()

            x = np.random.uniform(-10, 10, self.x_shape).astype(self.dtype)

            theta = np.zeros(self.theta_shape).astype(self.dtype)

            if len(self.grid_shape) == 4:
                for i in range(self.theta_shape[0]):
                    for j in range(2):
                        for k in range(3):
                            theta[i, j, k] = np.random.rand(1)[0]
                grid = AffineGrid(theta, self.grid_shape).astype(self.dtype)

                self.inputs = {'X': x, 'Grid': grid}
                self.attrs = {
                    'use_cudnn': self.use_cudnn,
                    "align_corners": self.align_corners,
                    "padding_mode": self.padding_mode,
                    "mode": self.mode,
                }
                self.outputs = {
                    'Output': GridSampler(
                        x,
                        grid,
                        self.align_corners,
                        self.mode,
                        self.padding_mode,
                    )
                }
            else:
                for i in range(self.theta_shape[0]):
                    for j in range(3):
                        for k in range(4):
                            theta[i, j, k] = np.random.rand(1)[0]
                grid = AffineGrid3D(theta, self.grid_shape)
                self.inputs = {'X': x, 'Grid': grid}
                self.attrs = {
                    'use_cudnn': self.use_cudnn,
                    "align_corners": self.align_corners,
                    "padding_mode": self.padding_mode,
                    "mode": self.mode,
                }
                self.outputs = {
                    'Output': GridSampler3D(
                        x,
                        grid,
                        self.align_corners,
                        self.mode,
                        self.padding_mode,
                    )
                }

        def initTestCase(self):
            self.x_shape = (2, 3, 8, 8)
            self.grid_shape = (2, 7, 9, 2)
            self.theta_shape = (2, 2, 3)
            self.align_corners = True
            self.padding_mode = "zeros"
            self.mode = "bilinear"

        def init_dtype(self):
            self.dtype = self.in_type

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            if hasattr(self, "no_need_check_grad") and self.no_need_check_grad:
                return
            self.check_grad_with_place(self.place, ['X', 'Grid'], 'Output')

    class TestGridSample1(TestXPUGridSamplerOp):
        def initTestCase(self):
            self.x_shape = (2, 3, 5, 6)
            self.grid_shape = (2, 8, 9, 2)
            self.theta_shape = (2, 2, 3)
            self.align_corners = False
            self.padding_mode = "zeros"
            self.mode = "bilinear"

    class TestGridSample2(TestXPUGridSamplerOp):
        def initTestCase(self):
            self.x_shape = (2, 3, 5, 6)
            self.grid_shape = (2, 8, 9, 2)
            self.theta_shape = (2, 2, 3)
            self.align_corners = False
            self.padding_mode = "border"
            self.mode = "bilinear"

    class TestGridSample3(TestXPUGridSamplerOp):
        def initTestCase(self):
            self.x_shape = (2, 3, 5, 6)
            self.grid_shape = (2, 8, 9, 2)
            self.theta_shape = (2, 2, 3)
            self.align_corners = False
            self.padding_mode = "reflection"
            self.mode = "bilinear"

    class TestGridSample4(TestXPUGridSamplerOp):
        def initTestCase(self):
            self.x_shape = (2, 3, 5, 6)
            self.grid_shape = (2, 8, 9, 2)
            self.theta_shape = (2, 2, 3)
            self.align_corners = True
            self.padding_mode = "reflection"
            self.mode = "bilinear"

    class TestGridSample5(TestXPUGridSamplerOp):
        def initTestCase(self):
            self.x_shape = (2, 3, 5, 6)
            self.grid_shape = (2, 8, 9, 2)
            self.theta_shape = (2, 2, 3)
            self.align_corners = False
            self.padding_mode = "reflection"
            self.mode = "nearest"

    class TestGridSample6(TestXPUGridSamplerOp):
        def initTestCase(self):
            self.x_shape = (2, 3, 128, 128)
            self.grid_shape = (2, 130, 130, 2)
            self.theta_shape = (2, 2, 3)
            self.align_corners = False
            self.padding_mode = "reflection"
            self.mode = "bilinear"

    class TestGridSample7(TestXPUGridSamplerOp):
        def initTestCase(self):
            self.x_shape = (2, 3, 128, 128)
            self.grid_shape = (2, 130, 130, 2)
            self.theta_shape = (2, 2, 3)
            self.align_corners = True
            self.padding_mode = "zeros"
            self.mode = "bilinear"

    # 3d grid_sample_grad is not supported yet
    @unittest.skipIf(
        core.get_xpu_device_version(0) == core.XPUVersion.XPU3,
        "grid_sample3d for XPU3 is not supported",
    )
    class TestGridSample3DBilinear(TestXPUGridSamplerOp):
        def initTestCase(self):
            self.x_shape = (2, 3, 5, 6, 7)
            self.grid_shape = (2, 8, 9, 10, 3)
            self.theta_shape = (2, 3, 4)
            self.align_corners = True
            self.padding_mode = "zeros"
            self.mode = "bilinear"

            self.no_need_check_grad = True

    @unittest.skipIf(
        core.get_xpu_device_version(0) == core.XPUVersion.XPU3,
        "grid_sample3d for XPU3 is not supported",
    )
    class TestGridSample3DNearest(TestXPUGridSamplerOp):
        def initTestCase(self):
            self.x_shape = (2, 3, 5, 6, 7)
            self.grid_shape = (2, 8, 9, 10, 3)
            self.theta_shape = (2, 3, 4)
            self.align_corners = True
            self.padding_mode = "zeros"
            self.mode = "nearest"

            self.no_need_check_grad = True

    @unittest.skipIf(
        core.get_xpu_device_version(0) == core.XPUVersion.XPU3,
        "grid_sample3d for XPU3 is not supported",
    )
    class TestGridSample3DBorder(TestXPUGridSamplerOp):
        def initTestCase(self):
            self.x_shape = (2, 3, 5, 6, 7)
            self.grid_shape = (2, 8, 9, 10, 3)
            self.theta_shape = (2, 3, 4)
            self.align_corners = True
            self.padding_mode = "border"
            self.mode = "nearest"

            self.no_need_check_grad = True

    @unittest.skipIf(
        core.get_xpu_device_version(0) == core.XPUVersion.XPU3,
        "grid_sample3d for XPU3 is not supported",
    )
    class TestGridSample3DReflection(TestXPUGridSamplerOp):
        def initTestCase(self):
            self.x_shape = (2, 3, 5, 6, 7)
            self.grid_shape = (2, 8, 9, 10, 3)
            self.theta_shape = (2, 3, 4)
            self.align_corners = True
            self.padding_mode = "reflection"
            self.mode = "bilinear"

            self.no_need_check_grad = True

    @unittest.skipIf(
        core.get_xpu_device_version(0) == core.XPUVersion.XPU3,
        "grid_sample3d for XPU3 is not supported",
    )
    class TestGridSample3DAlignCornersFalse(TestXPUGridSamplerOp):
        def initTestCase(self):
            self.x_shape = (2, 3, 5, 6, 7)
            self.grid_shape = (2, 8, 9, 10, 3)
            self.theta_shape = (2, 3, 4)
            self.align_corners = False
            self.padding_mode = "reflection"
            self.mode = "bilinear"

            self.no_need_check_grad = True


support_types = get_xpu_op_support_types('grid_sampler')
for stype in support_types:
    create_test_class(globals(), XPUTestGridSamplerOP, stype)

if __name__ == '__main__':
    unittest.main()
