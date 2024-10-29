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

import utils

import paddle
from paddle import nn


class RotaryPosEmb(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, cos, sin, position_ids):
        x = x.reshape([1, 4096, 32, 128 * 3])
        s1 = paddle.split(x, 3, axis=-1)
        q = s1[0]
        k = s1[1]
        v = s1[2]

        cos = cos.squeeze(axis=[0, 2])  # [seq_len, dim]
        sin = sin.squeeze(axis=[0, 2])  # [seq_len, dim]

        cos = cos[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]
        sin = sin[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed, v

    def rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return paddle.concat([-x2, x1], axis=-1)  # shape is the same as x


class TestRotaryPosEmb(unittest.TestCase):
    def setUp(self):
        paddle.seed(2022)
        self.prepare_data()

    def prepare_data(self):
        # self.q = paddle.randn([61, 2048, 8, 96], dtype="float32")
        # self.q.stop_gradient = False

        # self.k = paddle.randn([61, 2048, 8, 96], dtype="float32")
        # self.k.stop_gradient = False
        self.x = paddle.randn([1, 4096, 4096 * 3], dtype="float16")
        self.x.stop_gradient = False

        self.cos = paddle.randn([1, 4096, 1, 128], dtype="float16")
        self.cos.stop_gradient = True

        self.sin = paddle.randn([1, 4096, 1, 128], dtype="float16")
        self.sin.stop_gradient = True

        self.position_ids = paddle.arange(end=4096, dtype="int64").unsqueeze(0)
        self.position_ids.stop_gradient = True

    def check_jit_kernel_info(self, static_fn):
        utils.check_jit_kernel_number(static_fn, 1)
        utils.check_jit_kernel_structure(static_fn, {utils.JIT_KERNEL_NAME: 1})

    def eval(self, use_cinn):
        paddle.seed(2022)
        net = RotaryPosEmb()
        net = utils.apply_to_static(net, use_cinn)
        # net.eval()
        out = net(self.x, self.cos, self.sin, self.position_ids)

        loss = out[0] + out[1] + out[2]
        loss = loss.sum()

        loss.backward()
        # TODO(phlrain): Need to Fuse to one Kernel
        # if use_cinn:
        #     self.check_jit_kernel_info(net.forward)
        return out

    def test_eval(self):
        cinn_outs = self.eval(use_cinn=True)
        # dy_outs = self.eval(use_cinn=False)

        # for cinn_out, dy_out in zip(cinn_outs, dy_outs):
        #     np.testing.assert_allclose(
        #         cinn_out.numpy(), dy_out.numpy(), atol=1e-6, rtol=1e-6
        #     )


if __name__ == '__main__':
    unittest.main()
