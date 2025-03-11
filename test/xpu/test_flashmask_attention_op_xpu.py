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

import numpy as np

import paddle
from paddle.base import core
from paddle.nn.functional.flash_attention import (
    flashmask_attention,
)


def naive_attention(query, key, value, mask):
    if query.dtype != paddle.float32:
        query = paddle.cast(query, 'float32')
        key = paddle.cast(key, 'float32')
        value = paddle.cast(value, 'float32')
        mask = paddle.cast(mask, 'float32')

    assert query.dtype == paddle.float32
    assert key.dtype == paddle.float32
    assert value.dtype == paddle.float32
    assert mask.dtype == paddle.float32

    scale = 1.0 / np.sqrt(query.shape[-1])
    query = paddle.transpose(query, [0, 2, 1, 3])
    key = paddle.transpose(key, [0, 2, 1, 3])
    value = paddle.transpose(value, [0, 2, 1, 3])
    product = paddle.matmul(x=query, y=key, transpose_y=True)

    product = paddle.scale(product, scale)

    if mask is not None:
        mask[mask == -np.inf] = -3.3895313892515355e37
        product = product + mask

    weights = paddle.nn.functional.softmax(product)

    out = paddle.matmul(weights, value)
    out = paddle.transpose(out, [0, 2, 1, 3])
    return out


def flashmask_to_densemask(startend_row_indices, dtype, causal=True):
    if startend_row_indices is None:
        return None
    bz, num_head, seq_len, bound_num = startend_row_indices.shape
    m = paddle.zeros((bz, num_head, seq_len, seq_len), dtype=dtype)
    has_end = (causal and bound_num == 2) or ((not causal) and bound_num == 4)
    for bi in range(bz):
        for hi in range(num_head):
            for j in range(seq_len):
                downstart = startend_row_indices[bi, hi, j, 0]
                if has_end:
                    downend = startend_row_indices[bi, hi, j, 1]
                    m[bi, hi, downstart:downend, j] = -np.inf
                else:
                    m[bi, hi, downstart:, j] = -np.inf
                if causal:
                    m[bi, hi, :j, j] = -np.inf
                else:
                    if has_end:
                        upstart = startend_row_indices[bi, hi, j, 2]
                        upend = startend_row_indices[bi, hi, j, 3]
                        m[bi, hi, upstart:upend, j] = -np.inf
                    else:
                        upend = startend_row_indices[bi, hi, j, 1]
                        m[bi, hi, :upend, j] = -np.inf
    return m


def generate_causal_mask(B, L, H, D, doc_seq_lens):
    total_seq_len = np.sum(doc_seq_lens)
    assert total_seq_len <= L
    assert len(doc_seq_lens) >= 3
    padding = L - np.sum(doc_seq_lens)
    doc_seq_lens[-1] += padding
    seq_cusums = np.cumsum(doc_seq_lens)

    startend_row_indices = np.repeat(seq_cusums, doc_seq_lens)
    startend_row_indices = (
        paddle.to_tensor(startend_row_indices, dtype=paddle.int32)
        .reshape((1, 1, L, 1))
        .repeat_interleave(B, 0)
    )

    causal = True
    return startend_row_indices, causal


def generate_share_question_mask(B, L, H, D, doc_seq_lens):
    seq_cusums = np.cumsum(doc_seq_lens)
    seq_cusums = np.append(seq_cusums, 128)

    total_seq_len = np.sum(doc_seq_lens)
    assert total_seq_len <= L
    assert len(doc_seq_lens) >= 3
    padding = L - total_seq_len

    startend_row_indices = [total_seq_len] * doc_seq_lens[0]

    cur_len_so_far = doc_seq_lens[0]
    for idx in range(1, len(doc_seq_lens)):
        cur_len_so_far += doc_seq_lens[idx]
        startend_row_indices.extend([cur_len_so_far] * doc_seq_lens[idx])

    if padding > 0:
        startend_row_indices.extend([cur_len_so_far] * padding)
    startend_row_indices = (
        paddle.to_tensor(startend_row_indices, dtype=paddle.int32)
        .reshape((1, 1, L, 1))
        .repeat_interleave(B, 0)
    )

    causal = True
    return startend_row_indices, causal


@unittest.skipIf(
    core.is_compiled_with_xpu()
    and core.get_xpu_device_version(0) < core.XPUVersion.XPU3,
    "run test when xpu's compute capability >= xpu3.",
)
class TestFlashAttentionAPI(unittest.TestCase):
    def setUp(self):
        self.place = paddle.XPUPlace(0)
        self.shape = (1, 128, 2, 32)
        self.seq_lod = [31, 28, 69]
        self.dropout = 0.0
        self.return_softmax = False

    def test_all(self):
        self.run_case(
            dtype="float16",
            tolerance=5e-4,
            tolerance_dv=1e-3,
            generate_mask_fn=generate_causal_mask,
        )
        self.run_case(
            dtype="bfloat16",
            tolerance=6e-3,
            tolerance_dv=1e-2,
            generate_mask_fn=generate_causal_mask,
        )
        self.run_case(
            dtype="float16",
            tolerance=5e-4,
            tolerance_dv=1e-3,
            generate_mask_fn=generate_share_question_mask,
        )
        self.run_case(
            dtype="bfloat16",
            tolerance=6e-3,
            tolerance_dv=1e-2,
            generate_mask_fn=generate_share_question_mask,
        )

    def run_case(self, dtype, tolerance, tolerance_dv, generate_mask_fn):
        print(f"Test case shape {self.shape} dtype {dtype}")

        # test dynamic
        paddle.disable_static()
        B = self.shape[0]
        L = self.shape[1]
        H = self.shape[2]
        D = self.shape[3]
        np.random.seed(2023)
        query = np.random.uniform(-1.0, 1.0, self.shape)
        key = np.random.uniform(-1.0, 1.0, self.shape)
        value = np.random.uniform(-1.0, 1.0, self.shape)

        q = paddle.to_tensor(
            query, place=self.place, dtype=dtype, stop_gradient=False
        )
        k = paddle.to_tensor(
            key, place=self.place, dtype=dtype, stop_gradient=False
        )
        v = paddle.to_tensor(
            value, place=self.place, dtype=dtype, stop_gradient=False
        )

        q_ = paddle.to_tensor(
            query, place=self.place, dtype=dtype, stop_gradient=False
        )
        k_ = paddle.to_tensor(
            key, place=self.place, dtype=dtype, stop_gradient=False
        )
        v_ = paddle.to_tensor(
            value, place=self.place, dtype=dtype, stop_gradient=False
        )

        startend_row_indices, is_causal = generate_mask_fn(
            B, L, H, D, self.seq_lod
        )
        dense_mask = flashmask_to_densemask(
            startend_row_indices, "float32", is_causal
        )

        out = flashmask_attention(
            q,
            k,
            v,
            startend_row_indices,
            dropout=self.dropout,
            causal=is_causal,
        )

        out_ = naive_attention(q_, k_, v_, dense_mask)

        # forward result
        float_out = paddle.cast(out, "float32")
        float_out_ = paddle.cast(out_, "float32")

        max_diff_forward = np.max(
            np.abs(float_out.numpy() - float_out_.numpy())
        )
        mean_diff_forward = np.mean(
            np.abs(float_out.numpy() - float_out_.numpy())
        )
        print("max_diff_forward:", max_diff_forward)
        print("mean_diff_forward:", mean_diff_forward)

        np.testing.assert_allclose(
            float_out, float_out_, rtol=tolerance, atol=tolerance
        )


if __name__ == '__main__':
    unittest.main()
