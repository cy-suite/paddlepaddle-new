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

from __future__ import annotations

import numpy as np


def normalize_kernel(st, ed, step, dim_size) -> tuple[
    int,
    int,
]:
    ZERO_DIM_RESULT = (0, 0)

    """
          0   1     2     3    ..  D-1  D
    -D-1 -D  -D+1  -D+2  -D+3  ...-1
    """
    # 0 dim size, just return
    if dim_size <= 0:
        return ZERO_DIM_RESULT

    # positive step
    if step > 0:
        # 0 dim size case 1
        if st >= dim_size:
            return ZERO_DIM_RESULT

        # 0 dim size case 2
        if ed <= -dim_size:
            return ZERO_DIM_RESULT

        # make st belongs: (-inf, -D-1)∪[0, D)
        if -dim_size <= st < 0:
            st += dim_size
        # make st belongs: [0, D)
        st = max(st, 0)

        # make ed belongs: [0, +inf)
        if -dim_size <= ed < 0:
            ed += dim_size
        # make ed belongs: [0, D]
        ed = min(ed, dim_size)

        # 0 dim size case 3
        if st >= ed:
            return ZERO_DIM_RESULT

        return (st, ed)

    # negative step
    else:
        # 0 dim size case 1
        if st <= -dim_size - 1:
            return ZERO_DIM_RESULT

        # 0 dim size case 2
        if ed >= dim_size - 1:
            return ZERO_DIM_RESULT

        # make st belongs: [0, D)∪[0, +inf)
        if -dim_size <= st < 0:
            st += dim_size
        # make st belongs: [0, D)
        st = min(st, dim_size - 1)

        # make ed belongs: [-inf, -D)∪[0, D)
        if -dim_size <= ed < 0:
            ed += dim_size
        # make ed belongs: [-D-1, -D)∪[0, D) ==> {-D-1}∪[0, D)
        ed = max(ed, -dim_size - 1)

        if ed == -dim_size - 1:
            # When ed=-D-1, it is symmetrical to when step is greater than 0 and ed=D.
            return (st, ed)

        # now only remain the case that ed belongs to: [0, D)
        # 0 dim size case 3
        if ed >= st:
            return ZERO_DIM_RESULT

        return (st, ed)


def normalize_start_end2(st, D):
    up_bound = D - 1
    low_bound = -D
    if st > up_bound:
        st = up_bound
    if st < low_bound:
        st = low_bound
    return st + D if (st < 0) else st


def normalize_end(ed, D):
    up_bound = D
    low_bound = -D - 1
    if ed > up_bound:
        ed = up_bound
    if ed < low_bound:
        ed = low_bound
    return ed + D if (ed < 0) else ed


def normalize_start_end(st, ed, step, D):
    """ """
    st_up_bound = D - 1
    st_low_bound = -D
    if st > st_up_bound and step > 0:
        return 0, 0
    if st < st_low_bound and step < 0:
        return 0, 0
    st = st + D if (st < 0) else st

    ed_up_bound = D
    ed_low_bound = -D - 1
    if ed > ed_up_bound and step < 0:
        return 0, 0
    if ed < ed_low_bound and step > 0:
        return 0, 0

    ed = ed + D if (ed < 0) else ed

    return st, ed


def get_symboli_outshape(x, st, ed, step, D):
    out_shape = []
    st, ed = normalize_start_end(st, ed, step, D)

    if step < 0 and st > ed:
        i = st
        while i > ed:
            out_shape.append(x[i])
            i += step
    elif step > 0 and st < ed:
        i = st
        while i < ed:
            out_shape.append(x[i])
            i += step

    return out_shape, st, ed


def test_symbolic(x, D):
    for st in range(-D - 2, D + 2):
        for ed in range(-D - 2, D + 2):
            for step in [-1, 1]:
                try:
                    vanilla_slice_result = x[st:ed:step]
                except Exception as e:
                    continue

                out_shape, norm_st, norm_ed = get_symboli_outshape(
                    x, st, ed, step, D
                )
                # normed_slice_result = x[norm_st:norm_ed:step]

                print(
                    f"vanilla_slice_result:{vanilla_slice_result}\nout_shape:{out_shape}"
                )
                print(
                    f"[PASS] st:{st}, ed:{ed}, step:{step} norm_st:{norm_st}, norm_ed:{norm_ed}"
                )
                np.testing.assert_array_equal(vanilla_slice_result, out_shape)


def test_kernel(x, D):
    for st in range(-D - 2, D + 2):
        for ed in range(-D - 2, D + 2):
            for step in [-1, 1]:
                try:
                    vanilla_slice_result = x[st:ed:step]
                except Exception as e:
                    continue

                norm_st, norm_ed = normalize_kernel(st, ed, step, D)
                normed_slice_result = x[norm_st:norm_ed:step]

                print(
                    f"vanilla_slice_result:{vanilla_slice_result}\nnormed_slice_result:{normed_slice_result}"
                )
                np.testing.assert_array_equal(
                    vanilla_slice_result, normed_slice_result
                )
                print(f"[PASS] {st}, {ed}, {step}: {norm_st}, {norm_ed}")


x = np.arange(10)
D = x.shape[0]

# test_kernel(x, D)
test_symbolic(x, D)
