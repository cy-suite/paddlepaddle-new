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

from test_case_base import TestCaseBase

from paddle.jit import sot


def create_simple_generator(a, b):
    yield a
    yield b


def simple_generator_user():
    gen = create_simple_generator(1, 2)
    x = next(gen)
    y = next(gen)
    return x, y


def create_simple_generator_with_breakgraph():
    yield 1
    sot.psdb.breakgraph()
    yield 2


def simple_generator_user_with_breakgraph():
    gen = create_simple_generator_with_breakgraph()
    x = next(gen)
    y = next(gen)
    return x + y


def create_simple_generator_with_outer_breakgraph():
    yield 1
    yield 2


def simple_generator_user_with_outer_breakgraph():
    gen = create_simple_generator_with_outer_breakgraph()
    x = next(gen)
    sot.psdb.breakgraph()  # ~~codegen 时候（reconstruct）~~需要准确生成一次 next(gen)，否则无法恢复这里的 side effect，应该对其专门实现 side effect，reconstruct 应该是无副作用的才对
    y = next(gen)
    return x + y


def genexpr_user():
    gen = (i for i in range(2))
    x = next(gen)
    y = next(gen)
    return x, y


def echo():
    recv = None
    recv = yield recv
    recv = yield recv
    recv = yield recv


def generator_send():
    s = echo()
    init = next(s)
    recv_1 = s.send(2)
    recv_2 = s.send(3)
    return init, recv_1, recv_2


def yield_from_generator():
    yield from create_simple_generator(1, 2)


def generator_yield_from_generator_user():
    s = yield_from_generator()
    x = next(s)
    y = next(s)
    return x, y


def yield_from_iterable():
    yield from [1, 2]


def generator_yield_from_iterable_user():
    s = yield_from_iterable()
    x = next(s)
    y = next(s)
    return x, y


class TestGenerator(TestCaseBase):
    def test_generator_simple(self):
        self.assert_results(simple_generator_user)

    def test_generator_send(self):
        self.assert_results(generator_send)

    def test_genexpr(self):
        self.assert_results(genexpr_user)

    def test_generator_yield_from_generator(self):
        self.assert_results(generator_yield_from_generator_user)

    def test_generator_yield_from_iterable(self):
        self.assert_results(generator_yield_from_iterable_user)


if __name__ == "__main__":
    unittest.main()
