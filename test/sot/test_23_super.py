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

import paddle
from paddle.jit.sot.psdb import check_no_breakgraph


class A:
    def add2(self, x):
        return x + 2

    def add3(self, x):
        return x + 3


class B(A):
    @check_no_breakgraph
    def test_super_no_args_add2(self, x):
        y = super().add2(x)
        return y

    @check_no_breakgraph
    def test_super_with_args_add3(self, x):
        y = super(B, self).add3(x)  # noqa: UP008
        return y

    @check_no_breakgraph
    def test_super_both_add5(self, x):
        return super().add2(x) + super(B, self).add3(x)  # noqa: UP008

    @check_no_breakgraph
    def test_self_name(me, x):
        # Test case where the instance is referred to as 'me' instead of 'self'
        return super(B, me).add2(x)  # noqa: UP008


class TestSingleInheritance(TestCaseBase):
    def setUp(self) -> None:
        super().setUp()
        self._instance = B()

    def test_super_no_args(self):
        x = paddle.to_tensor(33)
        self.assertEqual(x + 2, self._instance.test_super_no_args_add2(x))

    def test_super_with_args(self):
        x = paddle.to_tensor(33)
        self.assertEqual(x + 3, self._instance.test_super_with_args_add3(x))

    def test_super_both(self):
        x = paddle.to_tensor(33)
        self.assertEqual(2 * x + 5, self._instance.test_super_both_add5(x))


class X:
    def get(self):
        return "X"

    def add(self, x):
        return x + 5


class Y:
    def get(self):
        return "Y"

    def add(self, x):
        return x + 500


class Z(X, Y):
    @check_no_breakgraph
    def get(self):
        return super().get() + "Z"


class P(Y):
    @check_no_breakgraph
    def get(self):
        return super(P, self).get() + "P"  # noqa: UP008

    def add(self, x):
        return x + 50000


class Q(Z, P):
    @check_no_breakgraph
    def get(self):
        return super().get() + "Q"

    @check_no_breakgraph
    def getP(self):
        return super(P, self).get() + "Q"

    @check_no_breakgraph
    def getZ(self):
        return super(Z, self).get() + "Q"

    @check_no_breakgraph
    def add5_with_X(self, x):
        return super().add(x)


class TestMultipleInheritance(TestCaseBase):
    def setUp(self) -> None:
        super().setUp()
        self._instance = Q()

    def test_with_args(self):
        self.assertEqual("XZQ", self._instance.get())
        self.assertEqual("YQ", self._instance.getP())
        self.assertEqual("XQ", self._instance.getZ())

    def test_no_args(self):
        x = paddle.to_tensor(5.0)
        # Call order: Q -> Z -> X -> P -> Y
        self.assertEqual(x + 5, self._instance.add5_with_X(x))


class A_ATTR:
    a = 1


class B_ATTR(A_ATTR):
    @check_no_breakgraph
    def foo(self):
        return super().a


class TestSuperAttr(TestCaseBase):
    def setUp(self) -> None:
        super().setUp()
        self._instance = B_ATTR()

    def test_attr_equal(self):
        self.assertEqual(A_ATTR.a, self._instance.foo())


if __name__ == "__main__":
    unittest.main()
