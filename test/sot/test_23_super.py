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


# ---------------------- test single inheritance case ----------------------
class A:
    @check_no_breakgraph
    def add2(self, x):
        return x + 2

    @check_no_breakgraph
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
    def test_self_name_me(me, x):
        # Test case where the instance is referred to as 'me' instead of 'self'
        return super(B, me).add2(x)  # noqa: UP008

    @check_no_breakgraph
    def test_self_name_this(this, x):
        # Test case where the instance is referred to as 'this' instead of 'self'
        return super(B, this).add2(x)  # noqa: UP008


class TestSingleInheritance(TestCaseBase):
    def test_super_no_args(self):
        self.assert_results(B().test_super_no_args_add2, paddle.to_tensor(33))

    def test_super_with_args(self):
        self.assert_results(B().test_super_with_args_add3, paddle.to_tensor(33))

    def test_super_both(self):
        self.assert_results(B().test_super_both_add5, paddle.to_tensor(33))

    def test_super_self_name(self):
        self.assert_results(B().test_self_name_me, paddle.to_tensor(33))
        self.assert_results(B().test_self_name_this, paddle.to_tensor(33))


# ---------------------- test multiple inheritance case ----------------------
class X:
    @check_no_breakgraph
    def addx(self, x):
        return 1 + x


class Y:
    @check_no_breakgraph
    def addx(self, x):
        return 2 + x


class Z(X, Y):
    @check_no_breakgraph
    def addx(self, x):
        return super().addx(x) + 3 + x


class P(Y):
    @check_no_breakgraph
    def addx(self, x):
        return super(P, self).addx(x) + 4 + x  # noqa: UP008


class Q(Z, P):
    @check_no_breakgraph
    def addx(self, x):
        return super(Q, self).addx(x) + 5 + x  # noqa: UP008

    @check_no_breakgraph
    def addxP(self, x):
        return super(P, self).addx(x) + 5 + x

    @check_no_breakgraph
    def addxZ(self, x):
        return super(Z, self).addx(x) + 5 + x


# Inheritance diagram
# X     Y
#  \   / \
#   \ /   \
#    Z     P
#     \   /
#      \ /
#       Q
class TestMultipleInheritance(TestCaseBase):
    def test_with_args(self):
        x = paddle.to_tensor([1.0])
        self.assert_results(Q().addx, x)
        self.assert_results(Q().addxP, x)
        self.assert_results(Q().addxZ, x)


# ---------------------- test `super()` as input ----------------------
class Cls_super_as_inp:
    @check_no_breakgraph
    def fn(self, x):
        return x + 1


@check_no_breakgraph
def super_as_input(spr):
    return spr.fn(2)


class TestSuperAsInput1(TestCaseBase, Cls_super_as_inp):
    def test_super_as_input(self):
        self.assert_results(super_as_input, super())


class Cls_super_as_inp_A:
    @check_no_breakgraph
    def test_fn(self, x):
        return x + 1


class Cls_super_as_inp_B(Cls_super_as_inp_A):
    @check_no_breakgraph
    def test_fn(self, x):
        return x + 4


class Cls_super_as_inp_C(Cls_super_as_inp_B):
    @check_no_breakgraph
    def super_as_input(self, spr, x):
        return spr.test_fn(x)

    @check_no_breakgraph
    def test_super_as_input(self, x, cls):
        return self.super_as_input(super(cls, self), x)


class TestSuperAsInput2(TestCaseBase):
    def test_super_as_input(self):
        x = paddle.to_tensor(3)
        self.assert_results(
            Cls_super_as_inp_C().test_super_as_input, x, Cls_super_as_inp_C
        )
        self.assert_results(
            Cls_super_as_inp_C().test_super_as_input, x, Cls_super_as_inp_B
        )


# ---------------------- test case which has no functions ----------------------
class A_attr:
    a = 1


class B_attr(A_attr):
    a = 111


class C_attr(B_attr):
    @check_no_breakgraph
    def foo(self, x):
        return (
            super().a + x,
            super(C_attr, self).a + x,  # noqa: UP008
            super(B_attr, self).a + x,
        )


class TestSuperAttr(TestCaseBase):
    def test_attr_equal(self):
        x = paddle.to_tensor([4.0])
        self.assert_results(C_attr().foo, x)


if __name__ == "__main__":
    unittest.main()
