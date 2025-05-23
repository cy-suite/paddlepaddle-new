#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""This is the lib for gradient checker unittest."""

from collections.abc import Sequence
from itertools import product

import numpy as np

import paddle
from paddle import base
from paddle.autograd.backward_utils import ValueDict
from paddle.base import core
from paddle.base.backward import _append_grad_suffix_, _as_list
from paddle.base.framework import in_pir_mode


def _product(t):
    return int(np.prod(t))


# data type like int32, int64, bool, that do not requires grad
DTYPE_REQUIRES_GRAD = [
    paddle.float16,
    paddle.float32,
    paddle.float64,
    core.DataType.FLOAT16,
    core.DataType.FLOAT32,
    core.DataType.FLOAT64,
]


def dtype_to_np_dtype(dtype):
    if dtype == paddle.float32 or dtype == core.DataType.FLOAT32:
        return np.float32
    elif dtype == paddle.float64 or dtype == core.DataType.FLOAT64:
        return np.float64
    elif dtype == paddle.float16 or dtype == core.DataType.FLOAT16:
        return np.float16
    else:
        raise ValueError("Not supported data type " + str(dtype))


def _get_item(t, i, np_dtype):
    if np_dtype == np.float16:
        np_t = np.array(t).astype(np.float16)
        np_t = np_t.flatten()
        return np_t[i]
    elif np_dtype == np.float32:
        return t._get_float_element(i)
    elif np_dtype == np.float64:
        return t._get_double_element(i)
    else:
        raise ValueError("Not supported data type " + str(np_dtype))


def _set_item(t, i, e, np_dtype, place):
    if np_dtype == np.float16:
        np_t = np.array(t).astype(np.float16)
        shape = np_t.shape
        np_t = np_t.flatten()
        np_t[i] = e
        np_t = np_t.reshape(shape)
        t.set(np_t, place)
    elif np_dtype == np.float32:
        t._set_float_element(i, e)
    elif np_dtype == np.float64:
        t._set_double_element(i, e)
    else:
        raise ValueError("Not supported data type " + str(np_dtype))


def set_var_in_scope(scope, place, name, value, recursive_seq_len=None):
    t = scope.var(name).get_tensor()
    t.set(value, place)
    if recursive_seq_len:
        t.set_recursive_sequence_lengths(recursive_seq_len)
    return t


def var_to_np_array_in_scope(scope, place, name):
    return np.array(scope.var(name).get_tensor())


def make_jacobian(x, y_size, np_dtype):
    if isinstance(x, (base.framework.Variable, paddle.pir.Value)):
        return np.zeros([_product(x.shape), y_size], dtype=np_dtype)
    elif isinstance(x, Sequence):
        jacobians = list(
            filter(
                lambda t: t is not None,
                (make_jacobian(item, y_size, np_dtype) for item in x),
            )
        )
        return jacobians
    else:
        pass


def _compute_numerical_jacobian(program, x, y, place, scope, delta):
    """Computes the numeric Jacobian for dy/dx.

    Computes the numeric Jacobian by slightly perturbing the inputs and
    measuring the differences on the output.

    Args:
        program (Program): the network program.
        x (Variable): the input variables.
        y (list[Variable]): the output variables.
        place (base.CPUPlace or base.CUDAPlace): the device.
        scope (Scope): the scope used to run program.
        delta: the amount of perturbation we give to the input

    Returns:
        A list of 2-D numpy array, the list length is len(y).
        Each 2-D numpy array represents the Jacobian for dy_i/dx.
        It has "x_size" rows and "y_size" columns
        where "x_size" is the number of elements in x and
        "y_size" is the number of elements in each y_i.
    """
    if not isinstance(x, base.framework.Variable):
        raise TypeError('x is not Variable')

    # To compute the jacobian, treat x and y as one-dimensional vectors.
    y = _as_list(y)
    exe = base.Executor(place)

    def run():
        y_res = exe.run(program, scope=scope, fetch_list=y)
        return [yi.flatten() for yi in y_res]

    x_name = x.name
    x_shape = x.shape
    x_size = _product(x_shape)
    x_t = scope.find_var(x_name).get_tensor()

    np_type = dtype_to_np_dtype(x.dtype)
    jacobian = [make_jacobian(x, _product(yi.shape), np_type) for yi in y]

    for i in range(x_size):
        orig = _get_item(x_t, i, np_type)
        x_pos = orig + delta
        _set_item(x_t, i, x_pos, np_type, place)
        y_pos = run()

        x_neg = orig - delta
        _set_item(x_t, i, x_neg, np_type, place)
        y_neg = run()

        _set_item(x_t, i, orig, np_type, place)

        for j in range(len(y)):
            jacobian[j][i, :] = (y_pos[j] - y_neg[j]) / delta / 2.0

    return jacobian


def _compute_analytical_jacobian(program, x, y, place, scope):
    """Computes the analytical Jacobian for dy/dx.

    Args:
        program (Program): a Program with forward pass.
        x (Variable|list[Variable]): a variable or list of variable
        y (Variable): the target variable.
        place (base.CPUPlace or base.CUDAPlace): the device.
        scope (Scope): the scope used to run program.

    Returns:
        A list of 2-D numpy array. The list length is len(x).
        Each 2-D numpy array represents the Jacobian for dy/dx_i.
        It has "xi_size" rows and "dy_size" columns
        where "x_size" is the number of elements in x_i and
        "dy_size" is the number of elements in y.
    """
    if not isinstance(y, base.framework.Variable):
        raise TypeError('y is not Variable')

    dy_name = _append_grad_suffix_(y.name)

    np_type = dtype_to_np_dtype(y.dtype)
    # create dy Variable in Program
    dy = program.global_block().create_var(
        name=dy_name, shape=y.shape, dtype=np_type, persistable=True
    )
    # append backward
    dx = base.gradients(y, x, dy)

    # init dy tensor in scope
    value = np.zeros(y.shape, dtype=np_type)
    dy_t = set_var_in_scope(scope, place, dy_name, value)

    exe = base.Executor(place)

    y_size = _product(y.shape)

    x = _as_list(x)
    jacobian = make_jacobian(x, y_size, np_type)

    # filter None in dx for DX/DY may be None in kernel
    # only fetch not None dx in exe.run
    filted = [(i, dxi) for i, dxi in enumerate(dx) if dxi is not None]
    filted_idx, filted_dx = zip(*filted)

    for i in range(y_size):
        _set_item(dy_t, i, 1, np_type, place)

        dx_res = exe.run(program, scope=scope, fetch_list=filted_dx)

        for j in range(len(filted_dx)):
            dx_idx = filted_idx[j]
            if dx_res[j] is not None:
                jacobian[dx_idx][:, i] = dx_res[j].flatten()
            else:
                jacobian[dx_idx][:, i] = np.zeros(
                    dx[dx_idx].shape, dtype=np_type
                ).flatten()

        _set_item(dy_t, i, 0, np_type, place)

    return jacobian


def _compute_numerical_jacobian_pir(
    program, x, y, fetch_list, feeds, place, delta
):
    """Computes the numeric Jacobian for dy/dx.

    Computes the numeric Jacobian by slightly perturbing the inputs and
    measuring the differences on the output.

    Args:
        program (Program): the network program.
        x (Variable): the input variables.
        y (list[Variable]): the output variables.
        fetch_list (list[Variable]): the variables to fetch.
        feeds (dict): the feed dict.
        place (base.CPUPlace or base.CUDAPlace): the device.
        delta: the amount of perturbation we give to the input

    Returns:
        A list of 2-D numpy array, the list length is len(y).
        Each 2-D numpy array represents the Jacobian for dy_i/dx.
        It has "x_size" rows and "y_size" columns
        where "x_size" is the number of elements in x and
        "y_size" is the number of elements in each y_i.
    """
    if not isinstance(x, paddle.pir.Value):
        raise TypeError('x is not Value')

    # To compute the jacobian, treat x and y as one-dimensional vectors.
    y = _as_list(y)
    filted_ddx = [dxi for dxi in fetch_list if dxi is not None]
    exe = paddle.static.Executor(place)

    def run():
        res = exe.run(program, feeds, fetch_list=[filted_ddx, y])
        y_res = res[len(filted_ddx) :]
        return [yi.flatten() for yi in y_res]

    x_name = x.get_defining_op().attrs()['name']
    x_shape = x.shape
    x_size = _product(x_shape)
    if x.dtype in DTYPE_REQUIRES_GRAD:
        np_type = dtype_to_np_dtype(x.dtype)
        np_t = np.array(feeds[x_name]).astype(np_type)
        np_t = np_t.flatten()
        jacobian = [make_jacobian(x, _product(yi.shape), np_type) for yi in y]
    else:
        np_type = np.float32  # temporarily set to float32
        jacobian = [make_jacobian(x, _product(yi.shape), np_type) for yi in y]
        return jacobian

    for i in range(x_size):
        orig = np_t[i]
        x_pos = orig + delta
        np_t[i] = x_pos
        np_f = np_t.reshape(x_shape)
        feeds[x_name] = np_f
        y_pos = run()

        x_neg = orig - delta
        np_t[i] = x_neg
        np_f = np_t.reshape(x_shape)
        feeds[x_name] = np_f
        y_neg = run()

        np_t[i] = orig

        for j in range(len(y)):
            jacobian[j][i, :] = (y_pos[j] - y_neg[j]) / delta / 2.0

    return jacobian


def _compute_analytical_jacobian_pir(
    program, x, i, y, fetch_list, feeds, place
):
    """Computes the analytical Jacobian for dy/dx.

    Args:
        program (Program): a Program with forward pass.
        x (Variable|list[Variable]): a variable or list of variable
        i (int): the index of y.
        y (Variable): the target variable.
        fetch_list (list[Variable]): the variables to fetch.
        feeds (dict): the feed dict.
        place (base.CPUPlace or base.CUDAPlace): the device.

    Returns:
        A list of 2-D numpy array. The list length is len(x).
        Each 2-D numpy array represents the Jacobian for dy/dx_i.
        It has "xi_size" rows and "dy_size" columns
        where "x_size" is the number of elements in x_i and
        "dy_size" is the number of elements in y.
    """
    if not isinstance(x, (list, paddle.pir.Value)):
        raise TypeError('x is not Value or list of Value')

    np_type = dtype_to_np_dtype(y[i].dtype)
    exe = paddle.static.Executor(place)
    y_size = _product(y[i].shape)
    x = _as_list(x)
    jacobian = make_jacobian(x, y_size, np_type)

    # filter None in dx for DX/DY may be None in kernel
    # only fetch not None dx in exe.run

    filted = [(i, dxi) for i, dxi in enumerate(fetch_list) if dxi is not None]
    filted_idx, filted_dx = zip(*filted)

    # get the name in feeds of dyi
    name = f'dys_{i}'
    np_t = np.array(feeds[name]).astype(np_type)
    shape = np_t.shape
    np_t = np_t.flatten()
    for i in range(y_size):
        np_t[i] = 1
        np_f = np_t.reshape(shape)
        feeds[name] = np_f
        res = exe.run(program, feed=feeds, fetch_list=[filted_dx, y])
        dx_res = res[: len(filted_dx)]
        for j in range(len(filted_dx)):
            dx_idx = filted_idx[j]
            if dx_res[j] is not None:
                jacobian[dx_idx][:, i] = dx_res[j].flatten()
            else:
                jacobian[dx_idx][:, i] = np.zeros(
                    fetch_list[dx_idx].shape, dtype=np_type
                ).flatten()

        np_t[i] = 0
        np_f = np_t.reshape(shape)
        feeds[name] = np_f

    return jacobian


def grad_check(
    x,
    y,
    fetch_list=None,
    feeds=None,
    place=None,
    program=None,
    eps=1e-6,
    atol=1e-5,
    rtol=1e-3,
    raise_exception=True,
):
    """
    Check numerical and analytical gradients for dy/dx.
    Each Jacobian gradients is a 2-D array with shape [xi_size, yi_size].

    Args:
        x (Variable|list[Variable]): input variables to the program.
        y (Variable|list[Variable]): output variables to the program.
        x_init (numpy.array|list[numpy.array]|None): the init value for input x.
        place (base.CPUPlace or base.CUDAPlace): the device.
        program (Program|None): a Program with forward pass.
            If None, use base.default_main_program().
        eps (float): perturbation for finite differences.
        atol (float): absolute tolerance.
        rtol (float): relative tolerance.
        raise_exception (bool): whether to raise an exception if
            the check fails. Default is True.
    Returns:
        True if all differences satisfy numpy.allclose condition.
    """

    def fail_test(msg):
        if raise_exception:
            raise RuntimeError(msg)
        return False

    scope = base.executor.global_scope()

    if in_pir_mode():
        analytical = []
        for i in range(len(y)):
            name = f'dys_{i}'
            feeds.update(
                {
                    name: np.zeros(
                        y[i].shape, dtype=dtype_to_np_dtype(y[i].dtype)
                    )
                }
            )
        for i in range(len(y)):
            analytical.append(
                _compute_analytical_jacobian_pir(
                    program, x, i, y, fetch_list, feeds, place
                )
            )
        numerical = [
            _compute_numerical_jacobian_pir(
                program, xi, y, fetch_list, feeds, place, eps
            )
            for xi in x
        ]
    else:
        # [x_idx, y_idx]
        numerical = [
            _compute_numerical_jacobian(program, xi, y, place, scope, eps)
            for xi in x
        ]
        # [y_idx, x_idx]
        analytical = []
        for yi in y:
            prog = program.clone()

            clone_x = []
            clone_y = None
            for b in prog.blocks:
                if b.has_var(yi.name):
                    clone_y = b.var(yi.name)
                    break
            for xi in x:
                for b in prog.blocks:
                    if b.has_var(xi.name):
                        clone_x.append(b.var(xi.name))
                        break
            analytical.append(
                _compute_analytical_jacobian(
                    prog, clone_x, clone_y, place, scope
                )
            )
    for i, (x_idx, y_idx) in enumerate(
        product(*[range(len(x)), range(len(y))])
    ):
        a = analytical[y_idx][x_idx]
        n = numerical[x_idx][y_idx]
        if not np.allclose(a, n, rtol, atol):
            msg = (
                f'Jacobian mismatch for output {y_idx} in y '
                f'with respect to input {x_idx} in x on {place},\n'
                f'numerical:{n}\nanalytical:{a}\n'
            )
            return fail_test(msg)
    return True


def double_grad_check(
    x,
    y,
    x_init=None,
    y_grads=None,
    place=None,
    program=None,
    eps=1e-6,
    atol=1e-5,
    rtol=1e-3,
    raise_exception=True,
):
    """
    Check gradients of gradients. This function will append backward to the
    program before second order gradient check.

    Args:
        x (Variable|list[Variable]): input variables to the program.
        y (Variable|list[Variable]): output variables to the program.
        x_init (numpy.array|list[numpy.array]|None): the init value for input x.
        y_grads (numpy.array|list[numpy.array]|None): the gradients with respect to y.
        place (base.CPUPlace or base.CUDAPlace): the device.
        program (Program|None): a Program with forward pass.
            If None, use base.default_main_program().
        eps (float): perturbation for finite differences.
        atol (float): absolute tolerance.
        rtol (float): relative tolerance.
        raise_exception (bool): whether to raise an exception if
            the check fails. Default is True.
    Returns:
        True if all differences satisfy numpy.allclose condition.
    """
    # check input arguments
    x = _as_list(x)
    for v in x:
        v.stop_gradient = False
        v.persistable = True
    y = _as_list(y)
    for u in y:
        u.stop_gradient = False
        u.persistable = True

    x_init = _as_list(x_init)

    if in_pir_mode():
        program, (keys, values) = paddle.base.libpaddle.pir.clone_program(
            paddle.static.default_main_program()
        )
        op_map = ValueDict()
        for key, value in zip(keys, values):
            op_map[key] = value
        clone_x = []
        for xi in x:
            clone_x.append(op_map[xi])
        clone_y = []
        for yi in y:
            clone_y.append(op_map[yi])
        with paddle.static.program_guard(program):
            (
                grad_res,
                x,
                target_grads,
                fetch_list,
                feeds,
                ir_program,
            ) = get_pir_static_double_grad(
                clone_x, clone_y, x_init, y_grads, place
            )
        grad_check(
            x,
            target_grads,
            fetch_list,
            feeds,
            place,
            ir_program,
            eps,
            atol,
            rtol,
        )
    else:
        grad_res, x, target_grads, program = get_static_double_grad(
            x, y, x_init, y_grads, place
        )
        grad_check(x, target_grads, None, None, place, program, eps, atol, rtol)


# TODO(jiabin): We currently support only triple grad check here, extend this to support
# higher order differentiation later.


# check triple grad and two outputs of the triple Kernel
def triple_grad_check(
    x,
    y,
    x_init=None,
    y_grads=None,
    x_grads_grads=None,
    place=None,
    program=None,
    eps=1e-6,
    atol=1e-5,
    rtol=1e-3,
    raise_exception=True,
):
    """
    Check triple gradients. This function will append backward to the
    program before third order gradient check.

    Args:
        x (Variable|list[Variable]): input variables to the program.
        y (Variable|list[Variable]): output variables to the program.
        x_init (numpy.array|list[numpy.array]|None): the init value for input x.
        y_grads (numpy.array|list[numpy.array]|None): the gradients with respect to y.
        x_grads_grads (numpy.array|list[numpy.array]|None): the gradients with respect to your input.
        place (base.CPUPlace or base.CUDAPlace): the device.
        program (Program|None): a Program with forward pass.
            If None, use base.default_main_program().
        eps (float): perturbation for finite differences.
        atol (float): absolute tolerance.
        rtol (float): relative tolerance.
        raise_exception (bool): whether to raise an exception if
            the check fails. Default is True.
    Returns:
        True if all differences satisfy numpy.allclose condition.
    """
    # check input arguments
    x = _as_list(x)
    for v in x:
        v.stop_gradient = False
        v.persistable = True
    y = _as_list(y)
    for u in y:
        u.stop_gradient = False
        u.persistable = True

    x_init = _as_list(x_init)

    # x <=> [x, dout, ddx]
    if in_pir_mode():
        program, (keys, values) = paddle.base.libpaddle.pir.clone_program(
            paddle.static.default_main_program()
        )
        op_map = ValueDict()
        for key, value in zip(keys, values):
            op_map[key] = value
        clone_x = []
        for xi in x:
            clone_x.append(op_map[xi])
        clone_y = []
        for yi in y:
            clone_y.append(op_map[yi])
        with paddle.static.program_guard(program):
            (
                grad_res,
                x,
                target_grads,
                fetch_list,
                feeds,
                ir_program,
            ) = get_pir_static_triple_grad(
                clone_x, clone_y, x_init, y_grads, place, program
            )
        grad_check(
            x,
            target_grads,
            fetch_list,
            feeds,
            place,
            ir_program,
            eps,
            atol,
            rtol,
        )
    else:
        grad_res, x, target_grads, program = get_static_triple_grad(
            x, y, x_init, y_grads, place
        )
        grad_check(x, target_grads, None, None, place, program, eps, atol, rtol)


def get_static_double_grad(
    x, y, x_init=None, dy_init=None, place=None, program=None
):
    """
    Get Double Grad result of static graph.

    Args:
        x (Variable|list[Variable]): input variables to the program.
        y (Variable|list[Variable]): output variables to the program.
        x_init (numpy.array|list[numpy.array]|None): the init value for input x.
        dy_init (numpy.array|list[numpy.array]|None): the init value for output y.
        place (base.CPUPlace or base.CUDAPlace): the device.
        program (Program|None): a Program with forward pass.
            If None, use base.default_main_program().
    Returns:
        A list of numpy array that stores second derivative result calculated by static graph.
    """

    if program is None:
        program = paddle.static.default_main_program()
    scope = base.executor.global_scope()
    if dy_init is None:
        y_grads = []
        y_grads_init = []
        for yi in y:
            dyi_name = _append_grad_suffix_(yi.name)
            np_type = dtype_to_np_dtype(yi.dtype)
            dy = program.global_block().create_var(
                name=dyi_name, shape=yi.shape, dtype=np_type, persistable=True
            )
            dy.stop_gradient = False
            v = np.random.random(size=yi.shape).astype(np_type)
            set_var_in_scope(scope, place, dyi_name, v)
            y_grads.append(dy)
            y_grads_init.append(v)
    else:
        y_grads = []
        y_grads_init = dy_init
        for i in range(len(y)):
            yi = y[i]
            dyi_name = _append_grad_suffix_(yi.name)
            np_type = dtype_to_np_dtype(yi.dtype)
            dy = program.global_block().create_var(
                name=dyi_name, shape=yi.shape, dtype=np_type, persistable=True
            )
            dy.stop_gradient = False
            set_var_in_scope(scope, place, dyi_name, dy_init[i])
            y_grads.append(dy)

    # append first order grads
    dx = base.gradients(y, x, y_grads)

    # y_grads are the input of first-order backward,
    # so, they are also the input of second-order backward.
    x += y_grads
    x_init += y_grads_init

    # filter None in dx for DX/DY may be None in kernel
    filted_dx = [dxi for dxi in dx if dxi is not None]
    y = filted_dx

    # check input arguments
    x = _as_list(x)
    y = _as_list(y)

    for v in x:
        v.stop_gradient = False
        v.persistable = True
    for u in y:
        u.stop_gradient = False
        u.persistable = True
    if place is None:
        place = base.CPUPlace()

    # init variable in startup program
    exe = paddle.static.Executor(place)
    exe.run(paddle.static.default_startup_program())

    x_init = _as_list(x_init)
    # init inputs if x_init is not None
    if x_init:
        if len(x_init) != len(x):
            raise ValueError(
                f'len(x_init) (={len(x_init)}) is not the same'
                f' as len(x) (={len(x)})'
            )
        # init variable in main program
        for var, arr in zip(x, x_init):
            assert var.shape == arr.shape
        feeds = {k.name: v for k, v in zip(x, x_init)}

    dys = []
    for yi in y:
        np_type = dtype_to_np_dtype(yi.dtype)
        dy_name = _append_grad_suffix_(yi.name)
        # create dy Variable in Program
        dy = program.global_block().create_var(
            name=dy_name, shape=yi.shape, dtype=np_type, persistable=True
        )
        # init dy tensor in scope
        value = np.ones(yi.shape, dtype=np_type)
        dy_t = set_var_in_scope(scope, place, dy_name, value)
        dys.append(dy)

    # append second order backward
    ddx = base.gradients(y, x, dys)
    exe = paddle.static.Executor(place)

    # filter None in dx for DX/DY may be None in kernel
    # only fetch not None dx in exe.run
    filted = [(i, dxi) for i, dxi in enumerate(ddx) if dxi is not None]
    filted_idx, filted_ddx = zip(*filted)
    ddx_res = exe.run(program, feed=feeds, scope=scope, fetch_list=filted_ddx)

    return ddx_res, x, filted_dx, program


def get_pir_static_double_grad(
    x, y, x_init=None, dy_init=None, place=None, program=None
):
    """
    Get Double Grad result of static graph.

    Args:
        x (Variable|list[Variable]): input variables to the program.
        y (Variable|list[Variable]): output variables to the program.
        x_init (numpy.array|list[numpy.array]|None): the init value for input x.
        dy_init (numpy.array|list[numpy.array]|None): the init value for output y.
        place (base.CPUPlace or base.CUDAPlace): the device.
        program (Program|None): a Program with forward pass.
            If None, use base.default_main_program().
    Returns:
        A list of numpy array that stores second derivative result calculated by static graph.
    """
    if program is None:
        program = paddle.static.default_main_program()
    exe = paddle.static.Executor(place)
    exe.run(paddle.static.default_startup_program())
    if dy_init is None:
        y_grads = []
        y_grads_init = []
        for i in range(len(y)):
            yi = y[i]
            yi.persistable = True
            np_type = dtype_to_np_dtype(yi.dtype)
            dy = paddle.static.data(
                name=f'Dgrad_{i}',
                shape=yi.shape,
                dtype=np_type,
            )
            dy.stop_gradient = False
            dy.persistable = True
            v = np.random.random(size=yi.shape).astype(np_type)
            y_grads.append(dy)
            y_grads_init.append(v)
    else:
        y_grads = []
        y_grads_init = dy_init
        for i in range(len(y)):
            yi = y[i]
            yi.persistable = True
            np_type = dtype_to_np_dtype(yi.dtype)
            dy = paddle.static.data(
                name=f'Dgrad_{i}',
                shape=yi.shape,
                dtype=np_type,
            )
            dy.stop_gradient = False
            dy.persistable = True
            y_grads.append(dy)

    # append first order grads
    dx = base.gradients(y, x, y_grads)
    # y_grads are the input of first-order backward,
    # so, they are also the input of second-order backward.
    x += y_grads
    x_init += y_grads_init

    # filter None in dx for DX/DY may be None in kernel
    filted_dx = [dxi for dxi in dx if dxi is not None]
    y = filted_dx

    # check input arguments
    x = _as_list(x)
    y = _as_list(y)

    for v in x:
        v.stop_gradient = False
        v.persistable = True
    for u in y:
        u.stop_gradient = False
        u.persistable = True

    if place is None:
        place = base.CPUPlace()

    feeds = {}
    x_init = _as_list(x_init)
    # init inputs if x_init is not None
    if x_init:
        if len(x_init) != len(x):
            raise ValueError(
                f'len(x_init) (={len(x_init)}) is not the same'
                f' as len(x) (={len(x)})'
            )
        # init variable in main program
        for var, arr in zip(x, x_init):
            assert tuple(var.shape) == tuple(arr.shape)

        for i in range(len(x)):
            feeds.update({x[i].get_defining_op().attrs()['name']: x_init[i]})

    dys = []
    for i in range(len(y)):
        yi = y[i]
        np_type = dtype_to_np_dtype(yi.dtype)
        dy = paddle.static.data(
            name=f'dys_{i}',
            shape=yi.shape,
            dtype=np_type,
        )
        value = np.ones(yi.shape, dtype=np_type)
        feeds.update({f'dys_{i}': value})
        dys.append(dy)

    # append second order backward
    ddx = base.gradients(y, x, dys)

    # filter None in dx for DX/DY may be None in kernel
    # only fetch not None dx in exe.run
    filted = [(i, dxi) for i, dxi in enumerate(ddx) if dxi is not None]
    filted_idx, filted_ddx = zip(*filted)
    ddx_res = exe.run(
        program=program, feed=feeds, fetch_list=[filted_ddx, filted_dx]
    )
    res = ddx_res[: len(filted_ddx)]

    return res, x, y, ddx, feeds, program


def get_eager_double_grad(
    func, x_init=None, dy_init=None, place=None, return_mid_result=False
):
    """
    Get Double Grad result of dygraph.

    Args:
        func: A wrapped dygraph function that its logic is equal to static program
        x_init (numpy.array|list[numpy.array]|None): the init value for input x.
        dy_init (numpy.array|list[numpy.array]|None): the init value for gradient of output.
        place (base.CPUPlace or base.CUDAPlace): the device.
        return_mid_result (bool): A flag that controls the return content.
    Returns:
        If 'return_mid_result' set True.
        the second order derivative and the inputs of second order derivative's calculation
        will be returned for higher order derivative's calculation.
        If 'return_mid_result' set False.
        A list of numpy array that stores second derivative result calculated by dygraph.
    """
    if isinstance(place, base.CPUPlace):
        paddle.set_device("cpu")
    if isinstance(place, base.CUDAPlace):
        paddle.set_device("gpu")
    inputs = []
    dys = []
    for x in x_init:
        input_tensor = paddle.to_tensor(x)
        input_tensor.stop_gradient = False
        inputs.append(input_tensor)
    for dy in dy_init:
        dy_tensor = paddle.to_tensor(dy)
        dy_tensor.stop_gradient = False
        dys.append(dy_tensor)
    # calculate first derivative
    outputs = func(inputs)
    d_inputs = paddle.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=dys,
        create_graph=True,
        allow_unused=True,
    )
    d_inputs = [d_input for d_input in d_inputs if d_input is not None]

    # calculate second derivative
    inputs = inputs + dys
    ddys = []
    if return_mid_result:
        create_graph = True
    else:
        create_graph = False

    for d_input in d_inputs:
        d_input.stop_gradient = False
        ddy = paddle.ones(shape=d_input.shape, dtype=d_input.dtype)
        ddy.stop_gradient = False
        ddys.append(ddy)

    dd_inputs = paddle.grad(
        outputs=d_inputs,
        inputs=inputs,
        grad_outputs=ddys,
        create_graph=create_graph,
        allow_unused=True,
    )

    if return_mid_result:
        return [
            dd_input for dd_input in dd_inputs if dd_input is not None
        ], inputs + ddys
    else:
        return [
            dd_input.numpy() for dd_input in dd_inputs if dd_input is not None
        ]


def double_grad_check_for_dygraph(
    func,
    x,
    y,
    x_init=None,
    place=None,
    program=None,
    atol=1e-5,
    rtol=1e-3,
    raise_exception=True,
):
    """
    Check second order gradients of dygraph. This function will compare the
    second order gradients of dygraph and second order gradients of static graph
    to validate dygraph's correctness

    Args:
        func: A wrapped dygraph function that its logic is equal to static program
        x (Variable|list[Variable]): input variables to the program.
        y (Variable|list[Variable]): output variables to the program.
        x_init (numpy.array|list[numpy.array]|None): the init value for input x.
        place (base.CPUPlace or base.CUDAPlace): the device.
        atol (float): absolute tolerance.
        rtol (float): relative tolerance.
        raise_exception (bool): whether to raise an exception if
            the check fails. Default is True.
    """

    def fail_test(msg):
        if raise_exception:
            raise RuntimeError(msg)
        return False

    # check input arguments
    for v in x:
        v.stop_gradient = False
        v.persistable = True
    y = _as_list(y)
    for u in y:
        u.stop_gradient = False
        u.persistable = True
    y_grads_init = []
    for yi in y:
        np_type = dtype_to_np_dtype(yi.dtype)
        v = np.random.random(size=yi.shape).astype(np_type)
        y_grads_init.append(v)

    x_init = _as_list(x_init)

    paddle.disable_static()
    eager_double_grad = get_eager_double_grad(func, x_init, y_grads_init, place)
    paddle.enable_static()

    if in_pir_mode():
        static_double_grad, _, _, _, _, _ = get_pir_static_double_grad(
            x, y, x_init, y_grads_init, place
        )
    else:
        (
            static_double_grad,
            _,
            _,
            _,
        ) = get_static_double_grad(x, y, x_init, y_grads_init, place)

    if len(static_double_grad) != len(eager_double_grad):
        msg = (
            "The output grad tensor's number of static graph is different with dygraph, "
            "please check the python api unit test used."
        )
        raise RuntimeError(msg)

    for i in range(len(static_double_grad)):
        if not np.allclose(
            static_double_grad[i], eager_double_grad[i], rtol, atol
        ):
            msg = (
                'Check eager double result fail. Mismatch between static_graph double grad '
                f'and eager double grad on {place!s}, the output double grad tensor\'s index is : {i} \n'
                f'static:{static_double_grad[i]}\n eager:{eager_double_grad[i]}\n'
            )
            return fail_test(msg)


def get_static_triple_grad(
    x, y, x_init=None, dy_init=None, place=None, program=None
):
    """
    Get Triple Grad result of static graph.

    Args:
        x (Variable|list[Variable]): input variables to the program.
        y (Variable|list[Variable]): output variables to the program.
        x_init (numpy.array|list[numpy.array]|None): the init value for input x.
        dy_init (numpy.array|list[numpy.array]|None): the init value for output y.
        place (base.CPUPlace or base.CUDAPlace): the device.
        program (Program|None): a Program with forward pass.
            If None, use base.default_main_program().
    Returns:
        A list of numpy array that stores third derivative result calculated by static graph.
    """
    if program is None:
        program = paddle.static.default_main_program()
    scope = base.executor.global_scope()
    if dy_init is None:
        y_grads = []
        y_grads_init = []
        for yi in y:
            dyi_name = _append_grad_suffix_(yi.name)
            np_type = dtype_to_np_dtype(yi.dtype)
            dy = program.global_block().create_var(
                name=dyi_name, shape=yi.shape, dtype=np_type, persistable=True
            )
            dy.stop_gradient = False
            v = np.random.random(size=yi.shape).astype(np_type)
            set_var_in_scope(scope, place, dyi_name, v)
            y_grads.append(dy)
            y_grads_init.append(v)
    else:
        y_grads = []
        y_grads_init = dy_init
        for i in range(len(y)):
            yi = y[i]
            dyi_name = _append_grad_suffix_(yi.name)
            np_type = dtype_to_np_dtype(yi.dtype)
            dy = program.global_block().create_var(
                name=dyi_name, shape=yi.shape, dtype=np_type, persistable=True
            )
            dy.stop_gradient = False
            set_var_in_scope(scope, place, dyi_name, dy_init[i])
            y_grads.append(dy)

    # append first order grads
    dx = base.gradients(y, x, y_grads)

    # y_grads are the input of first-order backward,
    # so, they are also the input of second-order backward.
    x += y_grads
    x_init += y_grads_init
    y = dx

    x_grads_grads_init = []
    for dxi in dx:
        np_type = dtype_to_np_dtype(dxi.dtype)
        value = np.ones(dxi.shape, dtype=np_type)
        x_grads_grads_init.append(value)

    return get_static_double_grad(
        x, y, x_init, dy_init=x_grads_grads_init, place=place, program=program
    )


def get_pir_static_triple_grad(
    x, y, x_init=None, dy_init=None, place=None, program=None
):
    """
    Get Triple Grad result of static graph.

    Args:
        x (Variable|list[Variable]): input variables to the program.
        y (Variable|list[Variable]): output variables to the program.
        x_init (numpy.array|list[numpy.array]|None): the init value for input x.
        dy_init (numpy.array|list[numpy.array]|None): the init value for output y.
        place (base.CPUPlace or base.CUDAPlace): the device.
        program (Program|None): a Program with forward pass.
            If None, use base.default_main_program().
    Returns:
        A list of numpy array that stores third derivative result calculated by static graph.
    """
    if program is None:
        program = paddle.static.default_main_program()
    if dy_init is None:
        y_grads = []
        y_grads_init = []
        for i in range(len(y)):
            yi = y[i]
            yi.persistable = True
            np_type = dtype_to_np_dtype(yi.dtype)
            dy = paddle.static.data(
                name=f'Tgrad_{i}',
                shape=yi.shape,
                dtype=np_type,
            )
            dy.stop_gradient = False
            dy.persistable = True
            v = np.random.random(size=yi.shape).astype(np_type)
            y_grads.append(dy)
            y_grads_init.append(v)
    else:
        y_grads = []
        y_grads_init = dy_init
        for i in range(len(y)):
            yi = y[i]
            yi.persistable = True
            np_type = dtype_to_np_dtype(yi.dtype)
            dy = paddle.static.data(
                name=f'Tgrad_{i}',
                shape=yi.shape,
                dtype=np_type,
            )
            dy.stop_gradient = False
            dy.persistable = True
            y_grads.append(dy)

    # append first order grads
    dx = base.gradients(y, x, y_grads)

    # y_grads are the input of first-order backward,
    # so, they are also the input of second-order backward.
    x += y_grads
    x_init += y_grads_init
    y = dx

    x_grads_grads_init = []
    for dxi in dx:
        np_type = dtype_to_np_dtype(dxi.dtype)
        value = np.ones(dxi.shape, dtype=np_type)
        x_grads_grads_init.append(value)

    return get_pir_static_double_grad(
        x,
        y,
        x_init,
        dy_init=x_grads_grads_init,
        place=place,
        program=program,
    )


def get_eager_triple_grad(
    func, x_init=None, dy_init=None, place=None, return_mid_result=False
):
    """
    Get triple Grad result of dygraph.

    Args:
        func: A wrapped dygraph function that its logic is equal to static program
        x_init (numpy.array|list[numpy.array]|None): the init value for input x.
        dy_init (numpy.array|list[numpy.array]|None): the init value for gradient of output.
        place (base.CPUPlace or base.CUDAPlace): the device.
        return_mid_result (list[Tensor], list[Tensor]): If set True, the
    Returns:
        A list of numpy array that stores second derivative result calculated by dygraph
    """
    dd_y, dd_x = get_eager_double_grad(
        func, x_init, dy_init, place, return_mid_result=True
    )

    # calculate third derivative
    dddys = []
    for dd_yi in dd_y:
        dd_yi.stop_gradient = False
        dddy = paddle.ones(shape=dd_yi.shape, dtype=dd_yi.dtype)
        dddy.stop_gradient = False
        dddys.append(dddy)
    ddd_inputs = paddle.grad(
        outputs=dd_y, inputs=dd_x, grad_outputs=dddys, allow_unused=True
    )
    return [
        ddd_input.numpy() for ddd_input in ddd_inputs if ddd_input is not None
    ]


def triple_grad_check_for_dygraph(
    func,
    x,
    y,
    x_init=None,
    place=None,
    program=None,
    atol=1e-5,
    rtol=1e-3,
    raise_exception=True,
):
    """
    Check third order gradients of dygraph. This function will compare the
    third order gradients of dygraph and third order gradients of static graph
    to validate dygraph's correctness

    Args:
        func: A wrapped dygraph function that its logic is equal to static program
        x (Variable|list[Variable]): input variables to the program.
        y (Variable|list[Variable]): output variables to the program.
        x_init (numpy.array|list[numpy.array]|None): the init value for input x.
        place (base.CPUPlace or base.CUDAPlace): the device.
        atol (float): absolute tolerance.
        rtol (float): relative tolerance.
        raise_exception (bool): whether to raise an exception if
            the check fails. Default is True.
    """

    def fail_test(msg):
        if raise_exception:
            raise RuntimeError(msg)
        return False

    # check input arguments
    x = _as_list(x)
    for v in x:
        v.stop_gradient = False
        v.persistable = True
    y = _as_list(y)
    for u in y:
        u.stop_gradient = False
        u.persistable = True
    y_grads_init = []
    for yi in y:
        np_type = dtype_to_np_dtype(yi.dtype)
        v = np.random.random(size=yi.shape).astype(np_type)
        y_grads_init.append(v)

    x_init = _as_list(x_init)

    paddle.disable_static()
    eager_triple_grad = get_eager_triple_grad(func, x_init, y_grads_init, place)
    paddle.enable_static()

    if in_pir_mode():
        static_triple_grad, _, _, _, _, _ = get_pir_static_triple_grad(
            x, y, x_init, y_grads_init, place
        )
    else:
        (
            static_triple_grad,
            _,
            _,
            _,
        ) = get_static_triple_grad(x, y, x_init, y_grads_init, place)

    if len(static_triple_grad) != len(eager_triple_grad):
        msg = (
            "The output grad tensor's number of static graph is different with dygraph, "
            "please check the python api unit test used."
        )
        raise RuntimeError(msg)

    for i in range(len(static_triple_grad)):
        if not np.allclose(
            static_triple_grad[i], eager_triple_grad[i], rtol, atol
        ):
            msg = (
                'Check eager double result fail. Mismatch between static_graph double grad '
                f'and eager double grad on {place!s}, the output double grad tensor\'s index is : {i} \n'
                f'static:{static_triple_grad[i]}\n eager:{eager_triple_grad[i]}\n'
            )
            return fail_test(msg)
