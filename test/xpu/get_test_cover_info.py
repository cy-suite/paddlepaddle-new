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

import contextlib
import fcntl
import inspect
import os
import unittest

import numpy as np

import paddle
from paddle.base import core
from paddle.base.core import VarDesc
from paddle.base.libpaddle import DataType

type_dict_paddle_to_str = {
    VarDesc.VarType.BOOL: 'bool',
    VarDesc.VarType.UINT8: 'uint8',
    VarDesc.VarType.INT8: 'int8',
    VarDesc.VarType.INT16: 'int16',
    VarDesc.VarType.INT32: 'int32',
    VarDesc.VarType.INT64: 'int64',
    VarDesc.VarType.FP16: 'float16',
    VarDesc.VarType.BF16: 'bfloat16',
    VarDesc.VarType.FP32: 'float32',
    VarDesc.VarType.FP64: 'float64',
    VarDesc.VarType.COMPLEX128: 'complex128',
    VarDesc.VarType.COMPLEX64: 'complex64',
    DataType.BOOL: 'bool',
    DataType.UINT8: 'uint8',
    DataType.INT8: 'int8',
    DataType.INT16: 'int16',
    DataType.INT32: 'int32',
    DataType.INT64: 'int64',
    DataType.FLOAT16: 'float16',
    DataType.BFLOAT16: 'bfloat16',
    DataType.FLOAT32: 'float32',
    DataType.FLOAT64: 'float64',
    DataType.COMPLEX128: 'complex128',
    DataType.COMPLEX64: 'complex64',
}

type_dict_paddle_to_numpy = {
    VarDesc.VarType.BOOL: np.bool_,
    VarDesc.VarType.UINT8: np.uint8,
    VarDesc.VarType.INT8: np.int8,
    VarDesc.VarType.INT16: np.int16,
    VarDesc.VarType.INT32: np.int32,
    VarDesc.VarType.INT64: np.int64,
    VarDesc.VarType.FP16: np.float16,
    VarDesc.VarType.BF16: np.uint16,
    VarDesc.VarType.FP32: np.float32,
    VarDesc.VarType.FP64: np.float64,
    VarDesc.VarType.COMPLEX128: np.complex128,
    VarDesc.VarType.COMPLEX64: np.complex64,
    DataType.BOOL: np.bool_,
    DataType.UINT8: np.uint8,
    DataType.INT8: np.int8,
    DataType.INT16: np.int16,
    DataType.INT32: np.int32,
    DataType.INT64: np.int64,
    DataType.FLOAT16: np.float16,
    DataType.BFLOAT16: np.uint16,
    DataType.FLOAT32: np.float32,
    DataType.FLOAT64: np.float64,
    DataType.COMPLEX128: np.complex128,
    DataType.COMPLEX64: np.complex64,
}

type_dict_str_to_paddle = {
    'uint8': VarDesc.VarType.UINT8,
    'int8': VarDesc.VarType.INT8,
    'int16': VarDesc.VarType.INT16,
    'int32': VarDesc.VarType.INT32,
    'int64': VarDesc.VarType.INT64,
    'bfloat16': VarDesc.VarType.BF16,
    'float16': VarDesc.VarType.FP16,
    'float32': VarDesc.VarType.FP32,
    'float64': VarDesc.VarType.FP64,
    'bool': VarDesc.VarType.BOOL,
    'complex64': VarDesc.VarType.COMPLEX64,
    'complex128': VarDesc.VarType.COMPLEX128,
}

type_dict_str_to_numpy = {
    'uint8': np.uint8,
    'int8': np.int8,
    'int16': np.int16,
    'int32': np.int32,
    'int64': np.int64,
    'bfloat16': np.uint16,
    'float16': np.float16,
    'float32': np.float32,
    'float64': np.float64,
    'bool': np.bool_,
    'complex64': np.complex64,
    'complex128': np.complex128,
}

xpu_test_op_white_list = []
xpu_test_device_type_white_list = ['xpu1_float64']
xpu_test_op_type_white_list = [
    "grad_add_float32",  # no api for grad_add, skip
    "lamb_float16",
    "lars_momentum_float32",
    "resnet_unit",
    "resnet_unit_grad",
    "c_embedding_float32",  # unittests of collective ops do not using xpu testing framework
    "c_sync_comm_stream_float32",
    "c_sync_calc_stream_float32",
    "reshape2_bool",
    "reshape2_grad_bool",
]
xpu_test_device_op_white_list = []
xpu_test_device_op_type_white_list = []


class XPUOpTestWrapper:
    def create_classes(self):
        base_class = None
        classes = []
        return base_class, classes


def get_version_str(xpu_version):
    if xpu_version == core.XPUVersion.XPU1:
        return "xpu1"
    if xpu_version == core.XPUVersion.XPU2:
        return "xpu2"
    if xpu_version == core.XPUVersion.XPU3:
        return "xpu3"
    raise ValueError("unknown xpu version, not 1, 2, or 3")


def get_op_white_list():
    op_white_list = xpu_test_op_white_list
    if os.getenv('XPU_TEST_OP_WHITE_LIST') is not None:
        op_white_list.extend(
            os.getenv('XPU_TEST_OP_WHITE_LIST').strip().split(',')
        )
    return list(set(op_white_list))


def get_type_white_list():
    xpu_version = core.get_xpu_device_version(0)
    version_str = get_version_str(xpu_version)
    xpu1_type_white_list = []
    xpu2_type_white_list = []
    xpu3_type_white_list = []
    for device_type in xpu_test_device_type_white_list:
        device, t_type = device_type.split("_")
        if "xpu1" == device:
            xpu1_type_white_list.append(t_type)
        elif "xpu2" == device:
            xpu2_type_white_list.append(t_type)
        elif "xpu3" == device:
            xpu3_type_white_list.append(t_type)
    if version_str == "xpu1":
        type_white_list = xpu1_type_white_list
    elif version_str == "xpu2":
        type_white_list = xpu2_type_white_list
    elif version_str == "xpu3":
        type_white_list = xpu3_type_white_list

    if os.getenv('XPU_TEST_TYPE_WHITE_LIST') is not None:
        type_white_list.extend(
            os.getenv('XPU_TEST_TYPE_WHITE_LIST').strip().split(',')
        )
    return list(set(type_white_list))


def get_op_type_white_list():
    op_type_white_list = xpu_test_op_type_white_list
    if os.getenv('XPU_TEST_OP_TYPE_WHITE_LIST') is not None:
        op_type_white_list.extend(
            os.getenv('XPU_TEST_OP_TYPE_WHITE_LIST').strip().split(',')
        )
    return list(set(op_type_white_list))


def get_device_op_white_list():
    device_op_white_list = xpu_test_device_op_white_list
    if os.getenv('XPU_TEST_DEVICE_OP_WHITE_LIST') is not None:
        device_op_white_list.extend(
            os.getenv('XPU_TEST_DEVICE_OP_WHITE_LIST').strip().split(',')
        )
    return list(set(device_op_white_list))


def get_device_op_type_white_list():
    device_op_type_white_list = xpu_test_device_op_type_white_list
    if os.getenv('XPU_TEST_DEVICE_OP_TYPE_WHITE_LIST') is not None:
        device_op_type_white_list.extend(
            os.getenv('XPU_TEST_DEVICE_OP_TYPE_WHITE_LIST').strip().split(',')
        )
    return list(set(device_op_type_white_list))


def make_xpu_op_list(xpu_version):
    ops = []
    raw_op_list = core.get_xpu_device_op_list(xpu_version)
    version_str = get_version_str(xpu_version)
    op_white_list = get_op_white_list()
    type_white_list = get_type_white_list()
    op_type_white_list = get_op_type_white_list()
    device_op_white_list = get_device_op_white_list()
    device_op_type_white_list = get_device_op_type_white_list()
    print('op_white_list:', op_white_list)
    print('type_white_list:', type_white_list)
    print('op_type_white_list:', op_type_white_list)
    print('device_op_white_list:', device_op_white_list)
    print('device_op_type_white_list:', device_op_type_white_list)

    for op_name, type_list in raw_op_list.items():
        device_op_name = version_str + '_' + op_name
        if op_name in op_white_list or device_op_name in device_op_white_list:
            continue
        for op_type in type_list:
            if op_type == paddle.bfloat16:
                op_type = paddle.bfloat16

            if (
                type_dict_paddle_to_str[op_type] in type_white_list
                or op_type not in type_dict_paddle_to_str.keys()
            ):
                continue

            device_op_type_name = (
                device_op_name + '_' + type_dict_paddle_to_str[op_type]
            )
            if device_op_type_name in device_op_type_white_list:
                continue

            op_type_name = op_name + '_' + type_dict_paddle_to_str[op_type]
            if op_type_name in op_type_white_list:
                continue

            ops.append(op_type_name)
    return ops


def get_xpu_op_support_types(op_name, dev_id=0):
    xpu_version = core.get_xpu_device_version(dev_id)
    support_type_list = core.get_xpu_device_op_support_types(
        op_name, xpu_version
    )
    support_type_str_list = []
    for stype in support_type_list:
        if stype == paddle.bfloat16:
            support_type_str_list.append(
                type_dict_paddle_to_str[paddle.bfloat16]
            )
        else:
            support_type_str_list.append(type_dict_paddle_to_str[stype])
    ops = make_xpu_op_list(xpu_version)
    support_types = []
    for stype in support_type_str_list:
        op_name_type = op_name + "_" + stype
        if op_name_type in ops:
            support_types.append(stype)
    if len(support_types) == 0:
        print("WARNING: support_types is EMPTY for op", op_name)
    return support_types


def record_op_test(op_name, test_type):
    dirname = os.getenv('XPU_OP_LIST_DIR')
    filename = 'xpu_op_test'
    if dirname is not None:
        filename = os.path.join(dirname, filename)
    with open(filename, 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(op_name + '_' + test_type + '\n')


def is_empty_grad_op_type(xpu_version, op, test_type):
    xpu_op_list = core.get_xpu_device_op_list(xpu_version)
    grad_op = op + '_grad'
    if grad_op not in xpu_op_list.keys():
        return True

    grad_op_types = xpu_op_list[grad_op]
    paddle_test_type = type_dict_str_to_paddle[test_type]
    if paddle_test_type not in grad_op_types:
        return True

    return False


def create_test_class(
    func_globals,
    test_class,
    test_type,
    test_grad=True,
    ignore_device_version=[],
    test_device_version=[],
):
    xpu_version = core.get_xpu_device_version(0)
    if xpu_version in ignore_device_version:
        return

    if len(test_device_version) != 0 and xpu_version not in test_device_version:
        return

    test_class_obj = test_class()
    register_classes = inspect.getmembers(test_class_obj, inspect.isclass)
    op_name = test_class_obj.op_name
    no_grad = is_empty_grad_op_type(xpu_version, op_name, test_type)

    for test_class in register_classes:
        if test_class[0] == '__class__':
            continue
        class_obj = test_class[1]
        cls_name = f"{test_class[0]}_{test_type}"
        func_globals[cls_name] = type(
            cls_name,
            (class_obj,),
            {
                'in_type': type_dict_str_to_numpy[test_type],
                'in_type_str': test_type,
                'op_type_need_check_grad': True,
            },
        )

    if (
        hasattr(test_class_obj, 'use_dynamic_create_class')
        and test_class_obj.use_dynamic_create_class
    ):
        base_class, dynamic_classes = test_class_obj.dynamic_create_class()
        for dy_class in dynamic_classes:
            cls_name = f"{dy_class[0]}_{test_type}"
            attr_dict = dy_class[1]
            attr_dict['in_type'] = type_dict_str_to_numpy[test_type]
            attr_dict['in_type_str'] = test_type
            attr_dict['op_type_need_check_grad'] = True
            func_globals[cls_name] = type(cls_name, (base_class,), attr_dict)

    record_op_test(op_name, test_type)
    if not no_grad:
        record_op_test(op_name + '_grad', test_type)


def check_run_big_shape_test():
    def wrapper(cls):
        run_big_shape_test_flag = os.environ.get("FLAGS_xpu_big_shape_test")
        return unittest.skipIf(
            not (run_big_shape_test_flag and run_big_shape_test_flag == "true"),
            "skip big shape test.",
        )(cls)

    return wrapper


@contextlib.contextmanager
def xpu_matmul_quant_type_guard(dtype):
    # only fp32 is supported now
    assert dtype == "float"
    env_name = "XPU_PADDLE_FC_FLOAT"
    origin_env = os.getenv(env_name)
    os.environ[env_name] = "1"
    yield
    if origin_env is not None:
        os.environ[env_name] = origin_env
    else:
        del os.environ[env_name]


def get_test_cover_info():
    xpu_version = core.get_xpu_device_version(0)
    version_str = get_version_str(xpu_version)
    xpu_op_list = make_xpu_op_list(xpu_version)
    xpu_op_covered = []

    dirname = os.getenv('XPU_OP_LIST_DIR')
    filename = 'xpu_op_test'
    if dirname is not None:
        filename = os.path.join(dirname, filename)
    if os.path.exists(filename) and os.path.isfile(filename):
        with open(filename) as f:
            for line in f:
                test_op_name = line.strip()
                if test_op_name in xpu_op_list:
                    xpu_op_covered.append(test_op_name)
    diff_list = list(set(xpu_op_list).difference(set(xpu_op_covered)))
    total_len = len(set(xpu_op_list))
    covered_len = len(set(xpu_op_covered))
    print(f'{version_str} test: {covered_len}/{total_len}')
    if len(diff_list) != 0:
        print(
            "These ops need to be tested on {}! ops:{}".format(
                version_str, ','.join(diff_list)
            )
        )


if __name__ == '__main__':
    get_test_cover_info()
