# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

# flags for instructions
from enum import Enum


class CONVERT_VALUE_FLAG(Enum):
    CV_STR = 1
    CV_REPR = 2
    CV_ASCII = 3


class MAKE_FUNCTION_FLAG:
    MF_HAS_CLOSURE = 0x08
    MF_HAS_ANNOTATION = 0x04
    MF_HAS_KWDEFAULTS = 0x02
    MF_HAS_DEFAULTS = 0x01


class CALL_FUNCTION_EX_FLAG:
    CFE_HAS_KWARGS = 0x01


# see https://github.com/python/cpython/blob/3.12/Python/intrinsics.c#L211-L225
class IntrinsicsUnaryFunctions(Enum):
    INTRINSIC_1_INVALID = 0
    INTRINSIC_PRINT = 1  # no support, only non-interactive mode
    INTRINSIC_IMPORT_STAR = 2  # no support, `from module import *`
    INTRINSIC_STOPITERATION_ERROR = 3  # no support, generator or coroutine
    INTRINSIC_ASYNC_GEN_WRAP = 4  # no support, async
    INTRINSIC_UNARY_POSITIVE = 5
    INTRINSIC_LIST_TO_TUPLE = 6
    INTRINSIC_TYPEVAR = 7  # no support, PEP 695
    INTRINSIC_PARAMSPEC = 8  # no support, PEP 695
    INTRINSIC_TYPEVARTUPLE = 9  # no support, PEP 695
    INTRINSIC_SUBSCRIPT_GENERIC = 10  # no support, PEP 695
    INTRINSIC_TYPEALIAS = 11  # no support, PEP 695
