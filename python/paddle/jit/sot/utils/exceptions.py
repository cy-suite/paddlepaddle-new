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
from __future__ import annotations

import traceback

from .info_collector import BreakGraphReasonInfo


class BreakGraphReasonBase:
    """Base class for representing reasons why graph execution was interrupted.

    Attributes:
        reason_str (str): Description of the break reason
        file_path (str): Path to the file where break occurred
        line_number (int): Line number where break occurred
    """

    def __init__(
        self,
        reason_str,
        file_path="",
        line_number=-1,
    ):
        self.reason_str = reason_str
        self.file_path = file_path
        self.line_number = line_number

    def __repr__(self) -> str:
        return f"{self.reason_str}"


class UnsupportedIteratorBreak(BreakGraphReasonBase):
    """Break reason for unsupported custom iterator operations."""

    def __init__(self, reason_str, file_path="", line_number=-1):
        if "" == reason_str:
            reason_str = "Break graph when using user defined iterator"
        super().__init__(reason_str, file_path, line_number)


class UnsupportedTypeBreak(BreakGraphReasonBase):
    """Break reason for unsupported type operations."""

    pass


class UnsupportedSliceBreak(BreakGraphReasonBase):
    """Break reason for unsupported slice operations."""

    pass


class UnsupportedDynamicShapeBreak(BreakGraphReasonBase):
    """Break reason for unsupported dynamic shape operations.

    This class represents breaks that occur when dealing with tensor shapes
    that cannot be determined statically at compile time.

    Example scenarios:
        - Dynamic reshaping operations
        - Variable-length sequences
        - Runtime-dependent shape calculations
    """

    pass


class ConditionalBreak(BreakGraphReasonBase):
    """Break reason for conditional statement execution."""

    def __init__(self, file_path="", line_number=-1):
        reason_str = "OpcodeInlineExecutor want break graph when simulate `if`."
        super().__init__(
            reason_str,
            file_path,
            line_number,
        )


class InferMetaBreak(BreakGraphReasonBase):
    """Break reason during meta information inference phase."""

    pass


class BuiltinFunctionBreak(BreakGraphReasonBase):
    """Break reason for unsupported built-in function calls.

    Args:
        fn_name (str): Name of the builtin function
        arg_types (list): Types of the arguments passed to the function
        file_path (str): Path to the file where break occurred
        line_number (int): Line number where break occurred
    """

    def __init__(self, fn_name, arg_types, file_path="", line_number=-1):
        reason_str = f"Not support builtin function: {fn_name} with args: Args({arg_types})"
        super().__init__(
            reason_str,
            file_path,
            line_number,
        )


class UnsupportedFunctionBreak(BreakGraphReasonBase):
    """Break reason for unsupported user-defined function calls.

    Args:
        fn_name (str): Name of the unsupported function
        file_path (str): Path to the file where break occurred
        line_number (int): Line number where break occurred
    """

    def __init__(self, fn_name, file_path="", line_number=-1):
        reason_str = f"Break graph by unsupported function: {fn_name}"
        super().__init__(
            reason_str,
            file_path,
            line_number,
        )


class UnsupportedOperatorBreak(BreakGraphReasonBase):
    """Break reason for unsupported operator operations between different types.

    Args:
        left_type (str): Type of the left operand
        right_type (str): Type of the right operand
        operator (str): The operator that's not supported
        file_path (str): Path to the file where break occurred
        line_number (int): Line number where break occurred
    """

    def __init__(
        self, left_type, right_type, operator, file_path="", line_number=-1
    ):
        reason_str = f"Unsupported operator '{operator}' between {left_type} and {right_type}"
        super().__init__(reason_str, file_path, line_number)


class UnsupportedAttributeBreak(BreakGraphReasonBase):
    """Break reason for unsupported method and attribute access.

    This class represents breaks that occur when attempting to access
    unsupported attributes or methods of objects during graph execution.

    Example scenarios:
        - Accessing undefined attributes
        - Calling unsupported methods
        - Dynamic attribute access
    """

    pass


class SotErrorBreak(BreakGraphReasonBase):
    """Break reason for other general SoT errors."""

    pass


class CustomBreakReason(BreakGraphReasonBase):
    """Custom break reason for user-defined interruption scenarios."""

    pass


class UnspecifiedBreakReason(BreakGraphReasonBase):
    """Break reason for cases that don't fall into other categories."""

    pass


class SotErrorBase(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from ..opcode_translator.breakpoint import BreakpointManager

        BreakpointManager().on_event(f"{self.__class__.__name__}")

    def print(self):
        lines = traceback.format_tb(self.__traceback__)
        print("".join(lines))


class InnerError(SotErrorBase):
    pass


class HasNoAttributeError(InnerError):
    pass


class FallbackError(SotErrorBase):
    def __init__(self, msg, disable_eval_frame=False):
        super().__init__(msg)
        self.disable_eval_frame = disable_eval_frame


# raise in inline function call strategy.
class BreakGraphError(SotErrorBase):
    def __init__(self, reason: BreakGraphReasonBase | str = None):
        super().__init__()

        if isinstance(reason, str):
            reason = UnspecifiedBreakReason(reason)
        self.reason = reason
        BreakGraphReasonInfo.collect_break_graph_reason(reason)


def inner_error_default_handler(func, message_fn):
    """Wrap function and an error handling function and throw an InnerError."""

    def impl(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except SotErrorBase as e:
            raise e
        except Exception as e:
            message = message_fn(*args, **kwargs)
            origin_exception_message = "\n".join(
                traceback.format_exception(type(e), e, e.__traceback__)
            )
            raise InnerError(
                f"{message}\nOrigin Exception is: \n {origin_exception_message}"
            ) from e

    return impl


class ExportError(SotErrorBase):
    pass


class SotExtraInfo:
    SOT_EXTRA_INFO_ATTR_NAME = "__SOT_EXTRA_INFO__"

    def __init__(self, *, need_breakgraph: bool = False):
        self.need_breakgraph = need_breakgraph

    def set_need_breakgraph(self, need_breakgraph: bool):
        self.need_breakgraph = need_breakgraph

    def attach(self, err: BaseException):
        setattr(err, SotExtraInfo.SOT_EXTRA_INFO_ATTR_NAME, self)

    @staticmethod
    def default() -> SotExtraInfo:
        return SotExtraInfo()

    @staticmethod
    def from_exception(err: BaseException) -> SotExtraInfo:
        info = getattr(
            err, SotExtraInfo.SOT_EXTRA_INFO_ATTR_NAME, SotExtraInfo.default()
        )
        setattr(err, SotExtraInfo.SOT_EXTRA_INFO_ATTR_NAME, info)
        return info
