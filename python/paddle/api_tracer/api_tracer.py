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

import os

import numpy as np
import yaml


class HookAPIMap:
    pass


class ConfigDump:
    def __init__(self):
        self.file = open(os.path.dirname(__file__) + "/api_config.txt", "a+")

    def dump_config(self, api, input_args, input_kwargs, outputs):
        self.file.write(api + "(")
        for value in input_args:
            self.file.write(self.dump_item_str(api, value) + ", ")
        for key, value in input_kwargs.items():
            self.file.write(key + "=" + self.dump_item_str(api, value) + ", ")
        self.file.write(") -> ")
        if isinstance(outputs, (list, tuple)):
            for output in outputs:
                self.file.write(self.dump_item_str(api, output) + ", ")
        else:
            self.file.write(self.dump_item_str(api, outputs) + ", ")

        self.file.flush()

    def dump_item_str(self, api, item):
        import paddle

        type_mapping = {
            np.integer: int,
            np.floating: float,
            np.bool_: bool,
            np.complexfloating: complex,
            np.str_: str,
            np.bytes_: bytes,
            # np.unicode_: str,
        }
        for numpy_type, builtin_type in type_mapping.items():
            if isinstance(item, numpy_type):
                item = builtin_type(item)
                break

        if isinstance(item, paddle.Tensor):
            return "Tensor(" + str(item.shape) + "," + str(item.dtype)[7:] + ")"
        elif isinstance(item, paddle.dtype):
            return "Dtype(" + str(item)[7:] + ")"
        elif isinstance(item, list):
            return "list(" + str(item) + ")"
        elif isinstance(item, tuple):
            return "tuple(" + str(item) + ")"
        elif isinstance(item, slice):
            return (
                "slice("
                + str(item.start)
                + ","
                + str(item.stop)
                + ","
                + str(item.step)
                + ")"
            )
        elif isinstance(item, (bool, int, float, str, complex, bytes)):
            return (
                str(type(item))[
                    str(type(item)).index("'") + 1 : str(type(item)).rindex("'")
                ]
                + "("
                + str(item)
                + ")"
            )
        else:
            print("[api_tracer error] : dump_item_str ", api, ", item = ", item)


config_dump = ConfigDump()


class APITemplate:
    def __init__(self, api_name):
        self.api_name = api_name

    def __call__(self, *args, **kwargs):
        output = getattr(HookAPIMap, self.api_name)(*args, **kwargs)
        try:
            config_dump.dump_config(self.api_name, args, kwargs, output)
        except Exception as err:
            print(
                "[api_tracer error] : config_dump.dump_config ",
                self.api_name,
                str(err),
            )
        return output


def wrapped_api(api_name):
    def api_template(*args, **kwargs):
        return APITemplate(api_name)(*args, **kwargs)

    return api_template


def start_api_tracer():
    with open(os.path.dirname(__file__) + "/api.yaml", "r") as f:
        apis = yaml.safe_load(f)
        sample_apis = apis.get("sample")
        f.close()
    for api in sample_apis:
        parent_package, method_name = api.rsplit(".", maxsplit=1)
        try:
            setattr(HookAPIMap, api, getattr(eval(parent_package), method_name))
            setattr(eval(parent_package), method_name, wrapped_api(api))
        except Exception as err:
            print("[api_tracer error] : start_api_tracer ", api, str(err))
