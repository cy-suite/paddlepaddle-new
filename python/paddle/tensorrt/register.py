# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved
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


class ConverterOpRegistry:
    def __init__(self):
        self._registry = {}

    def register(self, op_name, trt_version=None):
        def decorator(func):
            if op_name not in self._registry:
                self._registry[op_name] = []
            self._registry[op_name].append((trt_version, func))
            return func

        return decorator

    def get(self, op_name, trt_version=None):
        if op_name not in self._registry:
            return None
        for version_range, func in self._registry[op_name]:
            if self._version_match(trt_version, version_range):
                return func
            else:
                raise ValueError(
                    f"Requested TensorRT version : {trt_version} does not match the range of pip installed tensorrt versions : {version_range}"
                )
        return self._registry.get(op_name)

    def _version_match(self, trt_version, version_range):
        def _normalize_version(version):
            return tuple(map(int, [*version.split('.'), '0', '0'][:3]))

        trt_version_tuple = _normalize_version(trt_version)

        if '=' in version_range:
            comparator, ref_version = version_range.split('=')
            ref_version_tuple = _normalize_version(ref_version)
            return (
                comparator == 'trt_version_ge'
                and trt_version_tuple >= ref_version_tuple
            ) or (
                comparator == 'trt_version_le'
                and trt_version_tuple <= ref_version_tuple
            )

        if 'x' in version_range:
            return trt_version_tuple[0] == int(version_range.split('.')[0])

        return False


converter_registry = ConverterOpRegistry()
