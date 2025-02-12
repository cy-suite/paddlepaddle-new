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

cd third_party/gloo
git fetch tags v0.0.3
cd -
cd third_party/protobuf
git fetch tags v21.12
cd -
cd third_party/gtest
git fetch tags release-1.8.1
cd -
cd third_party/pocketfft
git fetch tags release_for_eigen
cd -
cd third_party/pybind
git fetch tags v2.13.6
cd -
