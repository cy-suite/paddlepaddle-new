#!/bin/bash

# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

# Top-level build script called from Dockerfile

# Stop at any error, show all commands
set -ex

<setcuda>

wget https://github.com/openucx/ucx/archive/refs/tags/v1.17.0.tar.gz -O ./ucx-v1.17.0.tar.gz
wget https://github.com/openucx/ucc/archive/refs/tags/v1.3.0.tar.gz -O ./ucc-v1.3.0.tar.gz
tar -zxvf ucx-v1.17.0.tar.gz
tar -zxvf ucc-v1.3.0.tar.gz


cd ./ucx-1.17.0/
./autogen.sh
./contrib/configure-release --prefix=/usr/local/ucx --with-cuda=${CUDA_HOME} --enable-mt --enable-optimizations --enable-profiling --enable-stats --without-xpmem --without-knem --without-java
make -j8
make install

cd ../ucc-1.3.0/
./autogen.sh
./configure --prefix=/usr/local/ucc --with-ucx=/usr/local/ucx --with-cuda=${CUDA_HOME} --enable-profiling --enable-optimizations
make install
