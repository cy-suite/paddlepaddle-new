#!/bin/bash
mkdir -p build
pushd build
cmake .. -DWITH_MKL=ON -DWITH_MUSA=ON -DWITH_MCCL=ON -DWITH_GPU=OFF -DPY_VERSION=3.9 -DWITH_CINN=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=on 
make -j 128
pip install python/dist/paddlepaddle_musa-0.0.0-cp39-cp39-linux_x86_64.whl --force-reinstall
popd
