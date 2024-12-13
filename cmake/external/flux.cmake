# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

include(ExternalProject)

add_definitions(-DPADDLE_WITH_FLUX)

if(WITH_GPU)
  set(FLUX_PREFIX_DIR ${THIRD_PARTY_PATH}/flux)
  set(FLUX_SOURCE_SUBDIR ./)
  set(FLUX_INSTALL_DIR ${THIRD_PARTY_PATH}/install/flux)
  set(SOURCE_DIR ${PADDLE_SOURCE_DIR}/third_party/flux)
  set(FLUX_LIBRARIES
      "${FLUX_INSTALL_DIR}/lib/libflux${CMAKE_SHARED_LIBRARY_SUFFIX}"
      CACHE FILEPATH "flux Library" FORCE)

  set(FLUX_INCLUDE_DIR
      "${FLUX_INSTALL_DIR}/include"
      CACHE PATH "flux Directory" FORCE)
  set(FLUX_LIB_DIR
      "${FLUX_INSTALL_DIR}/lib"
      CACHE PATH "flux Library Directory" FORCE)

  list(JOIN FLUX_NVCC_ARCH_BIN " " FLUX_CUDAARCHS)

  ExternalProject_Add(
    extern_flux
    ${EXTERNAL_PROJECT_LOG_ARGS}
    SOURCE_DIR ${SOURCE_DIR}
    PREFIX ${FLUX_PREFIX_DIR}
    SOURCE_SUBDIR ${FLUX_SOURCE_SUBDIR}
    UPDATE_COMMAND ""
    PATCH_COMMAND ""
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${FLUX_INSTALL_DIR} -DENABLE_NVSHMEM=OFF
               -DFLUX_CUDAARCHS=${FLUX_CUDAARCHS}
               -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DBUILD_THS=OFF
    CMAKE_CACHE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${FLUX_INSTALL_DIR}
    BUILD_BYPRODUCTS ${FLUX_LIBRARIES})
endif()

message(STATUS "flux library: ${FLUX_LIBRARIES}")
get_filename_component(FLUX_LIBRARY_PATH ${FLUX_LIBRARIES} DIRECTORY)
include_directories(${FLUX_INCLUDE_DIR})

add_library(flux INTERFACE)
add_dependencies(flux extern_flux)
