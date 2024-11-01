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

set(OPENVINO_PROJECT "extern_openvino")
set(OPENVINO_PREFIX_DIR ${THIRD_PARTY_PATH}/openvino)
set(OPENVINO_INSTALL_DIR ${THIRD_PARTY_PATH}/install/openvino)
set(OPENVINO_INC_DIR
    "${OPENVINO_INSTALL_DIR}/runtime/include"
    CACHE PATH "OpenVINO include directory." FORCE)
set(SOURCE_DIR ${PADDLE_SOURCE_DIR}/third_party/openvino)

set(TBB_INC_DIR
    "${OPENVINO_INSTALL_DIR}/runtime/3rdparty/tbb/include"
    CACHE PATH "OpenVINO TBB include directory." FORCE)

# Introduce variables:
# * CMAKE_INSTALL_LIBDIR
include(GNUInstallDirs)
set(LIBDIR "runtime/lib/intel64")
set(TBBDIR "runtime/3rdparty/tbb/lib")

message(STATUS "Set ${OPENVINO_INSTALL_DIR}/${LIBDIR} to runtime path")
message(STATUS "Set ${OPENVINO_INSTALL_DIR}/${TBBDIR} to runtime path")
set(OPENVINO_LIB_DIR ${OPENVINO_INSTALL_DIR}/${LIBDIR})
set(TBB_LIB_DIR ${OPENVINO_INSTALL_DIR}/${TBBDIR})

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}" "${OPENVINO_LIB_DIR}"
                        "${TBB_LIB_DIR}")

include_directories(${OPENVINO_INC_DIR}
)# For OpenVINO code to include internal headers.

include_directories(${TBB_INC_DIR}
)# For OpenVINO TBB code to include third_party headers.

if(LINUX)
  set(OPENVINO_LIB
      "${OPENVINO_INSTALL_DIR}/${LIBDIR}/libopenvino.so.2450"
      CACHE FILEPATH "OpenVINO library." FORCE)
  set(TBB_LIB
      "${OPENVINO_INSTALL_DIR}/${TBBDIR}/libtbb.so.12"
      CACHE FILEPATH "TBB library." FORCE)
else()
  message(ERROR "Only support Linux.")
endif()

if(LINUX)
  set(BUILD_BYPRODUCTS_ARGS ${OPENVINO_LIB} ${TBB_LIB})
else()
  set(BUILD_BYPRODUCTS_ARGS "")
endif()

ExternalProject_Add(
  ${OPENVINO_PROJECT}
  ${EXTERNAL_PROJECT_LOG_ARGS}
  SOURCE_DIR ${SOURCE_DIR}
  DEPENDS ${OPENVINO_DEPENDS}
  PREFIX ${OPENVINO_PREFIX_DIR}
  UPDATE_COMMAND ""
  #BUILD_ALWAYS        1
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${OPENVINO_INSTALL_DIR}
             -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DENABLE_INTEL_GPU=OFF
             -DTHREADING=TBB
  CMAKE_CACHE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${OPENVINO_INSTALL_DIR}
  BUILD_BYPRODUCTS ${BUILD_BYPRODUCTS_ARGS})

message(STATUS "OpenVINO library: ${OPENVINO_LIB}")
message(STATUS "OpenVINO TBB library: ${TBB_LIB}")
add_definitions(-DPADDLE_WITH_OPENVINO)

add_library(openvino SHARED IMPORTED GLOBAL)
add_library(tbb SHARED IMPORTED GLOBAL)
set_property(TARGET openvino PROPERTY IMPORTED_LOCATION ${OPENVINO_LIB})
set_property(TARGET tbb PROPERTY IMPORTED_LOCATION ${TBB_LIB})
add_dependencies(openvino ${OPENVINO_PROJECT})
add_dependencies(tbb ${OPENVINO_PROJECT})
