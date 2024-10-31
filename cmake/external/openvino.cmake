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

# Introduce variables:
# * CMAKE_INSTALL_LIBDIR
include(GNUInstallDirs)
set(LIBDIR "lib/intel64")
if(CMAKE_INSTALL_LIBDIR MATCHES ".*lib64$")
  set(LIBDIR "lib64/intel64")
endif()

message(STATUS "Set ${OPENVINO_INSTALL_DIR}/${LIBDIR} to runtime path")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}"
                        "${OPENVINO_INSTALL_DIR}/${LIBDIR}")

include_directories(${OPENVINO_INC_DIR}
)# For OpenVINO code to include internal headers.

if(NOT WIN32)
  set(OPENVINO_FLAG
      "-Wno-error=strict-overflow -Wno-error=unused-result -Wno-error=array-bounds"
  )
  set(OPENVINO_FLAG "${OPENVINO_FLAG} -Wno-unused-result -Wno-unused-value")
  set(OPENVINO_CFLAG "${CMAKE_C_FLAGS} ${OPENVINO_FLAG}")
  set(OPENVINO_CXXFLAG "${CMAKE_CXX_FLAGS} ${OPENVINO_FLAG}")
  set(OPENVINO_CXXFLAG_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
  set(OPENVINO_CFLAG_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
  set(OPENVINO_LIB
      "${OPENVINO_INSTALL_DIR}/${LIBDIR}/libopenvino.so"
      CACHE FILEPATH "OpenVINO library." FORCE)
else()
  message(ERROR "Don't support Windows yet.")
endif()

if(LINUX)
  set(BUILD_BYPRODUCTS_ARGS ${OPENVINO_LIB})
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
  CMAKE_ARGS -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
             -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
             -DCMAKE_CXX_FLAGS=${OPENVINO_CXXFLAG}
             -DCMAKE_CXX_FLAGS_RELEASE=${OPENVINO_CXXFLAG_RELEASE}
             -DCMAKE_CXX_FLAGS_DEBUG=${CMAKE_CXX_FLAGS_DEBUG}
             -DCMAKE_C_FLAGS=${OPENVINO_CFLAG}
             -DCMAKE_C_FLAGS_DEBUG=${CMAKE_C_FLAGS_DEBUG}
             -DCMAKE_C_FLAGS_RELEASE=${OPENVINO_CFLAG_RELEASE}
             -DCMAKE_INSTALL_PREFIX=${OPENVINO_INSTALL_DIR}
             -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
             -DCMAKE_POSITION_INDEPENDENT_CODE=ON
             -DTHREADING=OMP
  CMAKE_CACHE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${OPENVINO_INSTALL_DIR}
  BUILD_BYPRODUCTS ${BUILD_BYPRODUCTS_ARGS})

message(STATUS "OpenVINO library: ${OPENVINO_LIB}")
add_definitions(-DPADDLE_WITH_OPENVINO)
# copy the real so.0 lib to install dir
# it can be directly contained in wheel or capi
if(LINUX)
  set(OPENVINO_SHARED_LIB ${OPENVINO_INSTALL_DIR}/libdnnl.so.3)
  add_custom_command(
    OUTPUT ${OPENVINO_SHARED_LIB}
    COMMAND ${CMAKE_COMMAND} -E copy ${OPENVINO_LIB} ${OPENVINO_SHARED_LIB}
    DEPENDS ${OPENVINO_PROJECT})
  add_custom_target(openvino_cmd ALL DEPENDS ${OPENVINO_SHARED_LIB})
endif()

# generate a static dummy target to track openvino dependencies
# for cc_library(xxx SRCS xxx.c DEPS openvino)
generate_dummy_static_lib(LIB_NAME "openvino" GENERATOR "openvino.cmake")

target_link_libraries(openvino ${OPENVINO_LIB})
add_dependencies(openvino ${OPENVINO_PROJECT} openvino_cmd)
