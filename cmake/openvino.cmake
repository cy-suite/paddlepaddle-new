if(NOT WITH_OPENVINO)
  return()
endif()

if(WIN32)
  message(
    SEND_ERROR "The current openvino backend doesn't support windows platform")
  return()
else()
  set(OPENVINO_ROOT
      "/usr"
      CACHE PATH "OPENVINO ROOT")
  set(OV_INFER_RT libopenvino_c.so)
  set(PADDLE_FRONTEND libopenvino_paddle_frontend.so)
endif()

find_path(
  OPENVINO_INCLUDE_DIR openvino.hpp
  PATHS ${OPENVINO_ROOT} ${OPENVINO_ROOT}/include
        ${OPENVINO_ROOT}/include/openvino $ENV{OPENVINO_ROOT}
        $ENV{OPENVINO_ROOT}/include/openvino
  NO_DEFAULT_PATH)

find_path(
  OPENVINO_LIBRARY_DIR
  NAMES ${OV_INFER_RT}
  PATHS ${OPENVINO_ROOT} ${OPENVINO_ROOT}/lib ${OPENVINO_ROOT}/lib/intel64
        $ENV{OPENVINO_ROOT} $ENV{OPENVINO_ROOT}/lib
        $ENV{OPENVINO_ROOT}/lib/intel64
  NO_DEFAULT_PATH
  DOC "Path to OpenVINO library.")

find_library(
  OPENVINO_LIBRARY
  NAMES ${OV_INFER_RT}
  PATHS ${OPENVINO_LIBRARY_DIR}
  NO_DEFAULT_PATH
  DOC "Path to OpenVINO library.")

find_library(
  PADDLE_FRONTEND_LIBRARY
  NAMES ${PADDLE_FRONTEND}
  PATHS ${OPENVINO_LIBRARY_DIR}
  NO_DEFAULT_PATH
  DOC "Path to OpenVINO Paddle frontend library.")

if(OPENVINO_INCLUDE_DIR
   AND OPENVINO_LIBRARY
   AND PADDLE_FRONTEND_LIBRARY)
  set(OPENVINO_FOUND ON)
  message(STATUS "OPENVINO_INCLUDE_DIR = ${OPENVINO_INCLUDE_DIR}")
  message(STATUS "OPENVINO_LIBRARY = ${OPENVINO_LIBRARY}")
  message(STATUS "PADDLE_FRONTEND_LIBRARY = ${PADDLE_FRONTEND_LIBRARY}")
else()
  set(OPENVINO_FOUND OFF)
  message(
    WARNING
      "OpenVINO is disabled. You are compiling PaddlePaddle with option -DWITH_OPENVINO=ON, but OpenVINO is not found, please configure path to OpenVINO with option -DOPENVINO_ROOT or install it."
  )
endif()

if(OPENVINO_FOUND)
  file(READ ${OPENVINO_INCLUDE_DIR}/core/version.hpp
       OPENVINO_VERSION_FILE_CONTENTS)
  string(REGEX MATCH "define OPENVINO_VERSION_MAJOR +([0-9]+)"
               OPENVINO_MAJOR_VERSION "${OPENVINO_VERSION_FILE_CONTENTS}")
  string(REGEX MATCH "define OPENVINO_VERSION_MINOR +([0-9]+)"
               OPENVINO_MINOR_VERSION "${OPENVINO_VERSION_FILE_CONTENTS}")
  string(REGEX MATCH "define OPENVINO_VERSION_PATCH +([0-9]+)"
               OPENVINO_PATCH_VERSION "${OPENVINO_VERSION_FILE_CONTENTS}")

  if("${OPENVINO_MAJOR_VERSION}" STREQUAL "")
    message(SEND_ERROR "Failed to detect OpenVINO version.")
  endif()

  string(REGEX REPLACE "define OPENVINO_VERSION_MAJOR +([0-9]+)" "\\1"
                       OPENVINO_MAJOR_VERSION "${OPENVINO_MAJOR_VERSION}")
  string(REGEX REPLACE "define OPENVINO_VERSION_MINOR +([0-9]+)" "\\1"
                       OPENVINO_MINOR_VERSION "${OPENVINO_MINOR_VERSION}")
  string(REGEX REPLACE "define OPENVINO_VERSION_PATCH +([0-9]+)" "\\1"
                       OPENVINO_PATCH_VERSION "${OPENVINO_PATCH_VERSION}")

  get_filename_component(OPENVINO_INCLUDE_DIR "${OPENVINO_INCLUDE_DIR}"
                         DIRECTORY)
  message(
    STATUS
      "Current OpenVINO header is ${OPENVINO_INCLUDE_DIR}/openvino/openvino.hpp. "
      "Current OpenVINO version is v${OPENVINO_MAJOR_VERSION}.${OPENVINO_MINOR_VERSION}.${OPENVINO_PATCH_VERSION} "
  )
  include_directories(${OPENVINO_INCLUDE_DIR})
  link_directories(${OPENVINO_LIBRARY} ${PADDLE_FRONTEND_LIBRARY})
  add_definitions(-DPADDLE_WITH_OPENVINO)
endif()
