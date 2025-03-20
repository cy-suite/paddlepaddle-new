set(CMAKE_FIND_DEBUG_MODE ON)
# flagcx.cmake
if(NOT WITH_FLAGCX)
  return()
endif()

set(FLAGCX_SOURCE_DIR "${PADDLE_SOURCE_DIR}/third_party/flagcx")
set(FLAGCX_BINARY_DIR "${PADDLE_SOURCE_DIR}/build/third_party/flagcx")
# set(FLAGCX_INCLUDE_DIR "${THIRD_PARTY_PATH}/flagcx/flagcx/include")
set(FLAGCX_LIB_DIR "${PADDLE_SOURCE_DIR}/build/third_party/flagcx/build/lib")

message(STATUS "Copying third-party source to build directory")
# execute_process(
#     COMMAND ${CMAKE_COMMAND} -E remove_directory ${FLAGCX_BINARY_DIR}
#     COMMAND ${CMAKE_COMMAND} -E copy_directory ${FLAGCX_SOURCE_DIR} ${FLAGCX_BINARY_DIR}
#     RESULT_VARIABLE COPY_RESULT
# )
execute_process(
    COMMAND rm -r ${FLAGCX_BINARY_DIR}
    COMMAND cp -r ${FLAGCX_SOURCE_DIR} ${FLAGCX_BINARY_DIR}/..
    RESULT_VARIABLE COPY_RESULT
)

if(NOT COPY_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to copy third-party source to build directory")
endif()

# Create a custom target to build the third-party library
message(STATUS "Building third-party library with its Makefile")
execute_process(
    COMMAND make 
    WORKING_DIRECTORY ${FLAGCX_BINARY_DIR}
    RESULT_VARIABLE BUILD_RESULT
)

if(NOT BUILD_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to build third-party library")
endif()

# set(FLAGCX_ROOT
#       $ENV{FLAGCX_ROOT}
#       CACHE PATH "FLAGCX_ROOT")
#   message(STATUS "FLAGCX_ROOT is ${FLAGCX_ROOT}")
  # generate_dummy_static_lib(LIB_NAME "flagcx" GENERATOR "flagcx.cmake")
  find_path(
    FLAGCX_INCLUDE_DIR flagcx.h
    PATHS ${FLAGCX_SOURCE_DIR}/flagcx/include
    NO_DEFAULT_PATH)
  
  message(STATUS "FLAGCX_INCLUDE_DIR is ${FLAGCX_INCLUDE_DIR}")
  include_directories(SYSTEM ${FLAGCX_INCLUDE_DIR})
  # set(FLAGCX_LIB
  #     "${FLAGCX_LIB_DIR}/libflagcx.so"
  #     CACHE FILEPATH "flagcx library." FORCE)
  # target_link_libraries(flagcx ${FLAGCX_LIB})
  add_library(flagcx INTERFACE)
  find_library(FLAGCX_LIB 
  NAMES flagcx libflagcx
  PATHS ${FLAGCX_LIB_DIR}
  DOC "My custom library"
)
add_dependencies(flagcx FLAGCX_LIB)
  message(STATUS "FLAGCX_LIB is ${FLAGCX_LIB}")