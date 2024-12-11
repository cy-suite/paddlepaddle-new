file(
  GLOB TEST_OPS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

function(openvino_inference_download INSTALL_DIR URL FILENAME)
  message(STATUS "Download inference test stuff from ${URL}/${FILENAME}")
  string(REGEX REPLACE "[-%./\\]" "_" FILENAME_EX ${FILENAME})
  string(REGEX MATCH "[^/\\]+$" DOWNLOAD_NAME ${FILENAME})
  set(EXTERNAL_PROJECT_NAME "extern_download_${FILENAME_EX}")
  set(UNPACK_DIR "${INSTALL_DIR}/src/${EXTERNAL_PROJECT_NAME}")

  get_property(TARGET_EXIST GLOBAL PROPERTY ${EXTERNAL_PROJECT_NAME})
  if(NOT "${TARGET_EXIST}" STREQUAL EXIST)
    ExternalProject_Add(
      ${EXTERNAL_PROJECT_NAME}
      ${EXTERNAL_PROJECT_LOG_ARGS}
      PREFIX ${INSTALL_DIR}
      URL ${URL}/${FILENAME}
      DOWNLOAD_DIR ${INSTALL_DIR}
      DOWNLOAD_NO_EXTRACT 1
      DOWNLOAD_NO_PROGRESS 1
      CONFIGURE_COMMAND ""
      BUILD_COMMAND ${CMAKE_COMMAND} -E chdir ${INSTALL_DIR} ${CMAKE_COMMAND} -E
                    tar xzf ${DOWNLOAD_NAME}
      UPDATE_COMMAND ""
      INSTALL_COMMAND "")
    set_property(GLOBAL PROPERTY ${EXTERNAL_PROJECT_NAME} "EXIST")
  endif()
endfunction()

function(download_model install_dir data_file)
  string(REGEX MATCH "[^/\\]+$" file_name ${data_file})
  if(NOT EXISTS ${install_dir}/${file_name})
    openvino_inference_download(
      ${install_dir} "http://paddle-inference-dist.bj.bcebos.com/openvino"
      ${data_file})
  endif()
endfunction()

set(INFERENCE_DEMO_INSTALL_DIR
    "${THIRD_PARTY_PATH}/inference_demo"
    CACHE STRING "A path setting inference demo download directories.")

set(OPENVINO_MODELS_DIR "${INFERENCE_DEMO_INSTALL_DIR}/openvino")

set(RESNET50_DIR "${OPENVINO_MODELS_DIR}/resnet50")
download_model(${RESNET50_DIR} "resnet50_model.tar.gz")
py_test_modules(
  test_openvino_resnet50 MODULES test_openvino_resnet50 ENVS
  "OPENVINO_MODEL_DIR=${RESNET50_DIR}/model;FLAGS_enable_pir_api=0")
set_tests_properties(test_openvino_resnet50 PROPERTIES TIMEOUT 120)

set(MOBILENET_DIR "${OPENVINO_MODELS_DIR}/mobilenet")
download_model(${MOBILENET_DIR} "mobilenetv1_model.tar.gz")

py_test_modules(
  test_openvino_mobilenet_fp32 MODULES test_openvino_mobilenet_fp32 ENVS
  "OPENVINO_MODEL_DIR=${MOBILENET_DIR}/model;FLAGS_enable_pir_api=0")
set_tests_properties(test_openvino_mobilenet_fp32 PROPERTIES TIMEOUT 120)

py_test_modules(
  test_openvino_mobilenet_fp16 MODULES test_openvino_mobilenet_fp16 ENVS
  "OPENVINO_MODEL_DIR=${MOBILENET_DIR}/model;FLAGS_enable_pir_api=0")
set_tests_properties(test_openvino_mobilenet_fp16 PROPERTIES TIMEOUT 120)

py_test_modules(
  test_openvino_mobilenet_bs2 MODULES test_openvino_mobilenet_bs2 ENVS
  "OPENVINO_MODEL_DIR=${MOBILENET_DIR}/model;FLAGS_enable_pir_api=0")
set_tests_properties(test_openvino_mobilenet_bs2 PROPERTIES TIMEOUT 120)
