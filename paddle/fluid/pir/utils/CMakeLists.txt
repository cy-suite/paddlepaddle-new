cc_library(
  pir_general_functions
  SRCS general_functions.cc analysis_info.cc name_analysis.cc
       sub_graph_detector.cc pd_op_to_kernel_utils.cc
  DEPS op_dialect op_dialect_vjp pir drr)
