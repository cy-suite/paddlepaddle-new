cc_library(
  vectorize
  SRCS vmap_transforms.cc batching_rules.cc
  DEPS final_dygraph_function phi_core pybind phi common utils)
