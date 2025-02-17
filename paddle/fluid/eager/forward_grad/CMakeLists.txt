cc_library(
  forward_grad
  SRCS forward_grad.cc forward_grad.h functionality.cc
  DEPS phi common grad_node_info)
