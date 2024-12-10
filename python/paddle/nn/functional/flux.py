from paddle import _C_ops
def gemm_reduce_scatter(
    input,
    weight,
    group
):
  transpose_weight = input.shape[1] == weight.shape[0]
  global_M = input.shape[0]
  global_N = weight.shape[1] if transpose_weight else weight.shape[0]
  fuse_reduction = False
  nnodes = 1
  ring_id = group.id
  nranks = group.nranks
  bias = None
  input_scale = None
  weight_scale = None
  output_scale = None

  return _C_ops.gemm_reduce_scatter(input,
                                    weight,
                                    bias,
                                    input_scale,
                                    weight_scale,
                                    output_scale,
                                    nnodes,
                                    global_M,
                                    global_N,
                                    transpose_weight,
                                    fuse_reduction,
                                    ring_id,
                                    nranks)

def all_gather_gemm(
    input,
    weight,
    group,
    deepcopy_input_parallel
):
  nnodes = 1
  transpose_weight = input.shape[1] == weight.shape[0]
  full_m = input.shape[0] * group.nranks
  k_dim = input.shape[1]
  n_dim = weight.shape[1] if transpose_weight else weight.shape[0]
  ring_id = group.id
  fast_accum = not deepcopy_input_parallel
  local_copy = False
  bias = None
  input_scale = None
  weight_scale = None
  output_scale = None
  output, input_parallel = _C_ops.all_gather_gemm(
                               input, weight, bias, input_scale, weight_scale, output_scale,
                               nnodes, full_m, n_dim, k_dim, ring_id,
                               fast_accum, transpose_weight, local_copy)
  return output, input_parallel
