from paddle import _C_ops
def gemm_rs(
    input,
    weight,
    bias,
    input_scale,
    weight_scale,
    output_scale,
    nnodes,
    ring_id,
    root_id,
    nranks,
    transpose_weight,
    fast_accum,
):
  global_M = input.shape[0]
  global_N = weight.shape[1] if transpose_weight else weight.shape[0]
  fuse_reduction = False
  return _C_ops.gemm_rs(input,
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
                        root_id,
                        nranks)
