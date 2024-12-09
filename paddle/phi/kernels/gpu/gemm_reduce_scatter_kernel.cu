#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/common/data_type.h"

#include "paddle/phi/core/distributed/comm_context_manager.h"
#include "paddle/phi/backends/dynload/flux.h"

#include "paddle/phi/core/utils/intrusive_ptr.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/dense_tensor.h"
// #include "paddle/fluid/distributed/collective/process_group_nccl.h"
#include "paddle/fluid/distributed/collective/process_group.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/slice.h"
// #include "paddle/phi/kernels/elementwise_kernel.h"
#include "paddle/phi/kernels/elementwise_add_kernel.h"
#include "paddle/phi/kernels/unsqueeze_kernel.h"
#include "paddle/phi/kernels/view_kernel.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"
#include "paddle/phi/kernels/impl/slice_kernel_impl.h"
#include "paddle/phi/kernels/gpu/flux_utils.h"
namespace phi {

template<typename InT, typename OutT>
class GemmRSWrapper {
public:
  const phi::GPUContext& dev_ctx;
  paddle::distributed::ProcessGroup* tp_group;
  // umiswing: weird name, but it's used in source code of process group.
  const phi::GPUContext* comm_ctx;
  // TODO(umiswing): i find nobody use phi::intrusive_ptr...
  // phi::intrusive_ptr<paddle::distributed::ProcessGroupNCCL> tp_group; // umiswing: not necessary to pass tp_group
  const int32_t nnodes;
  const int32_t max_m;
  const int32_t n_dim;
  const phi::DataType input_dtype = std::is_same<InT, phi::dtype::float16>::value ?
                                    phi::DataType::FLOAT16 :
                                    phi::DataType::BFLOAT16;
  const phi::DataType output_dtype = std::is_same<OutT, phi::dtype::float16>::value ?
                                     phi::DataType::FLOAT16 :
                                     phi::DataType::BFLOAT16;
  const bool transpose_weight;
  const bool fuse_reduction;

  const int32_t rank;
  const int32_t world_size;
  const int32_t local_world_size;
  const int32_t local_rank;
  const int32_t node_idx;

  // Symmetrically distributed tensor
  std::vector<DenseTensor> output_buffers; // OutT
  std::vector<DenseTensor> reduce_buffers; // OutT
  std::vector<DenseTensor> barrier_buffers; //UINT8 (c10::ScalarType::Byte)
#ifndef FLUX_SHM_USE_NVSHMEM
  // used for the cuda-ipc-barrier
  std::vector<DenseTensor> sync_buffers; // int32_t (c10::ScalarType::Int)
#endif
  DenseTensor output_buffer;
  DenseTensor reduce_buffer;
  DenseTensor barrier_buffer;
  DenseTensor gemm_buffer; // TODO(umiswing): find a way to unify declaration, phi has some requirements.
  std::vector<void *> output_scatter_ptrs;
  std::vector<void *> barrier_ptrs;
  std::vector<void *> output_buffer_ptrs;
  std::vector<void *> reduce_buffer_ptrs;
  std::vector<void *> barrier_buffer_ptrs;
  std::vector<int32_t *> sync_buffer_ptrs;
  bool no_nvlink;
  int sub_world_size;
  // phi::CUDAStream rs_stream_;
  cudaStream_t rs_stream_;
  cudaEvent_t event_;
  bool use_1d_ring;
  bool use_p2p_read;
  const bool is_fp8_gemm{false};

  GemmRSWrapper(
      const phi::GPUContext& dev_ctx,
      paddle::distributed::ProcessGroup* tp_group_,
      const phi::GPUContext* comm_ctx,
      int32_t nnodes,
      int32_t max_m,
      int32_t n_dim,
      bool transpose_weight,
      bool fuse_reduction)
      : dev_ctx(dev_ctx),
        comm_ctx(comm_ctx),
        tp_group(tp_group_),
        nnodes(nnodes),
        max_m(max_m),
        n_dim(n_dim),
        transpose_weight(transpose_weight),
        fuse_reduction(fuse_reduction),
        rank(tp_group->GetRank()),
        world_size(tp_group->GetSize()),
        local_world_size(world_size / nnodes),
        local_rank(rank % local_world_size),
        node_idx(rank / local_world_size),
        output_scatter_ptrs(world_size, nullptr),
        barrier_ptrs(world_size, nullptr),
        output_buffer_ptrs(world_size, nullptr),
        reduce_buffer_ptrs(world_size, nullptr),
        barrier_buffer_ptrs(world_size, nullptr),
        sync_buffer_ptrs(world_size, nullptr),
        no_nvlink(!has_nvlink()),
        rs_stream_(CreateReduceScatterStream()),  // private stream. never dup with gemm stream
        use_1d_ring(use_1d_ring_or_not()),
        use_p2p_read(use_p2p_read_or_not()) {

    PADDLE_ENFORCE(
        rank >= 0 && rank < world_size,
        "invalid rank: %d and world_size: %d",
        rank,
        world_size);
    PADDLE_ENFORCE(
        world_size % nnodes == 0,
        "invalid nnodes: world_size[%d] % nnodes[%d] !=0",
        world_size,
        nnodes);
    PADDLE_ENFORCE(
        !fuse_reduction || this->input_dtype == phi::DataType::FLOAT16,
        "Fuse reduction only support float16 type on SM80 due to instruction limitation.");
    this->init_output_buffer();
    PADDLE_ENFORCE_GPU_SUCCESS(cudaEventCreate(&event_));
  }

  ~GemmRSWrapper() {
    PADDLE_ENFORCE_GPU_SUCCESS(cudaEventDestroy(event_));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamDestroy(rs_stream_));
  }

  bool
  has_nvlink() {
    return true;
  }

  bool
  use_1d_ring_or_not() {
    phi::dynload::ensure_nvml_init_capi();
    int devid = phi::backends::gpu::GetCurrentDeviceId();
    std::string devname(phi::dynload::get_gpu_device_name_capi(devid));
    if (devname != "NVIDIA L20" && world_size == 8) {
      return false;
    }
    return true;
  }

  bool
  use_p2p_read_or_not() {
    phi::dynload::ensure_nvml_init_capi();
    int devid = phi::backends::gpu::GetCurrentDeviceId();
    std::string devname(phi::dynload::get_gpu_device_name_capi(devid));
    if (devname != "NVIDIA L20") {
      return true;
    }
    return false;
  }

#if 0
using Deleter = void (*)(phi::Allocation*);
using AllocationDeleter = void (*)(phi::Allocation*);
DenseTensor from_blob(void *data,
                      const std::vector<int64_t>& shape,
                      phi::DataType dtype,
                      phi::Place place,
                      const Deleter& deleter,
                      phi::DataLayout layout = phi::DataLayout::NCHW ) {
  PADDLE_ENFORCE_NOT_NULL(
      data, common::errors::InvalidArgument("data can not be nullptr."));

  // TODO(umiswing): this check looks nice
  // auto data_place = GetPlaceFromPtr(data);
  // phi::is_gpu_place(place);
#if 0
  PADDLE_ENFORCE(
      dev_ctx.GetPlace().GetType() == phi::AllocationType::GPU,
      "gemm_rs not on GPU");
#endif

  auto meta =
      phi::DenseTensorMeta(dtype, common::make_ddim(shape), layout);

  size_t size = SizeOf(dtype) * (meta.is_scalar ? 1 : product(meta.dims));

  auto alloc =
      // std::make_shared<phi::Allocation>(data, size, alloc_deleter, place/*data_place*/);
      std::make_shared<phi::Allocation>(data, size, deleter, place/*data_place*/);

  return DenseTensor(alloc, meta);
}

template<typename BufferT>
std::vector<DenseTensor>
cudaipc_create_tensor_list(
    const std::vector<int64_t> &shape) {

  PADDLE_ENFORCE_GE(
      phi::backends::gpu::GetGPUDeviceCount(),
      tp_group->GetSize(),
      common::errors::InvalidArgument("create_ipc_tensors should only be used intra node"));

  size_t size = sizeof(BufferT) * std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
#if 0
  size_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
#endif
  PADDLE_ENFORCE_NE(size, 0);

  void *ptr = nullptr;
  PADDLE_ENFORCE_GPU_SUCCESS(cudaMalloc(&ptr, size));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaMemset(ptr, 0, size)); // memset the allocated buffer

#if 0
  DenseTensor local_tensor = phi::Empty<BufferT>(dev_ctx, IntArray{shape});
  phi::funcs::SetConstant<GPUContext, BufferT> set_functor;
  set_functor(dev_ctx, &local_tensor, BufferT{0});
  void *ptr = local_tensor.data();
#endif

  cudaIpcMemHandle_t handle;
  PADDLE_ENFORCE_GPU_SUCCESS(cudaIpcGetMemHandle(&handle, ptr));

  DenseTensor handle_d = phi::Empty<uint8_t>(dev_ctx, {sizeof(cudaIpcMemHandle_t)});
  PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpy(
      handle_d.data(), &handle, sizeof(cudaIpcMemHandle_t), cudaMemcpyHostToDevice));
  long int handles_shape = sizeof(cudaIpcMemHandle_t) * tp_group->GetSize();
  DenseTensor handles_d = phi::Empty<uint8_t>(dev_ctx, {handles_shape});
  // TODO(umiswing): find a better way to wrap func params
  tp_group->AllGather(&handles_d, handle_d, 0, -1, true, true)->Wait();

  std::vector<cudaIpcMemHandle_t> handles_h(tp_group->GetSize());
  PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpy(
      handles_h.data(),
      handles_d.data(),
      sizeof(cudaIpcMemHandle_t) * tp_group->GetSize(),
      cudaMemcpyDeviceToHost));

  std::vector<void *> ptrs(tp_group->GetSize());
  for (int i = 0; i < tp_group->GetSize(); ++i) {
    if (i != tp_group->GetRank()) {
        PADDLE_ENFORCE_GPU_SUCCESS(cudaIpcOpenMemHandle(&ptrs[i], handles_h[i], cudaIpcMemLazyEnablePeerAccess));
    } else {
      ptrs[i] = ptr;
    }
  }

 phi::DataType dtype;
 if (std::is_same<BufferT, phi::dtype::float16>::value) dtype = phi::DataType::FLOAT16;
 else if(std::is_same<BufferT, phi::dtype::bfloat16>::value) dtype = phi::DataType::BFLOAT16;
 else if(std::is_same<BufferT, uint8_t>::value) dtype = phi::DataType::UINT8;
 else if(std::is_same<BufferT, int32_t>::value) dtype = phi::DataType::INT32;
 else throw std::runtime_error("cudaipc_create_tensor_list unexpected BufferT");

  std::vector<DenseTensor> tensors;
  for (int i = 0; i < tp_group->GetSize(); ++i) {
    if (i == tp_group->GetRank()) {
      DenseTensor local_tensor;
      local_tensor =
          from_blob(ptrs[i], shape, dtype, dev_ctx.GetPlace(), [](phi::Allocation* allocation) { cudaFree(allocation->ptr()); });
      tensors.emplace_back(local_tensor);
    } else {
      DenseTensor tensor;
      tensor =
          from_blob(ptrs[i], shape, dtype, dev_ctx.GetPlace(), [](phi::Allocation* allocation) { cudaIpcCloseMemHandle(allocation->ptr()); });
      tensors.emplace_back(tensor);
    }
  }

  return tensors;
}
#endif

  void
  init_output_buffer() {
    // update max_m and allocate buffer
  int64_t device_id = dev_ctx.GetPlace().GetDeviceId();
  int sm_version = backends::gpu::GetGPUComputeCapability(device_id);
  // printf("\nsm_version:%d, sm_version == 90:%d\n", sm_version, sm_version == 90);
    if (sm_version == 90 || no_nvlink || (sm_version == 80 && nnodes > 1)) {
      int reduce_m_dim = (sm_version == 90)
                             ? (max_m + world_size - 1) / world_size * nnodes * nnodes
                             : max_m;
      // static BuffersHolder<OutT> reduce_buffers_holder{{reduce_m_dim, n_dim},dev_ctx, tp_group};
      static BuffersHolder<OutT> reduce_buffers_holder{{max_m, n_dim},dev_ctx, tp_group};

      // reduce_buffers = reduce_buffers_holder.get_buffers({reduce_m_dim, n_dim});
      reduce_buffers = reduce_buffers_holder.get_buffers({max_m, n_dim});
      // reduce_buffers =
      //     cudaipc_create_tensor_list<OutT>({reduce_m_dim, n_dim});
      reduce_buffer = reduce_buffers[local_rank];
    }
    static BuffersHolder<OutT> output_buffers_holder{{max_m, n_dim}, dev_ctx, tp_group};
    if (sm_version == 80 && nnodes > 1  && input_dtype == phi::DataType::BFLOAT16) {
      // SM80 does not support the fuse reduction for the bfloat16 data type
      // we have to use the float32 global_red instruction when SM80 && nnodes>1 && input_type=bf16
      // Therefore, in this case, here double the size of the output_buffer.
      output_buffers = output_buffers_holder.get_buffers({max_m*2, n_dim});
#if 0
      output_buffers =
          cudaipc_create_tensor_list<OutT>({max_m * 2, n_dim});
#endif
    } else {
      output_buffers = output_buffers_holder.get_buffers({max_m, n_dim});
      // output_buffers = cudaipc_create_tensor_list<OutT>({max_m, n_dim});
    }
    output_buffer = output_buffers[local_rank];
    for (int i = 0; i < world_size; ++i) {
      if (i / local_world_size == rank / local_world_size) {
        output_scatter_ptrs[i] = output_buffers[i % local_world_size].data();
        // only check for ranks on the same node
        PADDLE_ENFORCE_NOT_NULL(
            output_scatter_ptrs[i],
            common::errors::InvalidArgument("nullptr buffr of rank " + std::to_string(i)));
      } else {
        output_scatter_ptrs[i] = nullptr;
      }
    }
    for(size_t i=0;i<reduce_buffers.size();i++) {
      reduce_buffer_ptrs[i] = reduce_buffers[i].data();
    }
    for(size_t i=0;i<output_buffers.size();i++) {
      output_buffer_ptrs[i] = output_buffers[i].data();
    }
    // printf("\nreduce_buffers.size():%d\n", reduce_buffers.size());
#ifndef FLUX_SHM_USE_NVSHMEM
    static BuffersHolder<int32_t> sync_buffers_holder{{this->world_size}, dev_ctx, tp_group};
    this->sync_buffers = sync_buffers_holder.get_buffers({this->world_size});
#if 0
    this->sync_buffers =
        cudaipc_create_tensor_list<int32_t>({this->world_size});
#endif
    phi::funcs::SetConstant<GPUContext, int32_t> set_functor;
    set_functor(this->dev_ctx, &this->sync_buffers[this->rank], 0);
    for(size_t i=0;i<sync_buffers.size();i++) {
      sync_buffer_ptrs[i] = static_cast<int32_t *>(sync_buffers[i].data());
    }
#endif
  }

  void
  lazy_init_barrier_buffer(int64_t buffer_size) {
#if 0
    if ((buffer_size == 0) ||
        (barrier_buffer.initialized() && buffer_size <= barrier_buffer.numel())) {
      return;
    }
#endif
    if (buffer_size == 0) {
      return;
    }

    static BuffersHolder<uint8_t> barrier_buffers_holder{{buffer_size}, dev_ctx, tp_group}; 
    barrier_buffers = barrier_buffers_holder.get_buffers({buffer_size});

#if 0
    barrier_buffers =
        cudaipc_create_tensor_list<uint8_t>({buffer_size});
#endif
    for(size_t i=0;i<barrier_buffers.size();i++) {
      barrier_buffer_ptrs[i] = barrier_buffers[i].data();
    }
    this->barrier_buffer = this->barrier_buffers[this->local_rank];
    for (int i = 0; i < world_size; ++i) {
      if (i / this->local_world_size == rank / this->local_world_size) {
        barrier_ptrs[i] = barrier_buffers[i % this->local_world_size].data();
        // only check for ranks on the same node
        PADDLE_ENFORCE_NOT_NULL(
            barrier_ptrs[i],
            common::errors::InvalidArgument("nullptr buffr of rank " + std::to_string(i)));
      } else {
        barrier_ptrs[i] = nullptr;
      }
    }
  }

  void
  lazy_init_gemm_buffer(int64_t buffer_size) {
    if (buffer_size <= 0) {
      return;
    }
    buffer_size = (buffer_size + 127) / 128 * 128;
    if (!gemm_buffer.initialized() || buffer_size > gemm_buffer.numel()) {
      gemm_buffer = phi::Empty<uint8_t>(dev_ctx,{buffer_size});
    }
  }

  cudaStream_t
  CreateReduceScatterStream() {
     cudaStream_t rs_stream = nullptr;
     int least_priority, greatest_priority;
     PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority));
     PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamCreateWithPriority(&rs_stream, cudaStreamNonBlocking, greatest_priority));
     return rs_stream;
#if 0
     return this->comm_ctx->stream();
#endif
  }

  void
#if 0
  flux_barrier_all_on_stream(
      cudaStream_t stream,
      paddle::optional<std::vector<DenseTensor>> sync_buffers,
      paddle::optional<int> rank) {
#endif
  flux_barrier_all_on_stream() {
  #ifdef FLUX_SHM_USE_NVSHMEM
    nvshmemx_barrier_all_on_stream(stream);
  #else
    std::vector<int32_t *> sync_buffer_ptrs;
    // std::vector<DenseTensor>& sync_buffers_val = sync_buffers.get();
    // FLUX_CHECK(sync_buffers_val[rank.get()].initialized());

    int world_size = sync_buffers.size();
    for (size_t i = 0; i < sync_buffers.size(); i++) {
      sync_buffer_ptrs.push_back(reinterpret_cast<int32_t *>(sync_buffers[i].data()));
    }
    phi::dynload::cudaipc_barrier_all_on_stream_impl_capi(dev_ctx.stream(), sync_buffer_ptrs.data(), rank, world_size);
  #endif
  }

};

template<typename T, typename Context>
void GemmReduceScatterKernel(const Context& dev_ctx,
                  const DenseTensor& input,
                  const DenseTensor& weight,
                  const paddle::optional<DenseTensor>& bias,
                  const paddle::optional<DenseTensor>& input_scale,
                  const paddle::optional<DenseTensor>& weight_scale,
                  const paddle::optional<DenseTensor>& output_scale,
#if 0
                  const std::vector<const DenseTensor*>& output_buffers,
                  const std::vector<const DenseTensor*>& reduce_buffers,
                  const std::vector<const DenseTensor*>& barrier_buffers,
                  const std::vector<const DenseTensor*>& sync_buffers,
#endif
                  const int32_t nnodes,
                  const int32_t max_m,
                  const int32_t n_dim,
                  bool transpose_weight,
                  bool fuse_reduction,
                  int ring_id,
                  int root_id,
                  int nranks,
                  DenseTensor* output) {
  PADDLE_ENFORCE_GE(
      root_id,
      0,
      common::errors::InvalidArgument(
          "The root_id (%d) for c_scatter_op must be non-negative.", root_id));
  PADDLE_ENFORCE_GE(
      ring_id,
      0,
      common::errors::InvalidArgument(
          "The ring_id (%d) for c_scatter_op must be non-negative.", ring_id));

  auto map = paddle::distributed::ProcessGroupMapFromGid::getInstance();

  paddle::distributed::ProcessGroup* pg = map->get(ring_id);

  PADDLE_ENFORCE_NE(pg,
                    nullptr,
                    common::errors::Unavailable(
                        "ProcessGroup is nullptr."));

  // umiswing: idk why it's called comm_ctx, but it is the name in source code of process group.
  // const phi::GPUContext* comm_ctx = static_cast<phi::GPUContext*>(pg->GetDeviceContext(input.place(), /*use_calc_stream=*/false));
  const phi::GPUContext* comm_ctx = static_cast<phi::GPUContext*>(pg->GetGemmRSContext(input.place()));

  PADDLE_ENFORCE_NE(comm_ctx,
                    nullptr,
                    common::errors::Unavailable(
                        "comm_ctx is nullptr."));

  static int num_blocks = 12;
  static bool use_barrier_queue = false;
  // static bool use_gemmk = no_nvlink;
  static bool use_gemmk = false;
  static bool use_cudaMemcpyAsync = false;
  static int n_split = 1;
  // static bool per_tile_flags = get_bool_from_env("FLUX_RS_PER_TILE_FLAGS", no_nvlink);
  static bool per_tile_flags = false;
  const bool fast_accum = false;

  GemmRSWrapper<T, T> gemm_rs_wrapper{dev_ctx,
                                      pg,
                                      comm_ctx,
                                      nnodes,
                                      max_m,
                                      n_dim,
                                      transpose_weight,
                                      fuse_reduction};

  const int32_t m = input.dims()[0];
  const int32_t k = input.dims()[1];
  const int32_t n = transpose_weight ? weight.dims()[1] : weight.dims()[0];
  const int32_t wk = transpose_weight ? weight.dims()[0] : weight.dims()[1];

  auto launcher = [&](const bool get_workspace_size_flag,
                      const bool get_barrier_workspace_size) -> size_t {
    return phi::dynload::gemm_rs(
        input.data(),
        weight.data(),
        bias.is_initialized() ? bias->data() : nullptr,
        input_scale.is_initialized() ? input_scale->data() : nullptr,
        weight_scale.is_initialized() ? weight_scale->data() : nullptr,
        output_scale.is_initialized() ? output_scale->data() : nullptr,
        gemm_rs_wrapper.gemm_buffer.initialized() ? gemm_rs_wrapper.gemm_buffer.data() : nullptr,
        gemm_rs_wrapper.reduce_buffer_ptrs.data(),
        gemm_rs_wrapper.output_scatter_ptrs.data(),
        gemm_rs_wrapper.barrier_ptrs.data(),
        gemm_rs_wrapper.output_buffer_ptrs.data(),
        gemm_rs_wrapper.barrier_buffer_ptrs.data(),
        gemm_rs_wrapper.sync_buffer_ptrs.data(),
        m, n,
        k,
        wk,
        gemm_rs_wrapper.nnodes,
        gemm_rs_wrapper.max_m,
        gemm_rs_wrapper.n_dim,
        gemm_rs_wrapper.rank,
        gemm_rs_wrapper.world_size,
        gemm_rs_wrapper.local_world_size,
        gemm_rs_wrapper.local_rank,
        gemm_rs_wrapper.node_idx,
        num_blocks,
        n_split,
        fast_accum,
        std::is_same<T, phi::dtype::bfloat16>::value,
        gemm_rs_wrapper.transpose_weight,
        gemm_rs_wrapper.fuse_reduction,
        gemm_rs_wrapper.use_1d_ring,
        gemm_rs_wrapper.use_p2p_read,
        gemm_rs_wrapper.is_fp8_gemm,
        use_barrier_queue,
        use_gemmk,
        use_cudaMemcpyAsync,
        per_tile_flags,
        gemm_rs_wrapper.no_nvlink,
        get_workspace_size_flag,
        get_barrier_workspace_size,
        dev_ctx.stream(),
        gemm_rs_wrapper.rs_stream_,
        gemm_rs_wrapper.event_);
  };

  auto get_workspace_size = [&]() -> size_t {
    return launcher(true, false);
  };

  auto get_barrier_workspace_size = [&]() -> size_t {
    return launcher(false, true);
  };

  size_t workspace_size = get_workspace_size();
  size_t barrier_workspace_size = get_barrier_workspace_size();

  gemm_rs_wrapper.lazy_init_gemm_buffer(workspace_size);
  gemm_rs_wrapper.lazy_init_barrier_buffer(barrier_workspace_size);

  auto gemm_rs = [&]() {
    launcher(false, false);
  };

  gemm_rs();

  // reduce impl
  // int local_world_size = world_size / gemm_rs_wrapper.nnodes;
  DenseTensor output_3d;
  phi::ViewShapeKernel(dev_ctx,
                       gemm_rs_wrapper.reduce_buffer,
                       // {gemm_rs_wrapper.nnodes, local_world_size, m / gemm_rs_wrapper.world_size, n},
                       {gemm_rs_wrapper.world_size, m / gemm_rs_wrapper.world_size, n},
                       &output_3d);
  
  // nnodes == 1
  // TODO(umiswing): only 1 node, so just hack bcz idk how to do inplace scatter in paddle.
  // DenseTensor output;
  output->Resize(common::make_dim(m / gemm_rs_wrapper.world_size, n));
  dev_ctx.template Alloc<T>(output);
  phi::SumKernel<T>(dev_ctx,
                    output_3d,
                    {0},
                    gemm_rs_wrapper.output_dtype,
                    false,
                    output);
}

} // namespace phi

PD_REGISTER_KERNEL(gemm_reduce_scatter,
                   GPU,
                   ALL_LAYOUT,
                   phi::GemmReduceScatterKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
}
