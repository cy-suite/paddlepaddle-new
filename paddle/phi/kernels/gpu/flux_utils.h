namespace phi {
inline void *
ptr_offset(void *ptr, ptrdiff_t offset) {
  return static_cast<char *>(ptr) + offset;
}

// All2All for nvlink mode. for NVLINK machine, default is 0
// Ring1D for 1d-ring. for PCI-e machine without GPUs cross NUMA nodes use ring 1d
// Ring2D for 2d-ring. for PCI-e machine with GPUs cross NUMA nodes defaults to ring_2d
// RingCustom for custom ring. for defining arbitrary ring at compile time
enum class AGRingMode {
  All2All = 0,
  Ring1D = 1,
  Ring2D = 2,
  RingCustom = 3,
  Auto = -1,
};

static AGRingMode
get_ring_mode(AGRingMode ring_mode) {
      return AGRingMode::All2All;
}

class CUDAEventHolder {
public:
  CUDAEventHolder(const bool disable_timing = false) {
    if(disable_timing) {
      PADDLE_ENFORCE_GPU_SUCCESS(cudaEventCreateWithFlags(&this->event, cudaEventDisableTiming));
    } else {
      PADDLE_ENFORCE_GPU_SUCCESS(cudaEventCreate(&event));
    }
  }

  ~CUDAEventHolder() {
    PADDLE_ENFORCE_GPU_SUCCESS(cudaEventDestroy(event));
  }
  cudaEvent_t event;
};

class BuffersHolder {
private:
  const GPUContext& dev_ctx;
  paddle::distributed::ProcessGroup* tp_group;
  size_t world_size;
  std::vector<void*> ptrs;
  size_t size_in_bytes;
  size_t offset_in_bytes;
  void * ptr;
  phi::DenseTensor local_tensor;
public:

  BuffersHolder(const std::vector<std::pair<const phi::DataType, const std::vector<int64_t>>>& shapes,
                const GPUContext& dev_ctx_,
                paddle::distributed::ProcessGroup* tp_group_) :
    dev_ctx(dev_ctx_),
    tp_group(tp_group_),
    world_size(tp_group->GetSize()),
    ptrs(world_size, nullptr) {

    this->size_in_bytes = calc_size(shapes);
    this->offset_in_bytes = 0;
    alloc();
  }

  void clear() {
    this->offset_in_bytes = 0;
  }

  void reserve(const std::vector<std::pair<const phi::DataType, const std::vector<int64_t>>>& shapes) {
    size_t require_size = calc_size(shapes);
    if(require_size > this->size_in_bytes) {
      this->size_in_bytes = require_size;
      release();
      alloc();
    }
    this->clear();
  }

  // TODO(umiswing): although BuffersHolder object is static, it's better to find a way to destruct it.

  std::vector<DenseTensor> get_buffers(const std::pair<phi::DataType, std::vector<int64_t>>& shape) {
    PADDLE_ENFORCE_LT(
      this->offset_in_bytes,
      this->size_in_bytes,
      common::errors::InvalidArgument("BuffersHolder is full!"));
    std::vector<DenseTensor> tensors(tp_group->GetSize(), DenseTensor{});
    for (int i = 0; i < tp_group->GetSize(); ++i) {
        DenseTensor tensor;
        tensor =
            from_blob(ptr_offset(ptrs[i], this->offset_in_bytes),
                      shape.second,
                      shape.first,
                      dev_ctx.GetPlace(),
                      [](phi::Allocation* allocation) {});
        tensors[i] = tensor;
    }

    this->offset_in_bytes += this->calc_size(shape);
    PADDLE_ENFORCE_LE(
      this->offset_in_bytes,
      this->size_in_bytes,
      common::errors::InvalidArgument("buffer out of bound!"));

    return tensors;

  }

private:
  void alloc() {
    local_tensor = phi::Empty<uint8_t>(dev_ctx, {static_cast<int64_t>(this->size_in_bytes)});
    phi::funcs::SetConstant<GPUContext, uint8_t> set_zero;
    set_zero(this->dev_ctx, &local_tensor, static_cast<int32_t>(0));
    this->ptr = local_tensor.data();

    cudaIpcMemHandle_t handle;
    PADDLE_ENFORCE_GPU_SUCCESS(cudaIpcGetMemHandle(&handle, ptr));

    DenseTensor handle_d = phi::Empty<uint8_t>(dev_ctx, {sizeof(cudaIpcMemHandle_t)});
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpy(
        handle_d.data(), &handle, sizeof(cudaIpcMemHandle_t), cudaMemcpyHostToDevice));
    long int handles_shape = sizeof(cudaIpcMemHandle_t) * tp_group->GetSize();
    DenseTensor handles_d = phi::Empty<uint8_t>(dev_ctx, {handles_shape});
    tp_group->AllGather(&handles_d, handle_d, 0, -1, true, true)->Wait();

    std::vector<cudaIpcMemHandle_t> handles_h(tp_group->GetSize());
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpy(
        handles_h.data(),
        handles_d.data(),
        sizeof(cudaIpcMemHandle_t) * tp_group->GetSize(),
        cudaMemcpyDeviceToHost));

    for (int i = 0; i < tp_group->GetSize(); ++i) {
      if (i != tp_group->GetRank()) {
          PADDLE_ENFORCE_GPU_SUCCESS(cudaIpcOpenMemHandle(&ptrs[i], handles_h[i], cudaIpcMemLazyEnablePeerAccess));
      } else {
        ptrs[i] = ptr;
      }
    }
    int64_t device_id = dev_ctx.GetPlace().GetDeviceId();
    distributed::BarrierOptions opts{};
    opts.device_id = device_id;
    this->tp_group->Barrier(opts)->Wait();
  }

  size_t calc_numel(const std::vector<int64_t>& shape) {
    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
  }

  size_t calc_size(const std::pair<const phi::DataType, const std::vector<int64_t>>& shape) {
    return SizeOf(shape.first) * calc_numel(shape.second);
  }

  size_t calc_size(const std::vector<std::pair<const phi::DataType, const std::vector<int64_t>>>& shapes) {
    size_t size = 0;
    for(const auto& shape : shapes) {
      size += calc_size(shape);
    }
    return size;
  }

  void release() {
    for(int i=0; i<world_size; ++i) {
      if(i != this->tp_group->GetRank()) {
        PADDLE_ENFORCE_GPU_SUCCESS(cudaIpcCloseMemHandle(this->ptrs[i]));
      }
    }
    int64_t device_id = dev_ctx.GetPlace().GetDeviceId();
    distributed::BarrierOptions opts{};
    opts.device_id = device_id;
    this->tp_group->Barrier(opts)->Wait();

    for(int i=0; i<world_size; ++i) {
      this->ptrs[i] = nullptr;
    }
  }

  using Deleter = void (*)(phi::Allocation*);
  DenseTensor from_blob(void *data,
                        const std::vector<int64_t>& shape,
                        phi::DataType dtype,
                        phi::Place place,
                        const Deleter& deleter,
                        phi::DataLayout layout = phi::DataLayout::NCHW ) {
    PADDLE_ENFORCE_NOT_NULL(
        data, common::errors::InvalidArgument("data can not be nullptr."));
  
    auto meta =
        phi::DenseTensorMeta(dtype, common::make_ddim(shape), layout);
  
    size_t size = SizeOf(dtype) * (meta.is_scalar ? 1 : product(meta.dims));
  
    auto alloc =
        std::make_shared<phi::Allocation>(data, size, deleter, place/*data_place*/);
  
    return DenseTensor(alloc, meta);
  }
};

namespace flux {
  template<typename T>
  phi::DataType dtype() {
    if(std::is_same<T, int32_t>::value) return phi::DataType::INT32;
    if(std::is_same<T, phi::bfloat16>::value) return phi::DataType::BFLOAT16;
    if(std::is_same<T, phi::float16>::value) return phi::DataType::FLOAT16;
    if(std::is_same<T, uint8_t>::value) return phi::DataType::UINT8;

    PADDLE_THROW(common::errors::Unimplemented(
        "unsupported buffer type "
        ));
  }

  static void RaiseNotSupportedError() {
    PADDLE_THROW(common::errors::Unimplemented(
        "Flux is unsupported, please check "
        "the GPU compability and CUDA Version."
        ));
  }
} // namespace flux
} // namespace phi
