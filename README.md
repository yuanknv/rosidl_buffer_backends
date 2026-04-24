# rosidl_buffer_backends

CUDA and PyTorch buffer backend implementations for `rosidl::Buffer`,
enabling zero-copy GPU memory sharing between ROS 2 publishers and
subscribers.

## Packages

- **cuda_buffer** -- Core CUDA buffer library (VMM-backed IPC memory pool,
  host endpoint manager, ReadHandle/WriteHandle with CUDA event sync).
- **cuda_buffer_backend** -- BufferBackend plugin for CUDA IPC transport.
- **cuda_buffer_backend_msgs** -- ROS 2 message definitions for CUDA buffer
  descriptors.
- **libtorch_vendor** -- Vendor package that downloads and installs the
  pre-built LibTorch C++ distribution.
- **torch_buffer** -- Device-agnostic PyTorch buffer library wrapping device
  backends with tensor metadata (shape, strides, dtype).
- **torch_buffer_backend** -- BufferBackend plugin for PyTorch tensors.
- **torch_buffer_backend_msgs** -- ROS 2 message definitions for Torch buffer
  descriptors.

## Prerequisites

- A ROS 2 Rolling development environment. See the upstream
  [Building ROS 2 on Ubuntu](https://docs.ros.org/en/rolling/Installation/Alternatives/Ubuntu-Development-Setup.html)
  guide for the canonical source-build flow, or use the pixi workflow
  shipped by the [`ros2/ros2`](https://github.com/ros2/ros2) meta-repo.
- CUDA Toolkit (>= 11.8) on the host.
- LibTorch: provided automatically by `libtorch_vendor` at build time if a
  system LibTorch isn't already visible.

Per-package build, test, and run details live in each backend's README:

- [`cuda_buffer_backend/README.md`](cuda_buffer_backend/README.md)
- [`torch_buffer_backend/README.md`](torch_buffer_backend/README.md)
- Demo: [`../rosidl_buffer_backends_tutorials/README.md`](../rosidl_buffer_backends_tutorials/README.md)

## API overview

### CUDA buffer backend (`cuda_buffer_backend`)

```cpp
#include "cuda_buffer/cuda_buffer_api.hpp"

// Publisher: allocate + write directly via kernel.
auto msg = cuda_buffer_backend::allocate_msg<sensor_msgs::msg::Image>(byte_count);
{
  auto wh = cuda_buffer_backend::from_buffer(msg.data, stream);
  my_kernel<<<...>>>(wh.get_ptr(), ...);
}  // wh destructor records the write event on `stream`

// Publisher: copy from an existing host/device pointer into a pre-allocated buffer.
{
  auto wh = cuda_buffer_backend::from_buffer(msg.data, stream);
  cuda_buffer_backend::to_buffer(host_ptr, byte_count, wh, stream,
    cudaMemcpyHostToDevice);
}

// Subscriber: read handle (waits on publisher's write event).
auto rh = cuda_buffer_backend::from_buffer(msg->data, stream);
use_data<<<...>>>(rh.get_ptr(), ...);

// Auto-promotion: passing a non-CUDA buffer allocates a fresh CUDA buffer
// and (for reads) copies H2D; the handle owns the new buffer via
// get_promoted_buffer().
auto rh_any = cuda_buffer_backend::from_buffer(cpu_or_other_buf, stream);
std::shared_ptr<rosidl::Buffer<uint8_t>> promoted = rh_any.get_promoted_buffer();
```

### Torch buffer backend (`torch_buffer_backend`)

```cpp
#include "torch_buffer/torch_buffer_api.hpp"

// Publisher: allocate + copy a tensor into the message.
auto msg = torch_buffer_backend::allocate_msg<sensor_msgs::msg::Image>(
  {H, W, C}, torch::kByte);
torch_buffer_backend::to_buffer(msg.data, tensor);

// Subscriber: safe default returns an independent clone.
at::Tensor t = torch_buffer_backend::from_buffer(msg->data);

// Subscriber: zero-copy view when the caller is certain it will not mutate
// the tensor in place. Caller must treat the returned tensor as read-only.
at::Tensor view = torch_buffer_backend::from_buffer(msg->data, /*clone=*/false);
```

The torch backend does not cross-device-promote: the returned tensor stays
on the same device as the underlying torch buffer (CUDA or CPU).

## License

Apache-2.0
