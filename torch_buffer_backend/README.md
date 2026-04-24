# torch_buffer_backend

PyTorch buffer backend plugin for the ROS 2 Buffer system. Wraps CUDA/CPU device buffers with tensor metadata (shape, strides, dtype) and provides `allocate_msg`, `from_buffer`, and `to_buffer` APIs for zero-copy GPU tensor sharing.

## Build

Requires a ROS 2 Rolling source workspace; see
[Building ROS 2 on Ubuntu](https://docs.ros.org/en/rolling/Installation/Alternatives/Ubuntu-Development-Setup.html)
for the canonical setup. The torch packages auto-detect available device
backends at compile time: for GPU acceleration, build and source
`cuda_buffer_backend` first so its CMake config is visible when
`torch_buffer` is configured. Without it, `torch_buffer` falls back to CPU
automatically.

After cloning this repo into your workspace's `src/` directory:

```bash
# Install system dependencies.
rosdep install --from-paths src --ignore-src -y \
  --skip-keys "fastcdr rti-connext-dds-7.7.0 urdfdom_headers qt6-svg-dev"

# (Optional) Build the CUDA device backend first for GPU support.
colcon build --symlink-install --packages-up-to cuda_buffer_backend && \
  source install/setup.sh

# Build the torch packages.
colcon build --symlink-install --packages-up-to torch_buffer_backend
source install/setup.sh
```

`libtorch_vendor` auto-detects the CUDA variant (`cu118`/`cu121`/`cu124`/`cu126`/`cu128`/`cu129`, else `cpu`) and picks the latest LibTorch release available for it; override with `--cmake-args -DLIBTORCH_CUDA_VERSION=cu124` or `-DLIBTORCH_VERSION=2.6.0` if needed.

## Test

```bash
colcon test --packages-select torch_buffer torch_buffer_backend
colcon test-result --verbose
```

> **Note:** Tests validate the CPU backend path only, since
> `torch_buffer_backend` has no explicit dependency on any device backend
> package. To verify GPU IPC end-to-end, use the
> [torch_backend_demo](https://github.com/yuanknv/torch_backend_demo).

## Packages

| Package | Description |
|---|---|
| `torch_buffer` | Core TorchBufferImpl wrapping device buffers with tensor metadata, and user-facing `allocate_msg`/`from_buffer`/`to_buffer` APIs |
| `torch_buffer_backend` | Plugin registration via `pluginlib`, endpoint discovery, and descriptor serialization |
| `torch_buffer_backend_msgs` | ROS 2 message definition for `TorchBufferDescriptor` |

## Usage

### Stream Setup

The user is responsible for setting a non-default CUDA stream before calling
`allocate_msg`, `to_buffer`, or `from_buffer`. This ensures the same stream is
shared between buffer operations and the user's own operators (model inference,
custom kernels, etc.), enabling correct event-based asynchronous synchronization.

A convenience helper is provided:

```cpp
torch_buffer_backend::StreamGuard guard = torch_buffer_backend::set_stream();
// All buffer operations AND user operators in this scope share the same stream.
```

This auto-detects CUDA/ROCm and sets a non-default stream from the pool. On
CPU-only builds it is a no-op. If no stream is set, a warning is logged and
operations fall back to synchronous execution on the default stream.

### Device Selection

`allocate_msg` accepts an optional device parameter. If omitted, it auto-detects
the best available device backend at runtime:

```cpp
// Auto-detect the best available device backend
auto msg = torch_buffer_backend::allocate_msg<sensor_msgs::msg::Image>(
  {480, 640, 3}, torch::kByte);

// Explicit device override (when the caller needs a specific device)
auto msg_gpu = torch_buffer_backend::allocate_msg<sensor_msgs::msg::Image>(
  {480, 640, 3}, torch::kByte, c10::kCUDA);
```

### Publisher: `allocate_msg` + `from_buffer` (direct write, zero-copy)

`from_buffer` on a non-const buffer returns a mutable tensor view of the
buffer's memory.

```cpp
#include "torch_buffer/torch_buffer_api.hpp"
#include "sensor_msgs/msg/image.hpp"

torch_buffer_backend::StreamGuard guard = torch_buffer_backend::set_stream();

sensor_msgs::msg::Image msg = torch_buffer_backend::allocate_msg<sensor_msgs::msg::Image>(
  {480, 640, 3}, torch::kByte);
msg.height = 480;
msg.width = 640;
msg.encoding = "rgb8";
msg.step = 640 * 3;

{
  at::Tensor output = torch_buffer_backend::from_buffer(msg.data);
  my_pipeline(output);  // user operators run on the same stream
}  // tensor destroyed -> WriteHandle records write event on stream

publisher->publish(msg);
```

### Publisher: `to_buffer` (copy from existing tensor)

`to_buffer(buffer, tensor)` copies a tensor's data into a pre-allocated
torch-backed `rosidl::Buffer` and updates the buffer's tensor metadata
(shape, strides, dtype) to match. It does **not** allocate: the caller is
expected to first obtain a buffer from `allocate_msg`. The copy kind
depends on the source and destination devices.

```cpp
torch_buffer_backend::StreamGuard guard = torch_buffer_backend::set_stream();

sensor_msgs::msg::Image msg = torch_buffer_backend::allocate_msg<sensor_msgs::msg::Image>(
  {480, 640, 3}, torch::kByte);
msg.height = 480;
msg.width = 640;
msg.encoding = "rgb8";
msg.step = 640 * 3;

at::Tensor result = model(input);
torch_buffer_backend::to_buffer(msg.data, result);  // copy into msg.data

publisher->publish(msg);
```

### Subscriber: `from_buffer` (read input tensor)

On the subscriber side (const buffer) `from_buffer` defaults to returning an
independent copy of the buffer contents so subscribers can safely mutate
the tensor without corrupting shared memory. For CUDA this is a D2D clone;
for CPU it is a host copy. If the subscriber is certain it will not mutate
the tensor in place, pass `clone=false` for a zero-copy view.

```cpp
#include "torch_buffer/torch_buffer_api.hpp"

void callback(const sensor_msgs::msg::Image::SharedPtr msg) {
  torch_buffer_backend::StreamGuard guard = torch_buffer_backend::set_stream();

  const rosidl::Buffer<uint8_t> & data = msg->data;
  at::Tensor input = torch_buffer_backend::from_buffer(data);
  at::Tensor result = model(input);
}
```

## Event Lifecycle

All CUDA synchronization is managed by the backend via WriteHandle/ReadHandle RAII:

| Stage | Mechanism |
|---|---|
| Publisher writes to buffer | WriteHandle in tensor deleter records write event on destroy |
| Subscriber reads from buffer | ReadHandle waits on write event at construction, records read event on destroy |
| Buffer recycled to pool | CudaBuffer destructor waits on all read events before returning to pool |

No `cudaStreamSynchronize` in the pipeline.

## Inter-process Behavior

For inter-process communication, the TorchBufferDescriptor carries the inner device buffer via endpoint-aware serialization:

| Condition | Path |
|---|---|
| Same host, CUDA VMM available | Zero-copy via CUDA IPC (nested CudaBufferDescriptor with IPC handles) |
| CPU buffer or VMM unavailable | CPU serialization via CDR |

## License

Apache-2.0
