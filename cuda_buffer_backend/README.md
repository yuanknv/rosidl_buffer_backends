# cuda_buffer_backend

CUDA buffer backend plugin for the ROS 2 Buffer system. Enables zero-copy GPU memory sharing between publishers and subscribers on the same host using CUDA VMM (Virtual Memory Management).

## Build

Requires a ROS 2 Rolling source workspace; see
[Building ROS 2 on Ubuntu](https://docs.ros.org/en/rolling/Installation/Alternatives/Ubuntu-Development-Setup.html)
for the canonical setup. After cloning this repo into your workspace's
`src/` directory:

```bash
# Install system dependencies (CUDA toolkit, etc.).
rosdep install --from-paths src --ignore-src -y \
  --skip-keys "fastcdr rti-connext-dds-7.7.0 urdfdom_headers qt6-svg-dev"

# Build the CUDA backend.
colcon build --symlink-install --packages-up-to cuda_buffer_backend
source install/setup.sh
```

## Test

```bash
colcon test --packages-select cuda_buffer cuda_buffer_backend
colcon test-result --verbose
```

## Packages

| Package | Description |
|---|---|
| `cuda_buffer` | Core CUDA buffer implementation: memory pool, IPC manager, host endpoint manager, and user-facing `allocate_msg`/`from_buffer`/`to_buffer` APIs |
| `cuda_buffer_backend` | Plugin registration via `pluginlib`, endpoint discovery, and descriptor serialization |
| `cuda_buffer_backend_msgs` | ROS 2 message definition for `CudaBufferDescriptor` |

## Usage

### Publisher (direct write, zero-copy)

```cpp
#include "cuda_buffer/cuda_buffer_api.hpp"
#include "sensor_msgs/msg/image.hpp"

const size_t data_size = 640 * 480 * 3;

sensor_msgs::msg::Image msg =
  cuda_buffer_backend::allocate_msg<sensor_msgs::msg::Image>(data_size);
msg.height = 480;
msg.width = 640;
msg.encoding = "rgb8";
msg.step = 640 * 3;

cuda_buffer_backend::WriteHandle wh =
  cuda_buffer_backend::from_buffer(msg.data, stream);
my_kernel<<<...>>>(wh.get_ptr(), ...);

publisher->publish(msg);
// wh destructor records write_event on stream when it goes out of scope
```

### Publisher (copy from existing pointer)

Use `to_buffer` to copy bytes from an existing pointer (host or device) into
a buffer that was already allocated (e.g. via `allocate_msg`). `to_buffer`
is a plain memcpy-through-a-WriteHandle and does **not** allocate.

```cpp
sensor_msgs::msg::Image msg =
  cuda_buffer_backend::allocate_msg<sensor_msgs::msg::Image>(data_size);
msg.height = 480;
msg.width = 640;
msg.encoding = "rgb8";
msg.step = 640 * 3;

{
  cuda_buffer_backend::WriteHandle wh =
    cuda_buffer_backend::from_buffer(msg.data, stream);

  // From a device pointer (D2D copy, default kind)
  cuda_buffer_backend::to_buffer(gpu_ptr, data_size, wh, stream);

  // Or from a host pointer (H2D copy)
  // cuda_buffer_backend::to_buffer(
  //   host_ptr, data_size, wh, stream, cudaMemcpyHostToDevice);
}  // wh destructor records the write event on `stream`

publisher->publish(msg);
```

### Subscriber (read from buffer, zero-copy)

```cpp
#include "cuda_buffer/cuda_buffer_api.hpp"

void callback(const sensor_msgs::msg::Image::SharedPtr msg) {
  const rosidl::Buffer<uint8_t> & data = msg->data;
  cuda_buffer_backend::ReadHandle rh =
    cuda_buffer_backend::from_buffer(data, stream);
  // ReadHandle constructor waits on publisher's write_event

  my_kernel<<<...>>>(rh.get_ptr(), ...);
}  // ReadHandle destructor signals publisher that GPU work is complete
```

### Auto-promoting non-CUDA buffers

`from_buffer` accept any `rosidl::Buffer<T>`, not just
CUDA-backed ones. If the source is a non-CUDA buffer (e.g. the CPU fallback
path), `from_buffer` allocates a new CUDA-backed `rosidl::Buffer<uint8_t>`
and returns a handle for it.

```cpp
#include "cuda_buffer/cuda_buffer_api.hpp"

void callback(const sensor_msgs::msg::Image::SharedPtr msg) {
  const rosidl::Buffer<uint8_t> & data = msg->data;
  cuda_buffer_backend::ReadHandle rh =
    cuda_buffer_backend::from_buffer(data, stream);

  my_kernel<<<...>>>(rh.get_ptr(), ...);
}
```

### `from_buffer` handle rules

`from_buffer` returns a **WriteHandle** when called with a non-const buffer, or a
**ReadHandle** when called with a const buffer. The overload is selected at compile
time based on const-ness of the reference:

```cpp
// Write path (publisher):
cuda_buffer_backend::WriteHandle wh = cuda_buffer_backend::from_buffer(msg.data, stream);

// Read path (subscriber):
const rosidl::Buffer<uint8_t> & data = msg->data;
cuda_buffer_backend::ReadHandle rh = cuda_buffer_backend::from_buffer(data, stream);
```

- A **WriteHandle** can only be acquired once per buffer. Attempting to acquire
  a second WriteHandle (or acquiring one after finalization) throws `CudaError`.
- To read a received buffer, always pass a **const reference**.
- If the source buffer is non-CUDA, the handle owns the promoted CUDA buffer;
  call `handle.get_promoted_buffer()` to retrieve it.

## IPC Behavior

The RMW layer calls `on_discovering_endpoint()` for each subscriber to decide between zero-copy IPC and CPU fallback:

| Condition | Path |
|---|---|
| Same host, same GPU, same user | Zero-copy via CUDA VMM IPC |
| Different GPU, different user, different host, or VMM unavailable | CPU fallback via `to_cpu()` |

The publisher's pool checks a shared-memory refcount before recycling a block, ensuring all IPC subscribers have released their handles.

## License

Apache-2.0
