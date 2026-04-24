# torch_tensor_bridge (prototype, DLPack-aligned)

Prototype alternative to `torch_buffer_backend`. Instead of introducing a
framework-named `"torch"` buffer backend that wraps the existing CUDA/CPU
buffer with tensor metadata plus a dedicated descriptor for RMW
serialization, this prototype treats a tensor as a **plain, DLPack-shaped
ROS 2 message** and provides a small bridge library that converts
directly between that message and an `at::Tensor`.

The message schema follows [DLPack](https://dmlc.github.io/dlpack/latest/)
exactly, so any DLPack-compatible framework (PyTorch, TensorFlow, JAX,
CuPy, ONNX Runtime, TensorRT, MXNet, RAPIDS, ...) can plug in via a
thin wrapper and interoperate over the wire without re-encoding
metadata.

## Packages

| Package | Description |
|---|---|
| `torch_tensor_msgs` | Defines `Tensor.msg` with DLPack-aligned fields: `{dtype_code, dtype_bits, dtype_lanes}`, `{device_type, device_id}`, `shape[]`, `strides[]`, `byte_offset`, `data[]` |
| `torch_tensor_bridge` | Header-only library: `allocate_tensor`, `from_tensor_msg`, `to_tensor_msg`, `StreamGuard`, and DLPack-dtype/device helpers |

There is **no** pluginlib plugin, no `BufferImplBase` subclass, and no
custom descriptor. The `uint8[] data` field maps to
`rosidl::Buffer<uint8_t>`, which transparently uses whichever buffer
backend is registered at runtime (`cuda_buffer_backend` for GPU
zero-copy, CPU fallback otherwise).

## The `Tensor.msg` schema

```
# DLDataType
uint8  dtype_code        # DLPack DLDataTypeCode: 0=Int, 1=UInt, 2=Float, 4=BFloat, 6=Bool, ...
uint8  dtype_bits        # 8, 16, 32, 64, ...
uint16 dtype_lanes       # SIMD lanes; 1 for plain scalar

# DLDevice
int32 device_type        # DLPack DLDeviceType: 1=CPU, 2=CUDA, 3=CUDAHost, 10=ROCm, 13=CUDAManaged, ...
int32 device_id          # device ordinal

# DLTensor
int64[] shape
int64[] strides          # empty = contiguous (DLPack nullptr convention)
uint64  byte_offset      # view offset into `data`

# Underlying storage (may be larger than numel * element_size for views)
uint8[] data
```

This is a field-for-field transcription of `DLTensor`. Producing or
consuming a `DLManagedTensor` on either side is near-trivial.

## Build

```bash
# CUDA path (recommended): build cuda_buffer_backend first.
colcon build --symlink-install --packages-up-to cuda_buffer_backend
source install/setup.sh

colcon build --symlink-install --packages-up-to torch_tensor_bridge
source install/setup.sh
```

## Usage

### Publisher

```cpp
torch_tensor_bridge::StreamGuard guard = torch_tensor_bridge::set_stream();

auto msg = torch_tensor_bridge::allocate_tensor(
  {480, 640, 3}, torch::kByte);   // dtype_code=1(UInt), bits=8, device=CUDA
{
  at::Tensor t = torch_tensor_bridge::from_tensor_msg(msg);
  my_pipeline(t);                 // runs on the guarded stream
}
publisher->publish(msg);
```

### Subscriber

```cpp
void cb(const torch_tensor_msgs::msg::Tensor::SharedPtr msg) {
  torch_tensor_bridge::StreamGuard guard = torch_tensor_bridge::set_stream();
  at::Tensor in = torch_tensor_bridge::from_tensor_msg(*msg, /*clone=*/false);
  auto out = model(in);
}
```

### View transport (zero-copy slice)

```cpp
auto msg = torch_tensor_bridge::allocate_tensor({16}, torch::kInt, c10::kCPU);
// ... fill the full buffer ...

// Publish only positions [4..7] as a 4-int view into the same storage.
msg.shape = {4};
msg.strides = {1};
msg.byte_offset = 4 * sizeof(int32_t);
publisher->publish(msg);
```

The subscriber's `from_tensor_msg` honors `byte_offset`, so the view
materializes without copying.

## Cross-framework interop

Because the schema is DLPack-shaped, adding a TensorFlow or JAX bridge
is a small wrapper per framework, consuming the same `Tensor.msg`:

```
  torch_tensor_bridge  ─┐
  tf_tensor_bridge    ─┼──  torch_tensor_msgs/Tensor  ──  cuda_buffer_backend
  jax_tensor_bridge   ─┘                                   cpu fallback
```

Publisher written against torch, subscriber written against TensorFlow:
the subscriber decodes the DLPack triple, imports the bytes into a
`tf::Tensor` via DLPack, and zero-copy GPU transport is delivered by
`cuda_buffer_backend` underneath. No new message, no backend
negotiation, no N&times;N descriptor-compatibility matrix.

## Design comparison

|                                 | `torch_buffer_backend` | `torch_tensor_bridge` |
|---|---|---|
| New buffer backend plugin                      | Yes (`"torch"` over `"cuda"`/`"cpu"`)          | No |
| New descriptor message                         | `TorchBufferDescriptor` (metadata + nested bytes) | No (metadata is `Tensor.msg` fields) |
| `BufferImplBase` subclass                      | `TorchBufferImpl<T>`                            | None |
| Metadata visible to `ros2 topic echo` / bags / Python | No                                       | Yes |
| Dtype encoding                                 | String (`"Float"`, `"Byte"`, ...)               | DLPack `{code, bits, lanes}` |
| Device on the wire                             | Implicit (current CUDA context)                 | Explicit `{device_type, device_id}` |
| Strided / view transport                       | Not supported (always `.contiguous()`)          | Supported via `byte_offset` |
| Cross-framework interop                        | Silo per framework (descriptors don't cross) | Any DLPack-compatible framework via `Tensor.msg` |
| GPU zero-copy IPC path                         | Delegates to `cuda_buffer_backend`              | Delegates to `cuda_buffer_backend` |
| Extensions for quantized / sparse / ragged     | Requires descriptor rev                         | Compose new ROS messages around `Tensor.msg` |

## Testing

```bash
colcon test --packages-select torch_tensor_msgs torch_tensor_bridge
colcon test-result --verbose
```

The gtest covers the CPU path end-to-end, including a `byte_offset`
view round-trip. The two launch tests (`_intra_`, `_inter_pubsub_fastrtps_launch.py`)
exercise pub/sub through RMW; with `cuda_buffer_backend` installed, the
`data` buffer is CUDA-backed and the inter-process test goes through
CUDA IPC. Without it, `data` falls back to CPU via CDR.

## Guardrails against the "works locally, slow remotely" failure mode

Because `Tensor.msg` makes shape / dtype / device **visible** in the
schema, standard policy tooling can detect over-the-wire tensor use:

- CI lints that flag messages with `Tensor` fields subscribed across
  hosts without explicit opt-in.
- RMW-level policy extensions (via `on_discovering_endpoint` of the
  underlying `cuda_buffer_backend`) to refuse cross-host bridging of
  large device-backed payloads by default.
- One-shot logs on subscription match reporting resolved transport
  (`CUDA-IPC` vs `CDR fallback`) so silent CPU fallbacks become
  observable at deploy time rather than at debugging time.

Adding these guardrails is out of scope for this prototype but is
straightforward on top of the explicit DLPack metadata.

## License

Apache-2.0
