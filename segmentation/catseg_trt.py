"""Fixed-buffer TensorRT inference for the online CAT-Seg worker.

The exported CAT-Seg graph is split into a CLIP image encoder and a semantic
aggregator.  Both engines have fixed shapes, so all CUDA buffers are allocated
once at startup and reused for every frame.  This is important on Jetson Linux
r36.4.7, where repeated large cudaMalloc calls can hit the NvMap allocation
limit.
"""

from __future__ import annotations

import ctypes
import importlib
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np

DEFAULT_TENSORRT_PYTHON_PATH = "/usr/lib/python3.10/dist-packages"
INPUT_HEIGHT = 384
INPUT_WIDTH = 384


def _load_tensorrt(python_path: str | None):
    try:
        return importlib.import_module("tensorrt")
    except ImportError:
        if python_path and python_path not in sys.path:
            # Append rather than prepend: the catseg environment must keep its
            # NumPy-compatible OpenCV package ahead of the system cv2 package.
            sys.path.append(python_path)
        try:
            return importlib.import_module("tensorrt")
        except ImportError as second_error:
            raise RuntimeError(
                "TensorRT Python bindings are unavailable; tried the active "
                f"environment and {python_path!r}"
            ) from second_error


class _CudaAllocation:
    def __init__(self, runtime: "_CudaRuntime", nbytes: int):
        self._runtime = runtime
        self._pointer = ctypes.c_void_p()
        runtime.check(runtime.cudaMalloc(ctypes.byref(self._pointer), nbytes))

    def __int__(self) -> int:
        if self._pointer.value is None:
            raise RuntimeError("CUDA allocation has already been released")
        return self._pointer.value

    def close(self) -> None:
        if self._pointer.value is not None:
            self._runtime.check(self._runtime.cudaFree(self._pointer))
            self._pointer = ctypes.c_void_p()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class _CudaStream:
    def __init__(self, runtime: "_CudaRuntime"):
        self._runtime = runtime
        pointer = ctypes.c_void_p()
        runtime.check(runtime.cudaStreamCreate(ctypes.byref(pointer)))
        self.handle = pointer.value

    def synchronize(self) -> None:
        self._runtime.check(
            self._runtime.cudaStreamSynchronize(ctypes.c_void_p(self.handle))
        )

    def close(self) -> None:
        if self.handle:
            self._runtime.check(
                self._runtime.cudaStreamDestroy(ctypes.c_void_p(self.handle))
            )
            self.handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class _CudaRuntime:
    COPY_HOST_TO_DEVICE = 1
    COPY_DEVICE_TO_HOST = 2

    def __init__(self):
        self.library = None
        for library_name in ("libcudart.so", "libcudart.so.12"):
            try:
                self.library = ctypes.CDLL(library_name)
                break
            except OSError:
                continue
        if self.library is None:
            raise RuntimeError("could not load libcudart.so")

        self.cudaMalloc = self.library.cudaMalloc
        self.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.cudaMalloc.restype = ctypes.c_int
        self.cudaFree = self.library.cudaFree
        self.cudaFree.argtypes = [ctypes.c_void_p]
        self.cudaFree.restype = ctypes.c_int
        self.cudaMemcpyAsync = self.library.cudaMemcpyAsync
        self.cudaMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.cudaMemcpyAsync.restype = ctypes.c_int
        self.cudaStreamCreate = self.library.cudaStreamCreate
        self.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.cudaStreamCreate.restype = ctypes.c_int
        self.cudaStreamDestroy = self.library.cudaStreamDestroy
        self.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
        self.cudaStreamDestroy.restype = ctypes.c_int
        self.cudaStreamSynchronize = self.library.cudaStreamSynchronize
        self.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        self.cudaStreamSynchronize.restype = ctypes.c_int
        self.cudaGetErrorString = self.library.cudaGetErrorString
        self.cudaGetErrorString.argtypes = [ctypes.c_int]
        self.cudaGetErrorString.restype = ctypes.c_char_p

    def check(self, status: int) -> None:
        if status:
            message = self.cudaGetErrorString(status)
            detail = message.decode("utf-8") if message else "unknown CUDA error"
            raise RuntimeError(f"CUDA Runtime error {status}: {detail}")

    def allocate(self, nbytes: int) -> _CudaAllocation:
        return _CudaAllocation(self, nbytes)

    def stream(self) -> _CudaStream:
        return _CudaStream(self)

    def copy_host_to_device_async(
        self, destination: _CudaAllocation, source: np.ndarray, stream: _CudaStream
    ) -> None:
        self.check(
            self.cudaMemcpyAsync(
                ctypes.c_void_p(int(destination)),
                ctypes.c_void_p(source.ctypes.data),
                source.nbytes,
                self.COPY_HOST_TO_DEVICE,
                ctypes.c_void_p(stream.handle),
            )
        )

    def copy_device_to_host_async(
        self, destination: np.ndarray, source: _CudaAllocation, stream: _CudaStream
    ) -> None:
        self.check(
            self.cudaMemcpyAsync(
                ctypes.c_void_p(destination.ctypes.data),
                ctypes.c_void_p(int(source)),
                destination.nbytes,
                self.COPY_DEVICE_TO_HOST,
                ctypes.c_void_p(stream.handle),
            )
        )


class _TensorRTEngine:
    """A TensorRT execution context with persistent host/device buffers."""

    def __init__(
        self,
        path: Path,
        trt,
        cuda: _CudaRuntime,
        stream: _CudaStream,
        input_shapes: Mapping[str, tuple[int, ...]],
    ):
        if not path.is_file():
            raise FileNotFoundError(f"TensorRT engine not found: {path}")
        self._trt = trt
        self._cuda = cuda
        self._stream = stream
        self._logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(self._logger)
        self._engine = self._runtime.deserialize_cuda_engine(path.read_bytes())
        if self._engine is None:
            raise RuntimeError(f"could not deserialize TensorRT engine: {path}")
        self._context = self._engine.create_execution_context()
        if self._context is None:
            raise RuntimeError(f"could not create TensorRT context: {path}")

        self._input_specs: dict[str, tuple[tuple[int, ...], np.dtype]] = {}
        self._output_arrays: dict[str, np.ndarray] = {}
        self._device_buffers: dict[str, _CudaAllocation] = {}

        for name, shape in input_shapes.items():
            if not self._context.set_input_shape(name, tuple(shape)):
                raise ValueError(f"invalid input shape for {name}: {shape}")

        for index in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(index)
            shape = tuple(int(value) for value in self._context.get_tensor_shape(name))
            if any(value <= 0 for value in shape):
                raise ValueError(f"unresolved TensorRT shape for {name}: {shape}")
            dtype = np.dtype(trt.nptype(self._engine.get_tensor_dtype(name)))
            nbytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
            device = cuda.allocate(nbytes)
            self._device_buffers[name] = device
            self._context.set_tensor_address(name, int(device))
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self._input_specs[name] = (shape, dtype)
            else:
                self._output_arrays[name] = np.empty(shape, dtype=dtype)

        missing = set(input_shapes) - set(self._input_specs)
        if missing:
            raise ValueError(f"TensorRT engine is missing inputs: {sorted(missing)}")

    @property
    def output_shapes(self) -> dict[str, tuple[int, ...]]:
        return {name: value.shape for name, value in self._output_arrays.items()}

    def run(self, inputs: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        if set(inputs) != set(self._input_specs):
            raise ValueError(
                f"expected TensorRT inputs {sorted(self._input_specs)}, "
                f"received {sorted(inputs)}"
            )
        contiguous_inputs: list[np.ndarray] = []
        for name, value in inputs.items():
            shape, dtype = self._input_specs[name]
            contiguous = np.ascontiguousarray(value, dtype=dtype)
            if contiguous.shape != shape:
                raise ValueError(
                    f"TensorRT input {name} has shape {contiguous.shape}, expected {shape}"
                )
            contiguous_inputs.append(contiguous)
            self._cuda.copy_host_to_device_async(
                self._device_buffers[name], contiguous, self._stream
            )

        if not self._context.execute_async_v3(self._stream.handle):
            raise RuntimeError("TensorRT execution failed")
        for name, host in self._output_arrays.items():
            self._cuda.copy_device_to_host_async(
                host, self._device_buffers[name], self._stream
            )
        self._stream.synchronize()
        return self._output_arrays

    def close(self) -> None:
        for buffer in self._device_buffers.values():
            buffer.close()
        self._device_buffers.clear()
        self._output_arrays.clear()
        self._context = None
        self._engine = None
        self._runtime = None


def preprocess_bgr(image: np.ndarray) -> np.ndarray:
    """Convert a BGR frame to the fixed CAT-Seg TensorRT input tensor."""
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 BGR frame, got shape {array.shape}")
    resized = cv2.resize(array, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(np.transpose(rgb, (2, 0, 1))[None], dtype=np.float32)


def probabilities_to_mask(
    probabilities: np.ndarray,
    walkable_ids: Sequence[int],
    output_shape: tuple[int, int],
) -> np.ndarray:
    """Convert CAT-Seg class probabilities to a full-resolution bool mask."""
    scores = np.asarray(probabilities)
    if scores.ndim != 4 or scores.shape[0] != 1:
        raise ValueError(f"unexpected CAT-Seg probability shape: {scores.shape}")
    class_map = scores[0].argmax(axis=0)
    mask = np.isin(class_map, np.asarray(walkable_ids))
    height, width = output_shape
    if mask.shape != (height, width):
        mask = cv2.resize(
            mask.astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    return mask


class CatSegTensorRTPredictor:
    """Two-stage fixed-shape CAT-Seg TensorRT predictor."""

    def __init__(
        self,
        clip_engine: str | Path,
        aggregator_engine: str | Path,
        labels_path: str | Path,
        walkable_names: Sequence[str],
        tensorrt_python_path: str | None = DEFAULT_TENSORRT_PYTHON_PATH,
        warmup: int = 1,
    ):
        if warmup < 0:
            raise ValueError("TensorRT warmup must be non-negative")
        labels_path = Path(labels_path).expanduser()
        labels = json.loads(labels_path.read_text(encoding="utf-8"))
        missing = [name for name in walkable_names if name not in labels]
        if missing:
            raise ValueError(f"walkable classes not in {labels_path}: {', '.join(missing)}")
        self.walkable_ids = tuple(labels.index(name) for name in walkable_names)

        trt = _load_tensorrt(tensorrt_python_path)
        self._cuda = _CudaRuntime()
        self._stream = self._cuda.stream()
        self._clip = _TensorRTEngine(
            Path(clip_engine).expanduser(),
            trt,
            self._cuda,
            self._stream,
            {"image": (1, 3, INPUT_HEIGHT, INPUT_WIDTH)},
        )
        clip_shapes = self._clip.output_shapes
        required_clip_outputs = {"clip_features", "layer_4", "layer_8"}
        if not required_clip_outputs.issubset(clip_shapes):
            raise ValueError(
                "CLIP TensorRT engine outputs do not match CAT-Seg: "
                f"{sorted(clip_shapes)}"
            )
        self._aggregator = _TensorRTEngine(
            Path(aggregator_engine).expanduser(),
            trt,
            self._cuda,
            self._stream,
            {name: clip_shapes[name] for name in required_clip_outputs},
        )
        if "probabilities" not in self._aggregator.output_shapes:
            raise ValueError("aggregator TensorRT engine has no probabilities output")

        zero_image = np.zeros(
            (1, 3, INPUT_HEIGHT, INPUT_WIDTH), dtype=np.float32
        )
        for _ in range(warmup):
            self._infer_tensor(zero_image)

    def _infer_tensor(self, image: np.ndarray) -> np.ndarray:
        clip = self._clip.run({"image": image})
        outputs = self._aggregator.run(
            {
                "clip_features": clip["clip_features"],
                "layer_4": clip["layer_4"],
                "layer_8": clip["layer_8"],
            }
        )
        return outputs["probabilities"]

    def predict(self, image: np.ndarray) -> np.ndarray:
        tensor = preprocess_bgr(image)
        probabilities = self._infer_tensor(tensor)
        return probabilities_to_mask(
            probabilities, self.walkable_ids, image.shape[:2]
        )

    def close(self) -> None:
        aggregator = getattr(self, "_aggregator", None)
        if aggregator is not None:
            aggregator.close()
            self._aggregator = None
        clip = getattr(self, "_clip", None)
        if clip is not None:
            clip.close()
            self._clip = None
        stream = getattr(self, "_stream", None)
        if stream is not None:
            stream.close()
            self._stream = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
