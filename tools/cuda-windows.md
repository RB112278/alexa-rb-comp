# CUDA on Windows (the DLL hell chapter)

**What:** Notes on getting GPU-accelerated ONNX/CTranslate2 to actually find the right DLLs on Windows + Python 3.13. Easiest place in this project to lose half a day.

## The problem

Both `ctranslate2` (used by faster-whisper) and `onnxruntime-gpu` call `LoadLibraryW("cublas64_12.dll", ...)` from native code at first GPU op. On Windows + Python ≥ 3.8:

- The legacy `PATH` env var is **no longer searched** by `LoadLibrary` for security reasons.
- `os.add_dll_directory()` adds dirs to a *user* search path which **some loaders honor and some don't**. Specifically: **ctranslate2 does NOT honor it**.

Symptoms: `RuntimeError: Library cublas64_12.dll is not found or cannot be loaded` even though the DLL is on disk in `.venv/Lib/site-packages/nvidia/cublas/bin/`.

## The fix (used in `main.py`)

Pre-load every NVIDIA-bundled DLL into the process via `ctypes.WinDLL(absolute_path)` *before* importing anything that needs them. Once loaded, Windows keeps a process-wide table — when ctranslate2 later asks for `cublas64_12.dll` by name, the loader returns the already-loaded handle instead of searching disk.

```python
import os, sys, ctypes, importlib.util
from pathlib import Path

def _preload_nvidia_dlls():
    if sys.platform != "win32":
        return
    for pkg in ("nvidia.cublas", "nvidia.cudnn", "nvidia.cuda_nvrtc"):
        spec = importlib.util.find_spec(pkg)
        if not spec or not spec.submodule_search_locations:
            continue
        bin_dir = Path(spec.submodule_search_locations[0]) / "bin"
        if bin_dir.is_dir():
            os.add_dll_directory(str(bin_dir))   # belt-and-suspenders
            for dll in bin_dir.glob("*.dll"):
                try: ctypes.WinDLL(str(dll))
                except OSError: pass             # some need others loaded first

_preload_nvidia_dlls()
# ...now safe to: from faster_whisper import WhisperModel
```

## Required pip packages

```
nvidia-cublas-cu12        # cublas64_12.dll, cublasLt64_12.dll
nvidia-cudnn-cu12==9.*    # cudnn*64_9.dll
nvidia-cuda-nvrtc-cu12    # nvrtc, used by torch + onnxruntime
```

## What we DIDN'T install (and why)

For full `onnxruntime-gpu` CUDA EP, you'd also need:
```
nvidia-cufft-cu12         # cufft64_*.dll
nvidia-curand-cu12        # curand64_*.dll
nvidia-cusparse-cu12
nvidia-cusolver-cu12
```
…each ~100-300MB. We skipped them because Kokoro-onnx is fast enough on CPU (~RTF 0.3 with streaming) and the bottleneck is Claude's network round-trip, not TTS generation. faster-whisper only needs cuBLAS + cuDNN, which we have.

If GPU TTS becomes the bottleneck later: `pip install nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusparse-cu12 nvidia-cusolver-cu12` and add them to the preload loop.

## Verification

```bash
.venv/Scripts/python.exe -c "
import onnxruntime as ort
print(ort.get_available_providers())
"
# Expect: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```

Available ≠ usable. To check if CUDA EP actually loads at runtime, watch for `Failed to create CUDAExecutionProvider` warnings in stderr on first model run.

## Hardware in this project

```
GPU:       NVIDIA GeForce RTX 3080 Ti (12GB VRAM)
Driver:    591.86  →  CUDA 13.1 toolkit (back-compat with CUDA 12 runtime)
Compute:   8.6
```
