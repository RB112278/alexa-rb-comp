# faster-whisper

**What:** Speech-to-text. CTranslate2 reimplementation of OpenAI Whisper — 3-4× faster than the original `openai-whisper` package, fp16 on GPU, drop-in API.

**Why we picked it:** Best local STT speed/quality on consumer GPUs. Apache 2.0. CTranslate2 ships its own CUDA libraries (no system CUDA toolkit needed beyond cuBLAS + cuDNN, which we pip-install).

**Repo:** https://github.com/SYSTRAN/faster-whisper

## Install

```bash
pip install faster-whisper nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"
```

Whisper weights auto-download from HF on first use to `~/.cache/huggingface/`.

## How we use it (`main.py`)

```python
from faster_whisper import WhisperModel
whisper = WhisperModel("base.en", device="cuda", compute_type="float16")

segments, info = whisper.transcribe(
    audio_float32,         # numpy [-1, 1], 16kHz mono
    language="en",
    vad_filter=False,      # we already ran silero-vad upstream
)
text = " ".join(s.text.strip() for s in segments)
```

## Model size vs. quality

| Model | VRAM (fp16) | Latency | Quality |
|---|---|---|---|
| `tiny.en` | ~75 MB | <100ms | rough, noticeable errors |
| `base.en` | ~150 MB | ~200ms | **default — good balance** |
| `small.en` | ~500 MB | ~350ms | clean, recommended for accuracy |
| `medium.en` | ~1.5 GB | ~600ms | studio quality |
| `large-v3` | ~3 GB | ~1.2s | overkill for desktop voice |

## Gotchas

- **CUDA on Windows:** ctranslate2's loader does NOT honor `os.add_dll_directory()`. We work around this in `main.py:_preload_nvidia_dlls()` by `ctypes.WinDLL()`-loading every nvidia bundled DLL before importing whisper. Without this you get `cublas64_12.dll not found`.
- **`segments` is a generator** — iterating it is what actually runs inference. Don't time `.transcribe()` itself.
- **`vad_filter=True`** runs Silero internally and skips silent regions. We turn it off because we already endpoint upstream.

## Tunable env vars

| Var | Default | Meaning |
|---|---|---|
| `WHISPER_MODEL` | `base.en` | model size (see table above) |
| `WHISPER_DEVICE` | `cuda` | `cpu` to force CPU |
| `WHISPER_COMPUTE_TYPE` | `float16` | `int8_float16`, `int8`, `float32` |
