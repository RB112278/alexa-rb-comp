# kokoro-onnx

**What:** Text-to-speech. ONNX port of Kokoro-82M, the current best-in-class small TTS model (MOS 4.2). 54 voices across 9 languages. Streams audio as it generates.

**Why we picked it:**
- Apache 2.0 (vs. F5-TTS's CC-BY-NC and XTTS-v2's restrictive CPML)
- Tiny: 310MB ONNX + 27MB voices, <1GB VRAM
- Fast: ~0.3 RTF on CPU, ~0.05 on GPU
- 54 built-in voices, no fine-tuning needed for general use
- The original `kokoro` PyPI package depends on `spacy`/`thinc`/`blis` which won't compile on Python 3.13 without MSVC. The ONNX port avoids that entire dependency tree.

**Repo:** https://github.com/thewh1teagle/kokoro-onnx
**Original Kokoro:** https://huggingface.co/hexgrad/Kokoro-82M

## Install

```bash
pip install kokoro-onnx
```

Model files (download once, ~340MB total):
```bash
mkdir models
curl -L -o models/kokoro-v1.0.onnx https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -L -o models/voices-v1.0.bin   https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

## How we use it (`main.py`)

```python
from kokoro_onnx import Kokoro
k = Kokoro("models/kokoro-v1.0.onnx", "models/voices-v1.0.bin")
audio, sr = k.create("Hello Ramon.", voice="af_bella", speed=1.0)
# audio: float32 numpy, sr: 24000

# Or streaming:
async for audio_chunk, sr in k.create_stream(text, voice="af_bella"):
    sd.play(audio_chunk, sr)
```

We wrap this in `TTSPlayer` (in `main.py`) — a two-thread pipeline: sentence queue → generator thread → audio queue → playback thread. Lets the next sentence synthesize while the current one plays.

## Voice catalog (54 total)

- **`af_*` — American female:** alloy, aoede, **bella** ⭐, heart, jessica, kore, nicole, nova, river, sarah, sky
- **`am_*` — American male:** adam, echo, eric, fenrir, liam, michael, onyx, puck
- **`bf_*` — British female:** alice, emma, isabella, lily
- **`bm_*` — British male:** daniel, fable, george, lewis
- **`ef_*`, `em_*` — Spanish, `ff_*` — French, `hf_*`, `hm_*` — Hindi, `if_*`, `im_*` — Italian, `jf_*`, `jm_*` — Japanese, `pf_*`, `pm_*` — Brazilian Portuguese, `zf_*`, `zm_*` — Mandarin Chinese**

Get the full list with `Kokoro.get_voices()`. Default in this project: **`af_bella`** (warm, neutral American female — closest in tone to Samantha-from-*Her* among the built-ins).

## Voice cloning

**Kokoro alone cannot clone arbitrary voices** — only the bundled 54. For a true Samantha-from-Her voice clone, layer **F5-TTS** on top:
- Repo: https://github.com/SWivid/F5-TTS
- Zero-shot cloning from a 10-15s reference clip
- License: CC-BY-NC 4.0 (personal use only)

That's a future upgrade — `tools/f5-tts.md` will be added when we wire it.

## Gotchas

- **Sample rate is 24kHz**, not 16kHz like the mic chain. `sounddevice` handles the resampling.
- **CUDA EP for kokoro-onnx requires more libs** than just cuBLAS — also cuFFT, cuRAND. We didn't install those, so kokoro-onnx silently falls back to CPU. CPU is fast enough at RTF 0.3 with our streaming pipeline; GPU optimization is on the punch list.
- **`speed` accepts 0.5-2.0.** Below 0.7 sounds slurred, above 1.3 sounds rushed.

## Tunable env vars

| Var | Default | Meaning |
|---|---|---|
| `KOKORO_VOICE` | `af_bella` | any from `Kokoro.get_voices()` |
| `KOKORO_SPEED` | `1.0` | 0.5–2.0 |
| `KOKORO_MODEL_PATH` | `models/kokoro-v1.0.onnx` | path to ONNX model |
| `KOKORO_VOICES_PATH` | `models/voices-v1.0.bin` | path to voices file |
