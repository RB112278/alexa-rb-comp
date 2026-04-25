# silero-vad

**What:** Voice Activity Detector. Tells you in real time whether a chunk of audio is speech or silence. We use it to detect when the user has stopped talking, so we can stop recording and ship audio to Whisper — no fixed time window.

**Why we picked it:** Production standard for local VAD (used by NVIDIA Riva, LiveKit, Pipecat). Tiny ONNX model, runs on CPU at <1ms per chunk, MIT-licensed.

**Repo:** https://github.com/snakers4/silero-vad

## Install

```bash
pip install silero-vad
```

## How we use it (`main.py`)

```python
from silero_vad import load_silero_vad, VADIterator
import torch

vad_model = load_silero_vad()
vad_iter = VADIterator(
    vad_model,
    threshold=0.5,
    sampling_rate=16000,
    min_silence_duration_ms=800,
    speech_pad_ms=100,
)

# feed exactly 512-sample (32ms @ 16kHz) float32 chunks
event = vad_iter(torch.from_numpy(chunk_f32))
# event is None, {"start": idx}, or {"end": idx}
```

We end the recording when there's been ≥800ms of silence *after* speech started, or after a 15s hard cap.

## Gotchas

- **Chunk size MUST be 512 samples** at 16kHz. Anything else and it errors or returns garbage.
- **Audio must be float32 in [-1, 1]**, not int16. We convert in `_to_float32()`.
- **`vad_iter.reset_states()`** before every new utterance — otherwise stale internal state bleeds into the next turn.
- **`min_silence_duration_ms` tuning:** lower (500ms) = snappier but cuts off thinking pauses; higher (1200ms) = more patient but slower turn-taking.

## Tunable env vars

| Var | Default | Meaning |
|---|---|---|
| `VAD_THRESHOLD` | `0.5` | speech-confidence threshold per chunk |
| `VAD_MIN_SILENCE_MS` | `1500` | how much silence after speech ends the utterance — bump higher if you get cut off mid-thought |
