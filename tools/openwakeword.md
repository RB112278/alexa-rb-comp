# openwakeword

**What:** Always-on wake-word detector. Listens to mic frames, scores each against trained wake-word models, fires when confidence > threshold.

**Why we picked it:** Apache 2.0, pure Python, runs on CPU, ships pretrained models for `alexa` / `hey_jarvis` / `hey_mycroft` / `hey_rhasspy`. Custom wake words can be trained from synthetic data via the project's notebook.

**Repo:** https://github.com/dscripka/openWakeWord
**HF models:** https://huggingface.co/dscripka/openWakeWord

## Install

```bash
pip install openwakeword
```

Models auto-download to `~/.cache/openwakeword/` on first use.

## How we use it (`main.py`)

```python
from openwakeword.model import Model as WakeModel
wake = WakeModel(wakeword_models=["alexa"], inference_framework="onnx")

# loop: feed 1280-sample (80ms @ 16kHz) int16 mono frames
scores = wake.predict(frame)         # dict: {"alexa": 0.97, ...}
if scores["alexa"] >= 0.5:
    wake.reset()                     # clear internal buffer
    # ...handle wake event
```

## Gotchas

- **Frame size MUST be 1280 samples** (80ms @ 16kHz). The mic in our app reads 512-sample VAD chunks, so we coalesce 512→1280 before calling `predict`.
- **`wake.reset()`** clears internal state after a fire — without it the next prediction can spuriously re-trigger.
- **"alexa" is short** → false positives on similar words (Cassandra, tomorrow). If noisy, bump threshold to 0.6–0.7.
- **Custom wake words** require training. Realistic effort: 30–60 min on the 3080 Ti using the official notebook.

## Tunable env vars

| Var | Default | Meaning |
|---|---|---|
| `WAKE_WORD` | `alexa` | also: `hey_jarvis`, `hey_mycroft`, `hey_rhasspy` |
| `WAKE_THRESHOLD` | `0.5` | confidence required to fire (0.0–1.0) |
