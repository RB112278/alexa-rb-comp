# Alexa RB

Desktop voice agent. Wake word -> STT -> (later) router -> TTS.

Stack (MVP):
- **Wake word**: openwakeword (`hey_jarvis` pretrained model)
- **STT**: faster-whisper, CUDA 12, `base.en` by default
- **TTS**: pyttsx3 over Windows SAPI

Hardware target: Windows 11, NVIDIA GPU, Python 3.13.

## Install

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

First run will download the wake-word ONNX models and the Whisper weights.
Whisper `base.en` is ~150 MB; `small.en` is ~500 MB.

## Run

```powershell
python main.py
```

Say **"Hey Jarvis"**, wait for "Yes?", then speak. The agent prints and speaks the transcription back.

Stop with `Ctrl+C`.

## Config

Optional overrides via env vars (or copy `.env.example` to `.env` and load yourself):

| Var | Default | Notes |
|---|---|---|
| `WAKE_WORD` | `hey_jarvis` | also: `alexa`, `hey_mycroft` |
| `WHISPER_MODEL` | `base.en` | `tiny.en`, `small.en`, `medium.en`, `large-v3` |
| `WHISPER_DEVICE` | `cuda` | `cpu` to disable GPU |
| `WHISPER_COMPUTE_TYPE` | `float16` | `int8_float16`, `int8`, `float32` |

## Roadmap

1. ~~Mic loop -> wake word -> STT -> speak back~~ (this scaffold)
2. Silence-based endpointing (silero-vad) instead of fixed 5 s window
3. Deterministic intents: time, open app, clipboard, web search
4. LLM fallback (Claude) with tools
5. Custom "Hey Ramon" wake word trained via openwakeword notebook

Switch to git worktrees at step 3 — intents / LLM-tools / wake-training can fan out in parallel.
