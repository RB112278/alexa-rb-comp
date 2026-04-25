# Alexa RB

Local Windows desktop voice agent with Samantha-style persona. Wake word → Whisper STT → Claude Sonnet 4.6 (streaming) → Kokoro TTS, all stitched together with sub-second perceived latency via sentence-chunked streaming.

```
USER: "Alexa, what's the weather?"
   ↓ (openwakeword)
   ↓ (silero-vad detects end of speech)
   ↓ (faster-whisper on CUDA)
   ↓ (Claude Sonnet 4.6, streaming tokens)
   ↓ (sentence chunker → Kokoro TTS, generated in parallel with playback)
ASSISTANT: "It's 72 and sunny out. Want me to read the rest of the forecast?"
```

End-to-end target: ~600–800ms from user-stops-speaking to first-audio-out.

## Stack

| Layer | Pick |
|---|---|
| Wake word | `openwakeword` (`alexa` pretrained) |
| Endpointing | `silero-vad` |
| STT | `faster-whisper` `base.en` on CUDA fp16 |
| LLM | `claude-sonnet-4-6` (Anthropic API, streaming) |
| TTS | `kokoro-onnx` (54 voices, currently `af_bella`) |
| Audio | `sounddevice` |

Hardware target: Windows 11, NVIDIA GPU (RTX 3080 Ti tested), Python 3.13.

## Install

```powershell
git clone https://github.com/RB112278/alexa-rb-comp.git
cd alexa-rb-comp

py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Download Kokoro model files (~340 MB, one time):
```powershell
mkdir models
curl -L -o models/kokoro-v1.0.onnx https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -L -o models/voices-v1.0.bin   https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

Set up `.env` with your Claude API key:
```powershell
copy .env.example .env
notepad .env       # paste your sk-ant-... key
```

## Run

```powershell
python main.py
```

Say **"Alexa"**, wait for the *"Mhm?"* acknowledgment, then ask anything. Stop with `Ctrl+C`.

Whisper weights (~150 MB) and openwakeword models (~5 MB) auto-download on first run.

## Repo layout

```
.
├── main.py                  # the whole pipeline
├── smoke_test.py            # non-mic component check
├── requirements.txt
├── .env.example
├── models/                  # Kokoro ONNX + voices (gitignored, downloaded)
├── tools/                   # one-pager docs per external library
│   ├── openwakeword.md
│   ├── silero-vad.md
│   ├── faster-whisper.md
│   ├── anthropic-claude.md
│   ├── kokoro-onnx.md
│   ├── sounddevice.md
│   ├── cuda-windows.md
│   └── gh-cli.md
└── workflows/               # end-to-end flow docs
    ├── boot.md
    ├── wake-to-response.md
    ├── voice-cloning.md
    └── custom-wake-word.md
```

`tools/` and `workflows/` are reference material — read them when something breaks or before changing how a component is used.

## Tunables (env vars)

| Var | Default | Notes |
|---|---|---|
| `WAKE_WORD` | `alexa` | also: `hey_jarvis`, `hey_mycroft`, `hey_rhasspy` |
| `WAKE_THRESHOLD` | `0.5` | bump to 0.6-0.7 if false-firing |
| `VAD_MIN_SILENCE_MS` | `800` | how much silence before turn ends |
| `WHISPER_MODEL` | `base.en` | `small.en` for higher accuracy |
| `CLAUDE_MODEL` | `claude-sonnet-4-6` | `claude-haiku-4-5-20251001` for snappier |
| `CLAUDE_MAX_TOKENS` | `400` | shorter replies = less perceived lag |
| `KOKORO_VOICE` | `af_bella` | see `tools/kokoro-onnx.md` for the 54-voice list |
| `KOKORO_SPEED` | `1.0` | 0.5–2.0 |

## Roadmap

1. ✅ Wake word + Whisper + SAPI TTS scaffold
2. ✅ Replace SAPI with Kokoro (10× better voice)
3. ✅ Replace fixed 5s window with silero-vad endpointing
4. ✅ Claude Sonnet 4.6 streaming with sentence-chunked TTS
5. ✅ Realtime data via Perplexity tool use (`workflows/web-search.md`)
6. ☐ Custom "Hey Ramon" wake word (`workflows/custom-wake-word.md`)
7. ☐ Samantha-from-Her voice via F5-TTS clone (`workflows/voice-cloning.md`)
8. ☐ Barge-in (interrupt agent mid-sentence by speaking)
9. ☐ More tools — open apps, search files, control music, read clipboard

## Troubleshooting

- **`cublas64_12.dll not found`** → see `tools/cuda-windows.md`. Already handled in `main.py:_preload_nvidia_dlls()`.
- **Wake word fires on TV** → `WAKE_THRESHOLD=0.7`
- **Cuts you off mid-sentence** → `VAD_MIN_SILENCE_MS=1200`
- **Claude reads markdown literally** → check `CLAUDE_SYSTEM` in `main.py` includes "no markdown" instruction
- **Wrong mic** → `python -c "import sounddevice as sd; print(sd.query_devices())"`, then set `sd.default.device = (in_idx, out_idx)`
