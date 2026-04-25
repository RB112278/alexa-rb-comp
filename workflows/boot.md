# Workflow: Boot

What happens when you run `python main.py`.

```
┌─────────────────────────────────────────────────────────────────┐
│  1. _preload_nvidia_dlls()    │  ctypes-load every nvidia DLL   │
│                               │  so ctranslate2 can find them   │
│                               │  → see tools/cuda-windows.md    │
├─────────────────────────────────────────────────────────────────┤
│  2. load_dotenv()             │  reads .env into os.environ     │
├─────────────────────────────────────────────────────────────────┤
│  3. Check ANTHROPIC_API_KEY   │  exit if missing                │
├─────────────────────────────────────────────────────────────────┤
│  4. Load openwakeword model   │  ~5 MB, instant                 │
├─────────────────────────────────────────────────────────────────┤
│  5. Load faster-whisper       │  ~150 MB if base.en, ~1.5s      │
│     on CUDA fp16              │  warms up GPU                   │
├─────────────────────────────────────────────────────────────────┤
│  6. Load silero-vad           │  ~2 MB, instant                 │
├─────────────────────────────────────────────────────────────────┤
│  7. Load Kokoro ONNX          │  337 MB total, ~1.2s            │
│     start TTS worker threads  │  generator + playback threads   │
├─────────────────────────────────────────────────────────────────┤
│  8. Connect to Anthropic      │  no network call yet, lazy      │
├─────────────────────────────────────────────────────────────────┤
│  9. Open mic InputStream      │  starts callback thread feeding │
│     (16kHz mono int16, 512-   │  the audio queue                │
│     sample blocks)            │                                 │
├─────────────────────────────────────────────────────────────────┤
│ 10. Greeting via Kokoro       │  "Online. Say alexa when you    │
│     (queues sentence)         │   need me."                     │
├─────────────────────────────────────────────────────────────────┤
│ 11. Enter wake-word loop      │  ready                          │
└─────────────────────────────────────────────────────────────────┘
```

**Total cold boot time:** ~3–5 seconds on the 3080 Ti, dominated by faster-whisper CUDA warm-up. First-run is slower because Whisper weights download (~150MB).

## First-run downloads

These happen automatically on first launch and are cached:

| Asset | Where | Size |
|---|---|---|
| openwakeword `alexa_v0.1` model | `~/.cache/openwakeword/` | ~5 MB |
| faster-whisper `base.en` weights | `~/.cache/huggingface/` | ~150 MB |
| silero-vad weights | bundled with package | 0 MB extra |
| Kokoro ONNX + voices | `models/` (manual `curl` per README) | 337 MB |
