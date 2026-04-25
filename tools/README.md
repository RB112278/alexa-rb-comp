# Tools

Reference docs for every external library/component the voice agent depends on.
Each file covers: what it does, why we picked it, the API we use, and gotchas.

| Layer | Tool | File |
|---|---|---|
| Wake word | openwakeword | [openwakeword.md](openwakeword.md) |
| Endpointing | silero-vad | [silero-vad.md](silero-vad.md) |
| STT | faster-whisper | [faster-whisper.md](faster-whisper.md) |
| LLM | anthropic (Claude Sonnet 4.6) | [anthropic-claude.md](anthropic-claude.md) |
| TTS | kokoro-onnx | [kokoro-onnx.md](kokoro-onnx.md) |
| Audio I/O | sounddevice | [sounddevice.md](sounddevice.md) |
| CUDA on Windows | nvidia-cublas / cudnn / cuda_nvrtc | [cuda-windows.md](cuda-windows.md) |
| GitHub CLI | gh | [gh-cli.md](gh-cli.md) |
