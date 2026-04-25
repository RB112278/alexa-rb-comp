"""
Component smoke test. Does NOT open the mic or call Claude.

Verifies:
  1. CUDA DLL preload works
  2. sounddevice sees an input device
  3. openwakeword loads the alexa model
  4. faster-whisper loads on CUDA + transcribes silence without erroring
  5. silero-vad loads
  6. Kokoro generates + plays a short greeting
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path


# Mirror main.py CUDA DLL preload
def _preload_nvidia_dlls() -> None:
    if sys.platform != "win32":
        return
    import ctypes, importlib.util
    for pkg in ("nvidia.cublas", "nvidia.cudnn", "nvidia.cuda_nvrtc"):
        spec = importlib.util.find_spec(pkg)
        if not spec or not spec.submodule_search_locations:
            continue
        bin_dir = Path(spec.submodule_search_locations[0]) / "bin"
        if bin_dir.is_dir():
            os.add_dll_directory(str(bin_dir))
            for dll in bin_dir.glob("*.dll"):
                try:
                    ctypes.WinDLL(str(dll))
                except OSError:
                    pass


_preload_nvidia_dlls()

print("[1/5] sounddevice")
import sounddevice as sd
default_in = sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else sd.default.device
print(f"      default input idx: {default_in}")
print(f"      device 0: {sd.query_devices(0)['name']}")

print("[2/5] openwakeword (downloads ~5MB on first run)")
from openwakeword.utils import download_models
download_models(["alexa"])
from openwakeword.model import Model as WakeModel
wake = WakeModel(wakeword_models=["alexa"], inference_framework="onnx")
print(f"      models: {list(wake.models.keys())}")

print("[3/5] faster-whisper on CUDA (downloads ~150MB on first run)")
from faster_whisper import WhisperModel
t0 = time.time()
whisper = WhisperModel("base.en", device="cuda", compute_type="float16")
print(f"      loaded in {time.time()-t0:.1f}s")
import numpy as np
silence = np.zeros(16000, dtype=np.float32)
segs, info = whisper.transcribe(silence, language="en")
list(segs)  # force generator
print(f"      transcribed silence OK, lang={info.language}")

print("[4/5] silero-vad")
from silero_vad import load_silero_vad, VADIterator
vad_model = load_silero_vad()
vad_iter = VADIterator(vad_model, threshold=0.5, sampling_rate=16000)
print("      loaded OK")

print("[5/5] Kokoro TTS (you should hear: 'smoke test complete')")
from kokoro_onnx import Kokoro
kokoro = Kokoro("models/kokoro-v1.0.onnx", "models/voices-v1.0.bin")
audio, sr = kokoro.create("Smoke test complete.", voice="af_bella")
sd.play(audio, sr); sd.wait()

print("OK — all components healthy.")
