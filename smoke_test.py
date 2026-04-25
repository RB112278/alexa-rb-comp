"""One-shot import + GPU + model-load check. Does not open the mic."""
import os, sys, time
from pathlib import Path

# mirror main.py CUDA DLL setup (pre-load DLLs via ctypes)
import ctypes, importlib.util
for pkg in ("nvidia.cublas", "nvidia.cudnn", "nvidia.cuda_nvrtc"):
    spec = importlib.util.find_spec(pkg)
    if spec and spec.submodule_search_locations:
        bin_dir = Path(spec.submodule_search_locations[0]) / "bin"
        if bin_dir.is_dir():
            os.add_dll_directory(str(bin_dir))
            for dll in bin_dir.glob("*.dll"):
                try:
                    ctypes.WinDLL(str(dll))
                except OSError:
                    pass

print("[1/5] sounddevice")
import sounddevice as sd
default_in = sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else sd.default.device
print(f"      default input idx: {default_in}")
print(f"      device 0: {sd.query_devices(0)['name']}")

print("[2/5] openwakeword (downloads ~5MB on first run)")
from openwakeword.utils import download_models
download_models(["hey_jarvis"])
from openwakeword.model import Model as WakeModel
wake = WakeModel(wakeword_models=["hey_jarvis"], inference_framework="onnx")
print(f"      models: {list(wake.models.keys())}")

print("[3/5] faster-whisper on CUDA (downloads ~150MB on first run)")
from faster_whisper import WhisperModel
t0 = time.time()
whisper = WhisperModel("base.en", device="cuda", compute_type="float16")
print(f"      loaded in {time.time()-t0:.1f}s")

print("[4/5] dummy transcribe (1 sec of silence)")
import numpy as np
silence = np.zeros(16000, dtype=np.float32)
segs, info = whisper.transcribe(silence, language="en")
print(f"      lang={info.language} duration={info.duration:.2f}s segments={list(segs)}")

print("[5/5] TTS (you should hear: 'smoke test complete')")
import pyttsx3
e = pyttsx3.init("sapi5")
e.setProperty("rate", 185)
e.say("Smoke test complete.")
e.runAndWait()
print("OK")
