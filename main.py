"""
Alexa RB - desktop voice agent MVP
Loop: mic -> openwakeword -> faster-whisper -> SAPI TTS

Run:  python main.py
Say:  "Hey Jarvis", then a sentence.
"""
from __future__ import annotations

import os
import queue
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

# --- CUDA DLL discovery for faster-whisper on Windows -----------------------
# faster-whisper looks for cuBLAS + cuDNN at import time. The pip-installed
# nvidia-* packages drop DLLs in site-packages; we add those dirs to the loader
# search path before importing whisper.
def _add_nvidia_dll_dirs() -> None:
    if sys.platform != "win32":
        return
    import importlib.util
    for pkg in ("nvidia.cublas", "nvidia.cudnn"):
        spec = importlib.util.find_spec(pkg)
        if not spec or not spec.submodule_search_locations:
            continue
        bin_dir = Path(spec.submodule_search_locations[0]) / "bin"
        if bin_dir.is_dir():
            os.add_dll_directory(str(bin_dir))

_add_nvidia_dll_dirs()

from faster_whisper import WhisperModel  # noqa: E402
from openwakeword.model import Model as WakeModel  # noqa: E402

# --- Config -----------------------------------------------------------------
SAMPLE_RATE = 16_000
WAKE_FRAME_SAMPLES = 1280            # 80 ms @ 16 kHz, openwakeword expects this
WAKE_THRESHOLD = 0.5
WAKE_NAME = os.getenv("WAKE_WORD", "hey_jarvis")
UTTERANCE_SECONDS = 5.0              # fixed window after wake; silence-VAD comes later
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "base.en")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16")

# --- TTS --------------------------------------------------------------------
def make_tts():
    import pyttsx3
    engine = pyttsx3.init("sapi5")
    engine.setProperty("rate", 185)
    return engine

def speak(engine, text: str) -> None:
    print(f"[say] {text}")
    engine.say(text)
    engine.runAndWait()

# --- Audio capture ----------------------------------------------------------
def audio_stream() -> "queue.Queue[np.ndarray]":
    q: queue.Queue[np.ndarray] = queue.Queue()

    def cb(indata, frames, _time, status):
        if status:
            print(f"[mic] {status}", file=sys.stderr)
        q.put(indata[:, 0].copy())  # mono channel as int16

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=WAKE_FRAME_SAMPLES,
        callback=cb,
    )
    stream.start()
    return q

def record_utterance(q: "queue.Queue[np.ndarray]", seconds: float) -> np.ndarray:
    needed = int(SAMPLE_RATE * seconds)
    buf: list[np.ndarray] = []
    have = 0
    while have < needed:
        chunk = q.get()
        buf.append(chunk)
        have += len(chunk)
    audio = np.concatenate(buf)[:needed]
    # faster-whisper wants float32 in [-1, 1]
    return (audio.astype(np.float32) / 32768.0)

# --- Main loop --------------------------------------------------------------
def main() -> None:
    print(f"[boot] loading wake model: {WAKE_NAME}")
    wake = WakeModel(wakeword_models=[WAKE_NAME], inference_framework="onnx")

    print(f"[boot] loading whisper: {WHISPER_MODEL_NAME} on {WHISPER_DEVICE} ({WHISPER_COMPUTE_TYPE})")
    whisper = WhisperModel(
        WHISPER_MODEL_NAME,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE_TYPE,
    )

    print("[boot] loading TTS")
    tts = make_tts()

    print("[boot] opening mic")
    q = audio_stream()

    speak(tts, "Online. Say hey jarvis.")
    print(f"[ready] waiting for wake word '{WAKE_NAME}' (threshold {WAKE_THRESHOLD})")

    cooldown_until = 0.0
    while True:
        chunk = q.get()
        if time.time() < cooldown_until:
            continue
        scores = wake.predict(chunk)
        score = scores.get(WAKE_NAME, 0.0)
        if score >= WAKE_THRESHOLD:
            print(f"[wake] {WAKE_NAME} score={score:.2f}")
            # drain queue so old audio is not part of the utterance
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
            speak(tts, "Yes?")
            audio = record_utterance(q, UTTERANCE_SECONDS)
            print("[stt] transcribing...")
            t0 = time.time()
            segments, info = whisper.transcribe(audio, language="en", vad_filter=True)
            text = " ".join(s.text.strip() for s in segments).strip()
            dt = time.time() - t0
            print(f"[stt] ({dt:.2f}s, lang={info.language}): {text!r}")
            if text:
                speak(tts, f"You said: {text}")
            else:
                speak(tts, "I did not catch that.")
            wake.reset()
            cooldown_until = time.time() + 1.0  # avoid immediate re-trigger

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[exit] bye")
