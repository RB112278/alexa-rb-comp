"""
Alexa RB - desktop voice agent.

Pipeline:
  mic loop -> openwakeword -> silero-vad endpointing -> faster-whisper STT
    -> Claude Sonnet 4.6 (streaming) -> sentence chunker -> Kokoro TTS -> playback

Run:  python main.py
Say:  "Alexa", then ask anything.
"""
from __future__ import annotations

import os
import queue
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv

# --- CUDA DLL discovery for faster-whisper on Windows -----------------------
# faster-whisper -> ctranslate2 calls LoadLibraryW("cublas64_12.dll") at first
# GPU op. ctranslate2's loader does NOT honor os.add_dll_directory(), so we
# pre-load the DLLs with ctypes; once they're in the process, Windows resolves
# the later LoadLibraryW by module name from the already-loaded list.
def _preload_nvidia_dlls() -> None:
    if sys.platform != "win32":
        return
    import ctypes, importlib.util
    for pkg in ("nvidia.cublas", "nvidia.cudnn", "nvidia.cuda_nvrtc"):
        spec = importlib.util.find_spec(pkg)
        if not spec or not spec.submodule_search_locations:
            continue
        bin_dir = Path(spec.submodule_search_locations[0]) / "bin"
        if not bin_dir.is_dir():
            continue
        os.add_dll_directory(str(bin_dir))
        for dll in bin_dir.glob("*.dll"):
            try:
                ctypes.WinDLL(str(dll))
            except OSError:
                pass

_preload_nvidia_dlls()
# override=True so a project-local .env wins over any stale Windows env var
load_dotenv(override=True)

import torch  # noqa: E402  (silero-vad needs it; safe to import after dll preload)
from anthropic import Anthropic, APIStatusError, APIConnectionError  # noqa: E402
from faster_whisper import WhisperModel  # noqa: E402
from kokoro_onnx import Kokoro  # noqa: E402
from openwakeword.model import Model as WakeModel  # noqa: E402
from silero_vad import load_silero_vad, VADIterator  # noqa: E402

# --- Config -----------------------------------------------------------------
SAMPLE_RATE = 16_000                 # mic + STT + VAD all at 16kHz
WAKE_FRAME_SAMPLES = 1280            # 80 ms @ 16 kHz, openwakeword expects this
WAKE_THRESHOLD = float(os.getenv("WAKE_THRESHOLD", "0.5"))
WAKE_NAME = os.getenv("WAKE_WORD", "alexa")

# silero-vad runs on 512-sample (32ms) chunks at 16kHz
VAD_CHUNK_SAMPLES = 512
VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.5"))
VAD_MIN_SILENCE_MS = int(os.getenv("VAD_MIN_SILENCE_MS", "1500"))  # patience for thinking pauses
VAD_MIN_SPEECH_MS = 250
UTTERANCE_MAX_SECONDS = 20.0         # hard cap so we never listen forever

WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "base.en")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16")

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
CLAUDE_SYSTEM = (
    "You are Samantha, a warm, witty desktop voice assistant on Ramon's PC. "
    "Speak naturally and conversationally — short sentences, no markdown, no lists, "
    "no headings. Keep most answers to 1-3 sentences unless Ramon asks for detail. "
    "If you don't know something, say so plainly."
)
CLAUDE_MAX_TOKENS = int(os.getenv("CLAUDE_MAX_TOKENS", "400"))

KOKORO_MODEL_PATH = os.getenv("KOKORO_MODEL_PATH", "models/kokoro-v1.0.onnx")
KOKORO_VOICES_PATH = os.getenv("KOKORO_VOICES_PATH", "models/voices-v1.0.bin")
KOKORO_VOICE = os.getenv("KOKORO_VOICE", "af_bella")
KOKORO_SPEED = float(os.getenv("KOKORO_SPEED", "1.0"))
KOKORO_SR = 24_000

# Sentence-boundary regex: . ! ? followed by space or end-of-string
SENTENCE_END_RE = re.compile(r"([.!?])(\s|$)")
MIN_CHUNK_CHARS = 30                 # don't fire TTS for tiny fragments


# --- TTS worker -------------------------------------------------------------
class TTSPlayer:
    """
    Two-stage pipeline:
      sentences_q  -> generator thread (Kokoro)  -> audio_q  -> player thread (sounddevice)

    Sentence chunks are queued FIFO; audio plays in order while the next chunk
    is being synthesized in parallel.
    """

    _STOP = object()

    def __init__(self, kokoro: Kokoro, voice: str, speed: float) -> None:
        self.kokoro = kokoro
        self.voice = voice
        self.speed = speed
        self.sentences_q: queue.Queue = queue.Queue()
        self.audio_q: queue.Queue = queue.Queue(maxsize=8)
        self.gen_thread = threading.Thread(target=self._gen_loop, daemon=True)
        self.play_thread = threading.Thread(target=self._play_loop, daemon=True)
        self.gen_thread.start()
        self.play_thread.start()

    def say(self, text: str) -> None:
        """Queue a sentence/phrase to be spoken. Returns immediately."""
        text = text.strip()
        if text:
            self.sentences_q.put(text)

    def wait_idle(self, timeout: float | None = None) -> None:
        """Block until both queues are drained and current audio has finished."""
        deadline = None if timeout is None else time.time() + timeout
        while True:
            if (
                self.sentences_q.empty()
                and self.audio_q.empty()
                and not sd.get_stream().active if False else not _is_playing()
            ):
                return
            if deadline is not None and time.time() > deadline:
                return
            time.sleep(0.05)

    def _gen_loop(self) -> None:
        while True:
            text = self.sentences_q.get()
            if text is self._STOP:
                self.audio_q.put(self._STOP)
                return
            try:
                audio, sr = self.kokoro.create(text, voice=self.voice, speed=self.speed)
            except Exception as e:
                print(f"[tts:gen] error: {e}", file=sys.stderr)
                continue
            self.audio_q.put((audio, sr))

    def _play_loop(self) -> None:
        while True:
            item = self.audio_q.get()
            if item is self._STOP:
                return
            audio, sr = item
            try:
                sd.play(audio, sr)
                sd.wait()
            except Exception as e:
                print(f"[tts:play] error: {e}", file=sys.stderr)


def _is_playing() -> bool:
    try:
        return sd.get_stream().active
    except Exception:
        return False


# --- Audio capture ----------------------------------------------------------
def open_mic() -> "queue.Queue[np.ndarray]":
    q: queue.Queue[np.ndarray] = queue.Queue()

    def cb(indata, frames, _time, status):
        if status:
            print(f"[mic] {status}", file=sys.stderr)
        q.put(indata[:, 0].copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=VAD_CHUNK_SAMPLES,
        callback=cb,
    )
    stream.start()
    # keep a reference so the stream isn't GC'd
    q._stream = stream  # type: ignore[attr-defined]
    return q


def _to_float32(int16_audio: np.ndarray) -> np.ndarray:
    return int16_audio.astype(np.float32) / 32768.0


# --- Wake-word detection ----------------------------------------------------
def wait_for_wake(q: "queue.Queue[np.ndarray]", wake: WakeModel) -> None:
    """
    Block until the wake word fires. openwakeword wants 1280-sample chunks at
    16kHz; the mic gives us 512-sample VAD chunks, so we coalesce.
    """
    buf = np.empty(0, dtype=np.int16)
    while True:
        chunk = q.get()
        buf = np.concatenate([buf, chunk])
        while len(buf) >= WAKE_FRAME_SAMPLES:
            frame = buf[:WAKE_FRAME_SAMPLES]
            buf = buf[WAKE_FRAME_SAMPLES:]
            scores = wake.predict(frame)
            score = scores.get(WAKE_NAME, 0.0)
            if score >= WAKE_THRESHOLD:
                print(f"[wake] {WAKE_NAME} score={score:.2f}")
                wake.reset()
                return


# --- Endpointed recording ---------------------------------------------------
def record_until_silence(
    q: "queue.Queue[np.ndarray]",
    vad_iter: VADIterator,
) -> np.ndarray:
    """
    Record audio after wake word. silero-vad tracks speech vs. silence and
    fires an "end" event after VAD_MIN_SILENCE_MS of contiguous silence
    following speech — that's our turn-end signal. If speech resumes after
    "end", silero will fire another "start" and we keep recording.

    Hard cap at UTTERANCE_MAX_SECONDS so we never listen forever.
    """
    vad_iter.reset_states()
    collected: list[np.ndarray] = []
    speech_started = False
    saw_end = False
    start_time = time.time()

    while True:
        chunk_i16 = q.get()
        chunk_f32 = _to_float32(chunk_i16)
        if len(chunk_f32) != VAD_CHUNK_SAMPLES:
            continue
        speech_event = vad_iter(torch.from_numpy(chunk_f32), return_seconds=False)
        collected.append(chunk_i16)

        if speech_event is not None:
            if "start" in speech_event:
                speech_started = True
                # if user resumes after a pause, silero gives us another start —
                # forget the previous end so we keep going
                saw_end = False
            if "end" in speech_event:
                saw_end = True

        # Turn over only when we've seen speech AND silero has confirmed
        # the user has been silent long enough to fire "end"
        if speech_started and saw_end:
            break
        if time.time() - start_time >= UTTERANCE_MAX_SECONDS:
            break

    audio_i16 = np.concatenate(collected) if collected else np.zeros(0, dtype=np.int16)
    return _to_float32(audio_i16)


# --- LLM streaming + sentence chunking --------------------------------------
def stream_claude_to_tts(
    client: Anthropic,
    user_text: str,
    history: list[dict],
    tts: TTSPlayer,
) -> str:
    """
    Stream Claude's reply token-by-token. Whenever the buffer ends in a sentence
    boundary (and is at least MIN_CHUNK_CHARS long), flush the completed
    sentence(s) to TTS. Returns the full reply text.
    """
    history.append({"role": "user", "content": user_text})
    full_text_parts: list[str] = []
    buffer = ""

    def flush_sentences(force: bool = False) -> None:
        nonlocal buffer
        if not buffer.strip():
            return
        if force:
            tts.say(buffer)
            full_text_parts.append(buffer.strip())
            buffer = ""
            return
        # Find a sentence boundary at or after position MIN_CHUNK_CHARS-1 so
        # short openers ("Sure." "Of course.") get merged with the next clause
        # instead of being spoken alone.
        while True:
            search_from = max(0, MIN_CHUNK_CHARS - 1)
            m = SENTENCE_END_RE.search(buffer, search_from)
            if not m:
                break
            end_idx = m.end(1)
            sentence = buffer[:end_idx].strip()
            tts.say(sentence)
            full_text_parts.append(sentence)
            buffer = buffer[end_idx:].lstrip()

    with client.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=CLAUDE_MAX_TOKENS,
        system=CLAUDE_SYSTEM,
        messages=history,
    ) as stream:
        for text_delta in stream.text_stream:
            buffer += text_delta
            flush_sentences()

    # Flush any trailing fragment
    flush_sentences(force=True)
    full_reply = " ".join(p.strip() for p in full_text_parts).strip()
    history.append({"role": "assistant", "content": full_reply})
    return full_reply


# --- Main loop --------------------------------------------------------------
def main() -> None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("[fatal] ANTHROPIC_API_KEY not set. Copy .env.example to .env and fill it in.", file=sys.stderr)
        sys.exit(1)

    print(f"[boot] loading wake model: {WAKE_NAME}")
    wake = WakeModel(wakeword_models=[WAKE_NAME], inference_framework="onnx")

    print(f"[boot] loading whisper: {WHISPER_MODEL_NAME} on {WHISPER_DEVICE} ({WHISPER_COMPUTE_TYPE})")
    whisper = WhisperModel(WHISPER_MODEL_NAME, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)

    print("[boot] loading silero-vad")
    vad_model = load_silero_vad()
    vad_iter = VADIterator(
        vad_model,
        threshold=VAD_THRESHOLD,
        sampling_rate=SAMPLE_RATE,
        min_silence_duration_ms=VAD_MIN_SILENCE_MS,
        speech_pad_ms=100,
    )

    print(f"[boot] loading kokoro: {KOKORO_MODEL_PATH}, voice={KOKORO_VOICE}")
    kokoro = Kokoro(KOKORO_MODEL_PATH, KOKORO_VOICES_PATH)
    tts = TTSPlayer(kokoro, voice=KOKORO_VOICE, speed=KOKORO_SPEED)

    print("[boot] connecting to claude")
    claude = Anthropic()  # uses ANTHROPIC_API_KEY

    print("[boot] opening mic")
    q = open_mic()

    history: list[dict] = []

    tts.say(f"Online. Say {WAKE_NAME.replace('_', ' ')} when you need me.")
    print(f"[ready] waiting for wake word '{WAKE_NAME}' (threshold {WAKE_THRESHOLD})")

    while True:
        wait_for_wake(q, wake)
        # drain any audio queued before wake fired
        while not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                break

        tts.say("Mhm?")
        # let the brief acknowledgment finish before we start listening,
        # so the agent doesn't transcribe its own voice
        tts.wait_idle(timeout=1.5)

        print("[listen] recording until silence...")
        t0 = time.time()
        audio = record_until_silence(q, vad_iter)
        rec_dur = time.time() - t0
        print(f"[listen] captured {len(audio)/SAMPLE_RATE:.2f}s in {rec_dur:.2f}s")

        if len(audio) < SAMPLE_RATE * 0.3:
            print("[stt] too short, ignoring")
            continue

        print("[stt] transcribing...")
        t0 = time.time()
        segments, info = whisper.transcribe(audio, language="en", vad_filter=False)
        text = " ".join(s.text.strip() for s in segments).strip()
        print(f"[stt] ({time.time()-t0:.2f}s) you: {text!r}")

        if not text:
            tts.say("I did not catch that.")
            continue

        print("[claude] streaming...")
        t0 = time.time()
        try:
            reply = stream_claude_to_tts(claude, text, history, tts)
            print(f"[claude] ({time.time()-t0:.2f}s) reply: {reply!r}")
        except APIStatusError as e:
            print(f"[claude] HTTP {e.status_code}: {e.message}", file=sys.stderr)
            if e.status_code == 401:
                tts.say("My API key isn't working. Check your dot env file.")
            elif e.status_code == 429:
                tts.say("I'm being rate limited. Try again in a moment.")
            elif e.status_code >= 500:
                tts.say("Anthropic is having trouble. Try again soon.")
            else:
                tts.say("Something went wrong talking to Claude.")
            # remove the failed user turn from history so we don't poison context
            if history and history[-1].get("role") == "user":
                history.pop()
        except APIConnectionError as e:
            print(f"[claude] connection error: {e}", file=sys.stderr)
            tts.say("I can't reach the internet right now.")
            if history and history[-1].get("role") == "user":
                history.pop()

        # let the response finish playing before we listen for the next wake
        tts.wait_idle()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[exit] bye")
