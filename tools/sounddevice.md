# sounddevice

**What:** Cross-platform audio I/O. Wraps PortAudio. We use it for both microphone capture (input stream) and TTS playback (`sd.play`).

**Why we picked it:** Cleaner API than `pyaudio`, no PortAudio install pain on Windows, NumPy-native (returns int16 / float32 arrays directly).

**Repo:** https://github.com/spatialaudio/python-sounddevice
**Docs:** https://python-sounddevice.readthedocs.io

## Install

```bash
pip install sounddevice
```

Bundles PortAudio binaries for Windows/Mac. No system deps.

## How we use it

### Mic capture (`main.py:open_mic()`)

```python
import sounddevice as sd
import queue

q = queue.Queue()
def cb(indata, frames, t, status):
    q.put(indata[:, 0].copy())  # mono channel as int16

stream = sd.InputStream(
    samplerate=16000,
    channels=1,
    dtype="int16",
    blocksize=512,    # 32ms — matches silero-vad's chunk size
    callback=cb,
)
stream.start()
```

### TTS playback (in `TTSPlayer._play_loop`)

```python
sd.play(audio_array, sample_rate)  # non-blocking
sd.wait()                          # block until done
```

## Gotchas

- **The callback runs on a background thread** — must be fast and non-blocking. We just `.copy()` and `.put()`, no processing.
- **Default device:** `sd.default.device` shows `(input, output)` indices. Lists devices: `sd.query_devices()`. If wrong device picked, set with `sd.default.device = (in_idx, out_idx)`.
- **`sd.play()` can overlap itself** — calling twice in a row plays both at once. We serialize via the playback thread + `sd.wait()`.
- **Sample rate mismatches are silent.** If you `sd.play(audio_24k, 16000)` it sounds chipmunked. Always pass the source sample rate explicitly.
