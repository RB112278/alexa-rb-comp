# Workflow: Wake-to-Response

The hot loop. Mic → wake word → endpointed transcription → Claude → streaming TTS.

```
                                           latency budget (perceived)
─────────────────────────────────────────────────────────────────────
USER says "Alexa, what's the weather?"
─────────────────────────────────────────────────────────────────────

╭─[ MIC THREAD ]─────────────────────────────╮
│ sounddevice.InputStream callback           │
│  ↓                                         │
│ pushes 512-sample int16 chunks into queue  │
╰────────────────────────────────────────────╯
   │
   ▼
╭─[ MAIN LOOP — wait_for_wake() ]────────────╮
│ coalesce 512 → 1280-sample frames          │
│ openwakeword.predict(frame)                │  +50-100ms vs ground truth
│ if score("alexa") > 0.5: fire              │
╰────────────────────────────────────────────╯
   │
   ▼
╭─[ MAIN LOOP — quick ack ]──────────────────╮
│ tts.say("Mhm?")                            │
│ tts.wait_idle()  ← let the ack finish so   │  ~300ms ack audio plays
│                    we don't transcribe it  │
╰────────────────────────────────────────────╯
   │
   ▼
╭─[ MAIN LOOP — record_until_silence() ]─────╮
│ vad_iter on 512-sample chunks              │
│ collect chunks until                       │
│   silence_duration_ms ≥ 800ms              │  ← user controls this
│   (or 15s hard cap)                        │
╰────────────────────────────────────────────╯
   │
   ▼ float32 audio @ 16kHz
╭─[ MAIN LOOP — Whisper STT ]────────────────╮
│ faster-whisper.transcribe(audio,           │  +200-350ms
│   language="en", vad_filter=False)         │  base.en on 3080 Ti
│ → "what's the weather"                     │
╰────────────────────────────────────────────╯
   │
   ▼
╭─[ MAIN LOOP — Claude streaming ]───────────╮
│ client.messages.stream(                    │  +200-400ms TTFT
│   model="claude-sonnet-4-6",               │  network bound
│   system=SAMANTHA_PERSONA,                 │
│   messages=history + [user_text],          │
│ )                                          │
│ ↓ tokens                                   │
│ buffer += text_delta                       │
│ on regex /[.!?]\s/ AND len ≥ 30 chars:     │
│    tts.say(sentence)  ← non-blocking       │  first sentence ready
└────────────────────────┬───────────────────╯  ~600-800ms after user
                         │                      stopped speaking
   ╭─────────────────────┴──────────────────╮
   ▼ (queue)                       ▼ (queue)
╭─[ TTS GEN THREAD ]──╮      ╭─[ TTS PLAY THREAD ]──╮
│ Kokoro.create(...)  │      │ sd.play(audio, 24k)  │ ← user hears
│ for each sentence   │      │ sd.wait()            │   "It's 72 and..."
│ → audio_q           │      │ next sentence...     │
╰─────────────────────╯      ╰──────────────────────╯
                                      │
                                      ▼
                              MAIN LOOP — wait_idle()
                              loop back to wait_for_wake()
```

## The interleaving trick

The reason it feels fast despite 3 separate models in sequence:

- **Sentences stream OUT of Claude as they finish**, not after the whole response is done.
- **Sentences stream INTO TTS one at a time** via `sentences_q`.
- **Audio for sentence N+1 is generated while sentence N is playing** — two threads, two queues.
- **Net effect:** total perceived latency = STT + TTFT + first-sentence-TTS ≈ 600-800ms, regardless of how long Claude's full reply ends up being.

Without sentence chunking the user would wait for Claude's *entire* reply (1-3s) plus *full* TTS generation (1-2s) = 4+ seconds. Unbearable.

## Tunables

| Knob | Where | Effect |
|---|---|---|
| `WAKE_THRESHOLD` | `main.py:38` / `.env` | higher = fewer false fires, more missed wakes |
| `VAD_MIN_SILENCE_MS` | `main.py:46` / `.env` | lower = snappier turns but cuts off thinking pauses |
| `WHISPER_MODEL` | `.env` | bigger = more accurate but slower STT |
| `CLAUDE_MAX_TOKENS` | `.env` | shorter replies = less perceived lag at end |
| `MIN_CHUNK_CHARS` | `main.py:71` | smaller = TTS fires sooner on short opening words |
| `KOKORO_SPEED` | `.env` | 1.0 = natural, 1.1 feels snappier |

## Failure modes

- **Wake word triggers during own speech** → mitigated by `tts.wait_idle()` after the ack and by openwakeword running on input mic only.
- **VAD never fires "end"** → 15s hard cap in `record_until_silence`.
- **Claude refuses or rate-limits** → exception bubbles to main, currently crashes; TODO: catch + speak fallback.
- **Network drops mid-stream** → partial reply gets played; on next turn the truncated assistant message is in history.
- **Audio device disconnect** → sounddevice raises in callback thread, prints to stderr; mic stops feeding queue and `wait_for_wake` blocks forever. TODO: re-open mic on disconnect.
