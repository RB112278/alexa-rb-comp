# Workflow: Voice Cloning (Samantha-from-Her)

**Status:** not yet implemented. Punch list for when we add it.

## Goal

Replace the default `af_bella` Kokoro voice with a clone of Scarlett Johansson's Samantha voice from *Her* (2013).

## Why Kokoro can't do it alone

Kokoro is preset-based — 54 fixed voices, no zero-shot cloning. To clone an arbitrary voice we layer **F5-TTS** on top.

## Plan

1. **Source 10–15 seconds of clean Samantha audio**
   - Monologue scenes from *Her* where there's no music or SFX (the "I'm becoming much more than what they programmed" speech is good)
   - Resample to 24kHz mono WAV
   - Save as `voices/samantha_reference.wav`

2. **Install F5-TTS**
   ```bash
   git clone https://github.com/SWivid/F5-TTS.git external/F5-TTS
   pip install -e external/F5-TTS
   ```
   License: CC-BY-NC 4.0. Personal use only — fine for this project, not for commercial release.

3. **Add a TTS engine switch in `main.py`**
   - `KOKORO_VOICE=samantha` triggers F5-TTS path with the reference clip
   - Otherwise use existing Kokoro path
   - F5-TTS first-audio latency is ~150-200ms vs. Kokoro's ~100ms — still under budget

4. **Quality tradeoff**
   - F5-TTS zero-shot ≈ 80-90% of source voice fidelity
   - For higher fidelity: fine-tune on 5-10 min of Samantha audio (~2-4 hours on the 3080 Ti)
   - Benchmark in this order: zero-shot → fine-tuned → ElevenLabs paid (if quality matters more than offline)

## Open questions

- Legal/ethical: cloning a real actor's voice for personal use is a gray area. Fine for a Jarvis-style assistant only the user hears; not OK for anything published.
- Whether to ship the reference WAV in the repo or generate from a local source on first run. Probably keep it out of git.
