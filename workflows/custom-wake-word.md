# Workflow: Custom Wake Word ("Hey Ramon")

**Status:** not yet implemented. Punch list.

## Why

The pretrained openwakeword set is `alexa`, `hey_jarvis`, `hey_mycroft`, `hey_rhasspy`. Single-word custom names like "Samantha" are notoriously twitchy — false-trigger on any word with similar phonemes. A two-syllable phrase like "Hey Ramon" is far more robust.

## Plan

openwakeword ships an official training notebook that uses synthetic TTS data — no need to record yourself thousands of times.

1. **Clone the training tools**
   ```bash
   git clone https://github.com/dscripka/openWakeWord.git external/oww-train
   cd external/oww-train
   pip install -r requirements_training.txt
   ```

2. **Run their Colab-style notebook locally**
   - Notebook: `notebooks/automatic_model_training_simple.ipynb`
   - Inputs: target phrase string ("hey ramon"), number of synthetic positive samples (default 10K), pretrained negative dataset (downloads automatically — ~5GB)
   - Outputs: `hey_ramon.onnx` and `hey_ramon.tflite` files

3. **On the 3080 Ti**
   - TTS positive sample generation: ~10-20 min (uses Piper TTS to generate the wake phrase in many voices/inflections)
   - Training: ~15-30 min for the small classifier head on top of the frozen embedding model
   - Total: ~30-50 min end to end

4. **Drop the .onnx file into the project**
   ```bash
   cp external/oww-train/output/hey_ramon.onnx models/
   ```

5. **Use it**
   ```python
   wake = WakeModel(
       wakeword_models=["models/hey_ramon.onnx"],   # path instead of name
       inference_framework="onnx",
   )
   ```
   And update `.env`: `WAKE_WORD=hey_ramon`

6. **Tune threshold** — custom models often need higher thresholds (0.6-0.8) to avoid false fires. Test against TV / podcast audio for an hour and adjust.

## Alternative: Picovoice Porcupine

If openwakeword self-training proves twitchy:
- Free tier (3 users) supports custom wake words generated in their console
- Generate `samantha_windows.ppn` in 5 minutes at console.picovoice.ai
- Trade-off: not open source, requires access key
- Code change in `main.py`: ~30 lines, swap `WakeModel` for `pvporcupine.create(keyword_paths=[...], access_key=...)`

We'd document that as a separate `tools/porcupine.md` if/when we go that route.
