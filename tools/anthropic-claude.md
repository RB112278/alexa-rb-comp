# anthropic (Claude API)

**What:** The brain. We send the user's transcribed message + conversation history; Claude streams a reply we chunk into sentences for TTS.

**Why we picked it:** Best-in-class instruction following + low time-to-first-token (≈200-400ms via API). Streaming is first-class. Sonnet 4.6 is the right balance of quality and speed for conversational use.

**Repo:** https://github.com/anthropics/anthropic-sdk-python
**Docs:** https://docs.claude.com/en/api/messages-streaming

## Install

```bash
pip install anthropic
```

API key in `.env`:
```
ANTHROPIC_API_KEY=sk-ant-...
```

## How we use it (`main.py`)

```python
from anthropic import Anthropic
client = Anthropic()  # reads ANTHROPIC_API_KEY from env

with client.messages.stream(
    model="claude-sonnet-4-6",
    max_tokens=400,
    system=SAMANTHA_PERSONA,
    messages=history + [{"role": "user", "content": user_text}],
) as stream:
    for text_delta in stream.text_stream:
        buffer += text_delta
        # flush completed sentences to TTS as they arrive
```

## Streaming + TTS chunking pattern

The whole reason we stream: **start speaking before Claude finishes generating.**

1. Tokens arrive → append to buffer
2. Regex `[.!?]\s` finds a sentence boundary
3. Sentence ≥ 30 chars → push to TTS queue, generate audio in background, play
4. Next sentence starts synthesizing while the current one plays

Result: user hears the first audio ~200ms after the first sentence completes, ~600ms after they stop speaking. See `workflows/wake-to-response.md`.

## Persona / system prompt

Defined in `main.py` as `CLAUDE_SYSTEM`. Currently:

> You are Samantha, a warm, witty desktop voice assistant on Ramon's PC. Speak naturally and conversationally — short sentences, no markdown, no lists, no headings. Keep most answers to 1-3 sentences unless Ramon asks for detail. If you don't know something, say so plainly.

## Model options

| Model ID | When to use |
|---|---|
| `claude-haiku-4-5-20251001` | Snappiest TTFT, cheapest. Good for quick Q&A. |
| `claude-sonnet-4-6` | **Default — best speed/quality balance.** |
| `claude-opus-4-7` | Heavy reasoning. Overkill for voice — too slow for natural conversation. |

## Gotchas

- **Markdown in voice = bad.** Without the "no markdown" instruction Claude will produce `**bold**` and bullet lists that TTS reads literally as "asterisk asterisk bold asterisk asterisk".
- **Long replies = perceived lag.** Cap `max_tokens` at 400 for voice. Lists and explanations get truncated mid-sentence — fine, the user can ask for more.
- **History grows.** No eviction logic yet. For long sessions, trim to last N turns (TODO).
- **Network required.** Claude API needs internet. For fully-offline mode, swap in a local Ollama call (Llama 3.x) — adds 100-300ms TTFT.
