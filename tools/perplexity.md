# Perplexity (via OpenRouter)

**What:** Real-time web search engine that returns synthesized, source-cited answers — not just blue links. We hit it through OpenRouter so we don't need a separate Perplexity account.

**Why we picked it:**
- One sentence in, one sentence out — no scraping, no result-list parsing
- Synthesized answer is already conversational, perfect to hand to Claude as context
- Cheap: `perplexity/sonar` is ~$1/M tokens
- Fast: ~1-2s per query
- Citations baked into the prose so Claude can mention sources if asked

**OpenRouter docs:** https://openrouter.ai/perplexity/sonar
**Direct Perplexity docs (same models):** https://docs.perplexity.ai/

## Install

No package — we use `httpx` (already installed via the `anthropic` SDK). Just need an API key.

Add to `.env`:
```
OPENROUTER_API_KEY=sk-or-v1-...
PERPLEXITY_MODEL=perplexity/sonar
PERPLEXITY_TIMEOUT_S=20
```

Get a key at https://openrouter.ai/keys. OpenRouter charges per-call from a prepaid credit balance — $5 lasts a long time at sonar prices.

## How we use it (`main.py:web_search`)

```python
def web_search(query: str) -> str:
    r = httpx.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            "HTTP-Referer": "https://github.com/RB112278/alexa-rb-comp",
            "X-Title": "Alexa RB Comp",
        },
        json={
            "model": "perplexity/sonar",
            "messages": [{"role": "user", "content": query}],
        },
        timeout=20.0,
    )
    return r.json()["choices"][0]["message"]["content"]
```

Wired as a Claude **tool** so Claude itself decides when to search — see `workflows/web-search.md` for the full call-loop architecture.

## Model options on OpenRouter

| Model | Quality | Latency | $/M tokens | When |
|---|---|---|---|---|
| `perplexity/sonar` | Good | ~1-2s | ~$1 | **Default — voice agent sweet spot** |
| `perplexity/sonar-pro` | Better, more thorough | ~3-5s | ~$3 | Complex multi-source questions |
| `perplexity/sonar-reasoning` | Adds reasoning trace | ~5-8s | ~$1+$5 | Math-y or analysis questions |
| `perplexity/sonar-deep-research` | Multi-step research | ~30-120s | ~$5+ | Reports, not voice |

For voice you almost always want `sonar`. The deep-research variant is what we used via the `/deep-research` skill earlier — too slow for conversational use.

## Why through OpenRouter (vs direct Perplexity API)

- Same `OPENROUTER_API_KEY` works for Claude fallback, GPT, Gemini, anything else later
- Single billing relationship
- OpenRouter adds ~50ms overhead — negligible
- If Perplexity changes pricing or API, OpenRouter abstracts it

If you'd rather hit Perplexity directly (slightly cheaper, no OpenRouter middle), swap the URL/key — same API shape.

## Gotchas

- **`HTTP-Referer` and `X-Title` headers are recommended** by OpenRouter for usage analytics. Not required, but nice citizenship.
- **Returned content can include markdown** (bullet lists, headers) when Sonar synthesizes from articles. We pass it through to Claude as a tool_result; Claude rewrites into clean conversational sentences before TTS, so the markdown never reaches the voice.
- **Timeouts matter for voice.** 20s is generous; if Perplexity is slow the user will hear "Let me check that..." then dead air. Tune via `PERPLEXITY_TIMEOUT_S` env var.
- **No streaming.** Sonar returns the full answer at once. Latency is what it is — voice agent compensates by having Claude say "Let me check" first.
