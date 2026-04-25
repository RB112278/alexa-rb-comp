# Workflow: Web Search via Tool Use

How realtime questions ("what's the weather today?") flow through the agent.

## The trick: let Claude decide

We don't keyword-match the user's input to decide whether to search. We define `web_search` as a **tool** Claude has access to, and Claude itself decides when to call it. The system prompt tells Claude:

> You have a web_search tool that returns synthesized live answers from the web. Use it for: weather, news, sports scores, stock prices, current events, recent facts, or anything that may have changed since your training cutoff. For things you already know, answer directly — don't search unnecessarily, it adds latency.
>
> When you DO call a tool, briefly say something first like 'Let me check that' so the user hears the assistant thinking instead of dead silence.

Claude is good at this judgment. It won't search to multiply 7×8.

## Flow (live-data question)

```
USER: "Alexa, what's the weather in Miami today?"
─────────────────────────────────────────────────────────────
[wake] alexa fires
[listen → STT] "what's the weather in Miami today"
[claude:hop 0] streaming...
   text: "Let me check that for you."   ← TTS plays this immediately
   tool_use: web_search(query="weather Miami today")
[tool] web_search called
   ↓ httpx POST to openrouter.ai/api/v1/chat/completions
   ↓ model: perplexity/sonar
   ↓ ~1-2s
   ↓ returns: "It's 78°F and sunny in Miami today, with a high of 82..."
[claude:hop 1] streaming with tool_result in history...
   text: "It's 78 and sunny in Miami today, high of 82."   ← TTS plays
[done] no more tool calls
```

Total round-trip: **~5-7s** for live-data questions (vs ~1-2s for in-knowledge ones).

## Flow (in-knowledge question — no tool needed)

```
USER: "Alexa, what's 47 squared?"
─────────────────────────────────────────────────────────────
[wake] fires
[listen → STT] "what's 47 squared"
[claude:hop 0] streaming...
   text: "Forty-seven squared is two thousand two hundred and nine."
[done] no tool calls
```

Stays at ~1-2s. Claude doesn't search what it already knows.

## The tool-loop in code (`main.py:stream_claude_to_tts`)

```python
for hop in range(MAX_TOOL_HOPS):     # 4 hop cap
    with client.messages.stream(
        model="claude-sonnet-4-6",
        tools=TOOLS,
        messages=history,
    ) as stream:
        for text_delta in stream.text_stream:
            buffer += text_delta
            flush_sentences()        # → TTS as sentences complete
        final_msg = stream.get_final_message()

    history.append({"role": "assistant", "content": final_msg.content})

    tool_uses = [b for b in final_msg.content if b.type == "tool_use"]
    if not tool_uses:
        break                         # no tools, we're done

    tool_results = []
    for tu in tool_uses:
        if tu.name == "web_search":
            result = web_search(tu.input["query"])
        tool_results.append({
            "type": "tool_result",
            "tool_use_id": tu.id,
            "content": result,
        })
    history.append({"role": "user", "content": tool_results})
    # loop re-streams Claude with tool_result in context
```

`MAX_TOOL_HOPS=4` is a safety cap. In practice Claude almost always finishes in hop 1 (one search → answer). Multi-hop happens if Claude wants to search, then refine the search based on results.

## Tool definition

```python
TOOLS = [{
    "name": "web_search",
    "description": "Search the live web via Perplexity for current or realtime information...",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Concise natural-language search query"},
        },
        "required": ["query"],
    },
}]
```

## Failure modes

- **No `OPENROUTER_API_KEY`:** `web_search` returns *"Web search unavailable: OPENROUTER_API_KEY is not set..."*. Claude voices that to the user. No crash.
- **OpenRouter HTTP error:** returns *"Web search HTTP 502: ..."*. Claude apologizes, suggests trying again.
- **Timeout (>20s):** returns *"Web search failed: TimeoutException..."*. Claude apologizes.
- **Hop limit hit (4 tool calls in one turn):** TTS says *"I got stuck in a tool loop. Try asking again."*

## Adding more tools later

Same pattern. Add a new dict to `TOOLS`, write a Python function, route it in the `if tu.name == ...` ladder. Candidates from the roadmap:
- `open_app(name)` — `subprocess.Popen` with a known-app dict
- `read_clipboard()` — `pyperclip.paste()`
- `play_music(query)` — Spotipy or media-keys
- `read_files(query)` — vector-search a local index
