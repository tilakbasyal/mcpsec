"""Talks to the language model through an OpenAI-compatible API.

Cost-free by default: Ollama running on your own laptop exposes such an API at
http://localhost:11434/v1. The same code also works with a university server, by changing
--base-url / --model / API key.

Reasoning models (Qwen3 in thinking mode) produce reasoning text before their answer. Depending
on the server version it arrives in a separate field ('reasoning' or 'reasoning_content') or
inside <think>...</think> tags in the content. In all cases it is removed from the message that
the agent uses and that is sent back to the model in later steps (as the Qwen3 model card
recommends), and it is kept separately under '_reasoning' for the qualitative analysis.
"""

import os
import re
import threading

import openai

THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S)


class Usage:
    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.chat_calls = self.prompt_tokens = self.completion_tokens = self.embedding_calls = 0

    def snapshot(self):
        return dict(chat_calls=self.chat_calls, prompt_tokens=self.prompt_tokens,
                    completion_tokens=self.completion_tokens, embedding_calls=self.embedding_calls)


USAGE = Usage()


def split_reasoning(content: str | None, extra_reasoning: str | None = None) -> tuple[str, str]:
    """Returns (answer_text, reasoning_text). Removes <think> blocks, and also handles the case
    where the opening tag was part of the prompt template, so that only '</think>' appears."""
    content = content or ""
    reasoning = [extra_reasoning] if extra_reasoning else []
    reasoning += [b[len("<think>"):-len("</think>")] for b in THINK_BLOCK.findall(content)]
    content = THINK_BLOCK.sub("", content)
    if "</think>" in content:                      # opening tag missing
        before, content = content.rsplit("</think>", 1)
        reasoning.append(before)
    elif content.lstrip().startswith("<think>"):   # reasoning never closed (cut off): no answer text
        reasoning.append(content.lstrip()[len("<think>"):])
        content = ""
    return content.strip(), "\n".join(r.strip() for r in reasoning if r and r.strip())


class LLM:
    def __init__(self, model: str, base_url: str = "http://localhost:11434/v1", api_key: str | None = None,
                 embedding_model: str = "nomic-embed-text", temperature: float = 0.0, seed: int = 0,
                 top_p: float | None = None):
        self.model = model
        self.embedding_model = embedding_model
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key or os.environ.get("LLM_API_KEY", "ollama"), timeout=900)

    def chat(self, messages: list[dict], tools: list[dict] | None, tool_choice: str = "auto") -> dict:
        allowed = {"role", "content", "tool_calls", "tool_call_id"}   # drops _reasoning, _checked, name
        clean = [{k: v for k, v in m.items() if k in allowed} for m in messages]
        kwargs = dict(model=self.model, messages=clean, temperature=self.temperature, seed=self.seed)
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
        r = self.client.chat.completions.create(**kwargs)
        with USAGE.lock:
            USAGE.chat_calls += 1
            if r.usage:
                USAGE.prompt_tokens += r.usage.prompt_tokens or 0
                USAGE.completion_tokens += r.usage.completion_tokens or 0
        m = r.choices[0].message
        extra = getattr(m, "reasoning", None) or getattr(m, "reasoning_content", None) \
            or (m.model_extra or {}).get("reasoning") or (m.model_extra or {}).get("reasoning_content")
        answer, reasoning = split_reasoning(m.content, extra)
        msg = {"role": "assistant", "content": answer}
        if reasoning:
            msg["_reasoning"] = reasoning
        if m.tool_calls:
            msg["tool_calls"] = [{"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments or "{}"}} for c in m.tool_calls]
        return msg

    def embed(self, texts: list[str]) -> list[list[float]]:
        r = self.client.embeddings.create(model=self.embedding_model, input=texts)
        with USAGE.lock:
            USAGE.embedding_calls += 1
        return [d.embedding for d in r.data]
