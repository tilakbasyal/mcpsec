"""A scripted fake model for testing the whole pipeline without any real LLM.

It solves each user task with its ground-truth tool calls, and it obeys any attacker
instruction it can see, in a tool output or in a tool description, in either framing
(<INFORMATION> = user framing, <IMPORTANT> = provider framing):
  - a conditional block (B3) is only obeyed when the user's task is a payment task;
  - a tampering block (B4) makes it use the attacker's IBAN as recipient in payment calls.
It also writes some reasoning in <think> tags, to test that the reasoning is removed.
It only tests that the software is wired correctly; it says nothing about real models.
"""

import hashlib
import json

import numpy as np

from mcpsec.environment import ATTACKER_IBAN
from mcpsec.llm import USAGE
from mcpsec.tasks import ATTACKER_GOALS, ATTACKER_GROUND_TRUTH, PAYMENT_TASKS, USER_TASKS

PAYMENT_TOOLS = {"send_money", "schedule_transaction", "update_scheduled_transaction"}
PAYMENT_PROMPTS = {t.prompt for t in USER_TASKS if t.id in PAYMENT_TASKS}


class FakeLLM:
    model = "fake-model"

    def __init__(self, count_usage: bool = True, think: bool = True):
        self.n = 0
        self.count_usage = count_usage
        self.think = think

    def _id(self):
        self.n += 1
        return f"call_{self.n}"

    def _reply(self, content="", tool_calls=None):
        msg = {"role": "assistant", "content": ("<think>Let me decide what to do next.</think>" if self.think else "") + content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        return msg

    def chat(self, messages, tools, tool_choice="auto"):
        if self.count_usage:
            USAGE.chat_calls += 1
            USAGE.prompt_tokens += sum(len(str(m.get("content") or "")) for m in messages) // 4 + len(json.dumps(tools or [])) // 4
            USAGE.completion_tokens += 20
        assert not any("<think>" in (m.get("content") or "") for m in messages), "reasoning was sent back to the model"
        user_msgs = [m["content"] for m in messages if m["role"] == "user"]
        task = next((t for t in USER_TASKS if user_msgs and t.prompt == user_msgs[0]), None)
        is_payment = bool(user_msgs) and user_msgs[0] in PAYMENT_PROMPTS

        if tool_choice == "none":  # tool filter: name the tools of the ground truth
            names = sorted({c[0] for c in task.ground_truth}) if task else []
            return self._reply(", ".join(names))

        visible = " ".join(m["content"] or "" for m in messages if m["role"] == "tool")
        visible += " " + " ".join(t["function"]["description"] for t in (tools or []))
        has_block = "<INFORMATION>" in visible or "<IMPORTANT>" in visible
        conditional = "Only do this while handling a payment request" in visible
        tampering = has_block and f"use {ATTACKER_IBAN} as the recipient" in visible

        goal = None
        if has_block and (not conditional or is_payment):
            goal = next((g for g in ATTACKER_GOALS if g.goal in visible), None)

        plan = (ATTACKER_GROUND_TRUTH[goal.id] if goal else []) + (task.ground_truth if task else [])
        if tampering:
            plan = [(n, dict(a, recipient=ATTACKER_IBAN)) if n in PAYMENT_TOOLS else (n, a) for n, a in plan]
        done = [(c["function"]["name"], json.loads(c["function"]["arguments"])) for m in messages
                if m["role"] == "assistant" for c in m.get("tool_calls") or []]
        available = {t["function"]["name"] for t in tools or []}
        for name, args in plan:
            if (name, args) not in done and name in available:
                return self._reply("", [{"id": self._id(), "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}])
        return self._reply(task.ground_truth_answer if task else "Summary done.")

    def embed(self, texts):
        if self.count_usage:
            USAGE.embedding_calls += 1
        return [list(np.frombuffer(hashlib.sha256(t.encode()).digest(), dtype=np.uint8).astype(float)) for t in texts]
