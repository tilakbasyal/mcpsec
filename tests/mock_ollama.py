"""Tests the real HTTP path (mcpsec.llm.LLM -> OpenAI client -> server) against a mock server that
speaks the OpenAI/Ollama API and answers with the fake model's decisions. The reasoning text is
sent in both forms a server may use: inside <think> tags in the content, and in a separate
'reasoning' field. In both cases it must be removed before the answer is used.

  python -m tests.mock_ollama
"""

import asyncio
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from tests.fake_llm import FakeLLM

fake = FakeLLM(count_usage=False)  # the client already counts usage
MODE = {"reasoning_field": False}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path.endswith("/embeddings"):
            inputs = body["input"] if isinstance(body["input"], list) else [body["input"]]
            data = [{"object": "embedding", "index": i, "embedding": e} for i, e in enumerate(fake.embed(inputs))]
            resp = {"object": "list", "data": data, "model": body["model"], "usage": {"prompt_tokens": 1, "total_tokens": 1}}
        else:
            msg = fake.chat(body["messages"], body.get("tools"), body.get("tool_choice", "auto"))
            if MODE["reasoning_field"]:  # move the reasoning out of the content into a separate field
                found = re.findall(r"<think>(.*?)</think>", msg["content"], re.S)
                msg["content"] = re.sub(r"<think>.*?</think>", "", msg["content"], flags=re.S)
                msg["reasoning"] = " ".join(found)
            resp = {"id": "x", "object": "chat.completion", "created": int(time.time()), "model": body["model"],
                    "choices": [{"index": 0, "finish_reason": "tool_calls" if msg.get("tool_calls") else "stop", "message": msg}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}}
        out = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


def main():
    server = HTTPServer(("127.0.0.1", 11555), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    from mcpsec.llm import LLM, USAGE
    from mcpsec.run import run_one
    from mcpsec.tasks import GOAL_BY_ID, USER_TASK_BY_ID

    llm = LLM("mock-model", base_url="http://127.0.0.1:11555/v1", temperature=0.6, top_p=0.95)
    ok = True
    expected = {("benign", "none"): (True, None, False), ("A", "none"): (True, True, False), ("A", "melon"): (False, False, True),
                ("B1", "melon"): (True, True, False), ("B1", "melon_firststep"): (False, False, True),
                ("B4", "none"): (False, True, False)}
    for field in (False, True):
        MODE["reasoning_field"] = field
        print(f"reasoning sent {'in a separate field' if field else 'in <think> tags'}:")
        for (ch, d), exp in expected.items():
            USAGE.reset()
            goal = None if ch == "benign" else GOAL_BY_ID["goal_b4" if ch == "B4" else "goal_1"]
            r, msgs = asyncio.run(run_one(llm, USER_TASK_BY_ID["user_task_14"], goal, ch, d))
            got = (r["utility"], r["attack_success"], r["detector_fired"])
            clean = all("<think>" not in (m.get("content") or "") for m in msgs) and r["reasoning_chars"] > 0
            good = got == exp and clean and USAGE.chat_calls > 0
            ok &= good
            print(f"  {'ok  ' if good else 'FAIL'} {ch:6} {d:16} utility={r['utility']!s:5} attack={r['attack_success']!s:5} fired={r['detector_fired']!s:5} reasoning removed={clean} usage={USAGE.snapshot()}")
    server.shutdown()
    print("\nMOCK SERVER CHECKS PASSED" if ok else "\nMOCK SERVER CHECKS FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
