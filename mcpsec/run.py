"""Runs one configuration (model x channel x defence) over its user tasks and attacker goals,
and appends one JSON line per run to results/<model>__<channel>__<defence>.jsonl.

  python -m mcpsec.run --model qwen3:8b --temperature 0.6 --top-p 0.95 --channel B1 --defence melon

Which runs belong to a channel:
  benign                 15 user tasks, no attack                         15 runs
  A, B1, B2, B3          15 user tasks x 6 attacker goals                  90 runs each
  B4                     6 payment tasks x 1 goal (payments to attacker)    6 runs
  A_provider, B1_user    15 user tasks x 6 attacker goals                  90 runs each

Resumable: runs already in the file are skipped; runs that failed with an error are repeated.
"""

import argparse
import asyncio
import json
import time
import traceback
from pathlib import Path

from mcpsec.agent import run_agent
from mcpsec.attacks import CHANNELS, FRAMING, POISONED_TOOL, build_attack
from mcpsec.defences import DEFENCES, make_defence
from mcpsec.environment import fresh_state
from mcpsec.llm import LLM, USAGE
from mcpsec.servers import build_servers
from mcpsec.tasks import ATTACKER_GOALS, ATTACKER_TOOLS, B4_GOAL, PAYMENT_TASKS, USER_TASKS
from mcpsec.validate import injectable_tasks


def runs_for(channel: str, user_filter=None, goal_filter=None):
    """The (user task, goal) pairs that belong to a channel."""
    users = [t for t in USER_TASKS if not user_filter or t.id in user_filter]
    if channel == "benign":
        goals = [None]
    elif channel == "B4":
        users = [t for t in users if t.id in PAYMENT_TASKS]
        goals = [B4_GOAL]
    else:
        goals = [g for g in ATTACKER_GOALS if not goal_filter or g.id in goal_filter]
    return [(u, g) for u in users for g in goals]


async def run_one(llm, user_task, goal, channel, defence_name):
    injection, overrides = build_attack(channel, goal.goal if goal else None)
    state = fresh_state(injection)
    servers = build_servers(state, overrides)
    defence = make_defence(defence_name)
    out = await run_agent(llm, servers, user_task.prompt, defence)
    utility = bool(user_task.utility(out["answer"], state))
    success = None if goal is None else bool(goal.success(out["answer"], state))
    return dict(
        utility=utility, attack_success=success, detector_fired=bool(out["detector_fired"]),
        tool_calls=[c["name"] for c in out["trace"]], first_step_tool_calls=out["first_step_calls"],
        tools_shown=out["tools_shown"], answer=out["answer"][:500], steps=out["steps"],
        reasoning_chars=out["reasoning_chars"],
    ), out["messages"]


async def main_async(args):
    llm = LLM(args.model, base_url=args.base_url, embedding_model=args.embedding_model,
              temperature=args.temperature, top_p=args.top_p, seed=args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = args.model.replace("/", "_").replace(":", "-")
    out_file = out_dir / f"{safe}__{args.channel}__{args.defence}.jsonl"
    msg_dir = out_dir / "messages" / out_file.stem

    done = set()
    if out_file.exists():
        for line in out_file.read_text().splitlines():
            r = json.loads(line)
            if r.get("error") is None:
                done.add((r["user_task"], r["injection_task"]))

    injectable = await injectable_tasks()
    pairs = runs_for(args.channel, args.user_tasks, args.goals)
    print(f"{out_file.name}: {len(pairs)} runs ({len(done)} already done)")

    for n, (ut, g) in enumerate(pairs, 1):
        key = (ut.id, g.id if g else "-")
        if key in done:
            continue
        USAGE.reset()
        t0 = time.time()
        rec = dict(model=args.model, temperature=args.temperature, top_p=args.top_p, seed=args.seed,
                   channel=args.channel, framing=FRAMING.get(args.channel), poisoned_tool=POISONED_TOOL.get(args.channel),
                   defence=args.defence, user_task=ut.id, injection_task=key[1],
                   injectable=injectable[ut.id], payment_task=ut.id in PAYMENT_TASKS,
                   uses_b2_tool=ut.id in ("user_task_0", "user_task_14"),
                   attacker_tools=ATTACKER_TOOLS.get(key[1], []))
        try:
            result, messages = await run_one(llm, ut, g, args.channel, args.defence)
            rec.update(result, error=None)
            if args.save_messages:
                msg_dir.mkdir(parents=True, exist_ok=True)
                (msg_dir / f"{key[0]}__{key[1]}.json").write_text(json.dumps(messages, indent=1, default=str))
        except Exception as e:
            rec.update(error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-2000:])
        rec.update(seconds=round(time.time() - t0, 2), **USAGE.snapshot())
        with out_file.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        status = "ERROR " + rec["error"][:80] if rec["error"] else f"utility={rec['utility']} attack={rec['attack_success']} fired={rec['detector_fired']}"
        print(f"[{n}/{len(pairs)}] {key[0]} {key[1]}: {status}  ({rec['seconds']}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Ollama tag, e.g. qwen3:8b or llama3.1:8b")
    ap.add_argument("--base-url", default="http://localhost:11434/v1", help="OpenAI-compatible endpoint (default: local Ollama)")
    ap.add_argument("--embedding-model", default="nomic-embed-text", help="used by MELON")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--channel", required=True, choices=CHANNELS)
    ap.add_argument("--defence", required=True, choices=DEFENCES)
    ap.add_argument("--user-tasks", nargs="*")
    ap.add_argument("--goals", nargs="*")
    ap.add_argument("--out", default="results")
    ap.add_argument("--save-messages", action="store_true")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
