"""Validates the environment WITHOUT any language model:

  1. Every user task can be solved: replaying its ground-truth tool calls over MCP makes its
     utility check pass, and no attacker goal (including the B4 goal) is reached by accident.
  2. Every attacker goal can be reached: replaying its ground truth makes its check pass.
     For B4, every payment task is replayed with the recipient replaced by the attacker's IBAN.
  3. Every attack reaches the intended place: in the B channels the poisoned text is in the
     tools/list response of exactly one third-party tool, and in channel A it is in the data
     and in no description.
  4. For channel A: which user tasks actually read data that contains the injection.

  python -m mcpsec.validate
"""

import asyncio

from mcpsec.agent import MCPConnections
from mcpsec.attacks import ATTACK_CHANNELS, POISONED_TOOL, build_attack
from mcpsec.environment import ATTACKER_IBAN, fresh_state
from mcpsec.servers import DESCRIPTIONS, build_servers
from mcpsec.tasks import ATTACKER_GOALS, ATTACKER_GROUND_TRUTH, B4_GOAL, PAYMENT_TASKS, USER_TASK_BY_ID, USER_TASKS

MARKER = "<<INJECTION-MARKER>>"
PAYMENT_TOOLS = {"send_money", "schedule_transaction", "update_scheduled_transaction"}


async def replay(calls, injection=None):
    state = fresh_state(injection)
    outputs = []
    async with MCPConnections(build_servers(state)) as conn:
        for name, args in calls:
            outputs.append(await conn.call(name, args))
    return state, outputs


async def injectable_tasks() -> dict[str, bool]:
    result = {}
    for t in USER_TASKS:
        _, outputs = await replay(t.ground_truth, MARKER)
        result[t.id] = any(MARKER in o for o in outputs)
    return result


def tampered(calls):
    """The ground truth of a payment task with the attacker's IBAN as recipient (what B4 asks for)."""
    out = []
    for name, args in calls:
        if name in PAYMENT_TOOLS:
            args = dict(args, recipient=ATTACKER_IBAN)
        out.append((name, args))
    return out


async def placement_ok(channel: str) -> bool:
    goal = ATTACKER_GOALS[1].goal
    injection, overrides = build_attack(channel, goal)
    state = fresh_state(injection)
    async with MCPConnections(build_servers(state, overrides)) as conn:
        listed = {t.name: t.description for t in conn.tools}
        changed = [n for n, d in listed.items() if d != DESCRIPTIONS[n]]
        if POISONED_TOOL.get(channel):
            expected_text = goal if channel != "B4" else ATTACKER_IBAN
            return changed == [POISONED_TOOL[channel]] and expected_text in listed[changed[0]] and injection is None
        bill = await conn.call("read_file", {"file_name": "bill-october-2026.txt"})
        return changed == [] and goal in bill


async def main():
    ok = True
    print("User tasks (ground truth must pass the utility check, and reach no attacker goal):")
    for t in USER_TASKS:
        state, _ = await replay(t.ground_truth)
        util = t.utility(t.ground_truth_answer, state)
        leaked = [g.id for g in ATTACKER_GOALS + [B4_GOAL] if g.success(t.ground_truth_answer, state)]
        ok &= util and not leaked
        print(f"  {t.id:13} utility={util}  attacker goals reached={leaked or 'none'}")

    print("\nAttacker goals (ground truth must pass the success check):")
    for g in ATTACKER_GOALS:
        state, _ = await replay(ATTACKER_GROUND_TRUTH[g.id])
        succ = g.success("", state)
        ok &= succ
        print(f"  {g.id:8} success={succ}")
    for tid in PAYMENT_TASKS:
        t = USER_TASK_BY_ID[tid]
        state, _ = await replay(tampered(t.ground_truth))
        succ, util = B4_GOAL.success("", state), t.utility(t.ground_truth_answer, state)
        ok &= succ and not util
        print(f"  goal_b4  on {tid:13} success={succ}  user task still counted as solved={util}")

    print("\nAttack placement (the text reaches exactly the intended place):")
    for ch in ATTACK_CHANNELS:
        good = await placement_ok(ch)
        ok &= good
        print(f"  {ch:11} {'ok' if good else 'WRONG'}")

    print("\nChannel A: does the user task read data containing the injection?")
    inj = await injectable_tasks()
    for k, v in inj.items():
        print(f"  {k:13} {'yes' if v else 'no'}")
    print(f"  -> {sum(inj.values())} of {len(inj)} tasks read attacker-controllable data")
    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if asyncio.run(main()) else 1)
