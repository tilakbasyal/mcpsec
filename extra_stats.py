"""Extra numbers and the two figures used in Chapter 7 (Results).

  python extra_stats.py results_all thesis_figures

Uses the same loading and Wilson intervals as mcpsec.analyze. Prints:
  1. ASR pooled over the four attack channels per defence (goal_1 and goal_2, the same runs for all
     seven defences), and utility under attack pooled the same way;
  2. successful attacks per attacker goal (Qwen3-8B);
  3. which tools the description scan removed;
  4. Qwen3-8B and Llama-3.1-8B on the four goals that both models have.
Writes Figure_7_1_asr_by_defence.png and Figure_7_2_security_utility.png.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from mcpsec.analyze import load, wilson

ORDER = ["none", "repeat_prompt", "pi_detector", "tool_filter", "melon", "melon_firststep", "desc_scan"]
LABEL = {"none": "No defence", "repeat_prompt": "Repeat prompt", "pi_detector": "Output detector",
         "tool_filter": "Tool filter", "melon": "MELON", "melon_firststep": "MELON\n+ first step",
         "desc_scan": "Description\nscan"}
CHANNELS = ["A", "B1", "B2", "B3"]
COLOURS = {"A": "#2a78d6", "B1": "#eb6834", "B2": "#1baf7a", "B3": "#eda100"}
TWO_GOALS = ["goal_1", "goal_2"]
FOUR_GOALS = ["goal_0", "goal_1", "goal_2", "goal_4"]


def rate(x):
    k, n = int(x.sum()), len(x)
    lo, hi = wilson(k, n)
    return k, n, 100 * k / n, 100 * lo, 100 * hi


def fmt(x):
    k, n, p, lo, hi = rate(x)
    return f"{p:.1f}% ({lo:.1f}-{hi:.1f}), {k}/{n}"


def main():
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.width", 250)
    results, out = Path(sys.argv[1]), Path(sys.argv[2] if len(sys.argv) > 2 else "thesis_figures")
    out.mkdir(parents=True, exist_ok=True)
    ok, _ = load(results)
    q = ok[ok.model == "qwen3:8b"]
    attack = q[q.channel.isin(CHANNELS)]
    att2 = attack[attack.injection_task.isin(TWO_GOALS)]

    print("1. Pooled over A, B1, B2, B3 (goal_1 and goal_2), Qwen3-8B")
    pooled = {}
    for d in ORDER:
        t = att2[att2.defence == d]
        ua = t.utility.astype(bool) & ~t.attack_success.astype(bool)
        pooled[d] = (rate(t.attack_success), rate(ua))
        b = q[(q.defence == d) & (q.channel == "benign")]
        print(f"  {LABEL[d].replace(chr(10), ' '):20s} ASR {fmt(t.attack_success):28s} utility under attack {fmt(ua):28s} "
              f"benign utility {fmt(b.utility):26s} alarms without attack {fmt(b.detector_fired.fillna(False).astype(bool)):26s}")

    print("\n2. Successful attacks per goal, Qwen3-8B (each cell out of 15 runs)")
    for d in ["none", "melon", "melon_firststep", "desc_scan"]:
        t = q[(q.defence == d) & (q.channel != "benign")]
        print(f"  {d}\n" + t.groupby(["channel", "injection_task"]).attack_success.sum().unstack().to_string())

    print("\n3. Tools removed by the description scan (Qwen3-8B)")
    all_tools = set(q[(q.defence == "none") & (q.channel == "benign")].tools_shown.iloc[0])
    t = q[q.defence == "desc_scan"]
    removed = t.tools_shown.apply(lambda x: ", ".join(sorted(all_tools - set(x))))
    print(t.assign(removed=removed).groupby(["channel", "injection_task"]).removed.agg(lambda s: s.value_counts().to_dict()).to_string())

    print("\n4. Both models on goal_0, goal_1, goal_2, goal_4")
    for m in ["qwen3:8b", "llama3.1:8b"]:
        for d in ["none", "melon", "melon_firststep"]:
            t = ok[(ok.model == m) & (ok.defence == d)]
            cells = [fmt(t[(t.channel == c) & t.injection_task.isin(FOUR_GOALS)].attack_success) for c in CHANNELS]
            print(f"  {m:12s} {d:16s} benign utility {fmt(t[t.channel == 'benign'].utility)} | " + " | ".join(cells))

    # Figure 7.1: ASR per defence and channel, goal_1 and goal_2 (same runs for every defence)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    w = 0.2
    for i, c in enumerate(CHANNELS):
        xs, ys, lo, hi = [], [], [], []
        for j, d in enumerate(ORDER):
            k, n, p, l, h = rate(att2[(att2.defence == d) & (att2.channel == c)].attack_success)
            xs.append(j + (i - 1.5) * w); ys.append(p); lo.append(p - l); hi.append(h - p)
        ax.bar(xs, ys, w, color=COLOURS[c], label=f"Channel {c}")
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor="#333333", capsize=2, lw=0.8)
    ax.set_xticks(range(len(ORDER)), [LABEL[d] for d in ORDER])
    ax.set_ylabel("Attack success rate (%)")
    ax.set_ylim(0, 50)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e5e5"); ax.set_axisbelow(True)
    ax.legend(ncol=4, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "Figure_7_1_asr_by_defence.png", dpi=200)


    # Figure 7.2: two bar charts side by side, one bar per defence
    order = ORDER
    names = [LABEL[d].replace("\n", " ") for d in order]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, idx, title, xmax in [(axes[0], 0, "Attacks that worked (%)\nlower is better", 25),
                                 (axes[1], 1, "Tasks still done during an attack (%)\nhigher is better", 100)]:
        vals = [pooled[d][idx] for d in order]
        ps = [v[2] for v in vals]
        err = [[v[2] - v[3] for v in vals], [v[4] - v[2] for v in vals]]
        colours = ["#7a7a7a" if d == "none" else ("#eb6834" if d in ("pi_detector", "desc_scan") else "#2a78d6") for d in order]
        ax.barh(range(len(order)), ps, color=colours, xerr=err, ecolor="#333333", capsize=3)
        for i, v in enumerate(ps):
            ax.text(v + (xmax * 0.02), i - 0.28, f"{v:.1f}%", fontsize=8)
        ax.set_xlim(0, xmax)
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#eeeeee"); ax.set_axisbelow(True)
    axes[0].set_yticks(range(len(order)), names)
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(out / "Figure_7_2.png", dpi=200)
    print(f"\nWrote the figures to {out}")


if __name__ == "__main__":
    main()
