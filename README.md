# mcpsec — tool description poisoning vs. prompt injection defences, with real MCP servers

A free experimental environment. An AI agent helps "Emma" with her bank, files and email. All tools
are **real MCP servers** (official MCP Python SDK). The models run **for free on your own laptop**
with Ollama.

## The idea in one paragraph
Emma's agent is connected to three trusted MCP servers (`nordbank`, `files`, `mail`) and two
third-party MCP servers she installed (`fxrates` for currency rates, `billscan` for reading bills).
An attacker tries to make the agent do something harmful (change the password, send money, leak the
security code, ...). The attacker hides the instruction either **in data** the agent reads
(channel A) or **in the description of a third-party tool** (channels B1–B3). We measure how often
each attack works, with and without defences.

## Files
| File | What it is |
|---|---|
| `mcpsec/environment.py` | Emma's world: account, transactions, files, emails. Fresh copy for every run. |
| `mcpsec/servers.py` | The 5 MCP servers and their 15 tools. |
| `mcpsec/tasks.py` | 15 user tasks and 6 attacker goals, each with an automatic check (a B4 goal exists but is not used). |
| `mcpsec/attacks.py` | The attack channels: where the attack text is placed and how it is worded. |
| `mcpsec/agent.py` | The agent: MCP client + model loop. |
| `mcpsec/defences.py` | 7 defences, incl. MELON and our two additions. |
| `mcpsec/llm.py` | Talks to the model (Ollama). Removes reasoning text. Counts calls and tokens. |
| `mcpsec/validate.py` | Proves every task and goal can be solved, every check works and every attack lands in the right place (no model needed). |
| `mcpsec/run.py` | Runs one configuration, writes `results/*.jsonl`. Resumable. |
| `mcpsec/analyze.py` | Tables with 95% confidence intervals, the planned comparisons and figures for the thesis. |
| `tests/` | Free tests with a scripted fake model and a mock Ollama server. |

## Channels
| Channel | Where the attacker's text is | Wording | Runs |
|---|---|---|---|
| `benign` | nowhere (normal performance, false alarms) | – | 15 |
| `A` | in data: bill, emails, documents, transaction/payment notes | user (AgentDojo) | 90 |
| `B1` | description of `get_exchange_rate` (no task needs it) | provider | 90 |
| `B2` | description of `parse_bill` (useful for the bill tasks) | provider | 90 |
| `B3` | description of `get_exchange_rate`, only for payment requests (against MELON) | provider | 90 |
| `A_provider` | like A, provider wording (framing check, no defence only) | provider | 90 |
| `B1_user` | like B1, user wording (framing check, no defence only) | user | 90 |

## Defences
| Defence | What it does |
|---|---|
| `none` | no defence |
| `repeat_prompt` | repeats the user's task after every tool result |
| `pi_detector` | DeBERTa classifier checks every tool output (same model as AgentDojo) |
| `tool_filter` | the model first picks which tools it needs; others are hidden |
| `melon` | re-implementation of MELON (Zhu et al., 2025), following the official code |
| `melon_firststep` | **our change:** MELON that also checks the agent's very first action |
| `desc_scan` | **our addition:** the DeBERTa classifier checks tool *descriptions* when tools are loaded |

---

## Step by step

### 1. Install Ollama (free) and download the models
```bash
ollama pull qwen3:8b            # main model (reasoning mode is on by default), ~5 GB
ollama pull llama3.1:8b         # second model
ollama pull nomic-embed-text    # small embedding model used by MELON
ollama show qwen3:8b --parameters   # note the default top_k (should be 20) for the thesis
```
An 8B model needs about 8 GB of free memory.

### 2. Install Python 3.11+ and the packages
```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Free checks (no model needed)
```bash
python -m mcpsec.validate          # must end with: ALL CHECKS PASSED
python -m tests.fake_end_to_end    # all channels and defences with a scripted fake model
python -m tests.mock_ollama        # the real HTTP client against a mock server
```

### 4. Look at one real run before starting
```bash
python -m mcpsec.run --model qwen3:8b --temperature 0.6 --top-p 0.95 --channel B1 --defence none --user-tasks user_task_0 --save-messages
```
Open `results/messages/...` and check that the reasoning is stored under `_reasoning` and not in
`content`. Then delete the `results` folder.

### 5. Main experiment
```bash
bash run_all.sh                    # qwen3:8b, 2,805 runs
MODEL=llama3.1:8b bash run_all.sh  # llama3.1:8b, 1,125 runs
```
You can stop it any time (Ctrl+C) and restart; finished runs are skipped.
To split the work over three laptops, give each laptop a part of the defences, e.g.
`DEFENCES="none repeat_prompt pi_detector" bash run_all.sh`, and set `FRAMING_CHANNELS=""` on all
laptops but one. Afterwards copy all `results/*.jsonl` files into one folder.

### 6. Analyse
```bash
python -m mcpsec.analyze --results results --out analysis
```
`analysis/tables.md`: all metrics with 95% Wilson confidence intervals. `analysis/comparisons.md`: the
planned comparisons; a difference is marked "clear" only when the two intervals do not overlap.
`analysis/errors.csv`: runs to repeat.

## Things to report in the thesis
- Model tags and Ollama version; sampling: Qwen3 temperature 0.6, top-p 0.95, top-k from Ollama's
  default; Llama temperature 0; seed 0 for all runs.
- MELON is re-implemented and uses the free local embedding model `nomic-embed-text` instead of
  OpenAI's `text-embedding-3-large`. The threshold is 0.8 as in the official code.
- Channel A is compared with B on the 13 tasks whose data contains the injection; B3 is compared
  with B1 on the 6 payment tasks.
