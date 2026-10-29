"""Merge several results folders into one folder for the analysis.

  python merge_results.py results_all  FOLDER:PREFIX  FOLDER:PREFIX ...

Every *.jsonl file of FOLDER is copied to results_all as PREFIX__<name>.jsonl, so files with the
same name from different laptops never overwrite each other. The conversation files in
FOLDER/messages are copied into results_all/messages (their names contain the task and goal, so
they do not clash). The analysis reads all *.jsonl files in results_all and keeps one record per
model x channel x defence x task x goal.

Example:
  python merge_results.py results_all results:bhumika results_extra:bhumika_extra ^
      bina\\results:bina bina\\results_extra:bina_extra tilak\\results:tilak tilak\\results_extra:tilak_extra
"""

import shutil
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    dest = Path(sys.argv[1])
    dest.mkdir(parents=True, exist_ok=True)
    total_files = total_lines = 0
    for spec in sys.argv[2:]:
        folder, _, prefix = spec.rpartition(":")
        src = Path(folder)
        if not src.is_dir():
            raise SystemExit(f"Folder not found: {src}")
        files = sorted(src.glob("*.jsonl"))
        lines = 0
        for f in files:
            target = dest / f"{prefix}__{f.name}"
            if target.exists():
                raise SystemExit(f"Already exists, not overwriting: {target}")
            shutil.copy2(f, target)
            lines += sum(1 for line in f.read_text(encoding="utf-8").splitlines() if line.strip())
        msgs = src / "messages"
        n_msg = 0
        if msgs.is_dir():
            for m in msgs.rglob("*.json"):
                t = dest / "messages" / m.relative_to(msgs)
                t.parent.mkdir(parents=True, exist_ok=True)
                if not t.exists():
                    shutil.copy2(m, t)
                    n_msg += 1
        print(f"{src}: {len(files)} result files, {lines} lines, {n_msg} conversations  ->  prefix '{prefix}'")
        total_files += len(files)
        total_lines += lines
    print(f"\nDone: {total_files} result files with {total_lines} lines in {dest}")


if __name__ == "__main__":
    main()
