"""Judge calibration — HUMAN CHECKPOINT (PROJECT_SPEC.md §15.5).

    python -m evals.calibration export --run evals/reports/<run> --n 50
    # … a human fills the `human_hallucinated` column (1 = has an unsupported claim, 0 = clean)
    python -m evals.calibration score

`export` writes data/eval/judge_calibration.csv with the judge's verdict HIDDEN; the verdicts
go to data/eval/judge_calibration_key.json. `score` computes agreement, Cohen's kappa and the
2×2 table into data/eval/judge_calibration_scores.json, which the next harness report includes.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

from evals.judge import judge_inputs
from evals.metrics import cohens_kappa
from ticketpilot.config import PROJECT_ROOT

EVAL_DIR = PROJECT_ROOT / "data" / "eval"
CSV_PATH = EVAL_DIR / "judge_calibration.csv"
KEY_PATH = EVAL_DIR / "judge_calibration_key.json"
SCORES_PATH = EVAL_DIR / "judge_calibration_scores.json"
FIELDS = ["ticket_id", "reply", "evidence", "human_hallucinated", "human_notes"]


def _flagged(v: dict[str, Any]) -> bool:
    return any(not c["supported"] for c in v.get("claims", []))


def export(
    run_dir: Path, n: int, seed: int = 42, csv_path: Path = CSV_PATH, key_path: Path = KEY_PATH
) -> int:
    if csv_path.exists() and any(r.get("human_hallucinated") for r in read_csv(csv_path)):
        print(f"{csv_path} already has human labels; refusing to overwrite.", file=sys.stderr)
        return 1
    verdicts: dict[str, dict[str, Any]] = json.loads((run_dir / "judge.json").read_text())
    results = {
        json.loads(line)["ticket_id"]: json.loads(line)
        for line in (run_dir / "results.jsonl").read_text().splitlines()
        if line.strip()
    }
    rng = random.Random(seed)
    flagged = sorted(t for t, v in verdicts.items() if _flagged(v))
    clean = sorted(t for t, v in verdicts.items() if not _flagged(v))
    rng.shuffle(flagged)
    rng.shuffle(clean)
    # Stratify: up to half flagged, fill the rest with clean (and vice versa if short).
    want_flagged = min(len(flagged), n // 2)
    picked = flagged[:want_flagged] + clean[: n - want_flagged]
    if len(picked) < n:
        picked += flagged[want_flagged : want_flagged + (n - len(picked))]
    rng.shuffle(picked)
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for tid in picked:
            inp = judge_inputs(results[tid], include_ticket=False)
            evidence = "\n\n".join(f"[{s['source']}] {s['text']}" for s in inp["sources"])
            w.writerow(
                {
                    "ticket_id": tid,
                    "reply": inp["reply"],
                    "evidence": evidence,
                    "human_hallucinated": "",
                    "human_notes": "",
                }
            )
    key_path.write_text(
        json.dumps(
            {"run": str(run_dir), "verdicts": {t: _flagged(verdicts[t]) for t in picked}}, indent=1
        )
        + "\n"
    )
    print(
        f"Wrote {len(picked)} rows ({sum(1 for t in picked if _flagged(verdicts[t]))} judge-flagged, "
        f"verdicts hidden) → {csv_path}"
    )
    print(
        "\nHUMAN CHECKPOINT: open the CSV and fill `human_hallucinated` with 1 (reply contains a claim "
        "not supported by the evidence) or 0 (clean) for every row, then run:\n"
        "  python -m evals.calibration score\n"
        "The hallucination rate is only credible once this agreement (kappa) is measured."
    )
    return 0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def score(
    csv_path: Path = CSV_PATH, key_path: Path = KEY_PATH, out_path: Path = SCORES_PATH
) -> dict[str, Any] | None:
    rows = read_csv(csv_path)
    key = json.loads(key_path.read_text())["verdicts"]
    labeled = [r for r in rows if r["human_hallucinated"].strip() in ("0", "1")]
    missing = len(rows) - len(labeled)
    if missing:
        print(f"{missing}/{len(rows)} rows are not labeled yet (need 0 or 1).", file=sys.stderr)
        if not labeled:
            return None
    human = [int(r["human_hallucinated"]) for r in labeled]
    judge = [int(bool(key[r["ticket_id"]])) for r in labeled]
    k = cohens_kappa(human, judge)
    result = {
        "n": k["n"],
        "unlabeled": missing,
        "agreement": k["p_o"],
        "chance_agreement": k["p_e"],
        "kappa": k["kappa"],
        "table": {
            "human0_judge0": k["table"][0][0],
            "human0_judge1": k["table"][0][1],
            "human1_judge0": k["table"][1][0],
            "human1_judge1": k["table"][1][1],
        },
    }
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"n={result['n']}  agreement={result['agreement']:.3f}  kappa={result['kappa']:.3f}")
    print("              judge=0  judge=1")
    print(f"human=0     {k['table'][0][0]:7d}  {k['table'][0][1]:7d}")
    print(f"human=1     {k['table'][1][0]:7d}  {k['table'][1][1]:7d}")
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--run", type=Path, required=True)
    e.add_argument("--n", type=int, default=50)
    sub.add_parser("score")
    args = p.parse_args()
    if args.cmd == "export":
        return export(args.run, args.n)
    return 0 if score() is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
