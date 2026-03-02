#!/usr/bin/env python3
"""
Interactive terminal logger for rollout experiments (DP3 / CFM / EFM).

This tool is designed for manual result entry after each real-robot run.
It exports:
1) Raw per-trial CSV
2) Aggregated summary CSV (mean/std/p95 + pass rate)
3) YAML snapshot of the whole experiment
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


DEFAULT_POLICIES = ("dp3", "cfm", "efm")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "experiment_results"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def slugify(text: str) -> str:
    out = []
    for ch in text.strip().lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in (" ", "-", "_"):
            out.append("_")
    cleaned = "".join(out).strip("_")
    return cleaned or "experiment"


def parse_bool(value) -> bool:
    s = str(value).strip().lower()
    return s in ("1", "true", "t", "yes", "y", "pass", "p")


def metric_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "p95": float("nan")}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p95": float(np.percentile(arr, 95)),
    }


def fmt_num(value: float) -> str:
    if np.isnan(value):
        return ""
    return f"{value:.4f}"


@dataclass
class TrialResult:
    timestamp_utc: str
    experiment_name: str
    policy: str
    trial_id: int
    inference_time_ms: float
    rollout_duration_s: float
    steps_executed: int
    passed: bool
    abort_reason: str
    notes: str


class ExperimentLogger:
    def __init__(
        self,
        experiment_name: str,
        output_dir: Path,
        policies: List[str],
        target_trials_per_policy: int,
    ):
        self.experiment_name = experiment_name
        self.output_dir = output_dir
        self.policies = policies
        self.target_trials_per_policy = target_trials_per_policy
        self.trials: List[TrialResult] = []
        self.created_at_utc = utc_now_iso()
        self.active_folder: Optional[Path] = None

    def next_trial_id(self, policy: str) -> int:
        ids = [t.trial_id for t in self.trials if t.policy == policy]
        return (max(ids) + 1) if ids else 1

    def add_trial(self, trial: TrialResult) -> None:
        self.trials.append(trial)

    def remove_last(self) -> Optional[TrialResult]:
        if not self.trials:
            return None
        return self.trials.pop()

    def remove_trial(self, policy: str, trial_id: int) -> Optional[TrialResult]:
        policy = policy.strip().lower()
        for i, t in enumerate(self.trials):
            if t.policy == policy and int(t.trial_id) == int(trial_id):
                return self.trials.pop(i)
        return None

    def summary_rows(self) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        groups: Dict[str, List[TrialResult]] = {}
        for policy in self.policies:
            groups[policy] = [t for t in self.trials if t.policy == policy]

        for policy, group in groups.items():
            n_trials = len(group)
            n_pass = sum(1 for t in group if t.passed)
            n_fail = n_trials - n_pass
            pass_rate = (100.0 * n_pass / n_trials) if n_trials > 0 else float("nan")

            inf_stats = metric_stats([t.inference_time_ms for t in group])
            dur_stats = metric_stats([t.rollout_duration_s for t in group])
            step_stats = metric_stats([float(t.steps_executed) for t in group])

            fail_reasons = Counter(
                t.abort_reason.strip()
                for t in group
                if (not t.passed) and t.abort_reason.strip()
            )
            fail_reason_text = "; ".join(f"{k}:{v}" for k, v in sorted(fail_reasons.items()))

            rows.append(
                {
                    "experiment_name": self.experiment_name,
                    "policy": policy,
                    "n_trials": n_trials,
                    "n_pass": n_pass,
                    "n_fail": n_fail,
                    "pass_rate_percent": pass_rate,
                    "inference_time_mean_ms": inf_stats["mean"],
                    "inference_time_std_ms": inf_stats["std"],
                    "inference_time_p95_ms": inf_stats["p95"],
                    "rollout_duration_mean_s": dur_stats["mean"],
                    "rollout_duration_std_s": dur_stats["std"],
                    "rollout_duration_p95_s": dur_stats["p95"],
                    "steps_mean": step_stats["mean"],
                    "steps_std": step_stats["std"],
                    "steps_p95": step_stats["p95"],
                    "fail_reasons": fail_reason_text,
                }
            )

        return rows

    def progress_rows(self) -> List[Dict[str, object]]:
        rows = []
        for policy in self.policies:
            group = [t for t in self.trials if t.policy == policy]
            n = len(group)
            n_pass = sum(1 for t in group if t.passed)
            pass_rate = (100.0 * n_pass / n) if n else float("nan")
            rows.append(
                {
                    "policy": policy,
                    "trials": n,
                    "target": self.target_trials_per_policy,
                    "remaining": max(0, self.target_trials_per_policy - n),
                    "pass_rate_percent": pass_rate,
                }
            )
        return rows

    def load_trials(self, trials: List[TrialResult]) -> None:
        self.trials = list(trials)

    def set_active_folder(self, folder: Path) -> None:
        self.active_folder = folder.resolve()
        self.active_folder.mkdir(parents=True, exist_ok=True)

    def save_to_folder(self, folder: Path) -> Dict[str, Path]:
        folder = folder.resolve()
        folder.mkdir(parents=True, exist_ok=True)
        raw_csv = folder / "raw_trials.csv"
        summary_csv = folder / "summary_table.csv"
        yaml_path = folder / "results.yaml"

        self._write_raw_csv(raw_csv)
        self._write_summary_csv(summary_csv)
        self._write_yaml(yaml_path)

        return {"folder": folder, "raw_csv": raw_csv, "summary_csv": summary_csv, "yaml": yaml_path}

    def autosave(self) -> Dict[str, Path]:
        if self.active_folder is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            self.active_folder = self.output_dir / f"{slugify(self.experiment_name)}_{stamp}"
        return self.save_to_folder(self.active_folder)

    def export(self) -> Dict[str, Path]:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        folder = self.output_dir / f"{slugify(self.experiment_name)}_{stamp}"
        return self.save_to_folder(folder)

    def _write_raw_csv(self, path: Path) -> None:
        fields = [
            "timestamp_utc",
            "experiment_name",
            "policy",
            "trial_id",
            "inference_time_ms",
            "rollout_duration_s",
            "steps_executed",
            "passed",
            "abort_reason",
            "notes",
        ]
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for t in self.trials:
                row = asdict(t)
                row["passed"] = int(bool(row["passed"]))
                writer.writerow(row)

    def _write_summary_csv(self, path: Path) -> None:
        fields = [
            "experiment_name",
            "policy",
            "n_trials",
            "n_pass",
            "n_fail",
            "pass_rate_percent",
            "inference_time_mean_ms",
            "inference_time_std_ms",
            "inference_time_p95_ms",
            "rollout_duration_mean_s",
            "rollout_duration_std_s",
            "rollout_duration_p95_s",
            "steps_mean",
            "steps_std",
            "steps_p95",
            "fail_reasons",
        ]
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in self.summary_rows():
                writer.writerow({k: self._clean_value(v) for k, v in row.items()})

    def _write_yaml(self, path: Path) -> None:
        payload = {
            "experiment_name": self.experiment_name,
            "created_at_utc": self.created_at_utc,
            "target_trials_per_policy": self.target_trials_per_policy,
            "policies": self.policies,
            "num_trials": len(self.trials),
            "trials": [asdict(t) for t in self.trials],
            "summary": self.summary_rows(),
        }
        payload = self._replace_nan(payload)
        if yaml is not None:
            text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=False)
        else:
            # JSON is valid YAML 1.2 and keeps us dependency-free.
            text = json.dumps(payload, indent=2)
        path.write_text(text)

    @staticmethod
    def _clean_value(value):
        if isinstance(value, float) and np.isnan(value):
            return ""
        return value

    @staticmethod
    def _replace_nan(obj):
        if isinstance(obj, dict):
            return {k: ExperimentLogger._replace_nan(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [ExperimentLogger._replace_nan(v) for v in obj]
        if isinstance(obj, float) and np.isnan(obj):
            return None
        return obj


def load_trials_from_csv(path: Path) -> List[TrialResult]:
    required_fields = {
        "timestamp_utc",
        "experiment_name",
        "policy",
        "trial_id",
        "inference_time_ms",
        "rollout_duration_s",
        "steps_executed",
        "passed",
        "abort_reason",
        "notes",
    }
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or not required_fields.issubset(set(reader.fieldnames)):
            raise ValueError(f"CSV missing required fields: {path}")
        trials: List[TrialResult] = []
        for row in reader:
            trials.append(
                TrialResult(
                    timestamp_utc=str(row.get("timestamp_utc", "")).strip() or utc_now_iso(),
                    experiment_name=str(row.get("experiment_name", "")).strip(),
                    policy=str(row.get("policy", "")).strip().lower(),
                    trial_id=int(row.get("trial_id", 0)),
                    inference_time_ms=float(row.get("inference_time_ms", 0.0)),
                    rollout_duration_s=float(row.get("rollout_duration_s", 0.0)),
                    steps_executed=int(row.get("steps_executed", 0)),
                    passed=parse_bool(row.get("passed", "")),
                    abort_reason=str(row.get("abort_reason", "")).strip(),
                    notes=str(row.get("notes", "")).strip(),
                )
            )
    return trials


def find_latest_experiment_raw_csv(output_dir: Path, experiment_name: str) -> Optional[Path]:
    prefix = f"{slugify(experiment_name)}_"
    candidates = [
        p for p in output_dir.glob(f"{prefix}*/raw_trials.csv")
        if p.is_file()
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def prompt_text(label: str, default: Optional[str] = None, allow_empty: bool = False) -> str:
    while True:
        suffix = f" [{default}]" if default is not None else ""
        raw = input(f"{label}{suffix}: ").strip()
        if raw:
            return raw
        if default is not None:
            return default
        if allow_empty:
            return ""
        print("Value required.")


def prompt_int(label: str, default: Optional[int] = None, min_value: Optional[int] = None) -> int:
    while True:
        raw = prompt_text(label, str(default) if default is not None else None)
        try:
            value = int(raw)
        except ValueError:
            print("Enter an integer.")
            continue
        if min_value is not None and value < min_value:
            print(f"Value must be >= {min_value}.")
            continue
        return value


def prompt_float(label: str, default: Optional[float] = None, min_value: Optional[float] = None) -> float:
    while True:
        default_text = f"{default:.4f}" if isinstance(default, float) else None
        raw = prompt_text(label, default_text)
        try:
            value = float(raw)
        except ValueError:
            print("Enter a number.")
            continue
        if not np.isfinite(value):
            print("Enter a finite number.")
            continue
        if min_value is not None and value < min_value:
            print(f"Value must be >= {min_value}.")
            continue
        return value


def prompt_bool_pass_fail() -> bool:
    while True:
        raw = prompt_text("Result (p/f)", "p").lower()
        if raw in ("p", "pass", "1", "true", "t", "y", "yes"):
            return True
        if raw in ("f", "fail", "0", "false", "n", "no"):
            return False
        print("Enter p or f.")


def choose_policy(policies: List[str]) -> str:
    print("\nChoose policy:")
    for i, p in enumerate(policies, start=1):
        print(f"  {i}. {p}")
    while True:
        raw = prompt_text("Policy (number or name)")
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(policies):
                return policies[idx - 1]
        lowered = raw.lower()
        if lowered in policies:
            return lowered
        print("Invalid policy.")


def print_progress(logger: ExperimentLogger) -> None:
    print("\nProgress:")
    print("policy | trials/target | remaining | pass_rate(%)")
    print("-" * 48)
    for row in logger.progress_rows():
        print(
            f"{row['policy']:<6} | "
            f"{row['trials']}/{row['target']:<10} | "
            f"{row['remaining']:<9} | "
            f"{fmt_num(float(row['pass_rate_percent']))}"
        )


def print_summary(logger: ExperimentLogger) -> None:
    print("\nCurrent summary:")
    print(
        "policy | n | pass_rate(%) | inf_mean(ms) | inf_std | inf_p95 | "
        "dur_mean(s) | steps_mean"
    )
    print("-" * 104)
    for row in logger.summary_rows():
        print(
            f"{str(row['policy']):<6} | "
            f"{int(row['n_trials']):<2} | "
            f"{fmt_num(float(row['pass_rate_percent'])):<12} | "
            f"{fmt_num(float(row['inference_time_mean_ms'])):<12} | "
            f"{fmt_num(float(row['inference_time_std_ms'])):<7} | "
            f"{fmt_num(float(row['inference_time_p95_ms'])):<7} | "
            f"{fmt_num(float(row['rollout_duration_mean_s'])):<11} | "
            f"{fmt_num(float(row['steps_mean']))}"
        )


def add_trial_interactive(logger: ExperimentLogger) -> bool:
    print("\nAdd trial result")
    policy = choose_policy(logger.policies)
    trial_default = logger.next_trial_id(policy)
    trial_id = prompt_int("Trial number", default=trial_default, min_value=1)

    existing = next(
        (t for t in logger.trials if t.policy == policy and int(t.trial_id) == int(trial_id)),
        None,
    )
    if existing is not None:
        print(
            f"Existing run found: {policy} trial #{trial_id} "
            f"({'PASS' if existing.passed else 'FAIL'}), "
            f"inf={existing.inference_time_ms:.4f}ms, "
            f"dur={existing.rollout_duration_s:.4f}s, "
            f"steps={existing.steps_executed}"
        )
        confirm = prompt_text("Overwrite existing run? (y/n)", default="n").lower()
        if confirm not in ("y", "yes"):
            print("Add cancelled (no overwrite).")
            return False
        logger.remove_trial(policy=policy, trial_id=trial_id)
        print(f"Overwriting {policy} trial #{trial_id}.")

    inference_time_ms = prompt_float("Inference time (ms)", min_value=0.0)
    rollout_duration_s = prompt_float("Rollout duration (s)", min_value=0.0)
    steps_executed = prompt_int("Steps executed", min_value=0)
    passed = prompt_bool_pass_fail()

    abort_reason = ""
    if not passed:
        abort_reason = prompt_text("Abort reason", allow_empty=False)

    notes = prompt_text("Notes", default="", allow_empty=True)

    trial = TrialResult(
        timestamp_utc=utc_now_iso(),
        experiment_name=logger.experiment_name,
        policy=policy,
        trial_id=trial_id,
        inference_time_ms=inference_time_ms,
        rollout_duration_s=rollout_duration_s,
        steps_executed=steps_executed,
        passed=passed,
        abort_reason=abort_reason,
        notes=notes,
    )
    logger.add_trial(trial)
    print(f"Added: {policy} trial #{trial_id} ({'PASS' if passed else 'FAIL'})")
    return True


def print_menu() -> None:
    print("\nMenu:")
    print("  1) Add trial result")
    print("  2) Show progress")
    print("  3) Show summary")
    print("  4) Remove last entry")
    print("  5) Delete specific run")
    print("  6) Save snapshot now")
    print("  7) Save and exit")


def delete_trial_interactive(logger: ExperimentLogger) -> bool:
    print("\nDelete specific run")
    policy = choose_policy(logger.policies)
    trial_id = prompt_int("Trial number to delete", min_value=1)
    found = next((t for t in logger.trials if t.policy == policy and t.trial_id == trial_id), None)
    if found is None:
        print(f"No run found for policy={policy}, trial={trial_id}.")
        return False

    confirm = prompt_text(
        f"Delete {policy} trial #{trial_id}? (y/n)",
        default="n"
    ).lower()
    if confirm not in ("y", "yes"):
        print("Deletion cancelled.")
        return False

    removed = logger.remove_trial(policy=policy, trial_id=trial_id)
    if removed is None:
        print("Run was not deleted (not found).")
        return False
    print(f"Deleted: {removed.policy} trial #{removed.trial_id} ({'PASS' if removed.passed else 'FAIL'})")
    return True


def run_interactive(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    policies = [p.strip().lower() for p in args.policies.split(",") if p.strip()]
    if not policies:
        raise ValueError("No policies configured.")

    print("Rollout Experiment Logger")
    print("-" * 26)

    resume_csv: Optional[Path] = None
    loaded_trials: List[TrialResult] = []
    if args.resume_csv:
        resume_csv = Path(args.resume_csv).resolve()
        if not resume_csv.exists():
            raise FileNotFoundError(f"Resume CSV not found: {resume_csv}")
        loaded_trials = load_trials_from_csv(resume_csv)

    exp_name = args.experiment_name
    if not exp_name and loaded_trials:
        exp_names = {t.experiment_name for t in loaded_trials if t.experiment_name}
        if len(exp_names) == 1:
            exp_name = next(iter(exp_names))
    if not exp_name:
        exp_name = prompt_text("Experiment name")

    if args.resume_latest and resume_csv is None:
        resume_csv = find_latest_experiment_raw_csv(output_dir, exp_name)
        if resume_csv is not None:
            loaded_trials = load_trials_from_csv(resume_csv)

    target_trials = args.target_trials_per_policy
    if target_trials is None:
        if loaded_trials:
            by_policy: Dict[str, int] = {}
            for t in loaded_trials:
                if t.policy:
                    by_policy[t.policy] = by_policy.get(t.policy, 0) + 1
            inferred = max(by_policy.values()) if by_policy else 10
            target_trials = max(1, inferred)
        else:
            target_trials = prompt_int("Target trials per policy", default=10, min_value=1)

    logger = ExperimentLogger(
        experiment_name=exp_name,
        output_dir=output_dir,
        policies=policies,
        target_trials_per_policy=target_trials,
    )

    if resume_csv is not None:
        if loaded_trials:
            exp_names = {t.experiment_name for t in loaded_trials if t.experiment_name}
            if len(exp_names) == 1 and logger.experiment_name not in exp_names:
                logger.experiment_name = next(iter(exp_names))
            for t in loaded_trials:
                if t.policy and t.policy not in logger.policies:
                    logger.policies.append(t.policy)
            logger.load_trials(loaded_trials)
        print(f"[resume] Loaded {len(logger.trials)} trials from: {resume_csv}")

    if resume_csv is not None:
        session_folder = resume_csv.resolve().parent
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        session_folder = output_dir / f"{slugify(logger.experiment_name)}_{stamp}"
    logger.set_active_folder(session_folder)
    paths = logger.autosave()

    print(f"\nExperiment: {logger.experiment_name}")
    print(f"Policies: {', '.join(logger.policies)}")
    print(f"Target trials/policy: {logger.target_trials_per_policy}")
    print(f"Output dir: {output_dir}")
    print(f"Autosave folder: {paths['folder']}")

    while True:
        print_menu()
        choice = prompt_text("Select option")
        if choice == "1":
            changed = add_trial_interactive(logger)
            if changed:
                paths = logger.autosave()
                print(f"[autosave] {paths['raw_csv']}")
        elif choice == "2":
            print_progress(logger)
        elif choice == "3":
            print_summary(logger)
        elif choice == "4":
            removed = logger.remove_last()
            if removed is None:
                print("No entries to remove.")
            else:
                print(
                    f"Removed last entry: {removed.policy} trial #{removed.trial_id} "
                    f"({'PASS' if removed.passed else 'FAIL'})"
                )
                paths = logger.autosave()
                print(f"[autosave] {paths['raw_csv']}")
        elif choice == "5":
            changed = delete_trial_interactive(logger)
            if changed:
                paths = logger.autosave()
                print(f"[autosave] {paths['raw_csv']}")
        elif choice == "6":
            paths = logger.autosave()
            print(f"Exported to: {paths['folder']}")
            print(f"  raw:     {paths['raw_csv']}")
            print(f"  summary: {paths['summary_csv']}")
            print(f"  yaml:    {paths['yaml']}")
        elif choice == "7":
            paths = logger.autosave()
            print(f"Exported to: {paths['folder']}")
            print("Done.")
            return
        else:
            print("Invalid option.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Terminal UI to log rollout results and export paper tables."
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name. If omitted, prompt interactively.",
    )
    parser.add_argument(
        "--target-trials-per-policy",
        type=int,
        default=None,
        help="Expected trials per policy (for progress tracking).",
    )
    parser.add_argument(
        "--policies",
        type=str,
        default="dp3,cfm,efm",
        help="Comma-separated policy names.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where CSV/YAML exports are written.",
    )
    parser.add_argument(
        "--resume-csv",
        type=str,
        default=None,
        help="Path to an existing raw_trials.csv to continue logging.",
    )
    parser.add_argument(
        "--resume-latest",
        action="store_true",
        default=False,
        help="Resume from latest raw_trials.csv matching --experiment-name in output-dir.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_interactive(args)


if __name__ == "__main__":
    main()
