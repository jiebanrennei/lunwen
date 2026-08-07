import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _arg_name(key):
    return f"--{key}"


def _extend_args(cmd, args):
    for key, value in args.items():
        if value is None or value is False:
            continue
        flag = _arg_name(key)
        if value is True:
            cmd.append(flag)
        else:
            cmd.extend([flag, str(value)])


def _load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _dataset_enabled(name, selected):
    return not selected or name in selected


def _safe_run_name(name):
    name = (name or "batch").strip()
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    return name.strip("._-") or "batch"


def _arg_key(key):
    return key.strip().lstrip("-").replace("-", "_")


def _dataset_key(name):
    return name.strip().upper()


def main():
    parser = argparse.ArgumentParser(description="Run train_ig.py for datasets defined in a JSON config.")
    parser.add_argument("--config", default="batch_datasets.json", help="Path to batch config JSON")
    parser.add_argument("--datasets", default=None,
                        help="Comma-separated dataset names to run; default runs all configured datasets")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing them")
    parser.add_argument("--run_name", default=None,
                        help="Name prefix for this batch log directory, e.g. ilssc_seed")
    parser.add_argument("--profile", default=None,
                        help="Named profile from config.profiles for dataset-specific overrides")
    parser.add_argument("--dataset_arg", nargs=3, action="append", default=[],
                        metavar=("DATASET", "KEY", "VALUE"),
                        help="Override one train_ig.py argument for one dataset, e.g. --dataset_arg ACM ilssc_warmup_epochs 30")
    parser.add_argument("--train_arg", action="append", default=[],
                        help="Extra argument appended to every train_ig.py command; repeat for each token")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    repo_dir = config_path.parent
    cfg = _load_config(config_path)

    selected = None
    if args.datasets:
        selected = {x.strip() for x in args.datasets.split(",") if x.strip()}

    output_dir = repo_dir / cfg.get("output_dir", "batch_runs")
    output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in cfg.get("env", {}).items()})

    python_bin = cfg.get("python", sys.executable)
    train_script = repo_dir / cfg.get("train_script", "train_ig.py")
    base_args = cfg.get("base_args", {})
    datasets = cfg.get("datasets", [])
    profiles = cfg.get("profiles", {})
    profile_cfg = None
    if args.profile:
        if args.profile not in profiles:
            available = ", ".join(sorted(profiles)) or "<none>"
            raise ValueError(f"Unknown profile: {args.profile}. Available profiles: {available}")
        profile_cfg = profiles[args.profile]

    dataset_arg_overrides = {}
    for dataset_name, key, value in args.dataset_arg:
        dataset_arg_overrides.setdefault(_dataset_key(dataset_name), {})[_arg_key(key)] = value

    if not datasets:
        raise ValueError("No datasets configured in batch config")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = _safe_run_name(args.run_name or args.profile)
    run_dir = output_dir / f"{run_name}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)
    summary_path = run_dir / "batch_summary.log"

    print(f"[batch] run_dir -> {run_dir}")

    with open(summary_path, "w", encoding="utf-8") as summary:
        if args.profile:
            summary.write(f"[profile] {args.profile}\n")
        for item in datasets:
            name = item["name"]
            if not _dataset_enabled(name, selected):
                continue

            merged_args = dict(base_args)
            merged_args.update(item.get("args", {}))
            if profile_cfg is not None:
                merged_args.update(profile_cfg.get("base_args", {}))
                merged_args.update(profile_cfg.get("datasets", {}).get(name, {}))
            merged_args.update(dataset_arg_overrides.get(_dataset_key(name), {}))
            merged_args["dataset"] = name

            cmd = [python_bin, str(train_script)]
            _extend_args(cmd, merged_args)
            cmd.extend(args.train_arg)

            log_path = run_dir / f"{name}.log"
            line = " ".join(cmd)
            print(f"[batch] start {name}")
            print(f"[batch] log -> {log_path}")
            print(line)
            summary.write(f"[{name}] {line}\n")
            summary.flush()

            if args.dry_run:
                continue

            with open(log_path, "w", encoding="utf-8") as log_f:
                log_f.write(f"[command] {line}\n")
                log_f.flush()
                proc = subprocess.run(
                    cmd,
                    cwd=str(repo_dir),
                    env=env,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    text=True,
                )

            summary.write(f"[{name}] returncode={proc.returncode} log={log_path}\n")
            summary.flush()
            print(f"[batch] done {name}, returncode={proc.returncode}")
            if proc.returncode != 0:
                raise SystemExit(proc.returncode)

    print(f"[batch] summary -> {summary_path}")


if __name__ == "__main__":
    main()
