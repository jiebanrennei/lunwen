import argparse
import json
import os
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


def main():
    parser = argparse.ArgumentParser(description="Run train_ig.py for datasets defined in a JSON config.")
    parser.add_argument("--config", default="batch_datasets.json", help="Path to batch config JSON")
    parser.add_argument("--datasets", default=None,
                        help="Comma-separated dataset names to run; default runs all configured datasets")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing them")
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

    if not datasets:
        raise ValueError("No datasets configured in batch config")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = output_dir / f"batch_summary_{run_id}.log"

    with open(summary_path, "w", encoding="utf-8") as summary:
        for item in datasets:
            name = item["name"]
            if not _dataset_enabled(name, selected):
                continue

            merged_args = dict(base_args)
            merged_args.update(item.get("args", {}))
            merged_args["dataset"] = name

            cmd = [python_bin, str(train_script)]
            _extend_args(cmd, merged_args)

            log_path = output_dir / f"{name}_{run_id}.log"
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
