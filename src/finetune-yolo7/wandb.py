#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _load_wandb_sdk():
    script_dir = str(Path(__file__).resolve().parent)
    removed = False
    if script_dir in sys.path:
        sys.path.remove(script_dir)
        removed = True
    try:
        import wandb as wandb_sdk
    finally:
        if removed:
            sys.path.insert(0, script_dir)
    return wandb_sdk


def init_run(
    *,
    project: str,
    run_name: str,
    job_type: str,
    config_payload: dict[str, Any],
    tags: list[str],
    run_id: str | None = None,
):
    wandb_sdk = _load_wandb_sdk()
    settings = wandb_sdk.Settings(start_method="thread", console="off")
    return wandb_sdk.init(
        project=project,
        name=run_name,
        job_type=job_type,
        id=run_id,
        resume="allow" if run_id else None,
        config=config_payload,
        tags=tags,
        settings=settings,
    )


def log_metrics(run, metrics: dict[str, Any], step: int | None = None) -> None:
    if step is None:
        run.log(metrics)
    else:
        run.log(metrics, step=step)


def update_summary(run, summary: dict[str, Any]) -> None:
    for key, value in summary.items():
        run.summary[key] = value


def finish_run(run) -> None:
    run.finish()


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the local W&B helper import path.")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        _load_wandb_sdk()
        print("wandb helper is importable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
