from __future__ import annotations

from datetime import datetime
from pathlib import Path

from benchmark_common import config
from benchmark_common.metrics import read_json, write_json


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def artifacts_root() -> Path:
    return repo_root() / config.ARTIFACT_ROOT


def model_root(model_name: str) -> Path:
    return artifacts_root() / model_name


def runs_root(model_name: str) -> Path:
    return model_root(model_name) / "runs"


def latest_run_file(model_name: str) -> Path:
    return model_root(model_name) / "latest-run.json"


def make_run_name(model_name: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{model_name}-augment-e{config.EPOCHS}-img{config.IMG_SIZE}-{stamp}"


def write_latest_run(model_name: str, context: dict) -> None:
    write_json(latest_run_file(model_name), context)


def load_latest_run(model_name: str) -> dict:
    path = latest_run_file(model_name)
    if not path.exists():
        raise FileNotFoundError(f"Missing latest run file for {model_name}: {path}. Run setup first.")
    return read_json(path)
