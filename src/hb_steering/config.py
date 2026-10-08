from __future__ import annotations

import math
import re
import tomllib
from pathlib import Path

COMMIT = re.compile(r"[0-9a-f]{40}\Z")
SAFE_NAME = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,79}\Z")


def load_config(root: Path, path: str | None = None) -> tuple[dict, Path]:
    config_path = Path(path) if path else root / "configs/experiment.toml"
    if not config_path.is_absolute():
        config_path = root / config_path
    with config_path.open("rb") as stream:
        config = tomllib.load(stream)
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported config schema_version")
    for section in ("dataset", "models", "runtime", "steering", "judge"):
        if section not in config:
            raise ValueError(f"Missing config section: {section}")
    dataset = config["dataset"]
    for key in ("harmbench_revision", "xstest_revision"):
        if not COMMIT.fullmatch(dataset[key]):
            raise ValueError(f"dataset.{key} must be a full commit SHA")
    for key in ("harmbench_sha256", "xstest_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", dataset[key]):
            raise ValueError(f"Invalid dataset.{key}")
    if set(dataset["categories"]) != {"standard", "contextual"}:
        raise ValueError(
            "This initial dataset configuration requires standard and contextual together"
        )
    if not dataset["categories"] or len(set(dataset["categories"])) != len(dataset["categories"]):
        raise ValueError("dataset.categories must be nonempty and unique")
    if type(dataset["seed"]) is not int:
        raise ValueError("dataset.seed must be an integer")
    ratios = dataset["ratios"]
    if (
        len(ratios) != 3
        or any(not math.isfinite(x) or x <= 0 for x in ratios)
        or not math.isclose(sum(ratios), 1.0)
    ):
        raise ValueError("dataset.ratios must be three positive values summing to 1")
    runtime = config["runtime"]
    if runtime["device"] not in {"cuda", "cpu"}:
        raise ValueError("runtime.device must be cuda/cpu")
    for key in ("max_input_tokens", "max_new_tokens"):
        if type(runtime[key]) is not int or runtime[key] <= 0:
            raise ValueError(f"runtime.{key} must be a positive integer")
    if runtime.get("do_sample", False):
        raise ValueError("This initial pipeline uses deterministic do_sample=false")
    layers = config["steering"]["layers"]
    if not layers or any(type(x) is not int for x in layers) or len(set(layers)) != len(layers):
        raise ValueError("steering.layers must be unique integer decoder-block indices")
    if not math.isfinite(config["steering"]["coefficient"]):
        raise ValueError("steering.coefficient must be finite")
    if type(config["steering"]["normalize"]) is not bool:
        raise ValueError("steering.normalize must be true/false")
    if config["judge"]["hf_id"] != "cais/HarmBench-Llama-2-13b-cls":
        raise ValueError("This initial evaluator supports the official HarmBench 13B classifier")
    return config, config_path


def require_model(config: dict, key: str) -> dict:
    model = config["models"].get(key)
    if not model or not str(model.get("hf_id", "")).strip():
        raise ValueError(f"Set models.{key}.hf_id in configs/experiment.toml first")
    if not COMMIT.fullmatch(model.get("revision", "")):
        raise ValueError(f"Pin models.{key}.revision with ./scripts/pin-models first")
    return model


def require_judge(config: dict) -> dict:
    judge = config["judge"]
    if not COMMIT.fullmatch(judge.get("revision", "")):
        raise ValueError("Pin judge.revision with ./scripts/pin-models first")
    return judge


def safe_name(value: str) -> str:
    if not SAFE_NAME.fullmatch(value) or value in {".", ".."}:
        raise ValueError(
            "Use a run/model name containing only letters, digits, dot, dash, underscore"
        )
    return value
