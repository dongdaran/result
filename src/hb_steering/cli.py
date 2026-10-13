from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from .config import COMMIT, load_config, require_judge, require_model
from .prepare import prepare_dataset, read_prepared
from .storage import atomic_bytes, read_journal, run_lock


def pin_models(
    config: dict, config_path: Path, *, refresh: bool = False, model_path: Path | None = None
) -> None:
    model_path = model_path or config_path.parent.parent / "pyproject.toml"
    entries = [(f"tool.hb-steering.models.{key}", value) for key, value in config["models"].items()]
    text = model_path.read_text()
    judge_text = config_path.read_text()
    if config["judge"].get("method") != "rule":
        entries.append(("judge", config["judge"]))
    replacements = {}
    for section, model in entries:
        model_id = str(model.get("hf_id", "")).strip()
        if not model_id:
            print(f"{section}: pending HF model ID")
            continue
        if COMMIT.fullmatch(model.get("revision", "")) and not refresh:
            print(f"{section}: already pinned")
            continue
        headers = {"User-Agent": "harmbench-steering/0.1"}
        token = os.environ.get("HF_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(
            "https://huggingface.co/api/models/" + quote(model_id, safe="/"), headers=headers
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            revision = json.load(response)["sha"]
        if not COMMIT.fullmatch(revision):
            raise ValueError(f"Invalid HF revision returned for {section}")
        replacements[section] = revision
    changed_model = changed_judge = False
    for section, revision in replacements.items():
        is_judge = section == "judge"
        current = judge_text if is_judge else text
        pattern = rf"(?ms)(^\[{re.escape(section)}\]\s*\n)(.*?)(?=^\[|\Z)"
        match = re.search(pattern, current)
        if not match:
            raise ValueError(f"Missing TOML section: {section}")
        body, count = re.subn(
            r'(?m)^revision\s*=\s*"[^"]*"\s*$', f'revision = "{revision}"', match[2], count=1
        )
        if count != 1:
            raise ValueError(f"Expected a revision string in {section}")
        updated = current[: match.start(2)] + body + current[match.end(2) :]
        if is_judge:
            judge_text = updated
            changed_judge = True
        else:
            text = updated
            changed_model = True
        print(f"{section}: pinned {revision}")
    # All remote resolutions succeeded before mutating the user's config.
    if changed_model:
        atomic_bytes(model_path, text.encode())
    if changed_judge:
        atomic_bytes(config_path, judge_text.encode())


def preflight(config: dict, model_key: str | None = None) -> bool:
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise RuntimeError("Install GPU dependencies: uv sync --extra gpu") from exc
    device = config["runtime"]["device"]
    report = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "requested_device": device,
        "cuda_available": torch.cuda.is_available(),
        "pending_models": [key for key, model in config["models"].items() if not model["hf_id"]],
    }
    okay = device != "cuda" or torch.cuda.is_available()
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        report.update(
            gpu=torch.cuda.get_device_name(0),
            free_vram_gib=round(free / 2**30, 2),
            total_vram_gib=round(total / 2**30, 2),
            bf16_supported=torch.cuda.is_bf16_supported(),
        )
    if model_key:
        model = require_model(config, model_key)
        from transformers import AutoConfig, AutoTokenizer

        model_info = AutoConfig.from_pretrained(
            model["hf_id"], revision=model["revision"], trust_remote_code=False
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model["hf_id"], revision=model["revision"], trust_remote_code=False
        )
        if model.get("use_chat_template", True) and not tokenizer.chat_template:
            raise ValueError(
                "Target tokenizer has no chat template; explicitly set use_chat_template=false for a base model"
            )
        report.update(
            model=model["hf_id"],
            revision=model["revision"],
            architecture=model_info.model_type,
            context_window=getattr(model_info, "max_position_embeddings", None),
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not okay:
        print("CUDA is unavailable on this host. Run GPU experiments on Runpod.", file=sys.stderr)
    return okay


def _run_target_folds(run: dict, command: str) -> None:
    from .experiment import extract, generate
    from .models import HFRunner

    runner = None
    try:
        for fold in _selected_runs(run):
            label = f"fold {fold['fold']}" if "fold" in fold else "final external transfer"
            path = fold["directory"] / "vectors/directions.pt"
            responses = read_journal(fold["directory"] / "generations/responses.jsonl")
            pending_generation = set(responses) != {job["key"] for job in fold["jobs"]}
            needs_runner = (command != "generate" and not path.exists()) or (
                command != "extract" and pending_generation
            )
            if command == "generate" and not path.exists():
                raise ValueError(f"Extract {label} vectors before generation")
            if needs_runner and runner is None:
                runner = HFRunner(
                    require_model(run["config"], run["model_key"]), run["config"]["runtime"]
                )
            print(f"Target {label}", flush=True)
            if command != "generate":
                extract(fold, runner)
            if command != "extract":
                generate(fold, runner)
    finally:
        if runner is not None:
            runner.close()


def _score_folds(run: dict) -> None:
    from .evaluate import HarmBenchJudge, RuleJudge
    from .experiment import _environment, _validated_records, score

    judge = None
    try:
        for fold in _selected_runs(run):
            label = f"fold {fold['fold']}" if "fold" in fold else "final external transfer"
            generations, judgments = _validated_records(fold)
            if len(generations) != len(fold["jobs"]):
                raise ValueError(f"Complete {label} generation before scoring")
            if len(judgments) != len(fold["jobs"]) and judge is None:
                judge_config = require_judge(run["config"])
                judge = (
                    RuleJudge(judge_config)
                    if judge_config.get("method") == "rule"
                    else HarmBenchJudge(judge_config)
                )
            if judge is not None and hasattr(judge, "runner"):
                _environment(judge.runner, fold["directory"] / "logs/judge_environment.json")
            print(f"Score {label}", flush=True)
            score(fold, judge)
    finally:
        if judge is not None:
            judge.close()


def _selected_runs(run: dict) -> list[dict]:
    contexts = list(run["fold_runs"])
    if run.get("selected_fold") is None and run.get("transfer_run"):
        contexts.append(run["transfer_run"])
    return contexts


def _batch_size(value: str) -> int:
    size = int(value)
    if size <= 0:
        raise argparse.ArgumentTypeError("batch size must be a positive integer")
    return size


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Level-wise HarmBench steering experiments")
    result.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    result.add_argument("--config", help="Config path, relative to project root or absolute")
    sub = result.add_subparsers(dest="command", required=True)
    download = sub.add_parser("download", help="Download pinned data and create/reuse fixed splits")
    download.add_argument(
        "--resplit", action="store_true", help="Explicitly replace fixed split IDs"
    )
    sub.add_parser("check", help="Validate data, labels, split coverage, and hashes without GPU")
    pin = sub.add_parser("pin-models", help="Pin configured HF IDs to immutable revisions")
    pin.add_argument(
        "--refresh", action="store_true", help="Resolve new SHAs even for already-pinned IDs"
    )
    pre = sub.add_parser("preflight", help="Check CUDA and optionally model/tokenizer access")
    pre.add_argument("--model")
    for name in ("smoke", "extract", "generate", "score", "summarize", "run", "transfer"):
        command = sub.add_parser(name)
        command.add_argument("--model", required=True, help="A config model key, e.g. model_1")
        command.add_argument("--run-id", required=name != "smoke", default=None)
        if name != "transfer":
            command.add_argument("--fold", type=int, help="Process one fold; default is all folds")
        else:
            command.set_defaults(fold=None)
        command.add_argument("--limit-per-level", type=int)
        command.add_argument(
            "--batch-size", type=_batch_size,
            help="Override runtime.batch_size; saved in the effective run configuration",
        )
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    root = args.root.resolve()
    try:
        config, config_path = load_config(root, args.config)
        if args.command == "download":
            summary = prepare_dataset(root, config, resplit=args.resplit)
            print(json.dumps(summary, indent=2, ensure_ascii=False))
            print("Ready: fixed splits saved. Reuse this dataset for later runs.")
            return 0
        if args.command == "check":
            records, labels, benign, saved = read_prepared(root, config)
            external_summary = None
            if config.get("external", {}).get("enabled", False):
                from .external import read_external

                fit_ids = set(saved["benign"]["fit"])
                harmful, safe = read_external(
                    root, config, records, [r for r in benign if r["benign_id"] in fit_ids]
                )
                external_summary = {"harmful": len(harmful), "benign": len(safe)}
            print(
                json.dumps(
                    {
                        "status": "ok",
                        "harmbench": len(records),
                        "labels": len(labels),
                        "benign_safe": len(benign),
                        "fingerprint": saved["fingerprint"],
                        "n_folds": saved["harmbench"]["n_folds"],
                        "evaluation_protocol": "grouped_kfold_out_of_fold",
                        "external": external_summary,
                    },
                    indent=2,
                )
            )
            return 0
        if args.command == "pin-models":
            pin_models(config, config_path, refresh=args.refresh, model_path=root / "pyproject.toml")
            return 0
        if args.command == "preflight":
            return 0 if preflight(config, args.model) else 2

        from .experiment import setup_run, summarize, summarize_oof, summarize_transfer

        if args.batch_size is not None:
            config = copy.deepcopy(config)
            config["runtime"]["batch_size"] = args.batch_size
        require_model(config, args.model)
        if args.command in {"run", "score", "transfer"}:
            require_judge(config)
        if args.command == "smoke":
            config = copy.deepcopy(config)
            config["runtime"]["max_new_tokens"] = min(config["runtime"]["max_new_tokens"], 16)
            args.fold = args.fold if args.fold is not None else 1
            args.limit_per_level = 2
            if not args.run_id:
                args.run_id = "smoke-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        run = setup_run(
            root, config, config_path, args.model, args.run_id, args.fold, args.limit_per_level
        )
        with run_lock(run["directory"]):
            if args.command == "transfer":
                if not run.get("transfer_run"):
                    raise ValueError("Enable external and/or over_refusal evaluation for transfer")
                transfer_only = {**run, "fold_runs": [run["transfer_run"]]}
                transfer_only.pop("transfer_run")
                _run_target_folds(transfer_only, "run")
                _score_folds(transfer_only)
                transfer_report = summarize_transfer(run["transfer_run"])
                print(f"Transfer results: {run['transfer_run']['directory'] / 'metrics'}")
                return 0 if transfer_report["complete"] else 2
            if args.command in {"smoke", "run", "extract", "generate"}:
                _run_target_folds(run, args.command)
                if args.command == "smoke":
                    print(f"GPU smoke complete: {run['directory']} (no ASR judge run)")
                    return 0
                if args.command in {"extract", "generate"}:
                    return 0
            if args.command in {"run", "score"}:
                _score_folds(run)
                if args.command == "score":
                    return 0
            selected_reports = [summarize(fold) for fold in run["fold_runs"]]
            report = summarize_oof(run)
            print(f"Pooled results: {run['directory'] / 'metrics/asr.md'}")
            print(
                f"OOF coverage: {report['scored_jobs']}/{report['expected_jobs']}; "
                f"complete={report['complete']}; full_oof={report['full_oof']}"
            )
            if args.fold is not None:
                print(f"Selected fold {args.fold}: complete={selected_reports[0]['complete']}")
                return 0 if selected_reports[0]["complete"] else 2
            transfer_complete = True
            if run.get("transfer_run"):
                transfer_report = summarize_transfer(run["transfer_run"])
                transfer_complete = transfer_report["complete"]
                print(f"Transfer complete={transfer_complete}; full_transfer={transfer_report['full_transfer']}")
            return 0 if report["complete"] and transfer_complete else 2
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
