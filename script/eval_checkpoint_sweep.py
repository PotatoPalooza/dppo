#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


DPPO_ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT = DPPO_ROOT / "script" / "run.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a sweep of official DPPO checkpoints and save the best checkpoint."
    )
    parser.add_argument("--config-dir", required=True, help="Hydra config dir containing the eval config.")
    parser.add_argument("--config-name", required=True, help="Hydra eval config name.")
    parser.add_argument("--checkpoint-dir", required=True, help="Directory containing state_*.pt checkpoints.")
    parser.add_argument("--output-dir", required=True, help="Directory for per-checkpoint eval outputs and summary.")
    parser.add_argument("--device", default="cuda:0", help="Device override passed into the eval config.")
    parser.add_argument("--n-envs", type=int, default=None, help="Optional eval env count override.")
    parser.add_argument("--n-episodes", type=int, default=None, help="Optional completed-episode target override.")
    parser.add_argument("--n-steps", type=int, default=None, help="Optional eval rollout length override.")
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=None,
        help="Optional env.max_episode_steps override.",
    )
    parser.add_argument("--every-n", type=int, default=1, help="Evaluate every Nth checkpoint in sorted checkpoint order.")
    parser.add_argument("--start-index", type=int, default=None, help="Optional minimum checkpoint index.")
    parser.add_argument("--end-index", type=int, default=None, help="Optional maximum checkpoint index.")
    parser.add_argument(
        "--video-checkpoints",
        choices=("none", "best", "all"),
        default="none",
        help="Render no videos, only the final best checkpoint, or every evaluated checkpoint.",
    )
    parser.add_argument(
        "--render-num",
        type=int,
        default=1,
        help="Number of envs to render when videos are enabled.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse existing per-checkpoint metrics when eval_metrics.json already exists.",
    )
    parser.add_argument(
        "--copy-best-to",
        default=None,
        help="Optional destination path for a copy of the best checkpoint.",
    )
    return parser.parse_args()


def _checkpoint_index(path: Path) -> int | None:
    if not path.stem.startswith("state_"):
        return None
    suffix = path.stem.removeprefix("state_")
    return int(suffix) if suffix.isdigit() else None


def _collect_checkpoints(
    checkpoint_dir: Path,
    every_n: int,
    start_index: int | None,
    end_index: int | None,
) -> list[tuple[int, Path]]:
    available: list[tuple[int, Path]] = []
    for path in sorted(checkpoint_dir.glob("state_*.pt")):
        index = _checkpoint_index(path)
        if index is None:
            continue
        if start_index is not None and index < start_index:
            continue
        if end_index is not None and index > end_index:
            continue
        available.append((index, path))
    if every_n <= 1:
        return available
    return [item for ordinal, item in enumerate(available) if ordinal % every_n == 0]


def _metric_rank(metrics: dict[str, float | int | str]) -> tuple[float, float, float]:
    return (
        float(metrics["success_rate"]),
        float(metrics["best_reward_mean"]),
        float(metrics["return_mean"]),
    )


def _load_result_metrics(result_path: Path) -> dict[str, float | int]:
    with np.load(result_path) as payload:
        return {
            "num_episode": int(payload["num_episode"]),
            "success_rate": float(payload["eval_success_rate"]),
            "return_mean": float(payload["eval_episode_reward"]),
            "best_reward_mean": float(payload["eval_best_reward"]),
            "time": float(payload["time"]),
        }


def _run_eval(
    args: argparse.Namespace,
    *,
    checkpoint_path: Path,
    logdir: Path,
    save_video: bool,
) -> dict[str, float | int]:
    command = [
        sys.executable,
        str(RUN_SCRIPT),
        f"--config-dir={Path(args.config_dir).expanduser().resolve()}",
        f"--config-name={args.config_name}",
        f"base_policy_path={checkpoint_path}",
        f"logdir={logdir}",
        f"device={args.device}",
        f"env.save_video={str(save_video)}",
        f"render_num={args.render_num if save_video else 0}",
    ]
    if args.n_envs is not None:
        command.append(f"env.n_envs={args.n_envs}")
    if args.n_episodes is not None:
        command.append(f"n_episodes={args.n_episodes}")
    if args.n_steps is not None:
        command.append(f"n_steps={args.n_steps}")
    if args.max_episode_steps is not None:
        command.append(f"env.max_episode_steps={args.max_episode_steps}")
    subprocess.run(command, cwd=DPPO_ROOT, check=True)
    result_path = logdir / "result.npz"
    if not result_path.exists():
        raise FileNotFoundError(f"Missing eval result at {result_path}")
    return _load_result_metrics(result_path)


def _success_copy_path(base_path: Path, success_rate: float) -> Path:
    success_pct = int(max(0.0, float(success_rate)) * 100.0)
    return base_path.with_name(
        f"{base_path.stem}_{success_pct}_best{base_path.suffix}"
    )


def _format_metrics_log(metrics_rows: list[dict[str, float | int | str]]) -> str:
    lines = [
        "checkpoint_index\tsuccess_rate\tbest_reward_mean\treturn_mean\tnum_episode\ttime\tcheckpoint_path"
    ]
    for metrics in metrics_rows:
        lines.append(
            "\t".join(
                [
                    str(metrics["checkpoint_index"]),
                    f"{float(metrics['success_rate']):.6f}",
                    f"{float(metrics['best_reward_mean']):.6f}",
                    f"{float(metrics['return_mean']):.6f}",
                    str(metrics["num_episode"]),
                    f"{float(metrics['time']):.6f}",
                    str(metrics["checkpoint_path"]),
                ]
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    best_output_dir = output_dir / "best"
    best_output_dir.mkdir(parents=True, exist_ok=True)
    best_metrics_path = best_output_dir / "eval_metrics.json"
    best_checkpoint_record = best_output_dir / "checkpoint.txt"
    copy_target = (
        Path(args.copy_best_to).expanduser().resolve()
        if args.copy_best_to is not None
        else output_dir / "best_checkpoint.pt"
    )
    copy_target.parent.mkdir(parents=True, exist_ok=True)

    selected = _collect_checkpoints(
        checkpoint_dir=checkpoint_dir,
        every_n=max(1, args.every_n),
        start_index=args.start_index,
        end_index=args.end_index,
    )
    if not selected:
        raise FileNotFoundError(f"No checkpoints selected from {checkpoint_dir}")

    all_metrics: list[dict[str, float | int | str]] = []
    best_checkpoint_path: Path | None = None
    best_metrics: dict[str, float | int | str] | None = None

    for checkpoint_index, checkpoint_path in selected:
        checkpoint_output_dir = output_dir / f"state_{checkpoint_index}"
        checkpoint_output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = checkpoint_output_dir / "eval_metrics.json"
        if args.skip_existing and metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        else:
            metrics = _run_eval(
                args,
                checkpoint_path=checkpoint_path,
                logdir=checkpoint_output_dir,
                save_video=args.video_checkpoints == "all",
            )
            metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        metrics = {
            **metrics,
            "checkpoint_index": checkpoint_index,
            "checkpoint_path": str(checkpoint_path),
        }
        all_metrics.append(metrics)
        if best_metrics is None or _metric_rank(metrics) > _metric_rank(best_metrics):
            best_metrics = metrics
            best_checkpoint_path = checkpoint_path
            best_metrics_path.write_text(json.dumps(best_metrics, indent=2), encoding="utf-8")
            best_checkpoint_record.write_text(str(best_checkpoint_path), encoding="utf-8")
            shutil.copy2(best_checkpoint_path, _success_copy_path(copy_target, float(best_metrics["success_rate"])))
            shutil.copy2(best_checkpoint_path, copy_target)

    assert best_checkpoint_path is not None
    assert best_metrics is not None

    if args.video_checkpoints == "best":
        if not (
            args.skip_existing
            and best_metrics_path.exists()
            and best_checkpoint_record.exists()
            and best_checkpoint_record.read_text(encoding="utf-8").strip() == str(best_checkpoint_path)
        ):
            try:
                best_metrics = {
                    **_run_eval(
                        args,
                        checkpoint_path=best_checkpoint_path,
                        logdir=best_output_dir,
                        save_video=True,
                    ),
                    "checkpoint_index": int(_checkpoint_index(best_checkpoint_path) or -1),
                    "checkpoint_path": str(best_checkpoint_path),
                }
                best_metrics_path.write_text(json.dumps(best_metrics, indent=2), encoding="utf-8")
                best_checkpoint_record.write_text(str(best_checkpoint_path), encoding="utf-8")
            except subprocess.CalledProcessError as exc:
                print(
                    f"[warning] best-checkpoint video render failed: {exc}. "
                    "Keeping metrics and best-checkpoint selection without video.",
                    file=sys.stderr,
                )
    else:
        best_metrics_path.write_text(json.dumps(best_metrics, indent=2), encoding="utf-8")
        best_checkpoint_record.write_text(str(best_checkpoint_path), encoding="utf-8")

    shutil.copy2(best_checkpoint_path, copy_target)

    summary = {
        "config_dir": str(Path(args.config_dir).expanduser().resolve()),
        "config_name": args.config_name,
        "checkpoint_dir": str(checkpoint_dir),
        "output_dir": str(output_dir),
        "device": args.device,
        "n_envs": args.n_envs,
        "n_episodes": args.n_episodes,
        "n_steps": args.n_steps,
        "max_episode_steps": args.max_episode_steps,
        "every_n": args.every_n,
        "num_evaluated": len(all_metrics),
        "best_checkpoint": str(best_checkpoint_path),
        "best_checkpoint_copy": str(copy_target),
        "best_metrics": best_metrics,
    }
    (output_dir / "checkpoint_metrics.json").write_text(
        json.dumps(all_metrics, indent=2), encoding="utf-8"
    )
    (output_dir / "checkpoint_metrics.tsv").write_text(
        _format_metrics_log(all_metrics), encoding="utf-8"
    )
    (output_dir / "best_checkpoint.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
