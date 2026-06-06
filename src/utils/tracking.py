"""Weights & Biases experiment tracking utilities."""

from pathlib import Path

import wandb
import yaml


def init_run(
    config_path: str | Path,
    run_name: str,
    tags: list[str] | None = None,
    project: str = "zeineuski",
) -> None:
    """Initialize a WandB run with config loaded from YAML."""
    path = Path(config_path)
    config = yaml.safe_load(path.read_text()) if path.exists() else {}
    wandb.init(project=project, name=run_name, config=config, tags=tags or [])


def log_metrics(metrics: dict, step: int | None = None) -> None:
    """Log a dictionary of metrics to WandB."""
    wandb.log(metrics, step=step)


def log_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
    title: str = "Confusion Matrix",
) -> None:
    """Log a confusion matrix plot to WandB."""
    wandb.log(
        {
            title: wandb.plot.confusion_matrix(
                probs=None,
                y_true=y_true,
                preds=y_pred,
                class_names=class_names,
            )
        }
    )


def finish_run() -> None:
    """Finish the current WandB run."""
    wandb.finish()
