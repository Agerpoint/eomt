# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------

"""MLflow-aware checkpointing for exported inference weights."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger

BEST_ARTIFACT_FILENAME = "best.pt"


def _export_checkpoint(ckpt_path: str | Path, out_path: str | Path) -> None:
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping) or "state_dict" not in checkpoint:
        raise ValueError("Checkpoint does not contain a state_dict")

    state_dict = checkpoint["state_dict"]
    if not isinstance(state_dict, Mapping):
        raise TypeError("Checkpoint state_dict must be a mapping")

    cleaned_state_dict: dict[str, Any] = {}
    for key, value in state_dict.items():
        if not isinstance(key, str):
            raise TypeError("Checkpoint state_dict keys must be strings")
        if key.startswith("network."):
            key = key[len("network.") :]
        cleaned_state_dict[key] = value

    torch.save(cleaned_state_dict, out_path)


class MLFlowBestCheckpoint(ModelCheckpoint):
    """Export and upload the state dict whenever the global best improves."""

    def _save_checkpoint(self, trainer: Trainer, filepath: str) -> None:
        super()._save_checkpoint(trainer, filepath)

        if not trainer.is_global_zero or filepath != self.best_model_path:
            return

        out_path = Path(filepath).with_name(BEST_ARTIFACT_FILENAME)
        _export_checkpoint(filepath, out_path)

        mlflow_loggers = [
            logger for logger in trainer.loggers if isinstance(logger, MLFlowLogger)
        ]
        if not mlflow_loggers:
            raise RuntimeError("MLFlowBestCheckpoint requires an MLFlowLogger")

        for logger in mlflow_loggers:
            run_id = logger.run_id
            if run_id is None:
                raise RuntimeError("MLFlowLogger did not provide a run ID")
            logger.experiment.log_artifact(run_id, str(out_path))
