# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------

"""MLflow-aware checkpointing for exported inference weights."""

import datetime
from collections.abc import Mapping
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger

BEST_ARTIFACT_FILENAME = "best.pt"
ENCODER_TAGS = {
    "facebook/dinov3-vitb16-pretrain-lvd1689m": "dinov3b16",
    "facebook/dinov3-vitl16-pretrain-lvd1689m": "dinov3l16",
}


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

    def __init__(
        self,
        class_names: list[str],
        dirpath: str | Path | None = None,
        filename: str | None = None,
        monitor: str | None = None,
        verbose: bool = False,
        save_last: bool | Literal["link"] | None = None,
        save_top_k: int = 1,
        save_weights_only: bool = False,
        mode: str = "min",
        auto_insert_metric_name: bool = True,
        every_n_train_steps: int | None = None,
        train_time_interval: timedelta | None = None,
        every_n_epochs: int | None = None,
        save_on_train_epoch_end: bool | None = None,
        enable_version_counter: bool = True,
    ) -> None:
        super().__init__(
            dirpath=dirpath,
            filename=filename,
            monitor=monitor,
            verbose=verbose,
            save_last=save_last,
            save_top_k=save_top_k,
            save_weights_only=save_weights_only,
            mode=mode,
            auto_insert_metric_name=auto_insert_metric_name,
            every_n_train_steps=every_n_train_steps,
            train_time_interval=train_time_interval,
            every_n_epochs=every_n_epochs,
            save_on_train_epoch_end=save_on_train_epoch_end,
            enable_version_counter=enable_version_counter,
        )
        if not class_names or any(not name.strip() for name in class_names):
            raise ValueError("class_names must contain non-empty names")

        self.class_names = tuple(class_names)
        self.train_date = datetime.datetime.now().strftime("%m-%d-%Y")

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
            for key, value in self._build_tags(trainer).items():
                logger.experiment.set_tag(run_id, key, value)

    def _build_tags(self, trainer: Trainer) -> dict[str, Any]:
        model = getattr(trainer.lightning_module, "_orig_mod", trainer.lightning_module)
        if len(self.class_names) != model.num_classes:
            raise ValueError(
                "class_names count must match the model's num_classes: "
                f"{len(self.class_names)} != {model.num_classes}"
            )

        height, width = model.img_size
        if height != width:
            raise ValueError(
                f"MLflow resolution tag requires square images, got {model.img_size}"
            )

        backbone_name = model.network.encoder.backbone_name
        try:
            encoder = ENCODER_TAGS[backbone_name]
        except KeyError as error:
            raise ValueError(
                f"Unsupported encoder for MLflow tags: {backbone_name}"
            ) from error

        if self.best_model_score is None:
            raise RuntimeError("Best model score is unavailable for MLflow tags")
        best_model_score = self.best_model_score
        if isinstance(best_model_score, torch.Tensor):
            best_model_score = best_model_score.detach().cpu().item()

        return {
            "classes": list(self.class_names),
            "type": "semantic_segmentation",
            "arch": "eomt",
            "encoder": encoder,
            "resolution": height,
            "train_date": self.train_date,
            "val_metric": "IoU",
            "val_score": round(float(best_model_score), 3),
            "library": "pytorch",
        }
