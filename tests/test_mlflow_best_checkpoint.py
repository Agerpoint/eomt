# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------

"""Tests for MLflow best-checkpoint export and artifact behavior."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger

from training.mlflow_best_checkpoint import (
    BEST_ARTIFACT_FILENAME,
    MLFlowBestCheckpoint,
    _export_checkpoint,
)


class _RecordingMlflowClient:
    def __init__(self) -> None:
        self.artifacts: list[dict[str, Any]] = []

    def log_artifact(
        self,
        run_id: str,
        local_path: str,
        artifact_path: str | None = None,
    ) -> None:
        self.artifacts.append(
            {
                "run_id": run_id,
                "filename": Path(local_path).name,
                "artifact_path": artifact_path,
                "state_dict": torch.load(
                    local_path,
                    map_location="cpu",
                    weights_only=True,
                ),
            }
        )


class _FailingMlflowClient(_RecordingMlflowClient):
    def log_artifact(
        self,
        run_id: str,
        local_path: str,
        artifact_path: str | None = None,
    ) -> None:
        raise OSError("artifact upload failed")


class _FakeMLFlowLogger(MLFlowLogger):
    def __init__(
        self,
        client: _RecordingMlflowClient,
        run_id: str | None = "run-123",
    ) -> None:
        self._client = client
        self._fake_run_id = run_id

    @property
    def experiment(self) -> _RecordingMlflowClient:
        return self._client

    @property
    def run_id(self) -> str | None:
        return self._fake_run_id

    def after_save_checkpoint(
        self,
        checkpoint_callback: ModelCheckpoint,
    ) -> None:
        pass


class _FakeTrainer:
    def __init__(
        self,
        checkpoint: dict[str, Any],
        logger: _FakeMLFlowLogger | None,
        *,
        is_global_zero: bool = True,
    ) -> None:
        self.checkpoint = checkpoint
        self.loggers = [] if logger is None else [logger]
        self.is_global_zero = is_global_zero
        self.global_step = 1

    def save_checkpoint(self, filepath: str, weights_only: bool = False) -> None:
        torch.save(self.checkpoint, filepath)


class ExportCheckpointTests(unittest.TestCase):
    def test_exports_only_cleaned_cpu_state_dict_and_overwrites_output(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkpoint_path = root / "source.ckpt"
            output_path = root / BEST_ARTIFACT_FILENAME
            output_path.write_bytes(b"previous artifact")
            torch.save(
                {
                    "state_dict": {
                        "network.encoder.weight": torch.tensor([1.0]),
                        "criterion.empty_weight": torch.tensor([2.0]),
                    },
                    "optimizer_states": [{"step": 12}],
                    "epoch": 3,
                },
                checkpoint_path,
            )

            _export_checkpoint(checkpoint_path, output_path)

            state_dict = torch.load(
                output_path,
                map_location="cpu",
                weights_only=True,
            )
            self.assertEqual(
                set(state_dict),
                {"encoder.weight", "criterion.empty_weight"},
            )
            self.assertTrue(
                torch.equal(
                    state_dict["encoder.weight"],
                    torch.tensor([1.0]),
                )
            )
            self.assertTrue(
                torch.equal(
                    state_dict["criterion.empty_weight"],
                    torch.tensor([2.0]),
                )
            )
            self.assertTrue(
                all(value.device.type == "cpu" for value in state_dict.values())
            )

    def test_rejects_checkpoint_without_state_dict(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkpoint_path = root / "source.ckpt"
            torch.save({"epoch": 3}, checkpoint_path)

            with self.assertRaisesRegex(ValueError, "does not contain a state_dict"):
                _export_checkpoint(
                    checkpoint_path,
                    root / BEST_ARTIFACT_FILENAME,
                )


class MLFlowBestCheckpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.client = _RecordingMlflowClient()
        self.logger = _FakeMLFlowLogger(self.client)
        self.callback = MLFlowBestCheckpoint(
            dirpath=self.root,
            monitor="metrics/val_iou_all",
            mode="max",
            save_top_k=1,
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_logs_replacement_best_artifact_to_logger_run(self) -> None:
        trainer = _FakeTrainer(self._checkpoint(1.0), self.logger)
        first_checkpoint_path = self.root / "best-epoch-1.ckpt"
        self._save_as_best(trainer, first_checkpoint_path)

        trainer.checkpoint = self._checkpoint(2.0)
        trainer.global_step = 2
        second_checkpoint_path = self.root / "best-epoch-2.ckpt"
        self._save_as_best(trainer, second_checkpoint_path)

        self.assertEqual(len(self.client.artifacts), 2)
        self.assertEqual(
            [artifact["run_id"] for artifact in self.client.artifacts],
            ["run-123", "run-123"],
        )
        self.assertEqual(
            [artifact["filename"] for artifact in self.client.artifacts],
            [BEST_ARTIFACT_FILENAME, BEST_ARTIFACT_FILENAME],
        )
        self.assertEqual(
            [artifact["artifact_path"] for artifact in self.client.artifacts],
            [None, None],
        )
        self.assertTrue(
            torch.equal(
                self.client.artifacts[0]["state_dict"]["weight"],
                torch.tensor([1.0]),
            )
        )
        self.assertTrue(
            torch.equal(
                self.client.artifacts[1]["state_dict"]["weight"],
                torch.tensor([2.0]),
            )
        )
        local_best = torch.load(
            self.root / BEST_ARTIFACT_FILENAME,
            map_location="cpu",
            weights_only=True,
        )
        self.assertTrue(torch.equal(local_best["weight"], torch.tensor([2.0])))
        self.assertTrue(first_checkpoint_path.exists())
        self.assertTrue(second_checkpoint_path.exists())

    def test_does_not_export_non_best_or_last_checkpoint(self) -> None:
        trainer = _FakeTrainer(self._checkpoint(1.0), self.logger)
        self.callback.best_model_path = str(self.root / "existing-best.ckpt")

        self.callback._save_checkpoint(
            cast(Trainer, trainer),
            str(self.root / "top-k-but-not-best.ckpt"),
        )
        self.callback._save_checkpoint(
            cast(Trainer, trainer),
            str(self.root / "last.ckpt"),
        )

        self.assertFalse((self.root / BEST_ARTIFACT_FILENAME).exists())
        self.assertEqual(self.client.artifacts, [])

    def test_does_not_export_on_non_global_zero_process(self) -> None:
        trainer = _FakeTrainer(
            self._checkpoint(1.0),
            self.logger,
            is_global_zero=False,
        )
        checkpoint_path = self.root / "best.ckpt"

        self._save_as_best(trainer, checkpoint_path)

        self.assertTrue(checkpoint_path.exists())
        self.assertFalse((self.root / BEST_ARTIFACT_FILENAME).exists())
        self.assertEqual(self.client.artifacts, [])

    def test_fails_when_mlflow_logger_is_missing(self) -> None:
        trainer = _FakeTrainer(self._checkpoint(1.0), logger=None)
        checkpoint_path = self.root / "best.ckpt"

        with self.assertRaisesRegex(RuntimeError, "requires an MLFlowLogger"):
            self._save_as_best(trainer, checkpoint_path)

        self.assertTrue((self.root / BEST_ARTIFACT_FILENAME).exists())

    def test_fails_when_mlflow_logger_has_no_run_id(self) -> None:
        logger = _FakeMLFlowLogger(self.client, run_id=None)
        trainer = _FakeTrainer(self._checkpoint(1.0), logger)

        with self.assertRaisesRegex(RuntimeError, "did not provide a run ID"):
            self._save_as_best(trainer, self.root / "best.ckpt")

    def test_propagates_artifact_upload_failure(self) -> None:
        logger = _FakeMLFlowLogger(_FailingMlflowClient())
        trainer = _FakeTrainer(self._checkpoint(1.0), logger)

        with self.assertRaisesRegex(OSError, "artifact upload failed"):
            self._save_as_best(trainer, self.root / "best.ckpt")

    def _save_as_best(
        self,
        trainer: _FakeTrainer,
        checkpoint_path: Path,
    ) -> None:
        self.callback.best_model_path = str(checkpoint_path)
        self.callback._save_checkpoint(
            cast(Trainer, trainer),
            str(checkpoint_path),
        )

    @staticmethod
    def _checkpoint(weight: float) -> dict[str, Any]:
        return {
            "state_dict": {
                "network.weight": torch.tensor([weight]),
            },
            "optimizer_states": [{"step": 1}],
        }


if __name__ == "__main__":
    unittest.main()
