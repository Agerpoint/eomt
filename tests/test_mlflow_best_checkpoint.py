# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------

"""Tests for MLflow best-checkpoint export and artifact behavior."""

import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

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
        self.tags: list[dict[str, Any]] = []

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

    def set_tag(self, run_id: str, key: str, value: Any) -> None:
        self.tags.append(
            {
                "run_id": run_id,
                "key": key,
                "value": value,
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
        self._tracking_uri = "databricks"

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
        backbone_name: str = "facebook/dinov3-vitl16-pretrain-lvd1689m",
        is_global_zero: bool = True,
    ) -> None:
        self.checkpoint = checkpoint
        self.loggers = [] if logger is None else [logger]
        self.is_global_zero = is_global_zero
        self.global_step = 1
        self.lightning_module = SimpleNamespace(
            num_classes=2,
            img_size=(512, 512),
            network=SimpleNamespace(
                encoder=SimpleNamespace(
                    backbone_name=backbone_name,
                )
            ),
        )

    def save_checkpoint(self, filepath: str, weights_only: bool = False) -> None:
        torch.save(self.checkpoint, filepath)


class _FakePythonModel:
    pass


class _FakeTensorSpec:
    def __init__(self, dtype, shape, name) -> None:
        self.dtype = dtype
        self.shape = shape
        self.name = name


class _FakeSchema(list):
    pass


class _FakeModelSignature:
    def __init__(self, inputs, outputs) -> None:
        self.inputs = inputs
        self.outputs = outputs


class _FakePyfunc:
    PythonModel = _FakePythonModel

    def __init__(self) -> None:
        self.logged_models: list[dict[str, Any]] = []

    def log_model(self, **kwargs: Any) -> None:
        self.logged_models.append(kwargs)


class _FakeMlflow(ModuleType):
    def __init__(self) -> None:
        super().__init__("mlflow")
        self.pyfunc = _FakePyfunc()
        self.tracking_uris: list[str] = []
        self.started_runs: list[str] = []
        self._active_run = None

    def set_tracking_uri(self, tracking_uri: str) -> None:
        self.tracking_uris.append(tracking_uri)

    def active_run(self):
        return self._active_run

    @contextmanager
    def start_run(self, run_id: str):
        self.started_runs.append(run_id)
        previous_run = self._active_run
        self._active_run = SimpleNamespace(info=SimpleNamespace(run_id=run_id))
        try:
            yield self._active_run
        finally:
            self._active_run = previous_run


@contextmanager
def _fake_mlflow_modules():
    mlflow_module = _FakeMlflow()
    signature_module = ModuleType("mlflow.models.signature")
    signature_module.ModelSignature = _FakeModelSignature
    schema_module = ModuleType("mlflow.types.schema")
    schema_module.Schema = _FakeSchema
    schema_module.TensorSpec = _FakeTensorSpec

    modules = {
        "mlflow": mlflow_module,
        "mlflow.models": ModuleType("mlflow.models"),
        "mlflow.models.signature": signature_module,
        "mlflow.types": ModuleType("mlflow.types"),
        "mlflow.types.schema": schema_module,
    }
    with patch.dict(sys.modules, modules):
        sys.modules.pop("training.mlflow_model", None)
        try:
            yield mlflow_module
        finally:
            sys.modules.pop("training.mlflow_model", None)


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
            class_names=["background", "leaf"],
            dirpath=self.root,
            monitor="metrics/val_iou_all",
            mode="max",
            save_top_k=1,
        )
        self.callback.train_date = "07-29-2026"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_logs_replacement_best_artifact_to_logger_run(self) -> None:
        trainer = _FakeTrainer(self._checkpoint(1.0), self.logger)
        first_checkpoint_path = self.root / "best-epoch-1.ckpt"
        self._save_as_best(trainer, first_checkpoint_path, score=0.4567)

        trainer.checkpoint = self._checkpoint(2.0)
        trainer.global_step = 2
        second_checkpoint_path = self.root / "best-epoch-2.ckpt"
        self._save_as_best(trainer, second_checkpoint_path, score=0.7896)

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
        first_tags = {
            tag["key"]: tag["value"] for tag in self.client.tags[:9]
        }
        self.assertEqual(
            first_tags,
            {
                "classes": ["background", "leaf"],
                "type": "semantic_segmentation",
                "arch": "eomt",
                "encoder": "dinov3l16",
                "resolution": 512,
                "train_date": "07-29-2026",
                "val_metric": "IoU",
                "val_score": 0.457,
                "library": "pytorch",
            },
        )
        self.assertEqual(
            [
                tag["value"]
                for tag in self.client.tags
                if tag["key"] == "val_score"
            ],
            [0.457, 0.79],
        )

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
        self.assertEqual(self.client.tags, [])

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
        self.assertEqual(self.client.tags, [])

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

    def test_rejects_tag_metadata_that_does_not_match_model(self) -> None:
        trainer = _FakeTrainer(self._checkpoint(1.0), self.logger)
        trainer.lightning_module.img_size = (512, 640)
        self.callback.best_model_score = torch.tensor(0.5)

        with self.assertRaisesRegex(ValueError, "requires square images"):
            self.callback._build_tags(cast(Trainer, trainer))

        trainer.lightning_module.img_size = (512, 512)
        trainer.lightning_module.network.encoder.backbone_name = "unsupported"
        with self.assertRaisesRegex(ValueError, "Unsupported encoder"):
            self.callback._build_tags(cast(Trainer, trainer))

    def test_builds_encoder_metadata_for_supported_dinov3_backbones(self) -> None:
        self.callback.best_model_score = torch.tensor(0.5)
        cases = {
            "facebook/dinov3-vits16-pretrain-lvd1689m": "dinov3s16",
            "facebook/dinov3-vitb16-pretrain-lvd1689m": "dinov3b16",
            "facebook/dinov3-vitl16-pretrain-lvd1689m": "dinov3l16",
        }

        for backbone_name, expected_tag in cases.items():
            with self.subTest(backbone_name=backbone_name):
                trainer = _FakeTrainer(
                    self._checkpoint(1.0),
                    self.logger,
                    backbone_name=backbone_name,
                )

                tags = self.callback._build_tags(cast(Trainer, trainer))

                self.assertEqual(tags["encoder"], expected_tag)

    def test_rejects_unsupported_encoder_before_artifact_upload(self) -> None:
        trainer = _FakeTrainer(
            self._checkpoint(1.0),
            self.logger,
            backbone_name="unsupported",
        )

        with self.assertRaisesRegex(ValueError, "Unsupported encoder"):
            self._save_as_best(trainer, self.root / "best.ckpt")

        self.assertEqual(self.client.artifacts, [])
        self.assertEqual(self.client.tags, [])

    def test_logs_final_best_as_metadata_only_model(self) -> None:
        trainer = _FakeTrainer(self._checkpoint(1.0), self.logger)
        checkpoint_path = self.root / "best-epoch-3.ckpt"
        best_path = self.root / BEST_ARTIFACT_FILENAME
        checkpoint_path.write_bytes(b"lightning checkpoint")
        best_path.write_bytes(b"best weights")
        self.callback.best_model_path = str(checkpoint_path)
        self.callback.best_model_score = torch.tensor(0.87654)

        with _fake_mlflow_modules() as fake_mlflow:
            self.callback.on_fit_end(
                cast(Trainer, trainer),
                trainer.lightning_module,
            )

        self.assertEqual(fake_mlflow.tracking_uris, ["databricks"])
        self.assertEqual(fake_mlflow.started_runs, ["run-123"])
        self.assertEqual(len(fake_mlflow.pyfunc.logged_models), 1)
        logged_model = fake_mlflow.pyfunc.logged_models[0]
        self.assertEqual(logged_model["artifact_path"], "model")
        self.assertEqual(logged_model["artifacts"], {"best": str(best_path)})
        self.assertEqual(logged_model["python_model"].pt_file, str(best_path))
        self.assertEqual(
            logged_model["pip_requirements"],
            ["torch>=2.6.0,<2.8.0"],
        )
        self.assertEqual(
            logged_model["metadata"]["val_score"],
            0.877,
        )
        self.assertEqual(
            logged_model["input_example"].shape,
            (1, 3, 512, 512),
        )
        signature = logged_model["signature"]
        self.assertEqual(signature.inputs[0].shape, (-1, 3, -1, -1))
        self.assertEqual(signature.inputs[0].name, "input_tensor")
        self.assertEqual(signature.outputs[0].shape, (-1, 2, -1, -1))
        self.assertEqual(signature.outputs[0].name, "output_tensor")

    def test_final_model_logging_requires_best_weights(self) -> None:
        trainer = _FakeTrainer(self._checkpoint(1.0), self.logger)
        self.callback.best_model_path = str(self.root / "missing.ckpt")

        with self.assertRaisesRegex(RuntimeError, "Best weights are unavailable"):
            self.callback.on_fit_end(
                cast(Trainer, trainer),
                trainer.lightning_module,
            )

    def test_final_model_logging_rejects_another_active_run(self) -> None:
        trainer = _FakeTrainer(self._checkpoint(1.0), self.logger)
        checkpoint_path = self.root / "best.ckpt"
        checkpoint_path.write_bytes(b"checkpoint")
        (self.root / BEST_ARTIFACT_FILENAME).write_bytes(b"best weights")
        self.callback.best_model_path = str(checkpoint_path)
        self.callback.best_model_score = torch.tensor(0.5)

        with _fake_mlflow_modules() as fake_mlflow:
            fake_mlflow._active_run = SimpleNamespace(
                info=SimpleNamespace(run_id="different-run")
            )
            with self.assertRaisesRegex(RuntimeError, "another MLflow run"):
                self.callback.on_fit_end(
                    cast(Trainer, trainer),
                    trainer.lightning_module,
                )

    def _save_as_best(
        self,
        trainer: _FakeTrainer,
        checkpoint_path: Path,
        *,
        score: float = 0.5,
    ) -> None:
        self.callback.best_model_path = str(checkpoint_path)
        self.callback.best_model_score = torch.tensor(score)
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
