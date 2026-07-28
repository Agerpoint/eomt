# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------

"""Tests for folder-semantic MLflow config generation."""

import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import yaml
from PIL import Image

from scripts.generate_folder_semantic_mlflow_config import (
    compute_schedule,
    generate_mlflow_config,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class GeneratorImportTests(unittest.TestCase):
    def test_import_does_not_require_datasets_package(self) -> None:
        """The generator imports when every datasets import is blocked."""
        script = textwrap.dedent(
            """
            import builtins

            original_import = builtins.__import__

            def guarded_import(name, *args, **kwargs):
                if name == "datasets" or name.startswith("datasets."):
                    raise AssertionError(f"unexpected import: {name}")
                return original_import(name, *args, **kwargs)

            builtins.__import__ = guarded_import
            import scripts.generate_folder_semantic_mlflow_config
            """
        )

        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)


class TrainingScheduleTests(unittest.TestCase):
    def test_matches_documented_large_model_example(self) -> None:
        """The four-block schedule matches the documented worked example."""
        schedule = compute_schedule(
            num_train_images=100,
            batch_size=2,
            num_epochs=35,
            num_blocks=4,
        )

        self.assertEqual(schedule.steps_per_epoch, 50)
        self.assertEqual(schedule.total_steps, 1750)
        self.assertEqual(schedule.annealing_starts, (0, 438, 875, 1313))
        self.assertEqual(schedule.annealing_ends, (438, 875, 1313, 1750))
        self.assertEqual(schedule.warmup, (210, 315))

    def test_splits_base_model_schedule_into_three_windows(self) -> None:
        """The base schedule has three windows ending at total training steps."""
        schedule = compute_schedule(
            num_train_images=100,
            batch_size=2,
            num_epochs=35,
            num_blocks=3,
        )

        self.assertEqual(schedule.annealing_starts, (0, 583, 1167))
        self.assertEqual(schedule.annealing_ends, (583, 1167, 1750))

    def test_rounds_annealing_and_warmup_half_up(self) -> None:
        """Fractional schedule values use nearest-integer half-up rounding."""
        schedule = compute_schedule(
            num_train_images=25,
            batch_size=1,
            num_epochs=1,
            num_blocks=4,
        )

        self.assertEqual(schedule.annealing_starts, (0, 6, 13, 19))
        self.assertEqual(schedule.annealing_ends, (6, 13, 19, 25))
        self.assertEqual(schedule.warmup, (3, 5))

    def test_rejects_non_positive_inputs(self) -> None:
        """Every schedule input must be a positive integer."""
        cases = [
            {"num_train_images": 0},
            {"batch_size": 0},
            {"num_epochs": 0},
            {"num_blocks": 0},
        ]
        defaults = {
            "num_train_images": 4,
            "batch_size": 2,
            "num_epochs": 2,
            "num_blocks": 4,
        }

        for overrides in cases:
            with self.subTest(overrides=overrides):
                parameters = defaults | overrides
                with self.assertRaisesRegex(ValueError, "must be positive"):
                    compute_schedule(**parameters)

    def test_rejects_dataset_smaller_than_one_batch(self) -> None:
        """A schedule requires enough training images to form one full batch."""
        with self.assertRaisesRegex(ValueError, "at least batch_size"):
            compute_schedule(
                num_train_images=1,
                batch_size=2,
                num_epochs=1,
                num_blocks=4,
            )


class MlflowConfigGeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.dataset_root = self.root / "dataset"
        self._write_split("train", 5)
        self._write_split("val", 3)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_generates_large_config_with_all_overrides(self) -> None:
        """Large generation overrides inputs while preserving large settings."""
        output_path = self.root / "generated" / "large.yaml"

        result = self._generate(output_path=output_path)
        config = self._load_yaml(output_path)

        self.assertEqual(result, output_path)
        self.assertEqual(config["trainer"]["max_epochs"], 4)
        logger = config["trainer"]["logger"]["init_args"]
        self.assertEqual(logger["experiment_name"], "/Workspace/Shared/test")
        self.assertEqual(logger["run_name"], "test-run")
        checkpoint_callback = config["trainer"]["callbacks"][0]
        self.assertEqual(
            checkpoint_callback["class_path"],
            "training.mlflow_best_checkpoint.MLFlowBestCheckpoint",
        )
        checkpoint = checkpoint_callback["init_args"]
        self.assertEqual(
            checkpoint["dirpath"],
            "/local_disk0/eomt_checkpoints",
        )

        model = config["model"]["init_args"]
        self.assertEqual(model["attn_mask_annealing_start_steps"], [0, 2, 4, 6])
        self.assertEqual(model["attn_mask_annealing_end_steps"], [2, 4, 6, 8])
        self.assertEqual(model["warmup_steps"], [1, 1])
        self.assertEqual(model["llrd"], 1.0)
        self.assertEqual(
            model["network"]["init_args"]["encoder"]["init_args"]["backbone_name"],
            "facebook/dinov3-vitl16-pretrain-lvd1689m",
        )

        data = config["data"]["init_args"]
        self.assertEqual(data["path"], str(self.dataset_root.resolve()))
        self.assertEqual(data["batch_size"], 2)
        self.assertEqual(data["img_size"], [640, 640])
        self.assertEqual(data["num_classes"], 3)

    def test_generates_base_config_with_three_block_schedule(self) -> None:
        """Base generation preserves its backbone and creates three windows."""
        output_path = self.root / "base.yaml"

        self._generate(model_size="base", output_path=output_path)
        config = self._load_yaml(output_path)

        model = config["model"]["init_args"]
        self.assertEqual(model["attn_mask_annealing_start_steps"], [0, 3, 5])
        self.assertEqual(model["attn_mask_annealing_end_steps"], [3, 5, 8])
        network = model["network"]["init_args"]
        self.assertEqual(network["num_blocks"], 3)
        self.assertEqual(
            network["encoder"]["init_args"]["backbone_name"],
            "facebook/dinov3-vitb16-pretrain-lvd1689m",
        )

    def test_counts_only_training_images(self) -> None:
        """Validation images do not affect the generated global-step schedule."""
        output_path = self.root / "train-count.yaml"

        self._generate(output_path=output_path)
        config = self._load_yaml(output_path)

        starts = config["model"]["init_args"]["attn_mask_annealing_start_steps"]
        ends = config["model"]["init_args"]["attn_mask_annealing_end_steps"]
        self.assertEqual(starts, [0, 2, 4, 6])
        self.assertEqual(ends, [2, 4, 6, 8])

    def test_refuses_existing_output_unless_force_is_enabled(self) -> None:
        """Existing output remains unchanged until force is explicitly enabled."""
        output_path = self.root / "existing.yaml"
        output_path.write_text("original\n", encoding="utf-8")

        with self.assertRaises(FileExistsError):
            self._generate(output_path=output_path)
        self.assertEqual(output_path.read_text(encoding="utf-8"), "original\n")

        self._generate(output_path=output_path, force=True)
        self.assertIn("trainer", self._load_yaml(output_path))

    def test_rejects_invalid_parameters_without_writing_output(self) -> None:
        """Invalid generation parameters fail without creating output files."""
        cases = [
            ({"model_size": "giant"}, "model_size"),
            ({"num_epochs": 0}, "num_epochs"),
            ({"batch_size": 0}, "batch_size"),
            ({"image_size": 0}, "image_size"),
            ({"num_classes": 0}, "num_classes"),
            ({"mlflow_experiment_path": " "}, "mlflow_experiment_path"),
            ({"mlflow_run_name": ""}, "mlflow_run_name"),
        ]

        for index, (overrides, message) in enumerate(cases):
            with self.subTest(overrides=overrides):
                output_path = self.root / f"invalid-{index}.yaml"
                with self.assertRaisesRegex(ValueError, message):
                    self._generate(output_path=output_path, **overrides)
                self.assertFalse(output_path.exists())

    def test_rejects_invalid_dataset_without_writing_output(self) -> None:
        """Unpaired training samples fail validation before output is written."""
        dataset_root = self.root / "invalid-dataset"
        image_dir = dataset_root / "train" / "Images"
        mask_dir = dataset_root / "train" / "Masks"
        image_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)
        self._write_image(image_dir / "unmatched.png")
        output_path = self.root / "invalid-dataset.yaml"

        with self.assertRaisesRegex(ValueError, "images without masks"):
            self._generate(
                dataset_path=dataset_root,
                output_path=output_path,
            )

        self.assertFalse(output_path.exists())

    def test_rejects_unsupported_training_files(self) -> None:
        """Unsupported files fail structural validation before output is written."""
        unsupported_path = self.dataset_root / "train" / "Images" / "notes.txt"
        unsupported_path.write_text("not an image", encoding="utf-8")
        output_path = self.root / "unsupported-file.yaml"

        with self.assertRaisesRegex(ValueError, "Unsupported image file extension"):
            self._generate(output_path=output_path)

        self.assertFalse(output_path.exists())

    def test_rejects_duplicate_training_stems(self) -> None:
        """Duplicate image stems fail validation before output is written."""
        duplicate_path = self.dataset_root / "train" / "Images" / "sample-0.jpg"
        self._write_image(duplicate_path)
        output_path = self.root / "duplicate-stem.yaml"

        with self.assertRaisesRegex(ValueError, "Duplicate image filename stem"):
            self._generate(output_path=output_path)

        self.assertFalse(output_path.exists())

    def test_rejects_zero_step_generation_without_writing_output(self) -> None:
        """A batch larger than the train split fails before writing output."""
        output_path = self.root / "zero-step.yaml"

        with self.assertRaisesRegex(ValueError, "at least batch_size"):
            self._generate(
                batch_size=6,
                output_path=output_path,
            )

        self.assertFalse(output_path.exists())

    def test_rejects_non_yaml_output_path(self) -> None:
        """Output paths must use a YAML file extension."""
        output_path = self.root / "config.txt"

        with self.assertRaisesRegex(ValueError, "must end in .yaml or .yml"):
            self._generate(output_path=output_path)

        self.assertFalse(output_path.exists())

    def test_cli_generates_config_accepted_by_lightning_cli(self) -> None:
        """The command-line utility writes a config LightningCLI can parse."""
        output_path = self.root / "cli" / "generated.yaml"
        environment = os.environ.copy()
        environment["MPLCONFIGDIR"] = str(self.root / "matplotlib")
        generation = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.generate_folder_semantic_mlflow_config",
                "--model-size",
                "large",
                "--dataset-path",
                str(self.dataset_root),
                "--num-epochs",
                "4",
                "--batch-size",
                "2",
                "--image-size",
                "640",
                "--num-classes",
                "3",
                "--mlflow-experiment-path",
                "/Workspace/Shared/test",
                "--mlflow-run-name",
                "cli-run",
                "--output",
                str(output_path),
            ],
            cwd=REPOSITORY_ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(generation.returncode, 0, generation.stderr)
        self.assertIn("Training images: 5", generation.stdout)
        self.assertIn("Total steps: 8", generation.stdout)

        parsed = subprocess.run(
            [
                sys.executable,
                "main.py",
                "fit",
                "-c",
                str(output_path),
                "--print_config=skip_null",
            ],
            cwd=REPOSITORY_ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(parsed.returncode, 0, parsed.stderr)
        self.assertIn(
            "class_path: lightning.pytorch.loggers.MLFlowLogger",
            parsed.stdout,
        )
        self.assertIn("experiment_name: /Workspace/Shared/test", parsed.stdout)
        self.assertIn("run_name: cli-run", parsed.stdout)
        self.assertIn("max_epochs: 4", parsed.stdout)

    def _generate(self, **overrides: Any) -> Path:
        parameters = {
            "model_size": "large",
            "dataset_path": self.dataset_root,
            "num_epochs": 4,
            "batch_size": 2,
            "image_size": 640,
            "num_classes": 3,
            "mlflow_experiment_path": "/Workspace/Shared/test",
            "mlflow_run_name": "test-run",
            "output_path": self.root / "generated.yaml",
            "force": False,
        }
        parameters.update(overrides)
        return generate_mlflow_config(**parameters)

    def _write_split(self, split: str, sample_count: int) -> None:
        image_dir = self.dataset_root / split / "Images"
        mask_dir = self.dataset_root / split / "Masks"
        image_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)
        for index in range(sample_count):
            self._write_image(image_dir / f"sample-{index}.png")
            self._write_mask(mask_dir / f"sample-{index}.png")

    @staticmethod
    def _write_image(path: Path) -> None:
        Image.new("RGB", (2, 2), color=(10, 20, 30)).save(path)

    @staticmethod
    def _write_mask(path: Path) -> None:
        mask = Image.new("L", (2, 2))
        mask.putdata([0, 1, 2, 1])
        mask.save(path)

    @staticmethod
    def _load_yaml(path: Path) -> dict[str, Any]:
        with path.open(encoding="utf-8") as config_file:
            return yaml.safe_load(config_file)


if __name__ == "__main__":
    unittest.main()
