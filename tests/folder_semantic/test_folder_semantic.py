# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from PIL import Image

from datasets.folder_semantic import FolderSemantic, FolderSemanticDataset

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LARGE_WANDB_CONFIG_PATH = (
    REPOSITORY_ROOT
    / "configs"
    / "dinov3"
    / "folder"
    / "semantic"
    / "eomt_large_1280_wandb.yaml"
)
BASE_WANDB_CONFIG_PATH = (
    REPOSITORY_ROOT
    / "configs"
    / "dinov3"
    / "folder"
    / "semantic"
    / "eomt_base_1280_wandb.yaml"
)
LARGE_MLFLOW_CONFIG_PATH = (
    REPOSITORY_ROOT
    / "configs"
    / "dinov3"
    / "folder"
    / "semantic"
    / "eomt_large_1280_mlflow.yaml"
)
BASE_MLFLOW_CONFIG_PATH = (
    REPOSITORY_ROOT
    / "configs"
    / "dinov3"
    / "folder"
    / "semantic"
    / "eomt_base_1280_mlflow.yaml"
)


class FolderSemanticDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_preserves_zero_based_labels_and_ignores_255(self) -> None:
        image_dir, mask_dir = self._create_split("train")
        self._write_pair(
            image_dir,
            mask_dir,
            "sample",
            mask_values=[0, 1, 255, 1],
        )

        dataset = self._create_dataset(image_dir, mask_dir)
        _, target = dataset[0]

        self.assertEqual(target["labels"].tolist(), [0, 1])
        self.assertEqual(tuple(target["masks"].shape), (2, 2, 2))
        self.assertTrue(target["masks"][0, 0, 0])
        self.assertTrue(target["masks"][1, 0, 1])
        self.assertFalse(target["masks"][:, 1, 0].any())
        self.assertEqual(target["is_crowd"].tolist(), [False, False])

    def test_keeps_background_only_mask_by_default(self) -> None:
        image_dir, mask_dir = self._create_split("train")
        self._write_pair(
            image_dir,
            mask_dir,
            "background",
            mask_values=[0, 0, 0, 0],
        )

        dataset = self._create_dataset(image_dir, mask_dir)
        _, target = dataset[0]

        self.assertEqual(len(dataset), 1)
        self.assertEqual(target["labels"].tolist(), [0])
        self.assertTrue(target["masks"].all())

    def test_filters_background_only_mask_when_requested(self) -> None:
        image_dir, mask_dir = self._create_split("train")
        self._write_pair(
            image_dir,
            mask_dir,
            "background",
            mask_values=[0, 0, 0, 0],
        )
        self._write_pair(
            image_dir,
            mask_dir,
            "foreground",
            mask_values=[0, 1, 0, 1],
        )

        dataset = self._create_dataset(
            image_dir,
            mask_dir,
            check_empty_targets=True,
        )

        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset.samples[0][0].stem, "foreground")

    def test_rejects_split_when_filter_removes_every_sample(self) -> None:
        image_dir, mask_dir = self._create_split("train")
        self._write_pair(
            image_dir,
            mask_dir,
            "background",
            mask_values=[0, 0, 0, 0],
        )

        with self.assertRaisesRegex(
            ValueError,
            "No usable samples.*after filtering background-only masks",
        ):
            self._create_dataset(
                image_dir,
                mask_dir,
                check_empty_targets=True,
            )

    def test_rejects_all_ignore_mask(self) -> None:
        image_dir, mask_dir = self._create_split("train")
        self._write_pair(
            image_dir,
            mask_dir,
            "ignored",
            mask_values=[255, 255, 255, 255],
        )

        with self.assertRaisesRegex(ValueError, "only ignored pixels"):
            self._create_dataset(image_dir, mask_dir)

    def test_rejects_out_of_range_labels(self) -> None:
        image_dir, mask_dir = self._create_split("train")
        self._write_pair(
            image_dir,
            mask_dir,
            "invalid",
            mask_values=[0, 1, 2, 255],
        )

        with self.assertRaisesRegex(ValueError, "outside 0..1"):
            self._create_dataset(image_dir, mask_dir)

    def test_supports_all_documented_image_extensions(self) -> None:
        image_dir, mask_dir = self._create_split("train")
        for index, extension in enumerate(
            [".jpg", ".jpeg", ".png", ".tif", ".tiff"]
        ):
            self._write_pair(
                image_dir,
                mask_dir,
                f"sample_{index}",
                image_suffix=extension,
            )

        dataset = self._create_dataset(image_dir, mask_dir)

        self.assertEqual(len(dataset), 5)

    def test_rejects_image_without_mask(self) -> None:
        image_dir, mask_dir = self._create_split("train")
        self._write_image(image_dir / "unmatched.jpg")

        with self.assertRaisesRegex(ValueError, "images without masks: unmatched"):
            self._create_dataset(image_dir, mask_dir)

    def test_rejects_mask_without_image(self) -> None:
        image_dir, mask_dir = self._create_split("train")
        self._write_mask(mask_dir / "unmatched.png")

        with self.assertRaisesRegex(ValueError, "masks without images: unmatched"):
            self._create_dataset(image_dir, mask_dir)

    def test_pairs_stems_case_sensitively(self) -> None:
        image_dir, mask_dir = self._create_split("train")
        self._write_image(image_dir / "Sample.jpg")
        self._write_mask(mask_dir / "sample.png")

        with self.assertRaisesRegex(ValueError, "Image/mask pairing failed"):
            self._create_dataset(image_dir, mask_dir)

    def test_rejects_duplicate_image_stems(self) -> None:
        image_dir, mask_dir = self._create_split("train")
        self._write_image(image_dir / "duplicate.jpg")
        self._write_image(image_dir / "duplicate.png")
        self._write_mask(mask_dir / "duplicate.png")

        with self.assertRaisesRegex(ValueError, "Duplicate image filename stem"):
            self._create_dataset(image_dir, mask_dir)

    def test_rejects_missing_required_directory(self) -> None:
        image_dir = self.root / "train" / "Images"
        mask_dir = self.root / "train" / "Masks"

        with self.assertRaisesRegex(ValueError, "directory does not exist"):
            self._create_dataset(image_dir, mask_dir)

    def test_rejects_nested_directories(self) -> None:
        image_dir, mask_dir = self._create_split("train")
        (image_dir / "nested").mkdir()

        with self.assertRaisesRegex(ValueError, "Subdirectories are not supported"):
            self._create_dataset(image_dir, mask_dir)

    def test_rejects_unsupported_file_extension(self) -> None:
        image_dir, mask_dir = self._create_split("train")
        self._write_image(image_dir / "sample.bmp")
        self._write_mask(mask_dir / "sample.png")

        with self.assertRaisesRegex(ValueError, "Unsupported image file extension"):
            self._create_dataset(image_dir, mask_dir)

    def test_rejects_non_rgb_image(self) -> None:
        image_dir, mask_dir = self._create_split("train")
        self._write_image(image_dir / "grayscale.png", mode="L")
        self._write_mask(mask_dir / "grayscale.png")

        with self.assertRaisesRegex(ValueError, "Image must be RGB"):
            self._create_dataset(image_dir, mask_dir)

    def test_rejects_multichannel_mask(self) -> None:
        image_dir, mask_dir = self._create_split("train")
        self._write_image(image_dir / "sample.png")
        self._write_mask(mask_dir / "sample.png", mode="RGB")

        with self.assertRaisesRegex(
            ValueError,
            "Mask must be a single-channel integer image",
        ):
            self._create_dataset(image_dir, mask_dir)

    def test_rejects_dimension_mismatch(self) -> None:
        image_dir, mask_dir = self._create_split("train")
        self._write_image(image_dir / "sample.png", size=(2, 2))
        self._write_mask(mask_dir / "sample.png", size=(3, 2))

        with self.assertRaisesRegex(ValueError, "dimensions differ"):
            self._create_dataset(image_dir, mask_dir)

    def test_rejects_num_classes_that_conflicts_with_ignore_label(self) -> None:
        image_dir, mask_dir = self._create_split("train")

        with self.assertRaisesRegex(
            ValueError,
            "num_classes must be between 1 and 255",
        ):
            self._create_dataset(image_dir, mask_dir, num_classes=256)

    def _create_split(self, split: str) -> tuple[Path, Path]:
        image_dir = self.root / split / "Images"
        mask_dir = self.root / split / "Masks"
        image_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)
        return image_dir, mask_dir

    def _write_pair(
        self,
        image_dir: Path,
        mask_dir: Path,
        stem: str,
        image_suffix: str = ".png",
        mask_values: list[int] | None = None,
    ) -> None:
        self._write_image(image_dir / f"{stem}{image_suffix}")
        self._write_mask(
            mask_dir / f"{stem}.png",
            values=mask_values,
        )

    @staticmethod
    def _write_image(
        path: Path,
        mode: str = "RGB",
        size: tuple[int, int] = (2, 2),
    ) -> None:
        color = (10, 20, 30) if mode == "RGB" else 10
        Image.new(mode, size, color=color).save(path)

    @staticmethod
    def _write_mask(
        path: Path,
        mode: str = "L",
        size: tuple[int, int] = (2, 2),
        values: list[int] | None = None,
    ) -> None:
        mask = Image.new(mode, size)
        if values is None:
            values = [0, 1, 0, 1]
        if mode == "RGB":
            mask.putdata([(value, value, value) for value in values])
        else:
            mask.putdata(values)
        mask.save(path)

    @staticmethod
    def _create_dataset(
        image_dir: Path,
        mask_dir: Path,
        num_classes: int = 2,
        check_empty_targets: bool = False,
    ) -> FolderSemanticDataset:
        return FolderSemanticDataset(
            image_dir=image_dir,
            mask_dir=mask_dir,
            num_classes=num_classes,
            check_empty_targets=check_empty_targets,
        )


class FolderSemanticDataModuleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        for split in ("train", "val"):
            image_dir = self.root / split / "Images"
            mask_dir = self.root / split / "Masks"
            image_dir.mkdir(parents=True)
            mask_dir.mkdir(parents=True)
            Image.new("RGB", (2, 2), color=(10, 20, 30)).save(
                image_dir / "sample.png"
            )
            mask = Image.new("L", (2, 2))
            mask.putdata([0, 1, 0, 1])
            mask.save(mask_dir / "sample.png")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_dataloaders_produce_eomt_batch_contract(self) -> None:
        data_module = FolderSemantic(
            path=self.root,
            num_workers=0,
            batch_size=1,
            img_size=(2, 2),
            color_jitter_enabled=False,
            scale_range=(1.0, 1.0),
        )
        data_module.setup("fit")

        train_images, train_targets = next(iter(data_module.train_dataloader()))
        val_images, val_targets = next(iter(data_module.val_dataloader()))

        self.assertEqual(tuple(train_images.shape), (1, 3, 2, 2))
        self.assertEqual(len(train_targets), 1)
        self.assertEqual(
            set(train_targets[0]),
            {"masks", "labels", "is_crowd"},
        )
        self.assertEqual(len(val_images), 1)
        self.assertEqual(len(val_targets), 1)
        self.assertIsInstance(val_images[0], torch.Tensor)


class FolderSemanticConfigTests(unittest.TestCase):
    def test_base_config_parses_with_expected_values(self) -> None:
        output = self._print_config(config_path=BASE_WANDB_CONFIG_PATH)

        self.assertIn(
            "backbone_name: facebook/dinov3-vitb16-pretrain-lvd1689m",
            output,
        )
        self.assertIn("        num_q: 100", output)
        self.assertIn("        num_blocks: 3", output)
        self.assertIn("    llrd_l2_enabled: false", output)
        self.assertIn("    delta_weights: false", output)
        self.assertIn(
            "class_path: datasets.folder_semantic.FolderSemantic",
            output,
        )
        self.assertIn("    - 1280\n    - 1280", output)
        self.assertIn("    num_classes: 2", output)
        self.assertIn("      monitor: metrics/val_iou_all", output)

    def test_mlflow_configs_parse_with_expected_logger_values(self) -> None:
        configs = [
            (
                BASE_MLFLOW_CONFIG_PATH,
                "folder_semantic_eomt_base_1280_dinov3",
            ),
            (
                LARGE_MLFLOW_CONFIG_PATH,
                "folder_semantic_eomt_large_1280_dinov3",
            ),
        ]

        for config_path, run_name in configs:
            with self.subTest(config_path=config_path):
                output = self._print_config(config_path=config_path)

                self.assertIn(
                    "class_path: lightning.pytorch.loggers.MLFlowLogger",
                    output,
                )
                self.assertIn(
                    "experiment_name: "
                    "/Workspace/Shared/mlflow_experiments/eomt/experiment",
                    output,
                )
                self.assertIn(f"run_name: {run_name}", output)
                self.assertIn("    delta_weights: false", output)
                self.assertIn(
                    "class_path: datasets.folder_semantic.FolderSemantic",
                    output,
                )

    def test_default_config_parses_with_expected_values(self) -> None:
        output = self._print_config()

        self.assertIn(
            "class_path: datasets.folder_semantic.FolderSemantic",
            output,
        )
        self.assertIn("    - 1280\n    - 1280", output)
        self.assertIn("    num_classes: 2", output)
        self.assertIn("    check_empty_targets: false", output)
        self.assertIn("        num_q: 100", output)
        self.assertIn("    delta_weights: false", output)
        self.assertIn("      monitor: metrics/val_iou_all", output)
        self.assertIn(
            "      filename: best-{epoch}-{metrics/val_iou_all:.4f}",
            output,
        )
        self.assertIn("      auto_insert_metric_name: false", output)

    def test_cli_values_override_folder_defaults(self) -> None:
        output = self._print_config(
            "--data.img_size",
            "[1024, 1024]",
            "--data.num_classes",
            "4",
            "--data.check_empty_targets",
            "true",
        )

        self.assertIn("    - 1024\n    - 1024", output)
        self.assertIn("    num_classes: 4", output)
        self.assertIn("    check_empty_targets: true", output)

    def _print_config(
        self,
        *arguments: str,
        config_path: Path = LARGE_WANDB_CONFIG_PATH,
    ) -> str:
        with TemporaryDirectory() as matplotlib_config_directory:
            environment = os.environ.copy()
            environment["MPLCONFIGDIR"] = matplotlib_config_directory
            result = subprocess.run(
                [
                    sys.executable,
                    "main.py",
                    "fit",
                    "-c",
                    str(config_path),
                    "--data.path",
                    "/tmp/folder-semantic-test-data",
                    *arguments,
                    "--print_config=skip_null",
                ],
                cwd=REPOSITORY_ROOT,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout


if __name__ == "__main__":
    unittest.main()
