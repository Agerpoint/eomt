# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import tv_tensors

from datasets.lightning_data_module import LightningDataModule
from datasets.transforms import Transforms

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff"})
MASK_SUFFIXES = frozenset({".png"})
IGNORE_LABEL = 255


class FolderSemanticDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        image_dir: Path,
        mask_dir: Path,
        num_classes: int,
        check_empty_targets: bool,
        transforms: Callable | None = None,
    ) -> None:
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.num_classes = num_classes
        self.transforms = transforms

        if not 1 <= num_classes <= IGNORE_LABEL:
            raise ValueError(
                f"num_classes must be between 1 and {IGNORE_LABEL}, got "
                f"{num_classes}"
            )

        samples = self._pair_samples()
        self.samples = [
            sample
            for sample in samples
            if self._validate_sample(sample, check_empty_targets)
        ]

        if not self.samples:
            detail = (
                " after filtering background-only masks"
                if check_empty_targets
                else ""
            )
            raise ValueError(f"No usable samples found in {image_dir}{detail}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[tv_tensors.Image, dict[str, torch.Tensor]]:
        image_path, mask_path = self.samples[index]

        with Image.open(image_path) as image_file:
            image = tv_tensors.Image(image_file.convert("RGB"))

        with Image.open(mask_path) as mask_file:
            target_mask = tv_tensors.Mask(mask_file, dtype=torch.long)

        masks = []
        labels = []
        for label in target_mask[0].unique():
            label_id = label.item()
            if label_id == IGNORE_LABEL:
                continue

            masks.append(target_mask[0] == label)
            labels.append(label_id)

        if not masks:
            raise ValueError(f"Mask contains only ignored pixels: {mask_path}")

        target = {
            "masks": tv_tensors.Mask(torch.stack(masks)),
            "labels": torch.tensor(labels, dtype=torch.long),
            "is_crowd": torch.zeros(len(labels), dtype=torch.bool),
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def _pair_samples(self) -> list[tuple[Path, Path]]:
        images = self._files_by_stem(self.image_dir, IMAGE_SUFFIXES, "image")
        masks = self._files_by_stem(self.mask_dir, MASK_SUFFIXES, "mask")

        missing_masks = sorted(images.keys() - masks.keys())
        missing_images = sorted(masks.keys() - images.keys())
        if missing_masks or missing_images:
            details = []
            if missing_masks:
                details.append(f"images without masks: {', '.join(missing_masks)}")
            if missing_images:
                details.append(f"masks without images: {', '.join(missing_images)}")
            raise ValueError(
                f"Image/mask pairing failed for {self.image_dir.parent}: "
                + "; ".join(details)
            )

        if not images:
            raise ValueError(f"No image/mask pairs found in {self.image_dir.parent}")

        return [(images[stem], masks[stem]) for stem in sorted(images)]

    @staticmethod
    def _files_by_stem(
        directory: Path,
        allowed_suffixes: frozenset[str],
        kind: str,
    ) -> dict[str, Path]:
        if not directory.is_dir():
            raise ValueError(f"Required {kind} directory does not exist: {directory}")

        files_by_stem = {}
        for path in sorted(directory.iterdir()):
            if path.is_dir():
                raise ValueError(
                    f"Subdirectories are not supported in {kind} directory: {path}"
                )
            if not path.is_file():
                continue
            if path.suffix.lower() not in allowed_suffixes:
                raise ValueError(f"Unsupported {kind} file extension: {path}")
            if path.stem in files_by_stem:
                raise ValueError(
                    f"Duplicate {kind} filename stem '{path.stem}' in {directory}"
                )

            files_by_stem[path.stem] = path

        return files_by_stem

    def _validate_sample(
        self,
        sample: tuple[Path, Path],
        check_empty_targets: bool,
    ) -> bool:
        image_path, mask_path = sample

        with Image.open(image_path) as image:
            if image.mode != "RGB":
                raise ValueError(
                    f"Image must be RGB, got mode {image.mode!r}: {image_path}"
                )
            image_size = image.size

        with Image.open(mask_path) as mask:
            if not self._is_integer_single_channel(mask):
                raise ValueError(
                    f"Mask must be a single-channel integer image, got mode "
                    f"{mask.mode!r}: {mask_path}"
                )
            if mask.size != image_size:
                raise ValueError(
                    f"Image and mask dimensions differ for stem '{image_path.stem}': "
                    f"{image_size} != {mask.size}"
                )

            values = tv_tensors.Mask(mask, dtype=torch.long).unique()

        trainable_values = values[values != IGNORE_LABEL]
        if not trainable_values.numel():
            raise ValueError(f"Mask contains only ignored pixels: {mask_path}")

        invalid_values = trainable_values[
            (trainable_values < 0) | (trainable_values >= self.num_classes)
        ]
        if invalid_values.numel():
            invalid = ", ".join(str(value.item()) for value in invalid_values)
            raise ValueError(
                f"Mask contains labels outside 0..{self.num_classes - 1}: "
                f"{invalid} in {mask_path}"
            )

        has_foreground = bool((trainable_values > 0).any())
        return not check_empty_targets or has_foreground

    @staticmethod
    def _is_integer_single_channel(mask: Image.Image) -> bool:
        integer_mode = mask.mode in {"1", "L", "P", "I"} or mask.mode.startswith(
            "I;16"
        )
        return len(mask.getbands()) == 1 and integer_mode


class FolderSemantic(LightningDataModule):
    def __init__(
        self,
        path: str | Path,
        num_workers: int = 4,
        batch_size: int = 16,
        img_size: tuple[int, int] = (1280, 1280),
        num_classes: int = 2,
        color_jitter_enabled: bool = True,
        scale_range: tuple[float, float] = (0.5, 2.0),
        check_empty_targets: bool = False,
    ) -> None:
        super().__init__(
            path=path,
            batch_size=batch_size,
            num_workers=num_workers,
            num_classes=num_classes,
            img_size=img_size,
            check_empty_targets=check_empty_targets,
        )
        self.save_hyperparameters(ignore=["_class_path"])

        self.transforms = Transforms(
            img_size=img_size,
            color_jitter_enabled=color_jitter_enabled,
            scale_range=scale_range,
        )

    def setup(self, stage: str | None = None) -> LightningDataModule:
        dataset_root = Path(self.path)
        dataset_kwargs = {
            "num_classes": self.num_classes,
            "check_empty_targets": self.check_empty_targets,
        }

        self.train_dataset = FolderSemanticDataset(
            image_dir=dataset_root / "train" / "Images",
            mask_dir=dataset_root / "train" / "Masks",
            transforms=self.transforms,
            **dataset_kwargs,
        )
        self.val_dataset = FolderSemanticDataset(
            image_dir=dataset_root / "val" / "Images",
            mask_dir=dataset_root / "val" / "Masks",
            **dataset_kwargs,
        )

        return self

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            drop_last=True,
            collate_fn=self.train_collate,
            **self.dataloader_kwargs,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            collate_fn=self.eval_collate,
            **self.dataloader_kwargs,
        )
