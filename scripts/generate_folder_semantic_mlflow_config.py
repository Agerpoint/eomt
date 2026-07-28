# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------

"""Generate folder-semantic MLflow configs with dataset-scaled schedules."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIRECTORY = REPOSITORY_ROOT / "configs" / "dinov3" / "folder" / "semantic"
DEFAULT_CHECKPOINT_DIRECTORY = "/local_disk0/eomt_checkpoints"
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff"})
MASK_SUFFIXES = frozenset({".png"})
ModelSize = Literal["base", "large"]


@dataclass(frozen=True)
class _ModelVariant:
    """Describe the source template and query-processing block count."""

    template_path: Path
    num_blocks: int


@dataclass(frozen=True)
class TrainingSchedule:
    """Global-step schedule derived from the training dataset and run length."""

    steps_per_epoch: int
    total_steps: int
    annealing_starts: tuple[int, ...]
    annealing_ends: tuple[int, ...]
    warmup: tuple[int, int]


@dataclass(frozen=True)
class _PreparedConfig:
    output_path: Path
    config: dict[str, Any]
    num_train_images: int
    schedule: TrainingSchedule


MODEL_VARIANTS = {
    "base": _ModelVariant(
        template_path=CONFIG_DIRECTORY / "eomt_base_1280_mlflow.yaml",
        num_blocks=3,
    ),
    "large": _ModelVariant(
        template_path=CONFIG_DIRECTORY / "eomt_large_1280_mlflow.yaml",
        num_blocks=4,
    ),
}


def compute_schedule(
    num_train_images: int,
    batch_size: int,
    num_epochs: int,
    num_blocks: int,
) -> TrainingSchedule:
    """Calculate global-step annealing and warmup schedules.

    All fractional schedule values use nearest-integer rounding with ties
    rounded upward.

    Args:
        num_train_images: Number of paired samples in the training split.
        batch_size: Per-step batch size for single-device training.
        num_epochs: Number of complete training epochs.
        num_blocks: Number of query-processing blocks to anneal.

    Returns:
        The derived global-step schedule.

    Raises:
        ValueError: If an input is not positive or one batch cannot be formed.
    """
    _require_positive("num_train_images", num_train_images)
    _require_positive("batch_size", batch_size)
    _require_positive("num_epochs", num_epochs)
    _require_positive("num_blocks", num_blocks)

    steps_per_epoch = num_train_images // batch_size
    if steps_per_epoch == 0:
        raise ValueError(
            "num_train_images must be at least batch_size so one training "
            "batch can be formed"
        )

    total_steps = steps_per_epoch * num_epochs
    boundaries = tuple(
        _round_half_up(index * total_steps, num_blocks)
        for index in range(num_blocks + 1)
    )
    warmup = (
        _round_half_up(total_steps * 12, 100),
        _round_half_up(total_steps * 18, 100),
    )

    return TrainingSchedule(
        steps_per_epoch=steps_per_epoch,
        total_steps=total_steps,
        annealing_starts=boundaries[:-1],
        annealing_ends=boundaries[1:],
        warmup=warmup,
    )


def generate_mlflow_config(
    *,
    model_size: ModelSize,
    dataset_path: str | Path,
    num_epochs: int,
    batch_size: int,
    image_size: int,
    num_classes: int,
    mlflow_experiment_path: str,
    mlflow_run_name: str,
    output_path: str | Path,
    force: bool = False,
) -> Path:
    """Generate a folder-semantic MLflow config from a model template.

    Args:
        model_size: DINOv3 model variant, either ``base`` or ``large``.
        dataset_path: Root containing ``train/Images`` and ``train/Masks``.
        num_epochs: Number of epochs written to the trainer config.
        batch_size: Single-device training batch size.
        image_size: Square training image dimension.
        num_classes: Number of semantic classes, including background.
        mlflow_experiment_path: MLflow experiment name or workspace path.
        mlflow_run_name: MLflow run name.
        output_path: Destination ending in ``.yaml`` or ``.yml``.
        force: Whether to replace an existing destination.

    Returns:
        The path to the generated YAML file.

    Raises:
        FileExistsError: If the destination exists and ``force`` is false.
        ValueError: If parameters, dataset samples, or the template are invalid.
    """
    prepared = _prepare_config(
        model_size=model_size,
        dataset_path=dataset_path,
        num_epochs=num_epochs,
        batch_size=batch_size,
        image_size=image_size,
        num_classes=num_classes,
        mlflow_experiment_path=mlflow_experiment_path,
        mlflow_run_name=mlflow_run_name,
        output_path=output_path,
        force=force,
    )
    _write_config(prepared)
    return prepared.output_path


def _prepare_config(
    *,
    model_size: str,
    dataset_path: str | Path,
    num_epochs: int,
    batch_size: int,
    image_size: int,
    num_classes: int,
    mlflow_experiment_path: str,
    mlflow_run_name: str,
    output_path: str | Path,
    force: bool,
) -> _PreparedConfig:
    variant = _model_variant(model_size)
    _validate_parameters(
        num_epochs=num_epochs,
        batch_size=batch_size,
        image_size=image_size,
        num_classes=num_classes,
        mlflow_experiment_path=mlflow_experiment_path,
        mlflow_run_name=mlflow_run_name,
    )

    destination = Path(output_path).expanduser()
    _validate_destination(destination, force)

    dataset_root = Path(dataset_path).expanduser().resolve()
    num_train_images = _count_training_images(dataset_root)
    schedule = compute_schedule(
        num_train_images=num_train_images,
        batch_size=batch_size,
        num_epochs=num_epochs,
        num_blocks=variant.num_blocks,
    )
    config = _load_template(variant.template_path)
    _apply_overrides(
        config=config,
        dataset_root=dataset_root,
        num_epochs=num_epochs,
        batch_size=batch_size,
        image_size=image_size,
        num_classes=num_classes,
        mlflow_experiment_path=mlflow_experiment_path,
        mlflow_run_name=mlflow_run_name,
        schedule=schedule,
    )

    return _PreparedConfig(
        output_path=destination,
        config=config,
        num_train_images=num_train_images,
        schedule=schedule,
    )


def _validate_parameters(
    *,
    num_epochs: int,
    batch_size: int,
    image_size: int,
    num_classes: int,
    mlflow_experiment_path: str,
    mlflow_run_name: str,
) -> None:
    _require_positive("num_epochs", num_epochs)
    _require_positive("batch_size", batch_size)
    _require_positive("image_size", image_size)
    if not 1 <= num_classes <= 255:
        raise ValueError(f"num_classes must be between 1 and 255, got {num_classes}")
    _require_text("mlflow_experiment_path", mlflow_experiment_path)
    _require_text("mlflow_run_name", mlflow_run_name)


def _count_training_images(dataset_root: Path) -> int:
    train_root = dataset_root / "train"
    images = _files_by_stem(
        train_root / "Images",
        IMAGE_SUFFIXES,
        "image",
    )
    masks = _files_by_stem(
        train_root / "Masks",
        MASK_SUFFIXES,
        "mask",
    )

    missing_masks = sorted(images.keys() - masks.keys())
    missing_images = sorted(masks.keys() - images.keys())
    if missing_masks or missing_images:
        details = []
        if missing_masks:
            details.append(f"images without masks: {', '.join(missing_masks)}")
        if missing_images:
            details.append(f"masks without images: {', '.join(missing_images)}")
        raise ValueError(
            f"Image/mask pairing failed for {train_root}: " + "; ".join(details)
        )

    if not images:
        raise ValueError(f"No image/mask pairs found in {train_root}")

    return len(images)


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


def _apply_overrides(
    *,
    config: dict[str, Any],
    dataset_root: Path,
    num_epochs: int,
    batch_size: int,
    image_size: int,
    num_classes: int,
    mlflow_experiment_path: str,
    mlflow_run_name: str,
    schedule: TrainingSchedule,
) -> None:
    trainer = config["trainer"]
    logger = trainer["logger"]["init_args"]
    model = config["model"]["init_args"]
    data = config["data"]["init_args"]

    trainer["max_epochs"] = num_epochs
    logger["experiment_name"] = mlflow_experiment_path
    logger["run_name"] = mlflow_run_name
    _set_checkpoint_directory(trainer)
    model["attn_mask_annealing_start_steps"] = list(schedule.annealing_starts)
    model["attn_mask_annealing_end_steps"] = list(schedule.annealing_ends)
    model["warmup_steps"] = list(schedule.warmup)
    data["path"] = str(dataset_root)
    data["batch_size"] = batch_size
    data["img_size"] = [image_size, image_size]
    data["num_classes"] = num_classes


def _set_checkpoint_directory(trainer: dict[str, Any]) -> None:
    callbacks = trainer.get("callbacks", [])
    for callback in callbacks:
        if (
            callback.get("class_path")
            == "lightning.pytorch.callbacks.ModelCheckpoint"
        ):
            callback["init_args"]["dirpath"] = DEFAULT_CHECKPOINT_DIRECTORY
            return

    raise ValueError("Template is missing the ModelCheckpoint callback")


def _load_template(template_path: Path) -> dict[str, Any]:
    with template_path.open(encoding="utf-8") as template_file:
        config = yaml.safe_load(template_file)

    if not isinstance(config, dict):
        raise ValueError(f"Template must contain a YAML mapping: {template_path}")

    try:
        config["trainer"]["logger"]["init_args"]
        config["model"]["init_args"]
        config["data"]["init_args"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"Template is missing required configuration sections: {template_path}"
        ) from error

    return config


def _write_config(prepared: _PreparedConfig) -> None:
    yaml_text = yaml.safe_dump(
        prepared.config,
        sort_keys=False,
        default_flow_style=False,
    )
    prepared.output_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.output_path.write_text(yaml_text, encoding="utf-8")


def _validate_destination(output_path: Path, force: bool) -> None:
    if output_path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("output_path must end in .yaml or .yml")
    if output_path.exists() and not force:
        raise FileExistsError(
            f"Output already exists: {output_path}; pass force=True to replace it"
        )


def _model_variant(model_size: str) -> _ModelVariant:
    try:
        return MODEL_VARIANTS[model_size]
    except KeyError as error:
        choices = ", ".join(MODEL_VARIANTS)
        raise ValueError(f"model_size must be one of: {choices}") from error


def _require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _require_text(name: str, value: str) -> None:
    if not value.strip():
        raise ValueError(f"{name} must not be empty")


def _round_half_up(numerator: int, denominator: int) -> int:
    return (2 * numerator + denominator) // (2 * denominator)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a DINOv3 folder-semantic MLflow config with schedules "
            "scaled to the training dataset."
        )
    )
    parser.add_argument("--model-size", choices=MODEL_VARIANTS, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--num-epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--image-size", type=int, required=True)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--mlflow-experiment-path", required=True)
    parser.add_argument("--mlflow-run-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace the output file if it already exists.",
    )
    return parser


def _print_summary(prepared: _PreparedConfig) -> None:
    schedule = prepared.schedule
    print(f"Generated config: {prepared.output_path}")
    print(f"Training images: {prepared.num_train_images}")
    print(f"Steps per epoch: {schedule.steps_per_epoch}")
    print(f"Total steps: {schedule.total_steps}")
    print(f"Annealing starts: {list(schedule.annealing_starts)}")
    print(f"Annealing ends: {list(schedule.annealing_ends)}")
    print(f"Warmup steps: {list(schedule.warmup)}")


def main(arguments: list[str] | None = None) -> int:
    """Run the folder-semantic MLflow config generator CLI."""
    parser = _build_parser()
    args = parser.parse_args(arguments)

    try:
        prepared = _prepare_config(
            model_size=args.model_size,
            dataset_path=args.dataset_path,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            image_size=args.image_size,
            num_classes=args.num_classes,
            mlflow_experiment_path=args.mlflow_experiment_path,
            mlflow_run_name=args.mlflow_run_name,
            output_path=args.output,
            force=args.force,
        )
        _write_config(prepared)
    except (FileExistsError, OSError, ValueError, yaml.YAMLError) as error:
        parser.error(str(error))

    _print_summary(prepared)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
