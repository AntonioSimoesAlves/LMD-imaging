import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from .labeling import ManualLabeler, labels_to_txt


def define_dataset(img_path: Path, img_type: str) -> list[Path]:
    project_dir = Path.cwd()

    img_dir = project_dir.joinpath(img_path)
    img_list = sorted(img_dir.glob("*." + img_type))

    return img_list


@dataclass
class OutputPaths:
    base: Path
    image_train: Path
    image_validate: Path
    label_train: Path
    label_validate: Path

    def __init__(self, output_dir: Path) -> None:
        super().__init__()
        self.base = output_dir
        self.image_train = output_dir.joinpath("data", "images", "train")
        self.image_validate = output_dir.joinpath("data", "images", "val")
        self.label_train = output_dir.joinpath("data", "labels", "train")
        self.label_validate = output_dir.joinpath("data", "labels", "val")

    def create(self, remove_existing: bool = False) -> None:
        for path in (self.image_train, self.label_train):
            if remove_existing:
                shutil.rmtree(path)
            elif path.exists():
                raise RuntimeError(f"{path} already exists, pass -f to remove it anyway")

            path.mkdir(exist_ok=True, parents=True)


def prepare_dataset_and_generate_labels(
    dataset_dir: Path,
    output_dir: Path,
    remove_existing: bool = False,
) -> OutputPaths:
    input_images = define_dataset(dataset_dir, "png")

    output_paths = OutputPaths(output_dir)
    output_paths.create(remove_existing)

    labeler = ManualLabeler()
    for image, labels in zip(iter(input_images), labeler.batch_label(input_images)):
        if labels is None:
            continue

        shutil.copy(image, output_paths.image_train.joinpath(image.name))
        with output_paths.label_train.joinpath(image.stem + ".txt").open("w", encoding="utf-8") as fd:
            labels_to_txt(labels, fd)

    return output_paths


def train_dataset(dataset_paths: OutputPaths, yolo_model: str = "yolo11n-seg.yaml", **train_kwargs: Any) -> Path:
    yaml = {
        "path": str(dataset_paths.base.joinpath("data").resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 2,
        "names": ["Liquid Melt Pool", "Mushy Melt Pool"],
    }
    yaml_path = dataset_paths.base.joinpath("data.yaml")
    with yaml_path.open("w", encoding="utf-8") as fd:
        json.dump(yaml, fd, indent=4)

    runs_dir = dataset_paths.base.joinpath("runs")
    if runs_dir.exists():
        shutil.rmtree(runs_dir)

    train_kwargs["data"] = yaml_path
    if "model" in train_kwargs:
        del train_kwargs["model"]
    if "imgsz" not in train_kwargs:
        train_kwargs["imgsz"] = 640
    if "epochs" not in train_kwargs:
        train_kwargs["epochs"] = 300

    model = YOLO(yolo_model)
    model.train(**train_kwargs)

    return runs_dir.joinpath("segment", "train", "weights", "best.pt")
