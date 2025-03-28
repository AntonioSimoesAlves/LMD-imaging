import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import sklearn
from ultralytics import YOLO

from .labeling import ManualLabeler, labels_to_txt


def define_dataset(img_path, img_type) -> list[Path]:
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
        super().__init__(
            base=output_dir,
            image_train=output_dir.joinpath("data", "images", "train"),
            image_validate=output_dir.joinpath("data", "images", "val"),
            label_train=output_dir.joinpath("data", "labels", "train"),
            label_validate=output_dir.joinpath("data", "labels", "val"),
        )

    def create(self):
        for path in (self.image_train, self.image_validate, self.label_train, self.label_validate):
            if path.exists():
                shutil.rmtree(path)
            path.mkdir(exist_ok=True, parents=True)


def prepare_dataset_and_generate_labels(dataset_dir: Path, output_dir: Path, test_size: float = 0.1) -> OutputPaths:
    img_list = define_dataset(dataset_dir, "png")  # TODO: png
    img_train, img_test = sklearn.model_selection.train_test_split(img_list, test_size=test_size)

    output_paths = OutputPaths(output_dir)
    output_paths.create()

    labeler = ManualLabeler()
    for input_images, output_image_set, output_label_set in [
        (img_train, output_paths.image_train, output_paths.label_train),
        (img_test, output_paths.image_validate, output_paths.label_validate),
    ]:
        for input_image in input_images:
            label = labeler.label(input_image)
            shutil.copy(input_image, output_image_set.joinpath(input_image.name))
            with output_label_set.joinpath(input_image.stem + ".txt").open("w", encoding="utf-8") as fd:
                labels_to_txt(label, fd)

    return output_paths


def train_dataset(dataset_paths: OutputPaths, yolo_model: str = "yolo11n-seg.yaml", **train_kwargs) -> Path:
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
        train_kwargs["epochs"] = 40

    model = YOLO(yolo_model)
    model.train(**train_kwargs)

    return runs_dir.joinpath("segment", "train", "weights", "best.pt")
