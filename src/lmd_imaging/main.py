import shutil
import time
from pathlib import Path

from .labeling import labels_to_txt, regression_curves_to_txt, YoloLabeler
from .training import prepare_dataset_and_generate_labels, train_dataset

MODEL_TRAINING_DATASET = "micro_dataset"

PREDICTION_DATASET = "pico_dataset"

DO_TRAINING = False


def main() -> None:
    start_time = time.time()
    project_dir = Path.cwd()

    if DO_TRAINING:
        output_paths = prepare_dataset_and_generate_labels(project_dir.joinpath(MODEL_TRAINING_DATASET), project_dir)

    labeling_time = time.time()

    print("--- %s minutes for image labeling ---" % ((time.time() - start_time) / 60))

    if DO_TRAINING:
        model_weight_path = train_dataset(output_paths)
    else:
        model_weight_path = project_dir.joinpath("runs", "segment", "train", "weights", "best.pt")

    yolo_run_dir = project_dir.joinpath("runs", "segment")
    if yolo_run_dir.joinpath("predict").exists():
        shutil.rmtree(yolo_run_dir.joinpath("predict"))
    for name in ("labels", "regression-curve"):
        yolo_run_dir.joinpath("predict", name).mkdir(parents=True, exist_ok=True)

    yolo_labeler = YoloLabeler(model_weight_path, "cuda:0")
    for image_path in project_dir.joinpath(PREDICTION_DATASET).glob("*.png"):
        labels = yolo_labeler.label(image_path, save_predictions=(yolo_run_dir, "predict"))
        if labels is None:
            print(f"'{image_path}' could not be labeled")
            continue
        with yolo_run_dir.joinpath("predict", "labels", image_path.stem + ".txt").open("w", encoding="utf-8") as fd:
            labels_to_txt(labels, fd)
        with yolo_run_dir.joinpath("predict", "regression-curve", image_path.stem + ".txt").open(
            "w", encoding="utf-8"
        ) as fd:
            regression_curves_to_txt(labels, fd)

    print("--- %s minutes for YOLO execution ---" % ((time.time() - labeling_time) / 60))
