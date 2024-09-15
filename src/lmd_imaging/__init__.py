from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import sklearn.model_selection
from ultralytics import YOLO
import shutil
import torch

MAX_T = 2500
MELTING_T = 1350


def filter_contours(raw_contours, min_contour_size):

    filtered_contours = []

    for contour in raw_contours:
        if cv2.contourArea(contour) > min_contour_size:
            filtered_contours.append(contour)

    return filtered_contours


def save_contours_to_txt(contours, path: Path) -> None:

    with path.open("w") as f:
        for contour in contours:
            contour_str = " ".join([f"{point[0][0]} {point[0][1]}" for point in contour])
            if contour_str.count(" ") < 8:
                continue
            f.write("0 ")
            f.write(f"{contour_str}\n")


def img_contours(img_path: Path):
    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    img_height = img.shape[0]
    img_width = img.shape[1]

    img = img / 10

    img = np.float32(img)
    img = img / 2300
    img = np.clip(img, 0, 1)

    img_8bit = img * 255
    img_8bit = np.uint8(img_8bit)

    lower_threshold = MELTING_T / MAX_T * 255
    upper_threshold = 255

    img_8bit_blur = cv2.medianBlur(img_8bit, 5)

    mask = cv2.inRange(img_8bit_blur, lower_threshold, upper_threshold)

    contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)

    # plot the dataset with contours (recommended for very small amounts of dataset)

    # f, axarr = plt.subplots(1, 2)
    # axarr[0].imshow(img_8bit)
    # axarr[1].imshow(mask)
    # plt.show()

    normalized_contours = []
    for contour in contours:
        normalized_contours.append(np.divide(contour, [img_width, img_height]))

    return normalized_contours


def main() -> None:

    project_dir = Path.cwd()

    img_dir = project_dir.joinpath("dataset")
    img_list = sorted(img_dir.glob("*.png"))

    img_train, img_test = sklearn.model_selection.train_test_split(img_list, test_size=0.20)

    img_train_path = project_dir.joinpath("data", "images", "train")
    if img_train_path.exists():
        shutil.rmtree(img_train_path)

    img_train_path.mkdir(exist_ok=True, parents=True)

    for img in img_train:
        shutil.copy(img, img_train_path.joinpath(img.name))

    img_test_path = project_dir.joinpath("data", "images", "val")
    if img_test_path.exists():
        shutil.rmtree(img_test_path)

    img_test_path.mkdir(exist_ok=True, parents=True)

    for img in img_test:
        shutil.copy(img, img_test_path.joinpath(img.name))

    training_labels_dir = project_dir.joinpath("data", "labels", "train")
    if training_labels_dir.exists():
        shutil.rmtree(training_labels_dir)

    training_labels_dir.mkdir(exist_ok=True)

    for img_path in img_train:
        contours = img_contours(img_path)
        save_contours_to_txt(contours, training_labels_dir.joinpath(img_path.name).with_suffix(".txt"))

    testing_labels_dir = project_dir.joinpath("data", "labels", "val")
    if testing_labels_dir.exists():
        shutil.rmtree(testing_labels_dir)

    testing_labels_dir.mkdir(exist_ok=True)

    for img_path in img_test:
        contours = img_contours(img_path)
        save_contours_to_txt(contours, testing_labels_dir.joinpath(img_path.name).with_suffix(".txt"))

    yaml = {
        "path": str(project_dir.joinpath("data").resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["Outer"],
    }
    yaml_path = project_dir.joinpath("data.yaml")
    with yaml_path.open("w") as f:
        json.dump(yaml, f, indent=4)

    runs_dir = project_dir.joinpath("runs")
    if runs_dir.exists():
        shutil.rmtree(runs_dir)

    model = YOLO("yolov8n-seg.yaml")  # build a new model from scratch

    model.train(data="data.yaml", epochs=4, imgsz=640)  # train the model
