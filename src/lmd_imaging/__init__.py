from pathlib import Path
from ultralytics import YOLO
from PIL import Image
from typing import NamedTuple

import cv2
import matplotlib.pyplot as plt
import json
import sklearn.model_selection
import math
import shutil
import time
import itertools
import numpy as np


class SegmentLoopException(Exception):
    pass


class EndOfSegmentation(Exception):
    pass


class NoInterceptionException(Exception):
    pass


def calculate_moving_average(image, average_sample_size) -> list[float]:
    moving_average = []

    for i in range(len(image)):
        previous_values = []
        next_values = []
        current_value = image[i]
        for j in range(1, average_sample_size // 2 + 1):
            if (i - j) < 0:
                previous_values.append(image[i])
            else:
                previous_values.append(image[i - j])
            if (i + j) > len(image) - 1:
                next_values.append(image[i])
            else:
                next_values.append(image[i + j])
        moving_average.append(round((current_value + sum(previous_values) + sum(next_values)) / average_sample_size, 3))

    return moving_average


def define_dataset(img_path, img_type) -> list[Path]:
    project_dir = Path.cwd()

    img_dir = project_dir.joinpath(img_path)
    img_list = sorted(img_dir.glob("*." + img_type))

    return img_list


def image_to_numerical_value(img_path, y_coord) -> list:
    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    img_width = img.shape[1]

    im = Image.open(img_path)
    pix = im.load()

    image = []
    for x in range(img_width):
        image.append(pix[x, y_coord] / 10)

    return image


def calculate_deltas(img_width, moving_average) -> list:
    pixels_deltas = []

    for x in range(img_width):
        if x > 0:
            past_value = moving_average[x - 1]
            current_value = moving_average[x]
            delta = round(current_value - past_value, 3)
            pixels_deltas.append(delta)
        else:
            pixels_deltas.append(0)

    return pixels_deltas


def calculate_deltas_sum(pixels_deltas) -> list:
    deltas_sum = []

    for i in range(len(pixels_deltas)):
        if i < len(pixels_deltas) - 1:
            deltas_sum.append(
                10 * pixels_deltas[i] + pixels_deltas[i - 1] + pixels_deltas[i - 2] - pixels_deltas[i + 1]
            )
        else:
            deltas_sum.append(6 * pixels_deltas[i] + pixels_deltas[i - 1] + pixels_deltas[i - 2])

    return deltas_sum


def find_crossing_point(
    deltas_sum_moving_average, x_center, y_center, delta_threshold, y_coord, temperature_moving_average
) -> tuple[list[int], list[int]]:

    melt_pool_points_liquid_coordinate = []
    melt_pool_points_mushy_coordinate = []
    found_first_mushy_point = False

    for i in range(len(deltas_sum_moving_average)):
        if deltas_sum_moving_average[i] < delta_threshold:
            if len(deltas_sum_moving_average) >= i + 7:
                if (x_center[0] < i < x_center[1] and y_center[0] < y_coord < y_center[1]) or (
                    deltas_sum_moving_average[i - 5] > abs(delta_threshold)
                    or deltas_sum_moving_average[i - 4] > abs(delta_threshold)
                    or deltas_sum_moving_average[i - 3] > abs(delta_threshold)
                    or deltas_sum_moving_average[i - 2] > abs(delta_threshold)
                    or deltas_sum_moving_average[i - 1] > abs(delta_threshold)
                    # or deltas_sum_moving_average[i + 1] > abs(delta_threshold)
                    # or deltas_sum_moving_average[i + 2] > abs(delta_threshold)
                    # or deltas_sum_moving_average[i + 3] > abs(delta_threshold)
                    # or deltas_sum_moving_average[i + 4] > abs(delta_threshold)
                    # or deltas_sum_moving_average[i + 5] > abs(delta_threshold)
                    # or deltas_sum_moving_average[i + 6] > abs(delta_threshold)
                    # or deltas_sum_moving_average[i + 7] > abs(delta_threshold)
                ):
                    continue
            if temperature_moving_average[i] > MELTING_T:
                if len(melt_pool_points_liquid_coordinate) > 0:
                    if i - melt_pool_points_liquid_coordinate[-1] > 20:
                        melt_pool_points_liquid_coordinate.append(i)
                else:
                    melt_pool_points_liquid_coordinate.append(i)

    for i in range(len(deltas_sum_moving_average)):
        if temperature_moving_average[i] > MUSHY_MINIMUM and not found_first_mushy_point:
            if len(melt_pool_points_liquid_coordinate) == 0:
                continue
            found_first_mushy_point = True
            melt_pool_points_mushy_coordinate.append(i)
        if (
            temperature_moving_average[i] > MUSHY_MINIMUM
            and melt_pool_points_liquid_coordinate[0] == i + 1
            and found_first_mushy_point
            and len(melt_pool_points_liquid_coordinate) > 0
        ):
            if len(melt_pool_points_liquid_coordinate) == 1:
                if 1200 < temperature_moving_average[melt_pool_points_liquid_coordinate[-1] + 10] < 1400:
                    melt_pool_points_mushy_coordinate.append(i)
                else:
                    if len(melt_pool_points_mushy_coordinate) == 1:
                        melt_pool_points_mushy_coordinate.pop()
                        break
            elif len(melt_pool_points_liquid_coordinate) >= 2:
                melt_pool_points_mushy_coordinate.append(i)

    return melt_pool_points_liquid_coordinate, melt_pool_points_mushy_coordinate


class Point(NamedTuple):
    x: float
    y: float


SEGMENT_Y_MIN = 175
SEGMENT_Y_MAX = 270


def prepare_image(img_path: Path, output_path: Path) -> None:
    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    img_height = img.shape[0]
    img_width = img.shape[1]

    liquid_intersections = {}
    mushy_intersections = {}

    for y_label in range(SEGMENT_Y_MIN, SEGMENT_Y_MAX):
        image = image_to_numerical_value(img_path, y_label)
        moving_average = calculate_moving_average(image, 11)
        pixels_deltas = calculate_deltas(img_width, moving_average)
        deltas_sum = calculate_deltas_sum(pixels_deltas)
        deltas_sum_moving_average = calculate_moving_average(deltas_sum, 9)
        liquid_intersection_coordinate, mushy_intersection_coordinate = find_crossing_point(
            deltas_sum_moving_average, x_hotspot, y_hotspot, delta_threshold, y_label, moving_average
        )

        intersection = list(filter(lambda n: n != 0, liquid_intersection_coordinate))
        if intersection:
            liquid_intersections[y_label] = intersection

        intersection = list(filter(lambda n: n != 0, mushy_intersection_coordinate))
        if intersection:
            mushy_intersections[y_label] = intersection

    segment_image(img_path, liquid_intersections, mushy_intersections, output_path)


def segment_image(img_path: Path, liquid_intersections, mushy_intersections, output_path: Path) -> None:
    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    img_height = img.shape[0]
    img_width = img.shape[1]

    if len(liquid_intersections) == 0:
        with output_path.open("w") as f:
            f.write("\n")
        raise NoInterceptionException()
    if len(mushy_intersections) == 0:
        with output_path.open("w") as f:
            f.write("\n")
        raise NoInterceptionException()

    points: [Point] = []
    segments: [tuple[int, int]] = []

    left_end = None
    right_end = None

    for y, intersection in liquid_intersections.items():
        if not intersection:
            continue
        for x in intersection:
            match len(points):
                case 0:
                    points.append(Point(x, y))
                case 1:
                    points.append(Point(x, y))
                    first_point = points[0]
                    if first_point.x < x:
                        left_end = 0
                        right_end = 1
                    else:
                        left_end = 1
                        right_end = 0
                    segments.append((left_end, right_end))
                case index:
                    points.append(Point(x, y))

                    left_point = points[left_end]
                    right_point = points[right_end]
                    distance_to_left = math.sqrt((left_point.x - x) ** 2 + (left_point.y - y) ** 2)
                    distance_to_right = math.sqrt((right_point.x - x) ** 2 + (right_point.y - y) ** 2)

                    if distance_to_left < distance_to_right:
                        segments.append((left_end, index))
                        left_end = index
                    else:
                        segments.append((right_end, index))
                        right_end = index

    segments.append((right_end, left_end))

    liquid_mask_coordinates = []
    visited_segments = set()

    if NEWLABELS:
        with output_path.open("w") as f:
            f.write("0")

            current_segment = segments[0]
            first_point = current_segment[0]
            while True:
                if current_segment in visited_segments:
                    raise SegmentLoopException()
                visited_segments.add(current_segment)

                if current_segment[1] is None:
                    raise EndOfSegmentation
                segment_end_point = points[current_segment[1]]
                f.write(f" {segment_end_point.x / img_width} {segment_end_point.y / img_height}")
                liquid_mask_coordinates.append(segment_end_point.x)
                liquid_mask_coordinates.append(segment_end_point.y)
                previous_segment_start = current_segment[0]
                next_segment_start = current_segment[1]
                current_segment = next(filter(lambda segment: segment[0] == next_segment_start, segments), None)
                if current_segment is None:
                    break

            try:
                current_segment = next(
                    filter(
                        lambda segment: segment[1] == next_segment_start and segment[0] != previous_segment_start,
                        segments,
                    )
                )
            except StopIteration:
                raise EndOfSegmentation

            while True:
                if current_segment in visited_segments:
                    raise SegmentLoopException()
                visited_segments.add(current_segment)

                segment_end_point = points[current_segment[0]]
                f.write(f" {segment_end_point.x / img_width} {segment_end_point.y / img_height}")
                liquid_mask_coordinates.append(segment_end_point.x)
                liquid_mask_coordinates.append(segment_end_point.y)
                next_segment_start = current_segment[0]
                if next_segment_start == first_point:
                    break

                current_segment = next(filter(lambda segment: segment[1] == next_segment_start, segments))

            f.write("\n")

    points: [Point] = []
    segments: [tuple[int, int]] = []

    left_end = None
    right_end = None

    for y, intersection in mushy_intersections.items():
        if not intersection:
            continue
        for x in intersection:
            match len(points):
                case 0:
                    points.append(Point(x, y))
                case 1:
                    points.append(Point(x, y))
                    first_point = points[0]
                    if first_point.x < x:
                        left_end = 0
                        right_end = 1
                    else:
                        left_end = 1
                        right_end = 0
                    segments.append((left_end, right_end))
                case index:
                    points.append(Point(x, y))

                    left_point = points[left_end]
                    right_point = points[right_end]
                    distance_to_left = math.sqrt((left_point.x - x) ** 2 + (left_point.y - y) ** 2)
                    distance_to_right = math.sqrt((right_point.x - x) ** 2 + (right_point.y - y) ** 2)

                    if distance_to_left < distance_to_right:
                        segments.append((left_end, index))
                        left_end = index
                    else:
                        segments.append((right_end, index))
                        right_end = index

    segments.append((right_end, left_end))

    mushy_mask_coordinates = []
    visited_segments = set()

    if NEWLABELS:
        with output_path.open("a") as f:
            f.write("1")

            current_segment = segments[0]
            first_point = current_segment[0]
            while True:
                if current_segment in visited_segments:
                    raise SegmentLoopException()
                visited_segments.add(current_segment)

                if current_segment[1] is None:
                    raise EndOfSegmentation
                segment_end_point = points[current_segment[1]]
                f.write(f" {segment_end_point.x / img_width} {segment_end_point.y / img_height}")
                mushy_mask_coordinates.append(segment_end_point.x)
                mushy_mask_coordinates.append(segment_end_point.y)
                previous_segment_start = current_segment[0]
                next_segment_start = current_segment[1]
                current_segment = next(filter(lambda segment: segment[0] == next_segment_start, segments), None)
                if current_segment is None:
                    break

            try:
                current_segment = next(
                    filter(
                        lambda segment: segment[1] == next_segment_start and segment[0] != previous_segment_start,
                        segments,
                    )
                )
            except StopIteration:
                raise EndOfSegmentation

            while True:
                if current_segment in visited_segments:
                    raise SegmentLoopException()
                visited_segments.add(current_segment)

                segment_end_point = points[current_segment[0]]
                f.write(f" {segment_end_point.x / img_width} {segment_end_point.y / img_height}")
                mushy_mask_coordinates.append(segment_end_point.x)
                mushy_mask_coordinates.append(segment_end_point.y)
                next_segment_start = current_segment[0]
                if next_segment_start == first_point:
                    break

                current_segment = next(filter(lambda segment: segment[1] == next_segment_start, segments))

            f.write("\n")

    if DEBUG_SEGMENT:

        plt.figure(1, figsize=(9, 9))
        plt.xlim(0, 512)
        plt.ylim(290, 180)
        plt.title(str(img_path.name))
        plt.imshow(plt.imread(str(img_path)))
        for i in range(len(liquid_mask_coordinates)):
            if 2 * i > len(liquid_mask_coordinates) - 1:
                break
            if 2 * i / 255 > 1:
                plt.scatter(
                    [liquid_mask_coordinates[2 * i]], [liquid_mask_coordinates[2 * i + 1]], marker="*", color=(1, 0, 0)
                )
            else:
                plt.scatter(
                    [liquid_mask_coordinates[2 * i]],
                    [liquid_mask_coordinates[2 * i + 1]],
                    marker="*",
                    color=(1, 0, 0),
                )
        for i in range(len(mushy_mask_coordinates)):
            if 2 * i > len(mushy_mask_coordinates) - 1:
                break
            if 2 * i / 255 > 1:
                plt.scatter(
                    [mushy_mask_coordinates[2 * i]], [mushy_mask_coordinates[2 * i + 1]], marker="*", color=(0, 1, 0)
                )
            else:
                plt.scatter(
                    [mushy_mask_coordinates[2 * i]],
                    [mushy_mask_coordinates[2 * i + 1]],
                    marker="*",
                    color=(0, 1, 0),
                )
        plt.show()


def image_labelling(
    img_train_path: Path, img_train, img_test_path: Path, img_test, training_labels_dir: Path, testing_labels_dir: Path
) -> None:

    image_index = 0

    for img_path in img_train:
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        img_height = img.shape[0]
        img_width = img.shape[1]

        if KEEP_TRACK_OF_IMAGE:
            if image_index % 50 == 0:
                print("We're at the %sth training image" % (image_index + 1))

        image_index += 1

        shutil.copy(img_path, img_train_path.joinpath(img_path.name))

        for y_coord in range(y_min, y_max):
            image = image_to_numerical_value(img_path, y_coord)
            moving_average = calculate_moving_average(image, 11)
            deltas = calculate_deltas(img_width, moving_average)
            deltas_sum = calculate_deltas_sum(deltas)
            deltas_sum_moving_average = calculate_moving_average(deltas_sum, 11)

            liquid_crossing_point, mushy_crossing_point = find_crossing_point(
                deltas_sum_moving_average, x_hotspot, y_hotspot, delta_threshold, y_coord, moving_average
            )

            if DEBUG_Y_SWEEP:
                plt.figure(1, figsize=(10, 10))
                plt.subplot(3, 1, 1)
                plt.xlim([x_min, x_max])
                plt.ylim([y_max, y_min])
                plt.title(img_path.resolve().name)
                plt.imshow(img)
                for i in range(len(liquid_crossing_point)):
                    plt.scatter(liquid_crossing_point[i], y_coord, color="red")
                for i in range(len(mushy_crossing_point)):
                    plt.scatter(mushy_crossing_point[i], y_coord, color="yellow")
                plt.axhline(y_coord, color="red")
                plt.subplot(3, 1, 2)
                plt.plot(moving_average)
                plt.xlim([x_min, x_max])
                plt.subplot(3, 1, 3)
                plt.plot(deltas_sum_moving_average)
                for i in range(len(liquid_crossing_point)):
                    plt.axvline(liquid_crossing_point[i], color="red")
                plt.xlim([x_min, x_max])
                plt.show()

        if DO_SEGMENT:
            try:
                prepare_image(img_path, training_labels_dir.joinpath(img_path.stem + ".txt"))
            except SegmentLoopException:
                continue
            except NoInterceptionException:
                continue
            except EndOfSegmentation:
                continue

    image_index = 0

    for img_path in img_test:
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        img_height = img.shape[0]
        img_width = img.shape[1]

        if KEEP_TRACK_OF_IMAGE:
            if image_index % 50 == 0:
                print("We're at the %sth testing image" % (image_index + 1))

        image_index += 1

        shutil.copy(img_path, img_test_path.joinpath(img_path.name))

        for y_coord in range(y_min, y_max):
            image = image_to_numerical_value(img_path, y_coord)
            moving_average = calculate_moving_average(image, 11)
            deltas = calculate_deltas(img_width, moving_average)
            deltas_sum = calculate_deltas_sum(deltas)
            deltas_sum_moving_average = calculate_moving_average(deltas_sum, 13)

            liquid_crossing_point, mushy_crossing_point = find_crossing_point(
                deltas_sum_moving_average, x_hotspot, y_hotspot, delta_threshold, y_coord, moving_average
            )

            if DEBUG_Y_SWEEP:
                plt.figure(1, figsize=(10, 10))
                plt.subplot(3, 1, 1)
                plt.xlim([x_min, x_max])
                plt.ylim([y_max, y_min])
                plt.imshow(img)
                for i in range(len(liquid_crossing_point)):
                    plt.scatter(liquid_crossing_point[i], y_coord, color="red")
                for i in range(len(mushy_crossing_point)):
                    plt.scatter(mushy_crossing_point[i], y_coord, color="yellow")
                plt.axhline(y_coord, color="red")
                plt.subplot(3, 1, 2)
                plt.plot(moving_average)
                plt.xlim([x_min, x_max])
                plt.subplot(3, 1, 3)
                plt.plot(deltas_sum_moving_average)
                plt.axhline(delta_threshold, color="red")
                plt.xlim([x_min, x_max])
                plt.show()

        if DO_SEGMENT:
            try:
                prepare_image(img_path, testing_labels_dir.joinpath(img_path.stem + ".txt"))
            except SegmentLoopException:
                continue
            except NoInterceptionException:
                continue
            except EndOfSegmentation:
                continue


def extract_prediction_labels(label_path: Path) -> None:
    with label_path.open("r") as fd:
        for line in fd:
            class_, *coords = line.strip().split(" ")
            if class_ == "0":
                continue

            float()

            # todo: add strict parameter to batched() after updating to Python 3.13
            coords = [Point(*c) for c in itertools.batched(map(float, coords), 2)]

            rightmost_point_index = max(range(len(coords)), key=lambda i: coords[i].x)
            leftmost_point_index = min(range(len(coords)), key=lambda i: coords[i].x)
            assert leftmost_point_index < rightmost_point_index

            melt_pool_tail = coords[leftmost_point_index : rightmost_point_index + 1]

            if DEBUG_REGRESSION:
                plt.ylim(0.45, 0.28)
                plt.xlim(0.25, 0.56)
                plt.plot(
                    [coords[rightmost_point_index].x, coords[leftmost_point_index].x],
                    [coords[rightmost_point_index].y, coords[leftmost_point_index].y],
                    "k--",
                )
                plt.scatter(
                    [c.x for c in coords if c not in melt_pool_tail],
                    [c.y for c in coords if c not in melt_pool_tail],
                    color="red",
                )
                plt.scatter([c.x for c in melt_pool_tail], [c.y for c in melt_pool_tail], color="green")

                regression = np.polyfit([c.x for c in melt_pool_tail], [c.y for c in melt_pool_tail], 2)
                regression_line = np.linspace(coords[leftmost_point_index].x, coords[rightmost_point_index].x)
                plt.plot(
                    regression_line,
                    np.poly1d(regression)(regression_line),
                )
                print(regression)
                plt.show()


MAX_T = 2500
MELTING_T = 1100
MUSHY_MINIMUM = 1200

delta_threshold = -140

x_hotspot = [265, 304]
y_hotspot = [215, 245]

x_min = 100
x_max = 350

y_min = 190
y_max = 290

DEBUG_Y_SWEEP = False

DEBUG_SEGMENT = False

DEBUG_REGRESSION = False

NEWLABELS = False

DO_SEGMENT = False

KEEP_TRACK_OF_IMAGE = False

TRAIN_YOLO = False

RUN_YOLO = True

TRAINING_DATASET = "micro_dataset"

PREDICTION_DATASET = "pico_dataset"

EPOCHS = 40


def main() -> None:
    start_time = time.time()
    project_dir = Path.cwd()

    img_list = define_dataset(TRAINING_DATASET, "png")
    img_train, img_test = sklearn.model_selection.train_test_split(img_list, test_size=0.10)

    # Training images

    img_train_path = project_dir.joinpath("data", "images", "train")
    if img_train_path.exists() and NEWLABELS:
        shutil.rmtree(img_train_path)

    img_train_path.mkdir(exist_ok=True, parents=True)

    training_labels_dir = project_dir.joinpath("data", "labels", "train")
    if training_labels_dir.exists() and NEWLABELS:
        shutil.rmtree(training_labels_dir)

    training_labels_dir.mkdir(exist_ok=True, parents=True)

    # Testing images

    img_test_path = project_dir.joinpath("data", "images", "val")
    if img_test_path.exists() and NEWLABELS:
        shutil.rmtree(img_test_path)

    img_test_path.mkdir(exist_ok=True, parents=True)

    testing_labels_dir = project_dir.joinpath("data", "labels", "val")
    if testing_labels_dir.exists() and NEWLABELS:
        shutil.rmtree(testing_labels_dir)

    testing_labels_dir.mkdir(exist_ok=True, parents=True)

    if NEWLABELS or DEBUG_Y_SWEEP or DEBUG_SEGMENT or DO_SEGMENT:
        image_labelling(img_train_path, img_train, img_test_path, img_test, training_labels_dir, testing_labels_dir)

    labeling_time = time.time()

    print("--- %s minutes for image labeling ---" % ((time.time() - start_time) / 60))

    if RUN_YOLO:
        yaml = {
            "path": str(project_dir.joinpath("data").resolve()),
            "train": "images/train",
            "val": "images/val",
            "nc": 2,
            "names": ["Liquid Melt Pool", "Mushy Melt Pool"],
        }
        yaml_path = project_dir.joinpath("data.yaml")
        with yaml_path.open("w") as f:
            json.dump(yaml, f, indent=4)

        runs_dir = project_dir.joinpath("runs")
        if runs_dir.exists() and TRAIN_YOLO:
            shutil.rmtree(runs_dir)

        # Run YOLO on the available data

        if TRAIN_YOLO:
            model = YOLO("yolo11n-seg.yaml")
            model.train(data="data.yaml", epochs=EPOCHS, imgsz=640)
        else:
            model = YOLO(model="runs/segment/train/weights/best.pt")

        yolo_run_dir = project_dir.joinpath("runs", "segment")
        if yolo_run_dir.joinpath("predict").exists():
            shutil.rmtree(yolo_run_dir.joinpath("predict"))

        model.predict(
            source=PREDICTION_DATASET,
            save=True,
            project=yolo_run_dir,
            device="cuda:0",
            save_txt=True,
            name="predict",
        )

        prediction_labels_dir = yolo_run_dir.joinpath("predict", "labels")
        extract_prediction_labels(
            Path(
                r"C:\Users\Antonio "
                r"Alves\Desktop\FEUP\Bolsa\lmd-imaging\runs\segment\predict\labels\1707152506084292100.txt"
            )
        )
        extract_prediction_labels(
            Path(
                r"C:\Users\Antonio "
                r"Alves\Desktop\FEUP\Bolsa\lmd-imaging\runs\segment\predict\labels\1707152529453574600.txt"
            )
        )
        extract_prediction_labels(
            Path(
                r"C:\Users\Antonio "
                r"Alves\Desktop\FEUP\Bolsa\lmd-imaging\runs\segment\predict\labels\1707152808957778800.txt"
            )
        )
        extract_prediction_labels(
            Path(
                r"C:\Users\Antonio "
                r"Alves\Desktop\FEUP\Bolsa\lmd-imaging\runs\segment\predict\labels\1707152808974096900.txt"
            )
        )

    print("--- %s minutes for YOLO execution ---" % ((time.time() - labeling_time) / 60))
