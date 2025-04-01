import multiprocessing
from abc import ABC, abstractmethod
from collections.abc import Iterator, Iterable
from pathlib import Path
from typing import TextIO, NamedTuple

import cv2
import numpy as np


class Point(NamedTuple):
    x: float
    y: float


type Labels = dict[str, list[Point]]
LIQUID = "0"
MUSHY = "1"


class Labeler(ABC):
    @abstractmethod
    def label(self, image: cv2.typing.MatLike | Path) -> Labels | None:
        pass

    def batch_label(self, images: Iterable[cv2.typing.MatLike | Path]) -> Iterator[Labels | None]:
        with multiprocessing.Pool() as pool:
            return pool.map(self.label, images)


def labels_to_txt(labels: Labels, to: TextIO) -> None:
    for label, coords in labels.items():
        if coords is None:
            continue
        to.write(label)
        for point in coords:
            to.write(" ")
            to.write(str(point.x))
            to.write(" ")
            to.write(str(point.y))
        to.write("\n")


def calculate_point_regression_curve(points: list[Point]) -> list[float]:
    return np.polyfit(
        [c.x for c in points],
        [c.y for c in points],
        2,
    ).tolist()


def calculate_liquid_melt_pool_regression_curve(points: list[Point]) -> list[float]:
    liquid_downmost_point_index = max(range(len(points)), key=lambda i: points[i].y)
    liquid_leftmost_point_index = min(range(len(points)), key=lambda i: points[i].y)
    melt_pool_liquid_tail = points[liquid_leftmost_point_index : liquid_downmost_point_index + 1]

    return calculate_point_regression_curve(melt_pool_liquid_tail)


def calculate_mushy_melt_pool_regression_curve(points: list[Point]) -> list[float]:
    mushy_downmost_point_index = max(range(len(points)), key=lambda i: points[i].x)
    mushy_leftmost_point_index = min(range(len(points)), key=lambda i: points[i].x)
    melt_pool_mushy_tail = points[mushy_leftmost_point_index : mushy_downmost_point_index + 1]

    return calculate_point_regression_curve(melt_pool_mushy_tail)


def regression_curves_to_txt(labels: Labels, to: TextIO) -> None:
    for class_, f in [
        (LIQUID, calculate_liquid_melt_pool_regression_curve),
        (MUSHY, calculate_mushy_melt_pool_regression_curve),
    ]:
        if labels.get(class_) is None:
            continue
        to.write(class_)
        for regression_parameters in f(labels[class_]):
            to.write(" ")
            to.write(str(regression_parameters))
        to.write("\n")
