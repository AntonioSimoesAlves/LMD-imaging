from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt

from .common import (
    Labels,
    MUSHY,
    LIQUID,
    calculate_liquid_melt_pool_regression_curve,
    calculate_mushy_melt_pool_regression_curve,
)

X_MIN, X_MAX = 100, 360  # Sets boundaries for the plotting x-axis
Y_MIN, Y_MAX = 170, 290  # Sets boundaries for the plotting y-axis


def plot_labels(image: cv2.typing.MatLike | Path, labels: Labels, title: str | None = None) -> None:
    plt.figure(1, figsize=(12, 10))
    plt.xlim(X_MIN, X_MAX)
    plt.ylim(Y_MIN, Y_MAX)
    plt.gca().invert_yaxis()
    if title is not None:
        plt.title(title)
    if isinstance(image, Path):
        image = cv2.imread(str(image))
    plt.imshow(image)

    for mask in (labels[LIQUID], labels[MUSHY]):
        plt.scatter([p.x for p in mask], [p.y for p in mask], s=5)
    plt.show()


def plot_regression_curves(labels: Labels, image_path: Path) -> None:

    if labels is None:
        plt.figure(1, figsize=(10, 10))
        plt.title(image_path.name)
        plt.xlim(X_MIN, X_MAX)
        plt.ylim(Y_MIN, Y_MAX)
        plt.gca().invert_yaxis()
        plt.imshow(cv2.imread(str(image_path)))
        plt.show()
        return

    liquid_coords = labels[LIQUID]
    mushy_coords = labels[MUSHY]

    melt_pool_liquid_tail = None
    melt_pool_mushy_tail = None

    if liquid_coords is not None:
        liquid_downmost_point_index = max(range(len(liquid_coords)), key=lambda i: liquid_coords[i].y)
        liquid_leftmost_point_index = min(range(len(liquid_coords)), key=lambda i: liquid_coords[i].x)
        melt_pool_liquid_tail = liquid_coords[liquid_leftmost_point_index : liquid_downmost_point_index + 1]
        if not melt_pool_liquid_tail:
            melt_pool_liquid_tail_1 = liquid_coords[liquid_leftmost_point_index:]
            melt_pool_liquid_tail_2 = liquid_coords[: liquid_downmost_point_index + 1]
            melt_pool_liquid_tail = melt_pool_liquid_tail_1 + melt_pool_liquid_tail_2

    if mushy_coords is not None:
        mushy_downmost_point_index = max(range(len(mushy_coords)), key=lambda i: mushy_coords[i].x)
        mushy_leftmost_point_index = min(range(len(mushy_coords)), key=lambda i: mushy_coords[i].x)
        melt_pool_mushy_tail = mushy_coords[mushy_leftmost_point_index + 1 : mushy_downmost_point_index + 1]
        if not melt_pool_mushy_tail:
            melt_pool_mushy_tail_1 = mushy_coords[mushy_leftmost_point_index:]
            melt_pool_mushy_tail_2 = mushy_coords[: mushy_downmost_point_index + 1]
            melt_pool_mushy_tail = melt_pool_mushy_tail_1 + melt_pool_mushy_tail_2

    plt.figure(1, figsize=(10, 10))
    plt.title(image_path.name)
    plt.xlim(X_MIN, X_MAX)
    plt.ylim(Y_MIN, Y_MAX)
    plt.gca().invert_yaxis()
    plt.imshow(cv2.imread(str(image_path)))

    # if melt_pool_liquid_tail is not None:
    #     plt.scatter(
    #         [c.x for c in liquid_coords if c not in melt_pool_liquid_tail],
    #         [c.y for c in liquid_coords if c not in melt_pool_liquid_tail],
    #         color="cyan",
    #         s=5,
    #     )
    #     plt.scatter(
    #         [c.x for c in melt_pool_liquid_tail],
    #         [c.y for c in melt_pool_liquid_tail],
    #         color="cyan",
    #         s=5,
    #     )
    # if melt_pool_mushy_tail is not None:
    #     plt.scatter(
    #         [c.x for c in mushy_coords if c not in melt_pool_mushy_tail],
    #         [c.y for c in mushy_coords if c not in melt_pool_mushy_tail],
    #         color="lime",
    #         s=5,
    #     )
    #     plt.scatter(
    #         [c.x for c in melt_pool_mushy_tail],
    #         [c.y for c in melt_pool_mushy_tail],
    #         color="lime",
    #         s=5,
    #     )

    if melt_pool_liquid_tail is not None:
        liquid_regression_line = np.linspace(
            liquid_coords[liquid_leftmost_point_index].x, liquid_coords[liquid_downmost_point_index].x
        )
        liquid_regression = calculate_liquid_melt_pool_regression_curve(liquid_coords)
        plt.plot(
            liquid_regression_line,
            np.poly1d(np.array(liquid_regression))(liquid_regression_line),
            linewidth=3,
            color="blue",
        )

    if melt_pool_mushy_tail is not None:
        mushy_regression_line = np.linspace(
            mushy_coords[mushy_leftmost_point_index].x, mushy_coords[mushy_downmost_point_index].x
        )
        mushy_regression = calculate_mushy_melt_pool_regression_curve(mushy_coords)
        plt.plot(
            mushy_regression_line,
            np.poly1d(np.array(mushy_regression))(mushy_regression_line),
            linewidth=3,
            color="cyan",
        )

    plt.show()
