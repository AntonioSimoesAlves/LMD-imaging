from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt

from .common import (
    calculate_liquid_melt_pool_regression_curve,
    calculate_mushy_melt_pool_regression_curve,
    Labels,
    MUSHY,
    LIQUID,
)


def plot_labels(image: cv2.typing.MatLike | Path, labels: Labels, title: str | None = None) -> None:
    plt.figure(1, figsize=(9, 9))
    if title is not None:
        plt.title(title)
    if isinstance(image, Path):
        image = cv2.imread(str(image))
    plt.imshow(image)

    for mask in (labels[LIQUID], labels[MUSHY]):
        plt.gca().invert_yaxis()
        plt.scatter([p.x for p in mask], [p.y for p in mask])
    plt.show()


# GRAPH_OFFSET = 0.03
X_MIN, X_MAX = 60, 400
Y_MIN, Y_MAX = 130, 350


def plot_regression_curves(labels: Labels, image_path: Path) -> None:
    liquid_coords = labels[LIQUID]
    mushy_coords = labels[MUSHY]

    liquid_downmost_point_index = max(range(len(liquid_coords)), key=lambda i: liquid_coords[i].y)
    liquid_leftmost_point_index = min(range(len(liquid_coords)), key=lambda i: liquid_coords[i].y)
    melt_pool_liquid_tail = liquid_coords[liquid_leftmost_point_index : liquid_downmost_point_index + 1]

    mushy_downmost_point_index = max(range(len(mushy_coords)), key=lambda i: mushy_coords[i].x)
    mushy_leftmost_point_index = min(range(len(mushy_coords)), key=lambda i: mushy_coords[i].x)
    melt_pool_mushy_tail = mushy_coords[mushy_leftmost_point_index : mushy_downmost_point_index + 1]

    plt.xlim(X_MIN, X_MAX)
    plt.ylim(Y_MIN, Y_MAX)
    plt.gca().invert_yaxis()
    plt.imshow(plt.imread(str(image_path)))

    plt.scatter(
        [c.x for c in liquid_coords if c not in melt_pool_liquid_tail],
        [c.y for c in liquid_coords if c not in melt_pool_liquid_tail],
        color="blue",
    )
    plt.scatter(
        [c.x for c in mushy_coords if c not in melt_pool_mushy_tail],
        [c.y for c in mushy_coords if c not in melt_pool_mushy_tail],
        color="green",
    )
    plt.scatter(
        [c.x for c in melt_pool_liquid_tail],
        [c.y for c in melt_pool_liquid_tail],
        color="cyan",
    )
    plt.scatter([c.x for c in melt_pool_mushy_tail], [c.y for c in melt_pool_mushy_tail], color="lime")

    liquid_regression_line = np.linspace(
        liquid_coords[liquid_leftmost_point_index].x, liquid_coords[liquid_downmost_point_index].x
    )

    mushy_regression_line = np.linspace(
        mushy_coords[mushy_leftmost_point_index].x, mushy_coords[mushy_downmost_point_index].x
    )

    liquid_regression = calculate_liquid_melt_pool_regression_curve(liquid_coords)
    mushy_regression = calculate_mushy_melt_pool_regression_curve(mushy_coords)

    plt.plot(
        liquid_regression_line,
        np.poly1d(np.array(liquid_regression))(liquid_regression_line),
        linewidth=5,
        color="red",
    )
    plt.plot(
        mushy_regression_line,
        np.poly1d(np.array(mushy_regression))(mushy_regression_line),
        linewidth=5,
        color="red",
    )
    # plt.text(
    #     melt_pool_liquid_tail[int((len(melt_pool_liquid_tail) - 1) / 2)].x - 0.5 * GRAPH_OFFSET,
    #     melt_pool_liquid_tail[int((len(melt_pool_liquid_tail) - 1) / 2)].y - 1.2 * GRAPH_OFFSET,
    #     "y1 = "
    #     + str(round(liquid_regression[0], 2))
    #     + "x^2 "
    #     + (
    #         str(round(liquid_regression[1], 2))
    #         if liquid_regression[1] < 0
    #         else "+ " + str(round(liquid_regression[1], 2))
    #     )
    #     + "x "
    #     + (
    #         str(round(liquid_regression[2], 2))
    #         if liquid_regression[2] < 0
    #         else "+ " + str(round(liquid_regression[2], 2))
    #     ),
    #     fontsize=10,
    #     color="black",
    # )
    # plt.text(
    #     melt_pool_mushy_tail[int((len(melt_pool_mushy_tail) - 1) / 2)].x - 1 * GRAPH_OFFSET,
    #     melt_pool_mushy_tail[int((len(melt_pool_mushy_tail) - 1) / 2)].y + 1.2 * GRAPH_OFFSET,
    #     "y2 = "
    #     + str(round(mushy_regression[0], 2))
    #     + "x^2 "
    #     + (str(round(mushy_regression[1], 2)) if mushy_regression[1] < 0 else "+ " + str(round(mushy_regression[1], 2)))
    #     + "x "
    #     + (
    #         str(round(mushy_regression[2], 2)) if mushy_regression[2] < 0 else "+ " + str(round(mushy_regression[2], 2))
    #     ),
    #     fontsize=10,
    #     color="black",
    # )
    plt.show()
