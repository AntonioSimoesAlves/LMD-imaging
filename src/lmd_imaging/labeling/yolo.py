import math
from collections.abc import Iterator, Iterable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import ultralytics.engine.results
from ultralytics import YOLO

from .common import Labeler, Labels, Point, LIQUID, MUSHY


class YoloLabeler(Labeler):
    def __init__(self, model: Path | str, device: str | None = "cuda:0"):
        super().__init__()
        self.model = YOLO(model=model)
        self.device = device

    def label(
        self, image: cv2.typing.MatLike | Path, *, save_predictions: tuple[Path, str] | None = None
    ) -> Labels | None:
        if isinstance(image, np.ndarray) and (len(image.shape) < 3 or image.shape[2] != 3):
            raise ValueError("YOLO only supports BGR images")

        return yolo_results_to_labels(
            self.model.predict(
                source=image,
                device=self.device,
                show_boxes=False,
                **self._save_predictions_kwargs(save_predictions),
            )[0]
        )

    def batch_label(
        self, images: Path | Iterable[cv2.typing.MatLike], *, save_predictions: tuple[Path, str] | None = None
    ) -> Iterator[Labels | None]:
        if isinstance(images, Path) and not images.is_dir():
            raise ValueError("Path must be a directory")
        if hasattr(images, "__iter__"):
            for image in images:
                if isinstance(image, np.ndarray) and (len(image.shape) < 3 or image.shape[2] != 3):
                    raise ValueError("YOLO only supports BGR images")

        return map(
            yolo_results_to_labels,
            self.model.predict(
                source=images,
                device=self.device,
                show_boxes=False,
                **self._save_predictions_kwargs(save_predictions),
            ),
        )

    @staticmethod
    def _save_predictions_kwargs(save_predictions: tuple[Path, str] | None) -> dict[str, Any]:
        if save_predictions is not None:
            return {
                "save": True,
                "project": save_predictions[0],
                "name": save_predictions[1],
            }
        else:
            return {}


def yolo_results_to_labels(prediction_results: ultralytics.engine.results.Results | None) -> Labels | None:
    if prediction_results is None or prediction_results.masks is None:
        return None

    liquid_melt_pool_mask_index, mushy_melt_pool_mask_index = get_mask_indices(prediction_results)

    liquid_melt_pool_mask = (
        prediction_results.masks.xy[liquid_melt_pool_mask_index] if liquid_melt_pool_mask_index is not None else None
    )
    mushy_melt_pool_mask = (
        prediction_results.masks.xy[mushy_melt_pool_mask_index] if mushy_melt_pool_mask_index is not None else None
    )

    return {
        LIQUID: [Point(row[0], row[1]) for row in liquid_melt_pool_mask] if liquid_melt_pool_mask is not None else None,
        MUSHY: [Point(row[0], row[1]) for row in mushy_melt_pool_mask] if mushy_melt_pool_mask is not None else None,
    }


def get_mask_indices(prediction_results: ultralytics.engine.results.Results) -> tuple[int | None, int | None]:
    liquid = None
    mushy = None
    liquid_id = None
    mushy_id = None

    for key, value in prediction_results.names.items():
        if value == "Liquid Melt Pool":
            liquid_id = key
            if mushy_id is not None:
                break
        elif value == "Mushy Melt Pool":
            mushy_id = key
            if liquid_id is not None:
                break

    if liquid_id is None or mushy_id is None:
        raise ValueError("missing class names")

    assert str(liquid_id) == LIQUID
    assert str(mushy_id) == MUSHY

    current_biggest_liquid_mask_size = -1
    current_biggest_mushy_mask_size = -1

    mask_classes = prediction_results.boxes.cls.cpu().tolist()

    for i, class_id in enumerate(mask_classes):
        if math.isclose(class_id, liquid_id) and len(prediction_results.masks.xy[i]) > current_biggest_liquid_mask_size:
            current_biggest_liquid_mask_size = len(prediction_results.masks.xy[i])
            liquid = i
        if math.isclose(class_id, mushy_id) and len(prediction_results.masks.xy[i]) > current_biggest_mushy_mask_size:
            current_biggest_mushy_mask_size = len(prediction_results.masks.xy[i])
            mushy = i

    return liquid, mushy
