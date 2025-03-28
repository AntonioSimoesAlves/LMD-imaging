from collections.abc import Iterator, Iterable
from pathlib import Path

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
    def _save_predictions_kwargs(save_predictions: tuple[Path, str] | None) -> dict:
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

    liquid_melt_pool_mask = prediction_results.masks.xy[liquid_melt_pool_mask_index]
    mushy_melt_pool_mask = prediction_results.masks.xy[mushy_melt_pool_mask_index]

    return {
        LIQUID: [Point(row[0], row[1]) for row in liquid_melt_pool_mask],
        MUSHY: [Point(row[0], row[1]) for row in mushy_melt_pool_mask],
    }


def get_mask_indices(prediction_results: ultralytics.engine.results.Results) -> tuple[int, int]:
    liquid = None
    mushy = None

    for key, value in prediction_results.names.items():
        if value == "Liquid Melt Pool":
            liquid = key
            if mushy is not None:
                break
        elif value == "Mushy Melt Pool":
            mushy = key
            if liquid is not None:
                break

    assert str(liquid) == LIQUID
    assert str(mushy) == MUSHY

    if liquid is None or mushy is None:
        raise ValueError("Could not find mask names")
    return liquid, mushy
