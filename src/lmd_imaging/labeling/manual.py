import math
from pathlib import Path

import cv2
import numpy as np

from .common import Labels, Labeler, Point, LIQUID, MUSHY

DEBUG_Y_SWEEP = False

MAX_T = 2500
MELTING_T = 1100
MUSHY_MINIMUM = 1150

DELTA_WEIGHT = 10.0
DELTA_THRESHOLD = -140  # lower is more melt pool detection

# In order to avoid incorrect boundary points in hotspot at the center

X_HOTSPOT = [265, 304]
Y_HOTSPOT = [215, 245]

# Only segment the portion of the image that has the melt pool

SEGMENT_Y_MIN = 175
SEGMENT_Y_MAX = 280

LINE_MEAN_TEMPERATURE_THRESHOLD = 630  # Average of the horizontal line temperature. To detect the presence of the
# melt pool


class ManualLabeler(Labeler):
    def label(self, image: cv2.typing.MatLike | Path) -> Labels | None:
        if isinstance(image, Path):
            image = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)
        image_height, image_width = image.shape[:2]

        liquid_intersections = {}
        mushy_intersections = {}

        for y_coord in range(SEGMENT_Y_MIN, SEGMENT_Y_MAX):
            temps = extract_line_temperature(image, y_coord)
            if np.mean(temps) < LINE_MEAN_TEMPERATURE_THRESHOLD:
                continue

            temp_moving_averages = calculate_moving_average(temps, 11)
            deltas = calculate_deltas(image_width, temp_moving_averages)
            deltas_sum = calculate_deltas_sum(deltas)
            deltas_sum_moving_average = calculate_moving_average(deltas_sum, 9)

            liquid_crossing_point, mushy_crossing_point = find_crossing_point(
                deltas_sum_moving_average, X_HOTSPOT, Y_HOTSPOT, DELTA_THRESHOLD, y_coord, temp_moving_averages
            )

            intersection = list(filter(lambda n: n != 0, liquid_crossing_point))
            if intersection:
                liquid_intersections[y_coord] = intersection

            intersection = list(filter(lambda n: n != 0, mushy_crossing_point))
            if intersection:
                mushy_intersections[y_coord] = intersection

        try:
            return {
                LIQUID: self._segment(liquid_intersections, image_width, image_height),
                MUSHY: self._segment(mushy_intersections, image_width, image_height),
            }
        except (EndOfSegmentation, SegmentLoopException):
            return None

    @staticmethod
    def _segment(intersections: dict[int, list[int]], image_width: int, image_height: int) -> list[Point]:
        output = []

        points: list[Point] = []
        segments: list[tuple[int, int]] = []

        left_end = None
        right_end = None

        for y, intersection in intersections.items():
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
                        assert isinstance(left_end, int)
                        assert isinstance(right_end, int)

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

        if left_end is None or right_end is None:
            raise EndOfSegmentation()

        segments.append((right_end, left_end))

        visited_segments = set()

        current_segment = segments[0]
        initial_point = current_segment[0]
        while True:
            if current_segment in visited_segments:
                raise SegmentLoopException()
            visited_segments.add(current_segment)

            segment_end_point = points[current_segment[1]]
            output.append(Point(segment_end_point.x / image_width, segment_end_point.y / image_height))
            previous_segment_start = current_segment[0]
            next_segment_start = current_segment[1]

            s = next(filter(lambda segment: segment[0] == next_segment_start, segments), None)
            if s is None:
                break
            current_segment = s

        try:
            current_segment = next(
                filter(
                    lambda segment: segment[1] == next_segment_start and segment[0] != previous_segment_start,
                    segments,
                )
            )
        except StopIteration:
            raise EndOfSegmentation()

        while True:
            if current_segment in visited_segments:
                raise SegmentLoopException()
            visited_segments.add(current_segment)

            segment_end_point = points[current_segment[0]]
            output.append(Point(segment_end_point.x / image_width, segment_end_point.y / image_height))
            next_segment_start = current_segment[0]
            if next_segment_start == initial_point:
                break

            current_segment = next(filter(lambda segment: segment[1] == next_segment_start, segments))

        return output


def calculate_moving_average(line: np.ndarray | list[float], average_sample_size: int) -> list[float]:
    if isinstance(line, np.ndarray):
        line = line.tolist()  # TODO: avoid this
    moving_average = []

    for i in range(len(line)):
        previous_values = []
        next_values = []
        current_value = line[i]
        for j in range(1, average_sample_size // 2 + 1):
            if (i - j) < 0:
                previous_values.append(line[i])
            else:
                previous_values.append(line[i - j])
            if (i + j) > len(line) - 1:
                next_values.append(line[i])
            else:
                next_values.append(line[i + j])
        moving_average.append(round((current_value + sum(previous_values) + sum(next_values)) / average_sample_size, 3))

    return moving_average


def extract_line_temperature(image: cv2.typing.MatLike, y_coord: int) -> np.ndarray:
    """Fetches a horizontal pixel line and converts it into temperature data.

    Warning: scaling might change depending on image capture method used, and may need to be adjusted."""

    return image[y_coord].flatten() / 10


def calculate_deltas(img_width: int, moving_average: list[float]) -> list[float]:
    pixels_deltas = []

    for x in range(img_width):
        if x > 0:
            past_value = moving_average[x - 1]
            current_value = moving_average[x]
            delta = round(current_value - past_value, 3)
            pixels_deltas.append(delta)
        else:
            pixels_deltas.append(0.0)

    return pixels_deltas


def calculate_deltas_sum(pixels_deltas: list[float]) -> list[float]:
    deltas_sum = []

    # DELTA_WEIGHT adjusts the importance of the current pixel temperature delta compared to previous deltas

    for i in range(len(pixels_deltas)):
        if i < len(pixels_deltas) - 1:
            deltas_sum.append(
                DELTA_WEIGHT * pixels_deltas[i] + pixels_deltas[i - 1] + pixels_deltas[i - 2] - pixels_deltas[i + 1]
            )
        else:
            deltas_sum.append(DELTA_WEIGHT * pixels_deltas[i] + pixels_deltas[i - 1] + pixels_deltas[i - 2])

    return deltas_sum


def find_crossing_point(
    deltas_sum_moving_average: list[float],
    x_center: list[int],
    y_center: list[int],
    delta_threshold: float,
    y_coord: int,
    temperature_moving_average: list[float],
) -> tuple[list[int], list[int]]:
    melt_pool_points_liquid_coordinate: list[int] = []
    melt_pool_points_mushy_coordinate: list[int] = []
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
                ):
                    continue
            if temperature_moving_average[i] > MELTING_T:
                if len(melt_pool_points_liquid_coordinate) > 0:
                    if (
                        i - melt_pool_points_liquid_coordinate[-1] > 20
                    ):  # Avoids successive detections on pixels next to each other
                        melt_pool_points_liquid_coordinate.append(i)
                else:
                    melt_pool_points_liquid_coordinate.append(i)

    for i in range(len(deltas_sum_moving_average)):
        if temperature_moving_average[i] > MUSHY_MINIMUM and not found_first_mushy_point:  # First point that has
            # temperature above the defined minimum for mushy region
            if len(melt_pool_points_liquid_coordinate) == 0:  # Only detects mushy points if there are liquid points
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
                if 1200 < temperature_moving_average[melt_pool_points_liquid_coordinate[-1] + 10] < 1400:  # checks if
                    # the single liquid point is close to the potential mushy point (left side of the liquid melt pool)
                    melt_pool_points_mushy_coordinate.append(i)
                else:
                    if len(melt_pool_points_mushy_coordinate) == 1:  # if the liquid point is on the right side of the
                        # liquid melt pool, remove the mushy point and skip this line
                        melt_pool_points_mushy_coordinate.pop()
                        break
            elif len(melt_pool_points_liquid_coordinate) >= 2:
                melt_pool_points_mushy_coordinate.append(i)

    return melt_pool_points_liquid_coordinate, melt_pool_points_mushy_coordinate


class SegmentLoopException(Exception):
    pass


class EndOfSegmentation(Exception):
    pass
