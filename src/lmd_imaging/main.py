import csv
import itertools
import shutil
import time
from copy import deepcopy
from pathlib import Path
from typing import Iterator

import click
import matplotlib.pyplot as plt

from .labeling import YoloLabeler, labels_to_txt, Point, regression_curves_to_txt, Labels
from .labeling.plotting import plot_regression_curves, plot_labels
from .training import prepare_dataset_and_generate_labels, OutputPaths, train_dataset

MODEL = Path.cwd().joinpath("runs", "train", "weights", "best.pt")


@click.group()
def cli() -> None:
    pass


@cli.command(short_help="Generate YOLO-based txt labels for input_ files.")
@click.argument(
    "input_",
    type=click.Path(
        exists=True,
        readable=True,
        allow_dash=False,  # TODO
        path_type=Path,
    ),
)
@click.option(
    "--model",
    "-m",
    default=Path.cwd().joinpath("runs", "segment", "train", "weights", "best.pt"),
    show_default=True,
    type=click.Path(
        dir_okay=False,
        exists=True,
        readable=True,
        allow_dash=False,
        path_type=Path,
    ),
    help="Path to YOLO model",
)
@click.option(
    "--device",
    "-d",
    type=str,
    default="cpu",
    help="Specify which device to use for YOLO prediction.",
)
@click.option(
    "--output",
    default=Path.cwd().joinpath("yolo_labels"),
    type=click.Path(
        file_okay=False,
        exists=False,
        readable=True,
        path_type=Path,
    ),
    show_default=True,
    help="Where to write the labels to.",
)
@click.option(
    "--image-type",
    default=".png",
    type=click.Choice([".jpg", ".jpeg", ".png"]),
    show_default=True,
    help="Input image type.",
)
@click.option(
    "--verbose",
    type=(click.Choice(["none", "small", "full"])),
    default="none",
    show_default=True,
    help="Enable verbose mode. Small verbose only lists every 200 labelled images.",
)
def yolo_label(
    input_: Path,
    model: Path,
    device: str,
    output: Path,
    image_type=".png",
    verbose="none",
) -> None:

    verbose_image_limit = 100
    if verbose != "none" and verbose != "small" and verbose != "full":
        verbose = "none"

    if verbose == "small" and verbose_image_limit > len(list(Path(input_).glob("*" + image_type))):
        verbose = "none"

    if model is None:
        model = "yolo11n-seg.yaml"
    labeler = YoloLabeler(
        model=model, device=device, verbose=False if verbose == "none" or verbose == "small" else True
    )

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    if verbose == "small":
        image_count = 0

    if input_.is_dir():
        input_ = Path(input_)
        for image in sorted(input_.glob("*" + image_type)):
            if verbose == "small" and image_count % verbose_image_limit == 0:
                print(f"Have labelled {image_count} images out of {len(list(Path(input_).glob("*"+image_type)))}.")
            if str(image) == "-" or image.is_dir():
                raise ValueError("Directory and stdin inputs can only be passed as a single input")
            labels = labeler.label(image, verbose=False if verbose == "none" or verbose == "small" else True)
            output_path = output.joinpath(image.stem + ".txt")
            with output_path.open("w", encoding="utf-8") as fd:
                labels_to_txt(labels, fd)
            if verbose == "small":
                image_count += 1
        print(
            f"Have labelled all {len(list(Path(input_).glob("*"+image_type)))} images. Output is written to "
            f"{output}."
        )
    elif input_.is_file():
        labels = labeler.label(input_)
        output_path = output.joinpath(input_.stem + ".txt")
        with output_path.open("w", encoding="utf-8") as fd:
            labels_to_txt(labels, fd)


@cli.command(short_help="Generate segmentation masks for input_ files.")
@click.argument(
    "input_",
    type=click.Path(
        dir_okay=True,
        exists=True,
        readable=True,
        allow_dash=False,
        path_type=Path,
        file_okay=True,
    ),
)
@click.option(
    "--model",
    "-m",
    type=click.Path(
        dir_okay=False,
        exists=True,
        readable=True,
        allow_dash=False,
        path_type=Path,
    ),
    help="Path to YOLO model",
    default=Path.cwd().joinpath("runs", "segment", "train", "weights", "best.pt"),
    show_default=True,
)
@click.option(
    "--device",
    "-d",
    type=str,
    default="cpu",
    help="Specify which device to use for YOLO prediction.",
)
@click.option(
    "--output",
    default=Path.cwd().joinpath("runs", "segment"),
    type=click.Path(
        dir_okay=True,
        exists=False,
        readable=True,
        allow_dash=False,
        path_type=Path,
    ),
    show_default=True,
    help="Where to write the labels to.",
)
@click.option(
    "--run-name",
    type=click.Path(
        dir_okay=True,
        exists=False,
        readable=True,
        allow_dash=False,
        path_type=Path,
    ),
    default=Path.cwd().joinpath("runs", "predict"),
    show_default=True,
    help="Name of the folder to write to.",
)
def prediction(
    input_: Path,
    model: Path,
    device: str,
    output: Path,
    run_name: Path,
) -> Iterator[Labels | None]:

    if model is None:
        raise ValueError("No model found.")

    output_dir = output.joinpath(run_name.stem)
    if output_dir.exists():
        shutil.rmtree(output_dir)

    labeler = YoloLabeler(model=model, device=device)
    if input_.is_dir():
        labels = labeler.batch_label(
            input_,
            save_predictions=(output, str(run_name.stem)),
        )
    elif input_.is_file():
        labels = labeler.label(
            input_,
            save_predictions=(output, str(run_name.stem)),
        )
    elif not input_.exists():
        exit()

    print(f"Predictions have been saved to {output_dir}.")

    return labels


@cli.command(short_help="Generate YOLO-based txt files for regression parameters.")
@click.argument(
    "input_",
    type=click.Path(
        dir_okay=True,
        exists=True,
        readable=True,
        allow_dash=False,
        path_type=Path,
    ),
)
@click.option(
    "--output",
    default=Path.cwd().joinpath("runs", "segment", "predict", "regression_parameters"),
    show_default=True,
    type=click.Path(
        dir_okay=True,
        exists=False,
        readable=True,
        allow_dash=False,
        path_type=Path,
    ),
    help="Where to write the labels to.",
)
@click.option(
    "--write-csv",
    type=bool,
    default=False,
    show_default=True,
    is_flag=True,
    help="Write labels to CSV.",
)
@click.option(
    "--overwrite-df",
    type=bool,
    default=False,
    show_default=True,
    is_flag=True,
    help='Overwrite existing labels in "new_df.csv". Requires "df_1.csv" and "df_2.csv"',
)
def regression_parameters(
    input_: Path,
    output: Path,
    write_csv: bool,
    overwrite_df: bool,
) -> None:
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    if overwrite_df and not write_csv:
        raise ValueError(f"First write the intended csv. (use {"--write-csv"})")

    files_list = []

    if write_csv:
        with open("regression_parameters.csv", "w", encoding="utf-8", newline="") as csv_fd:
            writer = csv.writer(csv_fd, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["xiris_name", "a_liquid", "b_liquid", "c_liquid", "a_mushy", "b_mushy", "c_mushy"])

    if input_.is_file():
        labels = {}
        with input_.open("r", encoding="utf-8") as fd:
            files_list.append(input_.stem + ".png")
            for line in fd:
                class_, *coords = line.strip().split(" ")
                coords = [Point(*c) for c in itertools.batched(map(float, coords), 2)]
                labels[class_] = coords
            with output.joinpath(input_.stem + ".txt").open("w", encoding="utf-8") as fd2:
                regression_curves_to_txt(labels, fd2)
            if write_csv:
                with output.joinpath(input_.stem + ".txt").open("r", encoding="utf-8") as fd3:
                    line_to_add = []
                    files_list.append(input_.stem + ".png")
                    for line in fd3:
                        class_, *parameters = line.strip().split(" ")

                        line_to_add.append(parameters[0])
                        line_to_add.append(parameters[1])
                        line_to_add.append(parameters[2])

                    with open("regression_parameters.csv", "a", encoding="utf-8", newline="") as csv_fd:
                        writer = csv.writer(csv_fd)
                        writer.writerow([input_.stem + ".png", *line_to_add])
    elif input_.is_dir():
        for label_ in sorted(input_.glob("*.txt")):
            labels = {}
            with label_.open("r", encoding="utf-8") as fd:
                for line in fd:
                    class_, *coords = line.strip().split(" ")
                    coords = [Point(*c) for c in itertools.batched(map(float, coords), 2)]
                    labels[class_] = coords
                with output.joinpath(label_.stem + ".txt").open("w", encoding="utf-8") as fd2:
                    regression_curves_to_txt(labels, fd2)
                if write_csv:
                    with output.joinpath(label_.stem + ".txt").open("r", encoding="utf-8") as fd3:
                        line_to_add = []
                        files_list.append(label_.stem + ".png")
                        for line in fd3:
                            class_, *parameters = line.strip().split(" ")

                            line_to_add.append(parameters[0])
                            line_to_add.append(parameters[1])
                            line_to_add.append(parameters[2])

                        with open("regression_parameters.csv", "a", encoding="utf-8", newline="") as csv_fd:
                            writer = csv.writer(csv_fd)
                            writer.writerow([label_.stem + ".png", *line_to_add])

    print(f"CSV file created in {output}.")

    if overwrite_df:
        conversion_data = []
        with open("df_1.csv", "r", encoding="utf-8", newline="") as df_fd:
            reader = csv.reader(df_fd, delimiter=",")
            for row in reader:
                conversion_data.append(row)

        new_data = []
        with open("df_2.csv", "r", encoding="utf-8", newline="") as df_fd:
            reader = csv.reader(df_fd, delimiter=",")
            for row in reader:
                new_data.append(row)

        converted_data = deepcopy(new_data)

        for i in range(len(converted_data)):
            if i == 0:
                continue
            else:
                converted_data[i][0] = f"{conversion_data[i][0]}"
                converted_data[i][1] = f"{conversion_data[i][1]}"

        zero = "0.0"
        with open("new_df.csv", "w", encoding="utf-8", newline="") as df_fd:
            df_writer = csv.writer(df_fd, delimiter=",")
            with open("regression_parameters.csv", "r", encoding="utf-8", newline="") as csv_fd:
                regression_reader = csv.reader(csv_fd, delimiter=",")
                regression_reader = list(regression_reader)
                df_writer.writerow(
                    [
                        *new_data[0],
                        *[
                            regression_reader[0][1],
                            regression_reader[0][2],
                            regression_reader[0][3],
                            regression_reader[0][4],
                            regression_reader[0][5],
                            regression_reader[0][6],
                        ],
                    ]
                )

                j = 1
                for data_row in new_data:
                    if data_row[1] == "xiris_path":
                        continue
                    if data_row[1] in files_list:
                        for i in range(len(new_data)):
                            if regression_reader[i][0] == data_row[1]:
                                df_writer.writerow(
                                    [
                                        *converted_data[j],
                                        *[
                                            regression_reader[i][1] if len(regression_reader[i]) > 1 else zero,
                                            regression_reader[i][2] if len(regression_reader[i]) > 2 else zero,
                                            regression_reader[i][3] if len(regression_reader[i]) > 3 else zero,
                                            regression_reader[i][4] if len(regression_reader[i]) > 4 else zero,
                                            regression_reader[i][5] if len(regression_reader[i]) > 5 else zero,
                                            regression_reader[i][6] if len(regression_reader[i]) > 6 else zero,
                                        ],
                                    ]
                                )
                                break
                    j = j + 1


@cli.command(short_help="Plot label, mask or regression curves for input_ files on prediction directory.")
@click.argument(
    "input_",
    type=click.Path(
        dir_okay=True,
        exists=True,
        readable=True,
        allow_dash=False,
        path_type=Path,
    ),
)
@click.option(
    "--plot-mask",
    is_flag=True,
    default=False,
    show_default=True,
    type=bool,
    help="Plot the mask on top of the image.",
)
@click.option(
    "--plot-label",
    is_flag=True,
    default=False,
    show_default=True,
    type=bool,
    help="Plot the label on top of the image.",
)
@click.option(
    "--plot-regression",
    is_flag=True,
    default=False,
    show_default=True,
    type=bool,
    help="Plot the regression curves.",
)
@click.option(
    "--prediction-directory",
    default=Path.cwd().joinpath("runs", "segment", "predict"),
    show_default=True,
    type=click.Path(
        dir_okay=True,
        exists=False,
        readable=True,
        allow_dash=False,
        path_type=Path,
    ),
    help="Where the prediction labels are.",
)
@click.option(
    "--image-type",
    default=".png",
    show_default=True,
    type=(click.Choice([".jpg", ".jpeg", ".png"])),
    help="Input image type.",
)
@click.option(
    "--model",
    "-m",
    default=Path.cwd().joinpath("runs", "segment", "train", "weights", "best.pt"),
    show_default=True,
    type=click.Path(
        dir_okay=False,
        exists=True,
        readable=True,
        allow_dash=False,
        path_type=Path,
    ),
    help="Model location",
)
def plot(
    input_: Path,
    plot_mask: bool,
    plot_label: bool,
    plot_regression: bool,
    prediction_directory: Path,
    image_type: str,
    model: Path,
) -> None:

    x_min, x_max = 60, 400 if plot_mask else None
    y_min, y_max = 130, 350 if plot_mask else None

    if not (plot_mask or plot_label or plot_regression_curves):
        raise ValueError("No plot option chosen.")

    if input_.is_dir():
        for image in sorted(input_.glob("*" + image_type)):
            if plot_mask:
                if prediction_directory.resolve().joinpath(image.stem + ".jpg") not in sorted(
                    prediction_directory.glob("*.jpg")
                ):
                    raise ValueError(f"Prediction image {image.name} not found.")
                else:
                    plt.figure(1, figsize=(10, 10))
                    plt.title(image.name)
                    plt.xlim(x_min, x_max)
                    plt.ylim(y_min, y_max)
                    plt.gca().invert_yaxis()
                    plt.imshow(plt.imread(str(prediction_directory.resolve().joinpath(image.stem + ".jpg"))))
                    plt.show()
            if plot_label or plot_regression:
                labeler = YoloLabeler(model=model)
                labels = labeler.label(image)
                plot_labels(image, labels, title=image.name) if plot_label else None
                plot_regression_curves(labels, image) if plot_regression else None
    elif input_.is_file():
        if plot_mask:
            if prediction_directory.resolve().joinpath(input_.stem + ".jpg") not in sorted(
                prediction_directory.glob("*.jpg")
            ):
                raise ValueError(f"Prediction image {input_.name} not found.")
            else:
                plt.xlim(x_min, x_max)
                plt.ylim(y_min, y_max)
                plt.gca().invert_yaxis()
                plt.imshow(plt.imread(str(prediction_directory.resolve().joinpath(input_.stem + ".jpg"))))
                plt.show()
        if plot_label or plot_regression:
            labeler = YoloLabeler(model=model)
            labels = labeler.label(input_)
            plot_labels(input_, labels, title=input_.name) if plot_label else None
            plot_regression_curves(labels, input_) if plot_regression else None


@cli.command(short_help="Generate image labels for model training with the appropriate folder structure.")
@click.argument(
    "input_",
    type=click.Path(
        dir_okay=True,
        exists=True,
        readable=True,
        allow_dash=False,
        path_type=Path,
    ),
)
@click.option(
    "--add-manual",
    default=None,
    show_default=True,
    type=click.Path(
        dir_okay=True,
        exists=True,
        readable=True,
        allow_dash=False,
        path_type=Path,
    ),
    help="Path to folder containing training images and labels from manual labeling.",
)
@click.option(
    "--image-type",
    default=".png",
    show_default=True,
    type=(click.Choice([".jpg", ".jpeg", ".png"])),
    help="Input image type.",
)
@click.option(
    "-f",
    "--remove-existing",
    default=False,
    type=bool,
    is_flag=True,
    help="Remove existing images before generating.",
)
def generate_training_labels(input_: Path, add_manual: Path | None, image_type: str, remove_existing: bool) -> None:

    project_dir = Path.cwd()

    output_paths = prepare_dataset_and_generate_labels(input_, project_dir, remove_existing=remove_existing)

    if add_manual is not None:
        files = add_manual.iterdir()
        for file in files:
            if not file.is_file():
                continue
            if file.suffix == ".txt":
                shutil.copy(file, output_paths.label_train)
                shutil.copy(file.with_suffix(image_type), output_paths.image_train)
    print(f"Images stored in {output_paths.image_train} and labels stored in {output_paths.label_train}.")


@cli.command(
    short_help="Train YOLO model on available labels. All images and labels must be in their corresponding directories."
)
@click.option(
    "--epochs",
    default=300,
    show_default=True,
    help="Number of epochs to train the model.",
)
def train(epochs: int) -> None:
    start_time = time.time()
    project_dir = Path.cwd()

    yolo_dir = project_dir.joinpath("runs")
    if yolo_dir.exists():
        shutil.rmtree(yolo_dir)

    output_paths = OutputPaths(project_dir)
    model_weight_path = train_dataset(output_paths, epochs=epochs)

    print(f"Finished training in {(time.time() - start_time) / 60} minutes.")


def main() -> None:
    cli(max_content_width=120)
