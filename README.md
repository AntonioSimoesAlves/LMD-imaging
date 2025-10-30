# Laser Metal Deposition

This is a Python application that, with the aid of machine learning, identifies melt pool regions in a given dataset.

This is an academic project.

## Requirements

* Python 3.12

### Development requirements

* [uv](https://docs.astral.sh/uv/)

## Setup

### Instalation

```pwsh
uv sync
```

### Running

```pwsh
uv run lmd_imaging
```

Running the program will display all the commands that can be executed through the command line, alongside a small
description for each.

Running each command followed by ```--help``` will show all the options for each command and what they do.

## Command Documentation

### ```generate-training-labels```

Generates training labels for each image passed as input. The input can either be a single image or a directory of
images.

Once completed this command creates a ```data``` directory in the root folder with the correct format demanded by YOLO.
Once finished, the validation images and labels must also be placed **manually** in the corresponding ```val```
directories.

By default, only automatic labeling is performed utilizing this script. To add manually labeled images to the training
dataset use ```--add-manual``` followed by the directory path that contains both the images and labels inside (no
sub-directories).

```--image-type``` is a string that specifies the used images' type. The default is ```".png"```. The other available
options are ```".jpg"``` and ```".jpeg"```.

If this command is executed while ```data``` already exists, the command will not execute to avoid deleting the existing
images and labels. To bypass this, use ```-f``` or ```--remove-existing```.

### ```train```

Trains a YOLO segmentation model based on the training and validation images and labels already present in ```data```.

```--epochs``` is used to specify how many epochs to use for training. If no value is given, it defaults to 300.

The output will be located in ```\runs\segment\train``` in the root directory of the project. It overwrites previously
trained models so it's important to back those up when needed.

### ```prediction```

Generates segmentation masks for the input images using a YOLO model. The input can either be a single image or a
directory of images.

To specify which model to use, the option ```-m``` or ```--model``` takes a Path to the desired model. When omitted, the
default model used is located on ```\runs\segment\train\weights\best.pt``` in the root directory.

To avoid compatibility problems, the default device used is the CPU. To change this, used ```--device 0``` to use the
GPU.

The output path for the segmentation masks is ```\runs\segment``` but can be changed by passing a different Path to
```--run-name```.

### ```plot```

Plots mask, label or regression curve on top of the input images. The input can either be a single image or a
directory of images.

```--plot-mask``` plots the images located in the Path defined by ```--prediction-directory``` which defaults to
```runs\segment\predict```. The input image's stem (name with no suffix) must match the stem inside the ```predict```
directory, otherwise the command will raise a ValueError.

```--plot-label``` uses YOLO to generate a label for each input image and then plot it. For this reason, an already
trained YOLO model must be provided via the ```-m``` or ```--model``` option. If not given, the default is
```\runs\segment\train\weights\best.pt```.

```--plot-regression``` uses YOLO to generate a label but only plots the corresponding regression curves on top of the
input images. Similarly to before, a YOLO model must be provided with the default remaining the same as for
```--plot-label```.

Lastly, ```--image-type``` specifies the input images' file type, with the default being ```".png"```. The other
available options are ```".jpg"``` and ```".jpeg"```.

### ```yolo-label```

Generates label text files using a YOLO segmentation model for the given input. The input can either be a single image
or a directory of images.

Just as before, ```-m``` / ```--model```, ```-d``` / ```--device``` and ```--image-type``` are used to specify the
model, device and image type, respectively.

If no output is chosen via ```--output```, the default is ```\yolo_labels```.

To change the verbose mode, use ```--verbose``` followed by the desired output.

- ```--verbose none``` disables verbose and is the default if no command is given.
- ```--verbose small``` provides an update every time 100 images are labelled.
- ```--verbose full``` enables full YOLO verbose for every detection.

If the number of images in the input is less than 100, verbose is automatically set to ```none```.

### ```extract-regression-parameters```

Generates text files for the input labels provided using a YOLO segmentation model. The input can either be a single
text file or a directory of text files. Ideally used in conjunction with ```generate-training-labels``` or
```yolo-label``` as both reproduce the desired format for this command to work.

If no output is provided via ```--output``` the default is ```\runs\segment\predict\regression_parameters```

Using ```--write-csv``` writes the regression parameters for both mushy and liquid regions, as well the name of each
image.

```--overwrite-df``` is used to overwrite an existing dataframe called ```new_df.csv``` for it to be used in
EfficientNetV2. Requires ```df_1.csv``` and ```df_2.csv```.

## Intended Execution

**1.** The training labels should be generated using ```generate-training-labels```. This will create the required
subdirectories for YOLO. Once finished, the validation images and labels should be placed inside the created ```data```
directory appropriately.

**2.** The model can now be trained using the ```train``` command, if necessary, passing the ```-epochs``` command to
specify the desired number.

**3.** For visualization, using the ```prediction``` command will generate YOLO segmentation masks on the
specified output. Alternatively, using the ```plot``` command will also allow visualization of these masks, but now
inside Matplotlib. Additionally, the labels and regression curves can also be plotted using this command.

**4.** Generating the YOLO labels should be done with ```yolo-label```. This will ensure that the labels are in the
correct format for the last command. Ideally done with the full dataset.

**5.** Using ```extract-regression-parameters``` allows the export of the regression parameters to a CSV file.
