# Disclaimer
This program makes use of UV, Python package and project manager (https://docs.astral.sh/uv/).

After succesfully installing UV, to run the program, simply type `uv run lmd-imaging` in the project directory.

# Normal operation
Running the program will fetch the images in /dataset/, create segmentation masks according to predefined melting and maximum temperatures.

The masks are then split into training and testing folders inside the /data/ directory.

YOLOv8n-seg will then train and validate the model on said images.

Upon succesful completion of the model, the directory with the model data will open automatically.
