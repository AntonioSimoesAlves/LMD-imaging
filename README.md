Running the program will fetch the images in /dataset/, create segmentation masks according to predefined melting and maximum temperatures.

The masks are then split into training and testing folders inside the /data/ directory.

YOLOv8n-seg is then run to train and validate the model on said images.
