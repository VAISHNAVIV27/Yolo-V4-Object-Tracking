# Yolo-V4-Object-Tracking

Object tracking implemented with YOLOv4, DeepSort, and TensorFlow. YOLOv4 is a state of the art algorithm that uses deep convolutional neural networks to perform object detections. We can take the output of YOLOv4 feed these object detections into Deep SORT (Simple Online and Realtime Tracking with a Deep Association Metric) in order to create a highly accurate object tracker on low computing devices (Quad core CPU) 


# Getting Started
To get started, install the proper dependencies either via Anaconda or Pip.

# Pip
(TensorFlow 2 packages require a pip version >19.0.)

`TensorFlow CPU
pip install -r requirements.txt
`


# Downloading Official YOLOv4 Pre-trained Weights

Our object tracker uses YOLOv4 to make the object detections, which deep sort then uses to track. There exists an official pre-trained YOLOv4 object detector model that is able to detect 80 classes. For easy demo purposes we will use the pre-trained weights for our tracker. Download pre-trained yolov4.weights file:
Here we are using yolov4-tiny.weights, a smaller model that is faster at running detections on low computing systems but less accurate, download file here: https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights

Running the Tracker with YOLOv4-Tiny
The following commands will allow you to run yolov4-tiny model. Yolov4-tiny allows you to obtain a higher speed (FPS) for the tracker at a slight cost to accuracy. Make sure that you have downloaded the tiny weights file and added it to the 'data' folder in order for commands to work!

# save yolov4-tiny model
`python save_model.py --weights ./data/yolov4-tiny.weights --output ./checkpoints/yolov4-tiny-416 --model yolov4 --tiny`

# Run yolov4-tiny object tracker
`python object_tracker.py --weights ./checkpoints/yolov4-tiny-416 --model yolov4 --video ./data/video/test.mp4 --output ./outputs/tiny.avi --tiny`



