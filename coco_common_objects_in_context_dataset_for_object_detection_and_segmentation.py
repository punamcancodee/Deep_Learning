# -*- coding: utf-8 -*-
"""COCO: Common Objects in Context dataset for object detection and segmentation.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1VyQHZWAsJ_gnA_WTchwGRn9f8aLHu70F

The COCO (Common Objects in Context) dataset is a large-scale dataset designed for object detection, segmentation, and image captioning tasks. It contains over 200,000 labeled images and more than 1.5 million object instances.
"""

! pip install kaggle

from google.colab import drive
drive.mount('/content/drive')

! mkdir ~/.kaggle

! cp /content/drive/MyDrive/Kaggle_API/kaggle.json ~/.kaggle/

! chmod 600 ~/.kaggle/kaggle.json

! kaggle datasets download valentynsichkar/yolo-coco-data

ls -l

! unzip yolo-coco-data.zip

pip install opencv-python opencv-python-headless numpy

import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load COCO Names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Colors for bounding boxes
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Load an image
image_path = "path/to/your/image.jpg"
image = cv2.imread(image_path)
height, width, channels = image.shape

# Prepare the image for YOLO
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)

# Perform detection
outs = net.forward(output_layers)

# Analyze the results
class_ids = []
confidences = []
boxes = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply non-max suppression
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Draw bounding boxes
for i in indices:
    i = i[0]
    box = boxes[i]
    x, y, w, h = box[0], box[1], box[2], box[3]
    color = colors[class_ids[i]]
    label = str(classes[class_ids[i]])
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

# Display the result
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()