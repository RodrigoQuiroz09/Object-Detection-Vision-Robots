# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 16:32:54 2022

@author: rodri
"""

import cv2
import time
import numpy as np
import time

cap = cv2.VideoCapture('input/video_1.mp4')

# get the video frames' width and height for proper saving of videos
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# create the `VideoWriter()` object
out = cv2.VideoWriter('output/video_result_1.mp4',
                      cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))


with open('files/YOLO/object_detection_classes_coco.txt', 'r') as f:
    class_names = f.read().split('\n')

COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))

yolo_model = cv2.dnn.readNetFromDarknet(
    'files/YOLO/yolov3.cfg', 'files/YOLO/yolov3.weights')


ln = yolo_model.getLayerNames()
ln = [ln[i-1] for i in yolo_model.getUnconnectedOutLayers()]
print(ln)

font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0
# detect objects in each frame of the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_id += 1
    image = frame
    image_height, image_width, _ = image.shape

    blob = cv2.dnn.blobFromImage(
        image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    yolo_model.setInput(blob)

    layerOutputs = yolo_model.forward(ln)

    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > 0.7:
                centerX = int(detection[0] * image_width)
                centerY = int(detection[1] * image_height)
                w = int(detection[2] * image_width)
                h = int(detection[3] * image_height)

                x = int(centerX - (w / 2))
                y = int(centerY - (h / 2))

                boxes.append([x, y, w, h])
                confidences.append(confidence)
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.7, 0.4)

    # ensure at least one detection exists
    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y, w, h = boxes[i]

            color = COLORS[classIDs[i]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = str(class_names[classIDs[i]])
            cv2.putText(image, text, (x, y - 10), font, 1, color, 2)

    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    cv2.putText(image, "FPS: " + str(round(fps, 2)),
                (40, 40), font, .7, (0, 255, 255), 1)

    cv2.imshow("YOLO", image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
