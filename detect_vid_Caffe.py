# -*- coding: utf-8 -*-
"""
Created on Tue May 31 16:57:08 2022

@author: rodri
"""

import cv2
import time
import numpy as np

cap = cv2.VideoCapture('input/video_3.mp4')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('output/video_result_1.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))


with open('files/Caffe/object_detection_classes_coco.txt', 'r') as f:
    class_names = f.read().split('\n')
    
#class_names = [name.split(',')[0] for name in image_net_names]
labels = ["background", "aeroplane", "bicycle", "bird", 
"boat","bottle", "bus", "car", "cat", "chair", "cow", 
"diningtable","dog", "horse", "motorbike", "person", "pottedplant", 
"sheep","sofa", "train", "tvmonitor"]

COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))
#COLORS = np.random.uniform(0, 255, size=(len(labels), 3))
# load the neural network model
model = cv2.dnn.readNet(model='files/Caffe/VGG.caffemodel', config='files/Caffe/VGG.prototxt', framework='Caffe')

# cv2.dnn.readNet(model='files/Caffe/VGG.caffemodel', config='files/Caffe/VGG.prototxt', framework='Caffe')

count=0;
# detect objects in each frame of the video
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        
        image = frame

        height = int( 600)
        width = 1000
        dim = (width, height)
        image = cv2.resize(image, dim,interpolation=cv2.INTER_LINEAR)
        h, w, _ = image.shape
        # create blob from image
        blob = cv2.dnn.blobFromImage(image=image,scalefactor=1, size=(300, 300), mean=(106, 117,123 ))
        model.setInput(blob)
        output = model.forward()
        
            #blobFromImage(image=image, scalefactor=0.01, size=(224, 224), mean=(123, 117, 106))
        
        for detection in output[0, 0,:,:]:
            if count==0:
                print(detection)
            count=count+1
            confidence=detection[2]
            if confidence > 0.8:
          
                class_id =int( detection[1])

                class_name = class_names[int(class_id)-1]
                color = COLORS[int(class_id)]
                
                box = detection[3:7] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
            
                label = "{}: {:.2f}%".format(class_name, confidence * 100)
                cv2.rectangle(image, (centerX, centerY), (width, height), color, 2)
    
                y = centerY - 15 if centerY - 15 > 15 else centerY + 15
                cv2.putText(image, label, (centerX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
              

        cv2.imshow('image', image)
        out.write(image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()