import cv2
import time
import numpy as np
import argparse
import sys
import os

detection_counters = dict()
filtered_detection_countrs = list()
FONT = cv2.FONT_HERSHEY_SIMPLEX
command_args = []

def parametersErrorCheck():
    """Errors regarding the console parameters are checked with this method"""
    if command_args[0] == None: 
        print("No input video was introduced")
        sys.exit()
    if not os.path.exists(command_args[0]):
        print("Input video filename has not been found")
        sys.exit()
    if 0 > command_args[2] or command_args[2] > 1: 
        print("Confidence level should be between 0 and 1")
        sys.exit()

def caffePreparation(detection_counters):
    """Caffe model, as well as the class_names for object detection, are loaded and prepared"""
    with open('files/Caffe/object_detection_classes_coco.txt', 'r') as f:
        class_names = f.read().split('\n')
    
    for object_class in class_names:
        detection_counters[str(object_class)] = 0
        
    COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))
    
    model = cv2.dnn.readNet(model='files/Caffe/VGG.caffemodel', config='files/Caffe/VGG.prototxt', framework='Caffe')

    cap = cv2.VideoCapture(command_args[0])
    return (class_names, COLORS, model, cap)

def detectionActions(detection, class_names, COLORS, res, w, h, confidence):
    """All actions regarding the detected objects are handled in this method"""
    class_id = detection[1] # get the class id
    class_name = class_names[int(class_id)-1] # map the class id to the class
    color = COLORS[int(class_id)]
    box_x = detection[3] * w # get the bounding box coordinates (x)
    box_y = detection[4] * h # get the bounding box coordinates (y)
    box_width = detection[5] * w # get the bounding box width
    box_height = detection[6] * h # get the bounding box height
    box_text = class_name + " " + str(round(confidence, 2))
    detection_counters[class_name] +=1
    cv2.rectangle(res, (int(box_x), int(box_y)), (int(box_width), int(box_height)), color, thickness=2) # draw a rectangle around each detected object
    cv2.putText(res, box_text, (int(box_x), int(box_y - 5)), FONT, 1, color, 2)

def objectDetection(class_names, COLORS, model, cap):
    """Object detection is implemented with this method"""
    starting_time = time.time()
    frame_id = 0
    output_created = False
    output = None
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame_id += 1
            image = frame

            height = 600
            width = 1000
            dim = (width, height)
            res = cv2.resize(image, dim, interpolation=cv2.INTER_LINEAR)
            h, w, _ = res.shape
            if not output_created: 
                output_t = cv2.VideoWriter(f'output/{command_args[1]}', cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
                output_created = True

            blob = cv2.dnn.blobFromImage(image=image,scalefactor=1, size=(300, 300), mean=(106, 117,123 ))
            model.setInput(blob)
            output = model.forward()
            
            for detection in output[0, 0, :, :]:
                confidence = detection[2]
                if confidence > command_args[2]:
                   detectionActions(detection, class_names, COLORS, res, w, h, confidence)

            elapsed_time = time.time() - starting_time
            fps = frame_id / elapsed_time
            cv2.putText(res, "FPS: " + str(round(fps, 2)),
                        (40, 40), FONT, .7, (0, 255, 255), 1)
            cv2.imshow('Caffe', res)
            output_t.write(res)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        else:
            break
    output_t.release()
    print(f'Analyzed frames: {frame_id}')
    print("Objects identified, with the following format: [class, # of apperances, percentage of apperances]")
    for key in detection_counters.keys():
        if detection_counters[key] != 0: 
            filtered_detection_countrs.append([key, detection_counters[key], round((detection_counters[key]/frame_id)*100, 2)])

def executeProgram():
    """Single function that needs to be executed in order to perfomr the object detection"""
    parser = argparse.ArgumentParser(description="Object detection program with Caffe model")
    parser.add_argument("-i", help="Defines the video to be analyzed")
    parser.add_argument("-o", help="Defines the name of the output video. Default value is output.mp4", default="output.mp4")
    parser.add_argument("-c", help="Defines the confidence threshold for the detection model. Default value is 0.35", type=float, default=0.35)
    args = parser.parse_args()
    global command_args
    command_args = [args.i, args.o, args.c]
    parametersErrorCheck()
    class_names, COLORS, model, cap = caffePreparation(detection_counters)
    objectDetection(class_names, COLORS, model, cap)
    print(filtered_detection_countrs)
    cap.release()
    cv2.destroyAllWindows()
    
executeProgram()