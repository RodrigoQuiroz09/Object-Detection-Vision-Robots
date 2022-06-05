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

def yoloPreparation(detection_counters):
    """YOLO model, as well as the class_names for object detection, are loaded and prepared"""
    with open('files/YOLO/object_detection_classes_coco.txt', 'r') as f:
        class_names = f.read().split('\n')
        
    for object_class in class_names:
        detection_counters[str(object_class)] = 0
        
    COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))
    
    model = cv2.dnn.readNetFromDarknet('files/YOLO/yolov3.cfg', 'files/YOLO/yolov3.weights')

    cap = cv2.VideoCapture(command_args[0])
    return (class_names, COLORS, model, cap)

def objectDetection(class_names, COLORS, model, cap):
    """Object detection is implemented with this method"""
    starting_time = time.time()
    frame_id = 0
    output_created = False
    output = None
    ln = model.getLayerNames()
    ln = [ln[i-1] for i in model.getUnconnectedOutLayers()]
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        image = frame
        image_height, image_width, _ = image.shape

        blob = cv2.dnn.blobFromImage(
            image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        model.setInput(blob)
        layerOutputs = model.forward(ln)
        boxes = []
        confidences = []
        classIDs = []
        for output in layerOutputs:
            for detection in output:
                if not output_created: 
                    output_t = cv2.VideoWriter(f'output/{command_args[1]}', cv2.VideoWriter_fourcc(*'mp4v'), 30, (image_width,  image_height))
                    output_created = True
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > command_args[2]:
                    centerX = int(detection[0] * image_width)
                    centerY = int(detection[1] * image_height)
                    w = int(detection[2] * image_width)
                    h = int(detection[3] * image_height)

                    x = int(centerX - (w / 2))
                    y = int(centerY - (h / 2))
                    
                
                    boxes.append([x, y, w, h])
                    confidences.append(confidence)
                    classIDs.append(classID)
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, command_args[2], 0.4)
        if len(idxs) > 0:
            for i in idxs.flatten():
                x, y, w, h = boxes[i]
    
                color = COLORS[classIDs[i]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                text = str(class_names[classIDs[i]] + " " + str(round(confidences[i], 2)))
                detection_counters[class_names[classIDs[i]]] +=1
                cv2.putText(image, text, (x, y - 10), FONT, 1, color, 2)
                
        elapsed_time = time.time() - starting_time
        fps = frame_id / elapsed_time
        cv2.putText(image, "FPS: " + str(round(fps, 2)),(40, 40), FONT, .7, (0, 255, 255), 1)
        cv2.imshow("YOLO", image)
        output_t.write(frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    output_t.release()
    print(f'Analyzed frames: {frame_id}')
    print("Objects identified, with the following format: [class, # of apperances, percentage of apperances]")
    for key in detection_counters.keys():
        if detection_counters[key] != 0: 
            filtered_detection_countrs.append([key, detection_counters[key], round((detection_counters[key]/frame_id)*100, 2)])

def executeProgram():
    """Single function that needs to be executed in order to perfomr the object detection"""
    parser = argparse.ArgumentParser(description="Object detection program with Tensorflow model")
    parser.add_argument("-i", help="Defines the video to be analyzed")
    parser.add_argument("-o", help="Defines the name of the output video. Default value is output.mp4", default="output.mp4")
    parser.add_argument("-c", help="Defines the confidence threshold for the detection model. Default value is 0.7", type=float, default=0.7)
    args = parser.parse_args()
    global command_args
    command_args = [args.i, args.o, args.c]
    parametersErrorCheck()
    class_names, COLORS, model, cap = yoloPreparation(detection_counters)
    objectDetection(class_names, COLORS, model, cap)
    print(filtered_detection_countrs)
    cap.release() 
    cv2.destroyAllWindows()

executeProgram()