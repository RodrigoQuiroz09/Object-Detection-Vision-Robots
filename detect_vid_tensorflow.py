import cv2
import time
import numpy as np


# ==============================TensorFlowModel============================
with open('files/Tensorflow/object_detection_classes_coco.txt', 'r') as f:
    class_names = f.read().split('\n')

COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))

model = cv2.dnn.readNet(model='files/Tensorflow/frozen_inference_graph.pb',
                        config='files/Tensorflow/ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt',
                        framework='TensorFlow')

cap = cv2.VideoCapture('input/video_3.mp4')

# ==============================VideoCapture===============================

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('output/video_result_1.mp4',
                      cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))
count = 0
font = cv2.FONT_HERSHEY_SIMPLEX
starting_time = time.time()
frame_id = 0
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        frame_id += 1
        image = frame
        image_height, image_width, _ = image.shape
        ratio = image_height/image_width

        height = int(ratio*1000)
        width = 1000
        dim = (width, height)
        res = cv2.resize(image, dim, interpolation=cv2.INTER_LINEAR)
        h, w, _ = res.shape

        blob = cv2.dnn.blobFromImage(
            image=res, size=(300, 300), mean=(106, 115, 124))
        model.setInput(blob)
        output = model.forward()

        for detection in output[0, 0, :, :]:
            if count == 0:
                print(detection)
            count = count+1
            confidence = detection[2]
            if confidence > .25:
                # get the class id
                class_id = detection[1]
                # map the class id to the class
                class_name = class_names[int(class_id)-1]
                color = COLORS[int(class_id)]
                # get the bounding box coordinates
                box_x = detection[3] * w
                box_y = detection[4] * h
                # get the bounding box width and height
                box_width = detection[5] * w
                box_height = detection[6] * h
                # draw a rectangle around each detected object
                cv2.rectangle(res, (int(box_x), int(box_y)), (int(
                    box_width), int(box_height)), color, thickness=2)
                # put the FPS text on top of the frame
                cv2.putText(res, class_name, (int(box_x), int(
                    box_y - 5)), font, 1, color, 2)

        elapsed_time = time.time() - starting_time
        fps = frame_id / elapsed_time
        cv2.putText(res, "FPS: " + str(round(fps, 2)),
                    (40, 40), font, .7, (0, 255, 255), 1)
        cv2.imshow('Tensorflow', res)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
