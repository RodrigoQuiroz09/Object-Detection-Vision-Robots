
import cv2
import time
import numpy as np

cap = cv2.VideoCapture('input/video_1.mp4')
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

a=[]

count=0;
# detect objects in each frame of the video
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        
        image = frame
        height = 600
        width = 1000
        dim = (width, height)
        res = cv2.resize(image, dim, interpolation=cv2.INTER_LINEAR)
        h, w, _ = res.shape
        blur = cv2.GaussianBlur(res, (5, 5), 0)
        gray = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Further noise removal
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        
        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)
        
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1
        
        # Now, mark the region of unknown with zero
        markers[unknown == 255] = 0
        
        markers = cv2.watershed(res, markers)
        res[markers == -1] = [255, 0, 0]

 
              

        cv2.imshow('image', res)
        out.write(blur)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
