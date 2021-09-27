import cv2
import numpy as np
#C:\\Users\\SrinivasKhatravath\\Downloads\\vest_test_6.webm
net = cv2.dnn.readNet('C:/Users/SowmyaSri/Downloads/yolo_person_car.weights', 'C:/Users/SowmyaSri/Desktop/yolo_test/yolov3_cfg_2class.cfg')
out = cv2.VideoWriter('zoom.avi',cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (596,336))
classes = []
with open("classes.txt", "r") as f:
    classes = f.read().splitlines()
    
#path=0
#path="C:/Users/SowmyaSri/Downloads/mgccombokiosk/Recording.avi"
path="C:/Users/SowmyaSri/Downloads/test_e1.webm"
cap = cv2.VideoCapture(path)
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(100, 3))

while True:
    _, img = cap.read()
    img = cv2.resize(img, (596,336))
    height, width, _ = img.shape

    blob = cv2.dnn.blobFromImage(img, 1/255, (916, 916), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []
    #print("Hi")
    for output in layerOutputs:
        for detection in output:
            #print("inside")
            scores = detection[5:]
            #print(scores)
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            #print(confidence)
            if confidence > 0.4:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

    if len(indexes)>0:
        #print('hello')
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
	
            #print(label)
            confidence = str(round(confidences[i],2))
            #color = colors[i]
            if label =="car" or label =="person" :
            	
                print(label,confidence)
                cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
                # cv2.putText(img, label, (x-10, y -10), font, 3, (0,0,255), 3)
                cv2.putText(img, label+" "+confidence, (x-10, y-10), font, 2, (0,0,255), 2)
    out.write(img)
    cv2.imshow('Image', img)
    key = cv2.waitKey(1)
    if key==27:
        break
 	
cap.release()
cv2.destroyAllWindows()
