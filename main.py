from types import LambdaType
import cv2
import time
import numpy as np

COLORS = [(0,255,255), (255,255,0), (0,255,0), (255,0,0)]

pixelsPara1Cm = 15

class_names = []
with open("coco.names", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

cap = cv2.VideoCapture("video.mp4")

net = cv2.dnn.readNet("custom-yolov4-tiny-detector.cfg","custom-yolov4-tiny-detector_best.weights")

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416,416), scale=1/255)

while True:
    _, frame = cap.read()
    start = time.time()

    classes, scores, boxes = model.detect(frame, 0.1, 0.2)

    end = time.time()

    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]

        label = f"{class_names[classid]} : {score}"

        cv2.rectangle(frame, box, color, 2)
        h, w = (box[2]-box[0])/pixelsPara1Cm, (box[3]-box[1])/pixelsPara1Cm
        cv2.putText(frame, f"{round(h, 2)}x{round(w, 2)} cm", (box[0], box[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    fps_label = f"FPS: {round(1.0/(end-start), 2)}"

    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,0), 5)
    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,0), 3)

    cv2.imshow("detections", frame)

    if cv2.waitKey(1) == 27:
        break
cap.realease()
cv2.destroyAllWindows()
print("Rodou")