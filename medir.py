import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    _, threshold = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)


    kernel = np.ones((3,3),np.uint8)

    dilation = cv2.erode(threshold,kernel,iterations = 1)


    contours,_ = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    for (i,c) in enumerate(contours):
        (x,y,w,h) = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        cv2.putText(frame, "Faca um risco com 5 cm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255),2)
        cv2.putText(frame, str(w), (x+w+15, y+h+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255),1)
        

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

cincoCm = float(input("Qual o valor apareceu para os 5 cm?"))
print(f"{cincoCm/5.0} pixels para um cm")