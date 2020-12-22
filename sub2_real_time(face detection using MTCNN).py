import numpy as np
import cv2 as cv
import mtcnn
from matplotlib import pyplot as plt


detect_face = mtcnn.MTCNN()
cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    result = detect_face.detect_faces(frame_rgb)
    print(frame_rgb.shape)

    mask = np.zeros(frame_rgb.shape,np.uint8)


    for res in result:
        x1, y1, w, h = res['box']
        x2, y2 = x1+w , y1+h
        
        print("-------", res)
        
        cv.rectangle(frame_rgb, (x1,y1), (x2,y2), (0, 0, 255), thickness= 2)
        
        mask[y1:y2, x1:x2, : ] = 1
        
        frame_rgb = frame*mask[:,:,1,np.newaxis]
        
        cv.imshow("Frame", frame_rgb)
        
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    
    

    
