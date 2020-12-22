import numpy as np
import cv2 as cv
import mtcnn
from matplotlib import pyplot as plt



img = cv.imread('Team.JPEG')
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

detect_face = mtcnn.MTCNN()

result = detect_face.detect_faces(img_rgb)
print(img_rgb.shape)

mask = np.zeros(img_rgb.shape,np.uint8)

count = 0
for res in result:
    x1, y1, w, h = res['box']
    x2, y2 = x1+w , y1+h
    
    # print("-------", res)
    
    cv.rectangle(img_rgb, (x1,y1), (x2,y2), (0, 0, 0), thickness= 2)
    
    mask[y1:y2, x1:x2, :] = 1
    count += 1
    print("---------Total face detected-----------: ",count)

img_rgb = img_rgb*mask[:,:,1,np.newaxis]
plt.imshow(img_rgb),plt.colorbar(),plt.show()