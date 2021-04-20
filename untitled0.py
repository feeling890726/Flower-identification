import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread("C:/17flowers/bluebell/image_0263.jpg")
mask = np.zeros(img.shape[:2],np.uint8)

bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

rect = (75,75,600,600)
cv.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv.GC_INIT_WITH_RECT)

mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]

plt.imshow(img)
plt.show()
cv.imwrite("./new_img1.jpg", img)

