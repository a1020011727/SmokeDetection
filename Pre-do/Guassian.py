import cv2
import numpy as np

im = cv2.imread('../test_data/113.jpeg')
# im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
guass_blur = cv2.GaussianBlur(im,(3,3),0)
salt_blur = cv2.medianBlur(guass_blur,3)
cv2.imwrite("de-noised.png",guass_blur)
cv2.imshow("img",im)
cv2.waitKey(0)