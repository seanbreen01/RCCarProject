import numpy as np
import cv2
from cv2 import cuda
import matplotlib.pylab as plt
import time

start_time = time.time()

img = cv2.imread('20231023_172839.jpg')

grayImg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

imgMat = cv2.cuda_GpuMat(grayImg)
detector = cv2.cuda.createCannyEdgeDetector(low_thresh=100, high_thresh=110)
dstImg = detector.detect(imgMat)
canny = dstImg.download()

plt.subplot(121), plt.imshow(grayImg, cmap = 'gray'), plt.title('Original grey'), plt.axis('off')

plt.subplot(122), plt.imshow(canny, cmap = 'gray'), plt.title('Cuda Canny'), plt.axis('off')

plt.show()    



end_time = time.time()

print('Total time = ' + str(end_time - start_time))