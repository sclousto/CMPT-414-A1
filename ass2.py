import cv2
import numpy as np
import sys
from matplotlib import pyplot as plot
from scipy.ndimage.filters import sobel

img = cv2.imread('letters.png', 0)
refImg = cv2.imread('template_K.png', 0)


#image edges
test = cv2.medianBlur(img, 5)
test = cv2.GaussianBlur(test,(5,5),0)
test = cv2.medianBlur(test, 5)
test = cv2.medianBlur(test, 5)
test = cv2.medianBlur(test, 3)
test = cv2.medianBlur(test, 3)
test = cv2.Canny(img,20,100)
sobelx = cv2.Sobel(test,cv2.CV_64F,1,0,ksize=5)
sobelx = np.absolute(sobelx)
sobely = cv2.Sobel(test,cv2.CV_64F,0,1,ksize=5)
sobely = np.absolute(sobely)
#sobelx = sobel(test, axis=0, mode='constant')
#sobely = sobel(test, axis=1, mode='constant')
gradient1 = np.arctan(sobelx, sobely) * 180 / np.pi
#gradient1 = cv2.Laplacian(img,cv2.CV_64F)

#template edges
template = cv2.medianBlur(refImg, 5)
template = cv2.medianBlur(template, 5)
template = cv2.medianBlur(template, 5)
template = cv2.medianBlur(template, 5)
template = cv2.Canny(template,20,70)
templatex = cv2.Sobel(template,cv2.CV_64F,1,0,ksize=5)
templatex = np.absolute(templatex)
templatey = cv2.Sobel(template,cv2.CV_64F,0,1,ksize=5)
templatey = np.absolute(templatey)
#templatex = sobel(template, axis=0, mode='constant')
#templatey = sobel(template, axis=1, mode='constant')
gradient2 = np.arctan(templatex, templatey) * 180 / np.pi
#gradient2 = cv2.Laplacian(img,cv2.CV_64F)

refPointx = int(refImg.shape[0]/2)
refPointy = int(refImg.shape[1]/2)

#rTable
rTable = {}

angle = 0
angle = angle*180/np.pi

scale = 1

for(i,j), x in np.ndenumerate(template):
    if x:
        if not gradient2[i,j] in rTable.keys():
            rTable[gradient2[i,j]] = []
        dx = i - refPointx
        dy = j - refPointy
        xc = (np.cos(angle)*dx - np.sin(angle)*dy)*scale
        yc = (np.sin(angle)*dx + np.cos(angle)*dy)*scale
        rTable[gradient2[i,j]].append((xc,yc))

#voting                                          
acc = np.zeros(img.shape)
for(i,j), x in np.ndenumerate(img):
    if x:
        if gradient1[i,j] in rTable.keys():
            for r in rTable[gradient1[i,j]]: 
                        accI = i - r[0]
                        accJ = j - r[1]
                        #accI = i-(np.cos(angle)*dx - np.sin(angle)*dy)*scale
                        #accJ = j-(np.sin(angle)*dx + np.cos(angle)*dy)*scale
                        if accI < img.shape[0] and accJ < img.shape[1]:
                            acc[int(accI), int(accJ)] += 1

max1 = acc.max()
x, y = np.where(acc == max1)
print(x)
print(y)

plot.gray()

plot.subplot(2,2,1), plot.imshow(acc)

plot.subplot(2,2,2), plot.imshow(img)

plot.scatter(y, x, marker='o', color='r')

plot.subplot(2,2,3), plot.imshow(test)

plot.subplot(2,2,4), plot.imshow(template)

plot.show()

np.set_printoptions(threshold=sys.maxsize)
print(acc)
