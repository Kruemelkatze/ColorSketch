import numpy as np;
import cv2;
import pymeanshift as pms

filename = 'data/beach.jpg'

split = filename.split('.')
segmented = split[0] + '_segmented.' + ".".join(split[1:])

segmented_image = cv2.imread(segmented)
if segmented_image is None:
    img = cv2.imread(filename)
    print("Preprocessing")
    img = cv2.medianBlur(img, 5)
    img = cv2.pyrMeanShiftFiltering(img, 21, 51)

    print("Segmenting...")
    (spatial, range, density) = (10, 10, 450)
    (segmented_image, labels_image, number_regions) = pms.segment(img, spatial_radius=spatial,
                                                                  range_radius=range, min_density=density)
    cv2.imwrite(segmented, segmented_image)

cv2.imshow('segmented', segmented_image)
# cv2.destroyAllWindows()

# print("Contours...")
# imgray = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
# ret,thresh = cv2.threshold(imgray,127,255,0)
# _, contours, _ = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(segmented_image,contours,-1,(0,255,0),3)
# cv2.imshow('contours', segmented_image)


print("Opening...")
morph = segmented_image
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
# morph = cv2.erode(morph,kernel,iterations = 1)
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))

# morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

# morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)


cv2.imshow('morphed', morph)

cv2.waitKey(0)
cv2.destroyAllWindows()
