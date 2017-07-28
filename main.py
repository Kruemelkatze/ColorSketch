import numpy as np
import cv2
import pymeanshift as pms

filename = 'data/girl.png'

showAllImages = False

# Parameters
outlinesOnly = False  # instead of ink patches, draw outlines (for good results, adapt grayscaleMethod)
useSegmentedImage = False  # use segmented image as basis for inking

grayscaleMethod = 1  # 0: default, 1: MaxDecomposition, 2: desaturation
increaseContrast = False

# Some Fine-tuning parameters, most of them are defined inline
# Outlines only
dogSigma = 2  # the smaller gaussian kernel size for DoG
dogK = 1.6  # small/large gaussian kernel size for DoG
# Bilateral filter for inking
d = 50
sigmaColor = 75
sigmaSpace = 200
# Morphing, Post-processing
openingDiameter = 6


def rgb2gray(img):
    if grayscaleMethod == 2:
        # Desaturation
        sums = np.max(img, axis=2).astype(np.dtype('uint16')) + np.min(img, axis=2)
        return (sums / 2).astype(np.dtype('uint8'))
    if grayscaleMethod == 1:
        # MaxDecomposition
        return np.max(img, axis=2)
    else:
        # Default
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def segmentImage(img, path):
    print("Preprocessing...")
    pre = cv2.medianBlur(img, 5)
    pre = cv2.pyrMeanShiftFiltering(pre, 15, 25)
    if showAllImages:
        cv2.imshow("Pre-Segmentation", img)

    print("Segmenting...")
    (spatial, range, density) = (8, 8, 250)
    (segmented_image, labels_image, number_regions) = pms.segment(pre, spatial_radius=spatial,
                                                                  range_radius=range, min_density=density)
    segmented_image = cv2.medianBlur(segmented_image, 7)
    cv2.imwrite(path, segmented_image)
    return segmented_image


split = filename.split('.')
segmentedImagePath = split[0] + '_segmented.' + ".".join(split[1:])

print("Reading...")
img = cv2.imread(filename)
if img is None:
    print("File not found! Exiting...")
    exit(1)

print("Checking for existing segmented image...")
segmented_image = cv2.imread(segmentedImagePath)
if segmented_image is None:
    print("No existing segmented image --> Creating one")
    segmented_image = segmentImage(img, segmentedImagePath)

print("GrayScaling...")
if useSegmentedImage:
    imgGray = rgb2gray(segmented_image)
else:
    imgGray = rgb2gray(img)

cv2.imshow('Original', img)
cv2.imshow("Segmented", segmented_image)
if showAllImages:
    cv2.imshow('Grayscale', imgGray)

# Begin INKING
# Contrast
if increaseContrast:
    print("Increasing Contrast...")
    imgGray = cv2.equalizeHist(imgGray)
    if showAllImages:
        cv2.imshow('Increased Contrast', imgGray)

print("Bilateral Filter...")
bilat = cv2.bilateralFilter(imgGray, 9, sigmaColor, sigmaSpace)
if showAllImages:
    cv2.imshow('Bilateral filtered', bilat)

if not outlinesOnly:
    print("Adaptive Thresholding...")
    thresholded = cv2.adaptiveThreshold(bilat, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 115, 1)
    # thValue, thresholded = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # Variant for Otsu's Method
    if showAllImages:
        cv2.imshow('Adaptive threshold', thresholded)

    print("Morphing...")
    morph = thresholded
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (openingDiameter, openingDiameter))
    # Some variations...
    # morph = cv2.morphologyEx(morph, cv2.MORPH_GRADIENT, kernel)
    # morph = cv2.dilate(morph,kernel,iterations = 1)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
    inked = morph
    if showAllImages:
        cv2.imshow('Morphologically transformed', morph)
else:  # Outlines only: DoG
    print("Outlines with DoG...")
    blurredSmall = cv2.GaussianBlur(bilat, (0, 0), dogSigma)
    blurredLarge = cv2.GaussianBlur(bilat, (0, 0), dogSigma * dogK)
    dog = np.invert((blurredSmall - blurredLarge))
    inked = dog
    thValue, inked = cv2.threshold(inked, 200, 255, cv2.THRESH_BINARY)

    if showAllImages:
        cv2.imshow('Difference of Gaussians', dog)

print("Postprocessing...")
post = cv2.medianBlur(inked, 7)
if showAllImages:
    cv2.imshow('Postprocessed', post)

print("Combining...")
coloredInked = cv2.bitwise_and(segmented_image, segmented_image, mask=post)
cv2.imshow('End Result', coloredInked)

print("Done :-)")
cv2.waitKey(0)
cv2.destroyAllWindows()
