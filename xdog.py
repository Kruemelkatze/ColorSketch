import numpy as np
import cv2

# Default Parameters
t = 71
phi = 200
epsilon = -0.1
k = 1.6
sigma = 1.6


def fixtype(im):
    if im.dtype != 'uint8':
        return im.astype(np.dtype('uint8'))
    else:
        return im


path = "data/Lenna.png"

img = cv2.imread(path, 0)

blurredSmall = cv2.GaussianBlur(img, (0, 0), sigma)
blurredLarge = cv2.GaussianBlur(img, (0, 0), sigma * k)

# DOG
dog = blurredSmall - blurredLarge
xDog = blurredSmall - t * blurredLarge # Variant 1
xDog2 = (1 - t) * blurredSmall + t * blurredLarge # Variant 2, delivers very strange results

# xDog = fixtype(xDog)
# cv2.imshow("dog", dog)
cv2.imshow("xDog", xDog)
cv2.imshow("xDog2", xDog2)

# dog = fixtype(dog)
# cv2.imshow("dog", dog)

# Multiply
xDog = xDog2
xDogMult = xDog * img

# xDog Treshholded
xDogTresh = np.ones(xDogMult.shape)  # Fill with ones, for the case u >= e
# ones = dog >= epsilon
# xDog[ones] = 1
calc = dog < epsilon  # Logical Array
xDogTresh[calc] = 1 + np.tanh(phi * (xDogMult[calc] - epsilon))

# for i in range(0, len(dog)):
#     for j in range(0, len(dog[i])):
#         if dogt[i, j] < epsilon:
#             xDog[i, j] = 1
#         else:
#             xDog[i, j] = 1 + np.tanh(phi * (dogt[i, j]))

meanValue = np.mean(xDogTresh)
dark = xDogTresh <= meanValue  # Logical Array
bright = xDogTresh < meanValue  # Logical Array
xDogTresh[dark] = 0
xDogTresh[bright] = 255


# xDog = fixtype(xDog)
cv2.imshow("xDogTresh", xDogTresh)
cv2.waitKey(0)
cv2.destroyAllWindows()
