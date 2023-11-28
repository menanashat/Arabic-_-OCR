# import the necessary packages
from transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils

def scan(path,side):
        # load the image and compute the ratio of the old height
        # to the new height, clone it, and resize it
        image = cv2.imread(path)
        ratio = image.shape[0] / 500.0
        orig = image.copy()
        image = imutils.resize(image, height = 500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use Sobel edge detector
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        edged = np.uint8(np.sqrt(sobelx**2 + sobely**2))

        # find the contours in the edged image, keeping only the
        # largest ones, and initialize the screen contour
        cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

        # loop over the contours
        for c in cnts:
                # approximate the contour
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)

                # if our approximated contour has four points, then we
                # can assume that we have found our screen
                if len(approx) == 4:
                        screenCnt = approx
                        break

        if screenCnt is None:
                print("No valid contour found")
        # Handle this case as needed (e.g., return from the function, show an error message)
                return None                
        # apply the four point transform to obtain a top-down
        # view of the original image
        warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

        # show the original and scanned images
        newimg = cv2.resize(warped,(1000,630))
        if(side=="front"):
            cv2.imwrite("temp_front.jpg",newimg)
        elif(side=="back"):
            cv2.imwrite("temp_back.jpg",newimg)
        return newimg
