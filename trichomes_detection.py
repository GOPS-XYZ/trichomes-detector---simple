# import libraries
import numpy as np
import imutils
import cv2

# define the lower and upper boundaries of the "green" ball
greenLower = (29, 32, 6) #(29, 86, 6)
greenUpper = (64, 255, 255)

for i in range(1,4):
        # read the images
        frame = cv2.imread("trichomes- tobacco"+str(i)+".jpg")

        # resize the frame, blur it, and convert it to the HSV
        frame = imutils.resize(frame, width=600)
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # construct a mask for the color "green", then perform
        # a series of dilations and erosions to remove any small
        # blobs left in the mask
        mask = cv2.inRange(hsv, greenLower, greenUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]
        scenter = None

        # only proceed if at least one contour was found
        if len(cnts) > 0:
                # find the largest contour in the mask, then use
                # it to compute the minimum enclosing circle and
                # centroid
                for c in cnts:
                        # c = max(cnts, key=cv2.contourArea)
                        ((x, y), radius) = cv2.minEnclosingCircle(c)
                        M = cv2.moments(c)
                        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                        # only proceed if the radius meets a minimum size
                        if radius > 2 and radius < 25:
                                # draw the circle and centroid on the frame,
                                # then update the list of tracked points
                                cv2.circle(frame, (int(x), int(y)), int(radius),
                                           (0, 0, 255), 1)
                                # cv2.circle(frame, center, 5, (0, 0, 255), -1)
        # show the frame to our screen
        cv2.imshow("Frame",frame)
        cv2.imshow("Mask", mask)
			
        # Press any key of the keyboard for next image
        cv2.waitKey(0)
        

cv2.destroyAllWindows()
