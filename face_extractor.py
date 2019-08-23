import cv2
import numpy as np
import dlib
import pickle
import os
from imutils import face_utils
from imutils.face_utils import FaceAligner
from random import shuffle, randint
import cv2
import numpy as np
from imutils import contours
from imutils import face_utils
from mask_create import create_mask

shape_predictor_68_face_landmarks = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
landmarks = dlib.shape_predictor(shape_predictor_68_face_landmarks)

cap = cv2.VideoCapture(0)
fa = FaceAligner(landmarks, desiredFaceWidth=250)

while True:
    # Getting out image by webcam
	_, image = cap.read()
	image=cv2.flip(image, 1)
	# Converting the image to gray scale
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# Get faces into webcam's image
	rects = detector(gray, 0)

	# For each detected face, find the landmark.
	if len(rects) > 0:
		shape = landmarks(gray, rects[0])
		shape = face_utils.shape_to_np(shape)
		# Make the prediction and transfom it to numpy array
# Draw on our image, all the finded cordinate points (x,y)
		mask = create_mask(shape, gray)
		masked = cv2.bitwise_and(gray, mask)
		maskAligned = fa.align(mask, gray, rects[0])
		faceAligned = fa.align(masked, gray,rects[0])
		face = detector(faceAligned,1)
		if len(face) > 0:
			faceAligned = faceAligned[face[0].top():face[0].bottom(), face[0].left():face[0].right()]
			faceAligned = cv2.resize(faceAligned, (100, 100))
		image=faceAligned
	cv2.imshow("Output", image)


	k = cv2.waitKey(1) & 0xFF
	if k == ord('q'):
	    break

cv2.destroyAllWindows()
cap.release()
