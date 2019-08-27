import cv2
import numpy as np
import dlib
import os
from imutils import face_utils
from imutils.face_utils import FaceAligner
from mask_create import create_mask

dataset = 'dataset/' #dataset folder
label = int(input('Enter label: ')) #emotion label
if not os.path.exists(dataset+str(label)):
	os.mkdir(dataset+str(label))

#find index of next image
list_of_files=os.listdir(dataset+str(label))
image_index=0
if(len(list_of_files)>0):
	list_of_files=[file.split('.')[0] for file in list_of_files]
	image_index=max(list(map(int, list_of_files)))+1

#setup face detector, facial landmark detector and face aligner
shape_predictor_68_face_landmarks = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
landmarks = dlib.shape_predictor(shape_predictor_68_face_landmarks)
fa = FaceAligner(landmarks, desiredFaceWidth=100)

#start capturing video
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 400)

while True:
    # Getting out image by webcam
	_, image = cap.read()
	image=cv2.flip(image, 1)
	pic=image

	# Converting the image to gray scale
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# Get faces detected
	rects = detector(gray, 0)

	# find the landmarks.
	if len(rects) > 0:
		shape = landmarks(gray, rects[0])
		shape = face_utils.shape_to_np(shape)

		#create mask for face
		mask = create_mask(shape, gray)
		masked = cv2.bitwise_and(gray, mask)
		#align face
		maskAligned = fa.align(mask, gray, rects[0])
		faceAligned = fa.align(masked, gray,rects[0])
		face = detector(faceAligned,1)
		if len(face) > 0:
			faceAligned = faceAligned[face[0].top():face[0].bottom(), face[0].left():face[0].right()]
			faceAligned = cv2.resize(faceAligned, (100, 100))
		image=faceAligned
	#display face as well whole image
	cv2.imshow("Output", image)
	cv2.imshow("view", pic)

	button = cv2.waitKey(1) & 0xFF
	if button == ord('q'): #quit when q is pressed
	    break
	if button==ord('c'): #click picture when c is pressed
		save= cv2.waitKey(0) & 0xFF #displays clicked picture for review
		if save==ord('s'): #save picture when s is pressed
			cv2.imwrite(dataset+str(label)+'/'+str(image_index)+'.jpg', image)
			image_index += 1
		else:
			continue
cv2.destroyAllWindows()
cap.release()
