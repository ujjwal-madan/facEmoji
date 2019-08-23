import cv2
import numpy as np

def get_points(face_part, shape):
	points = [shape[point] for point in face_part]
	return np.array(points)

def create_mask(shape, img):
	height, width = img.shape
	mask = np.zeros((height, width), dtype=np.uint8)
	left_eyebrow = (17, 18, 19, 20, 21)
	right_eyebrow = (22, 24, 25, 26, 26)
	face=list(range(17))
	face.extend(right_eyebrow[::-1])
	face.extend(left_eyebrow[::-1])
	mask = cv2.fillPoly(mask, [get_points(face, shape)], 255)
	return mask
