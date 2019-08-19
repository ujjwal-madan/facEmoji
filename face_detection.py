import cv2
import dlib
detect=dlib.get_frontal_face_detector()
video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
while True:
    ret, frame = video_capture.read()
    frame=cv2.flip(frame, 1)
    faces = detect(frame,1)
    for face in faces:
        cv2.rectangle(frame,(face.left(), face.top()), (face.right(), face.bottom()),(0,250,0), 2)
    cv2.imshow('Video', frame)
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
