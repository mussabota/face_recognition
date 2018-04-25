import cv2
import config

haar_faces = config.HAAR_FACE_CASCADE

def detect_single_face(image):
    faces = haar_faces.detectMultiScale(image,
                                        scaleFactor=1.1,
                                        minNeighbors=5,
                                        minSize=(30, 30),
                                        flags=cv2.CASCADE_SCALE_IMAGE)

    if len(faces) != 1:
        return None
    return faces[0]

def detect_faces(image):
    faces = haar_faces.detectMultiScale(image,
                                        scaleFactor=1.1,
                                        minNeighbors=5,
                                        minSize=(30, 30),
                                        flags=cv2.CASCADE_SCALE_IMAGE)
    return faces
