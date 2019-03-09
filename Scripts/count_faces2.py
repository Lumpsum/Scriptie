import cv2
import dlib

img = cv2.imread('./Flickr/aaron carter/aaron carter0.jpg')
detector = dlib.get_frontal_face_detector()
dets = detector(img, 1)
print(len(dets))