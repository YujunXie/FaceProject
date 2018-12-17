import cv2
import numpy as np
import dlib
from PIL import Image

path = 'data/Image/100_1737.jpg'
im = np.array(Image.open(path))

'''
image = face_recognition.load_image_file('data/Image/100_1737.jpg')
face_locations = face_recognition.face_locations(image)

for face_location in face_locations:

    # Print the location of each face in this image
    top, right, bottom, left = face_location
    print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

    cv.imread('data/Image/100_1737.jpg')
    cv.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 3)
    cv.namedWindow("result", 0)
    cv.resizeWindow("result", 640, 480)
    cv.imshow("result", image)
    cv.waitKey(0)
'''
#hog feature
# face_detector = dlib.get_frontal_face_detector()
# faces = face_detector(im, upsample_num_times = 1)
# print("found {} face(s) in this photograph.".format(len(faces)))

#cnn feature
cnn_face_detection_model = "/Users/yujun/VirtualEnvs/face_recognition/lib/python3.5/site-packages" \
                            "/face_recognition_models/models/mmod_human_face_detector.dat"
cnn_face_detector = dlib.cnn_face_detection_model_v1(cnn_face_detection_model)
faces = cnn_face_detector(im, upsample_num_times = 1)
print("get faces")
print("found {} face(s) in this photograph.".format(len(faces)))

# Print the location of each face in this image
for face in faces:
    top, right, bottom, left = max(face.top(), 0), min(face.right(), im.shape[1]), min(face.bottom(), im.shape[0]), max(
        face.left(), 0)
    print(
        "A face is located at pixel location Left: {}, Top: {}, Right: {}, Bottom: {}".format(left, top, right, bottom))

    cv2.imread('data/Image/100_1737.jpg')
    cv2.rectangle(im, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 3)

cv2.namedWindow("result", 0)
cv2.resizeWindow("result", 640, 480)
cv2.imshow("result", im)
cv2.waitKey(0)

