__author__ = 'Rahul'
# -*- coding: utf-8 -*-

# Detect faces from an image and crop them

import numpy as np

import os
import cv2
import multiprocessing as mp
import urllib2


class Image(object):
    def __init__(self, path):
        # Uncomment if image path is on disk
        # self.filename = path.split("/")[-1]
        # self.img = np.array(cv2.imread(path), dtype=np.uint8)  # read image

        # If image path is a URL
        resource = urllib2.urlopen(path)
        self.img = cv2.imdecode(np.asarray(bytearray(resource.read()), dtype=np.uint8), -1)

        self.faces = []

        if self.img:
            self.size = self.img.shape  # get image shape (height,width,channels)

    def detect_faces(self):
        face_detector = CascadedDetector(
            cascade_fn=os.path.dirname(
                os.path.dirname(__file__)) + '/data/OpenCV/data/haarcascades/haarcascade_frontalface_default.xml')
        faces = face_detector.detect(self.img)

        face_detector = CascadedDetector(
            cascade_fn=os.path.dirname(
                os.path.dirname(__file__)) + '/data/OpenCV/data/haarcascades/haarcascade_frontalface_alt.xml')

        if len(faces) == 0:
            faces = face_detector.detect(self.img)
        else:
            new_faces = []
            for each_face in face_detector.detect(self.img):
                face_found = False

                for each_existing_face in faces:
                    if check_overlap(each_face, each_existing_face) or check_overlap(each_existing_face, each_face):
                        face_found = True
                        break

                if not face_found:
                    new_faces.append(each_face)

            if len(new_faces):
                faces = np.append(faces, new_faces, 0)

        for f_num, face_dimensions in enumerate(faces):
            x0, y0, x1, y1 = face_dimensions
            face_img = self.img[y0 - 0.7 * y0:y1 + 0.1 * y1, x0 - 0.1 * x0:x1 + 0.1 * x1]
            face = Face(face_img, face_dimensions, f_num)
            self.faces.append(face)

            # Uncomment if image of face is to be saved on disk
            # face.save_image(self.filename)

    def show_image(self):
        cv2.imshow("img", self.img)
        cv2.waitKey(0)


class Face(object):
    def __init__(self, img, dimensions, f_num=None):
        self.img = img
        self.dimensions = dimensions
        self.f_num = f_num

    def save_image(self, filename):
        extension = filename.split('.')[-1]

        if not os.path.exists("faces/{}".format(filename)):
            os.makedirs("faces/{}".format(filename))

        cv2.imwrite("faces/{}/{}.{}".format(filename, str(self.f_num), extension), self.img)


class CascadedDetector(object):
    def __init__(self, cascade_fn="./cascades/haarcascade_frontalface_alt2.xml", scale_factor=1.2, min_neighbors=5,
                 min_size=(30, 30)):
        if not os.path.exists(cascade_fn):
            raise IOError("No valid cascade found for path {}".format(cascade_fn))

        self.cascade = cv2.CascadeClassifier(cascade_fn)
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

    def detect(self, src):
        if np.ndim(src) == 3:
            src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

        src = cv2.equalizeHist(src)
        rects = self.cascade.detectMultiScale(src, scaleFactor=self.scale_factor, minNeighbors=self.min_neighbors,
                                              minSize=self.min_size)
        if not len(rects):
            return np.ndarray((0,))

        rects[:, 2:] += rects[:, :2]

        return rects


def check_overlap(face1, face2):
    # Check if two faces are overlapping
    a0, b0, a1, b1 = face1
    x0, y0, x1, y1 = face2
    overlap = False

    if x1 >= a0 >= x0 and y1 >= b0 >= y0:
        overlap = True
    elif x1 >= a1 >= x0 and y1 >= b1 >= y0:
        overlap = True
    elif x1 >= a0 >= x0 and y1 >= b1 >= y0:
        overlap = True
    elif x1 >= a1 >= x0 and y1 >= b0 >= y0:
        overlap = True

    return overlap


def start_detection(filename):
    image = Image('images/{}'.format(filename))
    image.detect_faces()


if __name__ == '__main__':
    n_cores = 4

    # Resize dimensions
    resize_width = 150

    # Path to images
    images = os.listdir('images')

    if '.DS_Store' in images:
        images.remove('.DS_Store')
        
    pool = mp.Pool(processes=n_cores)
    pool.map(start_detection, images)
