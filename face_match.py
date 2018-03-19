__author__ = 'Rahul'

# Check if two images belong to the same person

import os

import cv2
import numpy as np

np.set_printoptions(precision=2)

import openface
import urllib2

class OpenFace(object):
    def __init__(self):
        # Path to libraries and models
        openface_directory = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'packages', 'openface')
        dlib_models_directory = os.path.join(openface_directory, 'models', 'dlib')
        openface_models_directory = os.path.join(openface_directory, 'models', 'openface')
        face_predictor = os.path.join(dlib_models_directory, "shape_predictor_68_face_landmarks.dat")
        network_model = os.path.join(openface_models_directory, 'nn4.small2.v1.t7')

        self.image_dimensions = 96
        self.align = openface.AlignDlib(face_predictor)
        self.net = openface.TorchNeuralNet(network_model, self.image_dimensions, cuda=False)

    def get_representation(self, image_path, return_image = False):
        try:
            # Uncomment if image path is on disk
            #img = cv2.imread(image_path)

            # If image path is a URL
            #response = Image(image_path)
            #response.detect_faces()
            #resized_img = response.crop_and_resize_face(response,response.faces[0].dimensions)

            response = urllib2.urlopen(image_path).read()
            img = cv2.imdecode(np.asarray(bytearray(response), dtype=np.uint8), -1)

            rbg_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face_bounding_box = self.align.getLargestFaceBoundingBox(rbg_img)
            aligned_face = self.align.align(self.image_dimensions, rbg_img, face_bounding_box,
                                            landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

            rep = self.net.forward(aligned_face)

            
            return (rep, response) if return_image else rep
        except:
            return (None, response) if return_image else None

if __name__ == '__main__':
    # Path to images
    images = [('https://fm.cnbc.com/applications/cnbc.com/resources/img/editorial/2017/12/11/104891709-Bill_Gates_the_co-Founder.1910x1000.jpg',
               'https://pbs.twimg.com/profile_images/929933611754708992/ioSgz49P_400x400.jpg')]

    image = OpenFace()

    # Calculate distance between two images
    for img1, img2 in images:
        distance = image.get_representation(img1) - image.get_representation(img2)
        if np.dot(distance, distance) <= 0.99:
            print 'Same person', np.dot(distance, distance)
        else:
            print 'Different person', np.dot(distance, distance)
