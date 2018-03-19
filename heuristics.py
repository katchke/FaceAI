import os
import cv2
import multiprocessing as mp


# Define Image class

class Image(object):
    def __init__(self, filepath):
        self.filename = filepath.split("/")[-1]
        self.img = cv2.imread(filepath)  # read image

        if self.img:
            self.img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)  # convert to gray color
            self.size = self.img.shape  # get image shape (height,width,channels)

    def detect_faces(self, cascade):
        # Return dimensions of faces (if any)

        return cascade.detectMultiScale(self.img_gray, 1.03, 5) if self.img else []

    def show_image(self):
        cv2.imshow("img", self.img)
        cv2.waitKey(0)


# Define face object

class Face(object):
    def __init__(self, image, dimensions, f_num):
        self.image = image
        self.dimensions = dimensions
        self.f_num = f_num
        self.score = 0

    def calculate_score(self):
        x, y, width, height = self.dimensions
        face_gray = self.image.img_gray[y:y + height, x:x + width]  # crop gray image around the face

        eyes = eye_cascade.detectMultiScale(face_gray)  # return dimensions of eyes
        mouth = mouth_cascade.detectMultiScale(face_gray)  # return dimensions of mouth
        nose = nose_cascade.detectMultiScale(face_gray)  # return dimensions of nose
        left_ear = left_ear_cascade.detectMultiScale(face_gray)  # return dimensions of left ear
        right_ear = right_ear_cascade.detectMultiScale(face_gray)  # return dimensions of right ear

        n_eyes = len(eyes)

        # Scoring
        if n_eyes == 2 or n_eyes == 3:  # if 2 eyes are detected - sometimes three are detected!!!
            self.score += score_rules["eyes"] * 1.5
        elif n_eyes == 1:  # if only one eye is detected
            self.score += score_rules["eyes"]
        elif n_eyes == 0:
            eyes_glasses = eye_glasses_cascade.detectMultiScale(face_gray)  # return dimensions of eye glasses

            n_eye_glasses = len(eyes_glasses)

            if n_eye_glasses == 2:
                self.score += score_rules["eye glasses"] * 1.5
            elif n_eye_glasses == 1:
                self.score += score_rules["eye glasses"]

        if len(mouth) > 0:
            self.score += score_rules["mouth"]

        if len(nose) > 0:
            self.score += score_rules["nose"]

        if len(left_ear) > 0 or len(right_ear) > 0:
            pass

    def crop_and_resize_face(self):
        x, y, width, height = self.dimensions
        cropped_img = self.image.img[y * 0.8:(y + height) * 1.2, x * 0.8:(x + width) * 1.2]  # crop image
        cropped_img_width = cropped_img.shape[0]
        cropped_img_height = cropped_img.shape[1]

        aspect_ratio = float(cropped_img.shape[0]) / float(cropped_img.shape[1])  # aspect ratio of cropped image

        if cropped_img_width >= resize_width or cropped_img_height >= (
                resize_width * aspect_ratio):  # make sure that cropped image has minimum dimensions
            new_size = (resize_width, int(resize_width * aspect_ratio))  # required dimension
            resized_img = cv2.resize(cropped_img, new_size,
                                     interpolation=cv2.INTER_AREA)  # resize cropped image using cv2.INTER_AREA algo

            # Save image in directory
            if not os.path.exists("faces/{}".format(str(self.image.filename))):
                os.makedirs("faces/{}".format(str(self.image.filename)))

            cv2.imwrite("faces/{filename}/{num}".format(filename=str(self.image.filename), num=str(self.f_num)),
                        resized_img)


def detect(image_file):
    image = Image("images/{}".format(str(image_file)))  # create image object
    faces = image.detect_faces(face_cascade)  # get face dimensions

    # If no face is detected using face_cascade_alt, try another cascade - face_cascade_tree (increases processing time)
    if not faces:
        faces = image.detect_faces(face_cascade_tree)

    # Compute score for each face
    for f_num, each_face in enumerate(faces):
        face = Face(image, each_face, f_num)  # create face object
        face.calculate_score()

        # crop face if score is above threshold
        if face.score >= score_threshold:
            face.crop_and_resize_face()


if __name__ == '__main__':
    n_cores = 1
    score_rules = {"eyes": 15, "eye glasses": 10, "mouth": 10, "nose": 25}  # weights
    score_threshold = 35
    resize_width = 150  # required image dimension

    # load cascades
    face_cascade = cv2.CascadeClassifier("opencv-2.4.11/data/haarcascades/haarcascade_frontalface_default.xml")
    face_cascade_tree = cv2.CascadeClassifier("opencv-2.4.11/data/haarcascades/haarcascade_frontalface_alt_tree.xml")
    eye_cascade = cv2.CascadeClassifier("opencv-2.4.11/data/haarcascades/haarcascade_eye.xml")
    eye_glasses_cascade = cv2.CascadeClassifier("opencv-2.4.11/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml")
    mouth_cascade = cv2.CascadeClassifier("opencv-2.4.11/data/haarcascades/haarcascade_mcs_mouth.xml")
    nose_cascade = cv2.CascadeClassifier("opencv-2.4.11/data/haarcascades/haarcascade_mcs_nose.xml")
    left_ear_cascade = cv2.CascadeClassifier("opencv-2.4.11/data/haarcascades/haarcascade_mcs_leftear.xml")
    right_ear_cascade = cv2.CascadeClassifier("opencv-2.4.11/data/haarcascades/haarcascade_mcs_rightear.xml")

    # load images from directory
    images = os.listdir("images")

    if '.DS_Store' in images:
        images.remove('.DS_Store')

    # parallel processing
    pool = mp.Pool(processes=n_cores)
    pool.map(detect, images)
