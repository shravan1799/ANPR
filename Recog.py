import os
import sys
import cv2
import numpy as np
import re
from os.path import splitext
from local_utils import detect_lp
from keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder
import Run

flag = 1
final_string = ''


# Declaring functions


def load_model(path):
    try:
        path = splitext(path)[0]
        with open('%s.json' % path, 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json, custom_objects={})
        model.load_weights('%s.h5' % path)
        # print("Loading model successfully...")
        return model
    except Exception as e:
        print(e)


def preprocess_image(image_path, resize=False):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    if resize:
        img = cv2.resize(img, (224, 224))
    return img


def get_plate(image_path, Dmax=150, Dmin=400):
    vehicle = preprocess_image(image_path)
    ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    _, LpImg, _, cor = detect_lp(wpod_net, vehicle, bound_dim, lp_threshold=0.5)
    return vehicle, LpImg, cor


def sort_contours_2_line(cnts, reverse=False):
    i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts


def sort_contours(cnts, reverse=False):
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts


def predict_from_model(image, model, labels):
    image = cv2.resize(image, (80, 80))
    image = np.stack((image,) * 3, axis=-1)
    prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis, :]))])
    return prediction


def brute_force(ex):
    lenx = len(ex)
    if ex[0] == "6":
        list1 = list(ex)
        list1[0] = "G"
        ex = ''.join(list1)

    if ex[-1] == "A":
        list1 = list(ex)
        list1[-1] = "4"
        ex = ''.join(list1)

    if ex[-2] == "A":
        list1 = list(ex)
        list1[-2] = "4"
        ex = ''.join(list1)

    if ex[-3] == "A":
        list1 = list(ex)
        list1[-3] = "4"
        ex = ''.join(list1)

    if ex[-4] == "A":
        list1 = list(ex)
        list1[-4] = "4"
        ex = ''.join(list1)

    if ex[4] == "2":
        list1 = list(ex)
        list1[4] = "Z"
        ex = ''.join(list1)

    if ex[5] == "2":
        list1 = list(ex)
        list1[5] = "Z"
        ex = ''.join(list1)

    if ex[-1] == "Z":
        list1 = list(ex)
        list1[-1] = "2"
        ex = ''.join(list1)

    if ex[-2] == "Z":
        list1 = list(ex)
        list1[-2] = "2"
        ex = ''.join(list1)

    if ex[-3] == "Z":
        list1 = list(ex)
        list1[-3] = "2"
        ex = ''.join(list1)

    if ex[-4] == "Z":
        list1 = list(ex)
        list1[-4] = "2"
        ex = ''.join(list1)

    return ex


# Finished Declaring function

# Loading models and files


wpod_net_path = "wpod-net.json"
wpod_net = load_model(wpod_net_path)
json_file = open('MobileNets_character_recognition.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("License_character_recognition.h5")
# print("[INFO] Model loaded successfully...")
labels = LabelEncoder()
labels.classes_ = np.load('license_character_classes.npy')
# print("[INFO] Labels loaded successfully...")

# Loading files and models finished

test_image_path = Run.img
vehicle, LpImg, cor = get_plate(test_image_path)
# cv2.imwrite("plate.png", LpImg[0])
if len(LpImg):  # check if there is at least one license image
    # Scales, calculates absolute values, and converts the result to 8-bit.
    plate_image = cv2.convertScaleAbs(LpImg[0], alpha=255.0)
    cv2.imwrite("plate.png", plate_image)
    # convert to grayscale and blur the image
    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    # Applied inverse thresh_binary
    binary = cv2.threshold(blur, 180, 255,
                           cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)

test = plate_image.copy()
cont, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
test_roi = test.copy()
crop_characters = []
digit_w, digit_h = 50, 80

if flag == 1:
    for c in sort_contours_2_line(cont):
        (x, y, w, h) = cv2.boundingRect(c)
        ratio = h / w
        if 1 <= ratio <= 5.5:  # Only select contour with defined ratio
            if h / plate_image.shape[0] >= 0.2:
                cv2.rectangle(test_roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
                curr_num = thre_mor[y:y + h, x:x + w]
                curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                crop_characters.append(curr_num)

    final_string = ''
    for i, character in enumerate(crop_characters):
        title = np.array2string(predict_from_model(character, model, labels))
        final_string += title.strip("'[]")

    final_string = brute_force(final_string)

    q = re.match("^([A-Z][A-Z][0-9][0-9][A-Z][A-Z])", final_string)
    if q:
        pass
    else:
        fs = final_string[-10:-5]
        fs = ''.join(reversed(fs))
        fs1 = final_string[-5:]
        fs1 = ''.join(reversed(fs1))
        final_string = fs + fs1
        final_string = brute_force(final_string)
        x = len(final_string)
        if x == 10:
            q = re.match("^([A-Z][A-Z][0-9][0-9][A-Z][A-Z][0-9][0-9][0-9][0-9])$", final_string)
            if q:
                pass
            else:
                flag = 0
        else:
            flag = 0

if flag == 0:
    crop_characters = []
    for c in sort_contours(cont):
        (x, y, w, h) = cv2.boundingRect(c)
        ratio = h / w
        if 1 <= ratio <= 5.5:  # Only select contour with defined ratio
            if h / plate_image.shape[0] >= 0.2:
                cv2.rectangle(test_roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
                curr_num = thre_mor[y:y + h, x:x + w]
                curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                crop_characters.append(curr_num)

    final_string = ''
    for i, character in enumerate(crop_characters):
        title = np.array2string(predict_from_model(character, model, labels))
        final_string += title.strip("'[]")

    final_string = brute_force(final_string)
    x = len(final_string)
    if x == 10:
        q = re.match("^([A-Z][A-Z][0-9][0-9][A-Z][A-Z][0-9][0-9][0-9][0-9])$", final_string)
        if q:
            pass
        else:
            final_string = "4"
    else:
        final_string = "4"
