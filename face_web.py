import face_recognition
from flask import Flask, jsonify, request, redirect
import base64
import random
import cv2
import csv
import json
import pandas as pd
import numpy as np
from PIL import Image



FACE_SET_PATH = 'E:/faceimg/faceset/'
FACE_VERIFY_PATH = 'E:/faceimg/faceverify/'

app = Flask(__name__)

@app.route('/addFace.do', methods=['GET', 'POST'])
def add_faceset():
    print(request.method)
    result_add = {'status': 1}
    # print(request.form)
    if request.method == 'POST':
        base64_str = str(request.form['data'])
        username = str(request.form['id'])
        base64_str = base64_str.replace(' ', '+')
        imgData = base64.b64decode(base64_str)
        img_path = FACE_SET_PATH + username + str(random.randint(0, 500)) + '.jpg'
        face_img = open(img_path, 'wb')
        face_img.write(imgData)
        face_img.close()

        img1 = cv2.imread(img_path)
        img1 = cv2.resize(img1, (int(img1.shape[1] * 0.1), int(img1.shape[0] * 0.1)))
        cv2.imwrite(img_path, img1)

        img_temp = Image.open(img_path)

        if img_temp.width > img_temp.height:
            im_rotate = img_temp.rotate(270, expand=True)
            im_rotate.save(img_path)
        image = face_recognition.load_image_file(img_path)
        face_locations = face_recognition.face_locations(image)
        if len(face_locations) == 0:
            result_add['status'] = 0
            return json.dumps(result_add)
        face_encoding = face_recognition.face_encodings(image)[0]
        face_encoding = list(face_encoding)
        face_encoding.append(username)
        f = open('../result/faceCSV/face1.csv', 'a', newline='')
        write = csv.writer(f)
        write.writerow(face_encoding)
        f.close()
        result_add['status'] = 1
        return json.dumps(result_add)

@app.route('/verifyFace.do', methods=['GET', 'POST'])
def verify_face():
    print(request.method)
    face_feather = []
    user_label = []
    result_verify = {'status': False, 'username': ''}
    csv_file = pd.read_csv('../result/faceCSV/face1.csv')
    row = csv_file.shape[0]

    for i in range(row):
        temp = list(csv_file.loc[i])
        face_feather.append(np.asarray(temp[0:128]))
        user_label.append(''.join(temp[128:129]))

    if request.method == 'POST':
        base64_str = str(request.form['data'])
        base64_str = base64_str.replace(' ', '+')
        imgData = base64.b64decode(base64_str)
        img_path = FACE_VERIFY_PATH + 'verify' + str(random.randint(0, 500)) + '.jpg'
        face_img = open(img_path, 'wb')
        face_img.write(imgData)
        face_img.close()

        img1 = cv2.imread(img_path)
        img1 = cv2.resize(img1, (int(img1.shape[1] * 0.1), int(img1.shape[0] * 0.1)))
        cv2.imwrite(img_path, img1)
        img_temp = Image.open(img_path)
        if img_temp.width > img_temp.height:
            im_rotate = img_temp.rotate(270, expand=True)
            im_rotate.save(img_path)
        image = face_recognition.load_image_file(img_path)
        face_locations = face_recognition.face_locations(image)
        if len(face_locations) == 0:
            result_verify['status'] = False
            return json.dumps(result_verify)
        face_enconding = face_recognition.face_encodings(image)[0]
        face_distance = face_recognition.face_distance(face_feather, face_enconding)
        face_distance = list(face_distance)
        result_verify['status'] = True
        if min(face_distance) < 0.6:
            result_verify['username'] = user_label[face_distance.index(min(face_distance))]
        return json.dumps(result_verify)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8585, debug=True)
