import base64
import json
import numpy as np
from flask import Flask, request
from PIL import Image
import dlib
import cv2
import os
from werkzeug.utils import secure_filename
import csv
import time
from config import DLIB_FACE_RECOGNITION_DAT, SHAP_PREDICTOR_DAT, CSV, FACES_DIR, TMP_DIR
from io import BytesIO
import shutil
import uuid

app = Flask(__name__)

# 探测器
detector = dlib.get_frontal_face_detector()

# 预测器
predictor = dlib.shape_predictor(SHAP_PREDICTOR_DAT)
# 人脸特征抽取器
face_reco_model = dlib.face_recognition_model_v1(DLIB_FACE_RECOGNITION_DAT)

# csv作为人脸数据库
FACES_FEATURES_CSV_FILE = CSV
# 识别阈值
FACES_FEATURES_DISTANCE_THRESHOLD = 0.4
# 人脸图片文件夹
UPLOAD_FOLDER = FACES_DIR

# 临时图片文件夹
UPLOAD_FOLDER_TMP = TMP_DIR

# 上传文件格式
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def init():
    init0()
    response = json.dumps({"code": "00000", "message": "操作成功", "data": None})
    return response, 200, {"Content-Type": "application/json"}


def init0():
    extract_features_to_csv(UPLOAD_FOLDER)


def identify():
    img_file = request.files.get('pic')

    if not img_file:
        response = json.dumps({"code": "00001", "message": "未传文件", "data": None})
        return response, 200, {"Content-Type": "application/json"}

    allow = allowed_file(img_file.filename)
    if not allow:
        response = json.dumps({"code": "00002", "message": "文件类型不允许", "data": None})
        return response, 200, {"Content-Type": "application/json"}

    new_file_name = get_new_file_name(img_file.filename)

    if not os.path.exists(UPLOAD_FOLDER_TMP):
        os.makedirs(UPLOAD_FOLDER_TMP)

    save_path = os.path.join(UPLOAD_FOLDER_TMP, new_file_name)
    img_file.save(save_path)

    results = compare_face_features_with_database(get_csv_datas(), save_path)

    os.remove(save_path)

    response = json.dumps({"code": "00000", "message": "操作成功", "data": results})
    return response, 200, {"Content-Type": "application/json"}


def identify_base64():
    data = json.loads(request.get_data())
    pic_base64 = data['pic']
    pic = Image.open(BytesIO(base64.b64decode(pic_base64)))
    if not pic:
        response = json.dumps({"code": "00001", "message": "未传文件", "data": None})
        return response, 200, {"Content-Type": "application/json"}

    zeroName = "0000000." + pic.format.lower()

    allow = allowed_file(zeroName)
    if not allow:
        response = json.dumps({"code": "00002", "message": "文件类型不允许", "data": None})
        return response, 200, {"Content-Type": "application/json"}

    new_file_name = get_new_file_name(zeroName)

    if not os.path.exists(UPLOAD_FOLDER_TMP):
        os.makedirs(UPLOAD_FOLDER_TMP)

    save_path = os.path.join(UPLOAD_FOLDER_TMP, new_file_name)
    pic.save(save_path)

    results = compare_face_features_with_database(get_csv_datas(), save_path)

    os.remove(save_path)

    response = json.dumps({"code": "00000", "message": "操作成功", "data": results})
    return response, 200, {"Content-Type": "application/json"}


def uploadPic():
    identity = request.form.get("identity", type=str, default=None)
    save_path_dir = os.path.join(UPLOAD_FOLDER, identity)

    pics = request.files.getlist('pics')

    if not pics:
        response = json.dumps({"code": "00001", "message": "未传文件", "data": None})
        return response, 200, {"Content-Type": "application/json"}

    for pic in pics:
        allow = allowed_file(pic.filename)
        if allow:
            if not os.path.exists(save_path_dir):
                os.makedirs(save_path_dir)

            save_path = os.path.join(UPLOAD_FOLDER, identity, get_new_file_name(pic.filename))
            pic.save(save_path)

    mean_features = get_person_feature(save_path_dir)

    person_label = mean_features[1]
    person_features = mean_features[0]
    person_features = np.insert(person_features, 0, person_label, axis=0)

    update_csv(person_label, person_features)

    response = json.dumps({"code": "00000", "message": "操作成功", "data": None})
    return response, 200, {"Content-Type": "application/json"}


def extractEigenvalue():
    data = json.loads(request.get_data())
    pics_base64 = data['pics']

    tmp_dir = str(uuid.uuid1());
    save_path_dir = os.path.join(UPLOAD_FOLDER_TMP, tmp_dir)

    if not pics_base64:
        response = json.dumps({"code": "00001", "message": "未传文件", "data": None})
        return response, 200, {"Content-Type": "application/json"}

    for pic_base64 in pics_base64:
        pic = Image.open(BytesIO(base64.b64decode(pic_base64)))
        zeroName = "0000000." + pic.format.lower()
        allow = allowed_file(zeroName)
        if allow:
            if not os.path.exists(save_path_dir):
                os.makedirs(save_path_dir)

            save_path = os.path.join(UPLOAD_FOLDER_TMP, tmp_dir, get_new_file_name(zeroName))
            pic.save(save_path)

    mean_features = get_person_feature(save_path_dir)

    shutil.rmtree(save_path_dir)

    person_features = mean_features[0]

    response = json.dumps({"code": "00000", "message": "操作成功", "data": person_features.tolist()})
    return response, 200, {"Content-Type": "application/json"}


def uploadPicBase64():
    data = json.loads(request.get_data())
    identity = data['identity']
    pics_base64 = data['pics']
    save_path_dir = os.path.join(UPLOAD_FOLDER, identity)

    if not pics_base64:
        response = json.dumps({"code": "00001", "message": "未传文件", "data": None})
        return response, 200, {"Content-Type": "application/json"}

    for pic_base64 in pics_base64:
        pic = Image.open(BytesIO(base64.b64decode(pic_base64)))
        zeroName = "0000000." + pic.format.lower()
        allow = allowed_file(zeroName)
        if allow:
            if not os.path.exists(save_path_dir):
                os.makedirs(save_path_dir)

            save_path = os.path.join(UPLOAD_FOLDER, identity, get_new_file_name(zeroName))
            pic.save(save_path)

    mean_features = get_person_feature(save_path_dir)

    person_label = mean_features[1]
    person_features = mean_features[0]
    person_features = np.insert(person_features, 0, person_label, axis=0)

    update_csv(person_label, person_features)

    response = json.dumps({"code": "00000", "message": "操作成功", "data": None})
    return response, 200, {"Content-Type": "application/json"}


def get_new_file_name(filename):
    t = int(round(time.time() * 1000))
    suffix = filename.rsplit('.', 1)[1].lower()
    new_file_name = str(t) + '.' + suffix

    return secure_filename(new_file_name)


def update_csv(person_label, person_features):
    output_rows = []

    is_update = False
    with open(FACES_FEATURES_CSV_FILE, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if row[0] == person_label:
                output_rows.append(person_features)
                is_update = True
            else:
                output_rows.append(row)
    csvfile.close()

    if not is_update:
        output_rows.append(person_features)

    with open(FACES_FEATURES_CSV_FILE, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(output_rows)


def get_csv_datas():
    feature_lists = []

    with open(FACES_FEATURES_CSV_FILE) as f:
        for row in csv.reader(f, skipinitialspace=True):
            features = []
            for i in range(1, len(row)):
                features.append(row[i])
            feature_lists.append((row[0], features))

    return feature_lists


def get_person_feature(identity_dir):
    person_label = os.path.split(identity_dir)[-1]
    image_paths = [os.path.join(identity_dir, f) for f in os.listdir(identity_dir)]
    image_paths = list(filter(lambda x: os.path.isfile(x), image_paths))
    feature_list_of_person_x = []

    for image_path in image_paths:

        # 计算每一个图片的特征
        feature = get_128d_features_of_face(image_path)
        if feature == 0:
            continue

        feature_list_of_person_x.append(feature)

    # 计算当前人脸的平均特征
    features_mean_person_x = np.zeros(128, dtype=object, order='C')
    if feature_list_of_person_x:
        features_mean_person_x = np.array(feature_list_of_person_x, dtype=object).mean(axis=0)

    return features_mean_person_x, person_label


def allowed_file(filename):
    """
    判断文件类型是否允许上传
    """
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_128d_features_of_face(image_path):
    image = Image.open(image_path)
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    faces = detector(image, 1)

    if len(faces) != 0:
        shape = predictor(image, faces[0])
        face_descriptor = face_reco_model.compute_face_descriptor(image, shape)
    else:
        face_descriptor = 0
    return face_descriptor


def compare_face_features_with_database(datas, image_path):
    image = Image.open(image_path)
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    faces = detector(image, 1)

    compare_results = []
    if len(faces) != 0:
        for i in range(len(faces)):
            shape = predictor(image, faces[i])
            face_descriptor = face_reco_model.compute_face_descriptor(image, shape)
            face_feature_distance_list = []
            for face_data in datas:
                # 比对人脸特征，当距离小于 0.4 时认为匹配成功
                dist = get_euclidean_distance(face_descriptor, face_data[1])
                dist = round(dist, 4)

                if dist >= FACES_FEATURES_DISTANCE_THRESHOLD:
                    continue

                face_feature_distance_list.append((face_data[0], dist))

            # 按距离排序，取最小值进行绘制
            sorted(face_feature_distance_list, key=lambda x: x[1])
            if face_feature_distance_list:
                person_dist = face_feature_distance_list[0][1]
                person_label = face_feature_distance_list[0][0]
                compare_results.append({"label": person_label, "score": str(round(1 / (1 + person_dist), 2))})

    return compare_results


def get_euclidean_distance(feature_1, feature_2):
    feature_1 = np.array(feature_1, dtype='float64')
    feature_2 = np.array(feature_2, dtype='float64')

    return np.sqrt(np.sum(np.square(feature_1 - feature_2)))


def extract_features_to_csv(faces_dir):
    mean_features_list = list(get_mean_features_of_face(faces_dir))
    with open(FACES_FEATURES_CSV_FILE, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for mean_features in mean_features_list:
            person_features = mean_features[0]
            person_label = mean_features[1]
            person_features = np.insert(person_features, 0, person_label, axis=0)
            writer.writerow(person_features)


def get_mean_features_of_face(path):
    path = os.path.abspath(path)
    subDirs = [os.path.join(path, f) for f in os.listdir(path)]
    subDirs = list(filter(lambda x: os.path.isdir(x), subDirs))
    for index in range(0, len(subDirs)):
        subDir = subDirs[index]

        yield get_person_feature(subDir)
