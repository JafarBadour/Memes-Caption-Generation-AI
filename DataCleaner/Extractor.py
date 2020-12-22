import numpy as np

import cv2
import random
import tensorflow as tf
from sklearn.neighbors import KDTree

import PIL

import json

DATASET_PATH = '/home/jafar/Downloads/data'
FILE_JSON = DATASET_PATH + '/train.jsonl'

base_model = tf.keras.applications.EfficientNetB1(input_shape=(224, 224, 3), include_top=False)

base_model.trainable = False


def create_text(line):
    text = json.loads(line)['text']

    return text


def get_indexes(file_json=FILE_JSON):
    images_annotations = open(file_json).readlines()
    id2json_indexer = {}
    s = set()
    meme_index = []
    for row in images_annotations:

        traindict = json.loads(row)
        if not traindict['id'] in id2json_indexer:
            id2json_indexer[traindict['id']] = []
        id2json_indexer[traindict['id']].append({'text': traindict['text'], 'img': traindict['img']})
        meme_index.append(traindict['id'])
    return meme_index, id2json_indexer


def get_vector(img):
    try:
        extracted_features = base_model.predict(img)
    except:
        extracted_features = base_model.predict(image2np(img))
    return extracted_features.reshape(1, 7 * 7 * 1280)


def predeploy(file_json):
    ## pre-deployment
    meme_index, id2json_indexer = get_indexes(file_json)

    images = np.zeros((8500, 7, 7, 1280), np.float64)
    for cnt, k in enumerate(id2json_indexer.keys()):
        image_ = read_image(DATASET_PATH + '/' + id2json_indexer[k][0]['img'])
        image_ = np.array(image_)
        result = cv2.resize(image_, dsize=(224, 224), interpolation=cv2.INTER_AREA)
        result = image2np(result)
        images[cnt] = base_model.predict(result)

        if cnt % 50 == 0:
            print(cnt)
    images = images.reshape(8500, 7 * 7 * 1280)
    np.save('extracted_features', images)


def read_image(path):
    image_ = PIL.Image.open(path)
    image_.thumbnail((224, 224))

    return image_


def image2np(img):  # and resize
    zbr = np.array(img)

    result = zbr[:, :, :3]

    result = result.reshape((1, 224, 224, 3))
    return result


def load_dataset(path=DATASET_PATH, file_json=FILE_JSON):
    """

    :param path:
    :param file_json:
    :return:
    """
    predeploy(file_json)


if __name__ == '__main__':
    json = (load_dataset())
    print(base_model.output)
