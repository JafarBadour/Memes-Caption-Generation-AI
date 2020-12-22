import numpy as np
import cv2
import random
import tensorflow as tf
from sklearn.neighbors import KDTree

import PIL
import numpy as np
import json

from DataCleaner.Extractor import get_indexes, read_image, image2np



class KNNRandomGenerator:
    
    def __init__(self,images_features : np.ndarray, base_model):
        self.meme_index, self.id2json_indexer = get_indexes()
        self.base_model = base_model
        self.indexer = KDTree(images_features)

    def get_vector(self, img):
        try:
            extracted_features = self.base_model.predict(img)
        except:
            extracted_features = self.base_model.predict(image2np(img))
        return extracted_features.reshape(1, 7 * 7 * 1280)

    def process_image(self, path):

        K = 5

        # put text on image

        im = np.array(read_image(path))
        Color = [255-im[:,:,i].mean() for i in range(3)]
        font = cv2.FONT_HERSHEY_SIMPLEX
        imcopy = cv2.resize(im, dsize=(224, 224), interpolation=cv2.INTER_AREA)
        imcopy = image2np(imcopy)

        img = self.get_vector(imcopy.reshape((1, 224, 224, 3)))
        textr = ''
        _, choices = self.indexer.query(img, k=K)
        choices = choices[0]
        choice = random.choice(choices)
        print('choice is', choice)
        text =self.id2json_indexer[self.meme_index[choice]][0]['text']
        if len(im.shape) == 2:
            X, Y = im.shape
        else:
            X, Y, _ = im.shape[:3]

        print(text)
        print('Shape', im.shape)

        im = cv2.imread(path, 1)

        N = len(text.split(' '))
        lines = 0
        im = cv2.resize(im, dsize=(1000, 1000), interpolation=cv2.INTER_AREA)
        X, Y = 1000, 1000
        words = text.split(' ')
        words.append('')
        for cnt, c in enumerate(words[:-1]):

            textr += c + ' '

            if len(textr + words[cnt+1]) > 23 or cnt == len(words) - 2:
                cv2.putText(im, textr, (77, X - 400 + lines * 59), font, 2, (123, 0, 123), 4, cv2.LINE_AA)
                textr = ''
                lines += 1

        cv2.imwrite(f'/home/jafar/PycharmProjects/Artistic Neural Networks/generated/{choice}.jpg', im)

        return f'/generated/{choice}.jpg'