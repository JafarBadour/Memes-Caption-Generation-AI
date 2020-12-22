#!/usr/bin/env python
# coding: utf-8

# In[1]:
import cv2
import tensorflow as tf

# You'll generate plots of attention in order to see which parts of an image
# our model focuses on during captioning
import matplotlib.pyplot as plt

# Scikit-learn includes many helpful utilities
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import collections
import random
import re
import numpy as np
import os
import time
import json
from glob import glob
from PIL import Image
import pickle







# In[16]:



# In[17]:


import json
import os
if os.path.dirname(os.path.realpath('.')) != '/home/jafar/PycharmProjects/Artistic Neural Networks':
    DATASET_PATH = '/content/gdrive/MyDrive/ColabFiles/FacebookAI/data'
    FILE_JSON = DATASET_PATH + '/train.jsonl'
else:
    DATASET_PATH = '/home/jafar/Downloads/data'
    FILE_JSON = DATASET_PATH + '/train.jsonl'

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


# In[18]:


# Group all captions together having the same image ID.



# In[63]:




def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path


print('Visual attention step 2')

# ## Split the data into training and testing

# ## Create a tf.data dataset for training
# 

#  Our images and captions are ready! Next, let's create a tf.data dataset to use for training our model.

# In[32]:


# Feel free to change these parameters according to your system's configuration



class BahdanauAttention(tf.keras.Model):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, features, hidden):
    
    hidden_with_time_axis = tf.expand_dims(hidden, 1)


    attention_hidden_layer = (tf.nn.tanh(self.W1(features) +
                                         self.W2(hidden_with_time_axis)))


    score = self.V(attention_hidden_layer)


    attention_weights = tf.nn.softmax(score, axis=1)


    context_vector = attention_weights * features
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights


# In[35]:


class CNN_Encoder(tf.keras.Model):
  
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x


# In[36]:


class RNN_Decoder(tf.keras.Model):
  def __init__(self, embedding_dim, units, vocab_size):
    super(RNN_Decoder, self).__init__()
    self.units = units

    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc1 = tf.keras.layers.Dense(self.units)
    self.fc2 = tf.keras.layers.Dense(vocab_size)

    self.attention = BahdanauAttention(self.units)

  def call(self, x, features, hidden):
    
    context_vector, attention_weights = self.attention(features, hidden)


    x = self.embedding(x)


    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)


    output, state = self.gru(x)

    # shape == (batch_size, max_length, hidden_size)
    x = self.fc1(output)

    # x shape == (batch_size * max_length, hidden_size)
    x = tf.reshape(x, (-1, x.shape[2]))

    # output shape == (batch_size * max_length, vocab)
    x = self.fc2(x)

    return x, state, attention_weights

  def reset_state(self, batch_size):
    return tf.zeros((batch_size, self.units))


# In[37]:









class VisualAttentionGenerator:
    def __init__(self):
        meme_index, id2json_indexer = get_indexes()

        image_path_to_caption = collections.defaultdict(list)
        for id in meme_index:
            text = '<start> ' + id2json_indexer[id][0]['text'] + ' <end>'
            path = DATASET_PATH + '/' + id2json_indexer[id][0]['img']
            image_path_to_caption[path].append(text)

        # In[19]:

        image_paths = list(image_path_to_caption.keys())

        train_image_paths = image_paths[:8500]
        print(len(train_image_paths))

        # In[23]:

        train_captions = []
        img_name_vector = []

        for image_path in train_image_paths:
            caption_list = image_path_to_caption[image_path]
            train_captions.extend(caption_list)
            img_name_vector.extend([image_path] * len(caption_list))

        image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                        weights='imagenet')
        new_input = image_model.input
        hidden_layer = image_model.layers[-1].output


        def calc_max_length(tensor):
            return max(len(t) for t in tensor)

        # In[24]:

        # Choose the top 5000 words from the vocabulary
        top_k = 5000
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                          oov_token="<unk>",
                                                          filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
        tokenizer.fit_on_texts(train_captions)
        train_seqs = tokenizer.texts_to_sequences(train_captions)

        # In[25]:

        tokenizer.word_index['<pad>'] = 0
        tokenizer.index_word[0] = '<pad>'

        # In[26]:

        # Create the tokenized vectors
        # noinspection PyRedeclaration
        train_seqs = tokenizer.texts_to_sequences(train_captions)

        # In[27]:

        # Pad each vector to the max_length of the captions
        # If you do not provide a max_length value, pad_sequences calculates it automatically
        cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')

        # In[30]:

        # Calculates the max_length, which is used to store the attention weights
        self.max_length = calc_max_length(train_seqs)

        print('Visual attention step 4')
        BATCH_SIZE = 64
        BUFFER_SIZE = 1000
        embedding_dim = 256
        units = 512
        vocab_size = top_k + 1

        # Shape of the vector extracted from InceptionV3 is (64, 2048)
        # These two variables represent that vector shape
        features_shape = 2048
        self.attention_features_shape = 64

        print('Visual attention step 3')
        encoder = CNN_Encoder(embedding_dim)
        decoder = RNN_Decoder(embedding_dim, units, vocab_size)

        # In[38]:

        optimizer = tf.keras.optimizers.Adam()
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')
        print('Visual attention step 5')

        encoder.load_weights('../nnartsmodels/encoder.tf.index')
        print('Visual attention step 6')

        decoder.load_weights('../nnartsmodels/encoder.tf.index')
        print('Visual attention step 7')

        # image_features_extract_model.load_weights('../nnartsmodels/image_features_extract_model.tf.index')
        self.encoder = encoder
        self.decoder = decoder
        image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

        self.image_features_extract_model = image_features_extract_model
        self.tokenizer = tokenizer
        print('Visual attention step 8')


    def evaluate(self, image):
        attention_plot = np.zeros((self.max_length, self.attention_features_shape))

        hidden = self.decoder.reset_state(batch_size=1)

        temp_input = tf.expand_dims(load_image(image)[0], 0)
        img_tensor_val = self.image_features_extract_model(temp_input)
        img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

        features = self.encoder(img_tensor_val)

        dec_input = tf.expand_dims([self.tokenizer.word_index['<start>']], 0)
        result = []

        for i in range(self.max_length):
            predictions, hidden, attention_weights = self.decoder(dec_input, features, hidden)

            attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

            predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
            result.append(self.tokenizer.index_word[predicted_id])

            if self.tokenizer.index_word[predicted_id] == '<end>':
                return result, attention_plot

            dec_input = tf.expand_dims([predicted_id], 0)

        attention_plot = attention_plot[:len(result), :]
        return result, attention_plot


# In[46]:


    def plot_attention(self,image, result, attention_plot):
        temp_image = np.array(Image.open('../'+image))

        fig = plt.figure(figsize=(10, 10))

        len_result = len(result)
        for l in range(len_result):
            temp_att = np.resize(attention_plot[l], (8, 8))
            ax = fig.add_subplot(len_result//2, len_result//2, l+1)
            ax.set_title(result[l])
            img = ax.imshow(temp_image)
            ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

        import random
        r = random.randint(1, 100000000)
        plt.tight_layout()
        # plt.show()
        fig.savefig(f'../generated/attention_plot{r}.png')
        return f'../generated/attention_plot{r}.png'


    def process_picture(self, image, text):
        print(text)


        font = cv2.FONT_HERSHEY_SIMPLEX

        im = cv2.imread(image, 1)

        N = len(text)
        lines = 0
        im = cv2.resize(im, dsize=(1000, 1000), interpolation=cv2.INTER_AREA)
        X, Y = 1000, 1000
        words = text
        words = words[:14]
        words.append('')
        import random
        textr = ''
        r = random.randint(1, 100000000)
        for cnt, c in enumerate(words[:-1]):

            textr += c + ' '

            if len(textr + words[cnt + 1]) > 23 or cnt == len(words) - 2:
                cv2.putText(im, textr, (77, X - 400 + lines * 59), font, 2, (123, 0, 123), 4, cv2.LINE_AA)
                textr = ''
                lines += 1

        cv2.imwrite(f'/home/jafar/PycharmProjects/Artistic Neural Networks/generated/{r}.jpg', im)

        return f'/generated/{r}.jpg'







# In[ ]:



if __name__ == '__main__':
    v = VisualAttentionGenerator()

