API_TOKEN = '*********:AAFfs_KjpTy9M9VOqZL6IM*********yOgg'

from telegram.ext import Updater, CommandHandler

from telegram.ext import Updater, InlineQueryHandler, CommandHandler
import requests
import re
import time
import telepot
import numpy as np
from DataCleaner.Extractor import base_model
from Generator.KNN import KNNRandomGenerator


TelegramBot = telepot.Bot(API_TOKEN)

print("loading features, wait")
knn_generator = KNNRandomGenerator(np.load('../DataCleaner/extracted_features.npy'),base_model)
print('ready, bring them')
def main():
    last_date = int(open('last_date').read())

    while True:
        # print(TelegramBot.getMe())
        messages = TelegramBot.getUpdates()
        last_date_ = last_date
        for message_ in messages:
            message = message_['message']
            chat_id = message['from']['id']
            if message['date'] <= last_date:
                continue
            if not 'photo' in message or not 'file_id' in message['photo'][0]:
                # actually send some shit
                # print(message['photo'][0]['file_id'])
                TelegramBot.sendMessage(chat_id,'Send a picture, not as a file but as a picture gluck')
                continue
            x = TelegramBot.getFile(message['photo'][0]['file_id'])
            file_path = x['file_path']

            bf = requests.get(f'https://api.telegram.org/file/bot{API_TOKEN}/{file_path}').content
            open('/home/jafar/PycharmProjects/Artistic Neural Networks/' + file_path,'wb').write(bf)
            res = knn_generator.process_image('../' + file_path)
            TelegramBot.sendPhoto(chat_id, photo=open('..' + res, 'rb'))
            last_date_ = max(last_date_, message['date'])

        last_date = last_date_
        open('last_date', 'w').write(str(last_date))
        # TelegramBot.sendPhoto('398385320', photo= open('../test/1.jpg','rb'))
        time.sleep(2)

    # TelegramBot.sendPhoto()


if __name__ == '__main__':
    main()
