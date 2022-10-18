import os
import cv2
import numpy as np
from collections import Counter
from time import time
import tkinter.filedialog
from tkinter import *


def k_nearest_neighbors(predict, k):
    distances = []
    for image in training_data:
        distances.append([np.linalg.norm(image[0] - predict), image[1]]) # calcul de distance euclidienne
    distances.sort()
    votes = [i[1] for i in distances[:k]]
    votes = ''.join(str(e) for e in votes)
    votes = votes.replace(',', '')
    votes = votes.replace(' ', '')
    result = Counter(votes).most_common(1)[0][0]
    return result


def test():
    start = time()
    correct = 0
    total = 0
    skipped = 0
    for i in range(len(x_test)+1):
        try:
            prediction = k_nearest_neighbors(x_test[i], 5)
            if int(prediction) == y_test[i]:
                correct += 1
            total += 1
        except Exception as e:
            print('An exception occured')
            skipped += 1
    accuracy = correct/total
    end = time()
    print(end-start)
    print(accuracy)


def main():
    root = Tk()
    root.withdraw()
    root.update()
    filename = tkinter.filedialog.askopenfilename(title="Ouvrir fichier", filetypes=[('all files', '.*')]) # sélectionner la photo
    src = cv2.imread(cv2.samples.findFile(filename), cv2.IMREAD_COLOR) # charger la photo
    root.destroy()

    img = resize_img(src)
    pred = k_nearest_neighbors(img, 10)
    if pred == '0':
        print('Coin')
    else:
        print('Banknote')


def resize_img(img):
    dim = (150, 150)
    new_img = cv2.resize(img, dim)
    return new_img

if __name__=="__main__":
    coin_datadir_train = '../coins-dataset/classified/train'
    coin_datadir_test = '../coins-dataset/classified/test'
    note_datadir_train = '../banknote-dataset/classified/train'
    note_datadir_test = '../banknote-dataset/classified/test'

    categories = ['1c', '2c', '5c', '10c', '20c', '50c', '1e', '2e', '5e', '10e', '20e', '50e']
    coin_index = 8

    training_data = []

    for category in categories[:coin_index]:
        path = os.path.join(coin_datadir_train, category)
        label = 0
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img))
            training_data.append([img_array, label])

    for category in categories[coin_index:]:
        path = os.path.join(note_datadir_train, category)
        label = 1
        for img in os.listdir(path):
            img_array = resize_img(cv2.imread(os.path.join(path, img)))
            training_data.append([img_array, label])


    testing_data = []

    for category in categories[:coin_index]:
        path = os.path.join(coin_datadir_test, category)
        label = 0
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img))
            testing_data.append([img_array, label])

    for category in categories[coin_index:]:
        path = os.path.join(note_datadir_test, category)
        label = 1
        for img in os.listdir(path):
            img_array = resize_img(cv2.imread(os.path.join(path, img)))
            testing_data.append([img_array, label])


    x_train = []
    y_train = []

    for features, label in training_data:
        x_train.append(features)
        y_train.append(label)
        
    x_train = np.array(x_train)


    x_test = []
    y_test = []

    for features, label in testing_data:
        x_test.append(features)
        y_test.append(label)
        
    x_test = np.array(x_test)
    main()
