import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
import os
from tqdm import tqdm
import pickle

TRAIN_DIR = "../tracking_images"

TRAIN_SIZE = .8
TEST_SIZE = .1
VAL_SIZE = .1

#get training data
train_data = []
test_data = []
validation_data = []

train_labels = []
test_labels = []
validation_labels = []

resize_scale = 3
patterns = ['cascade', '423', 'columns', 'twoInLH', 'twoInRh']

if not os.path.exists("data_cache/train_data.npy"):
    for i, name in tqdm(enumerate(patterns), total=len(patterns)):
        for path in os.listdir(TRAIN_DIR):
            if name in path:
                full_path = os.path.join(TRAIN_DIR, path)
                nimg = len(os.listdir(full_path))
                for image_in in range(nimg):
                    img = cv2.imread(os.path.join(full_path, str(image_in) + ".png"), 0)
                    h, w = img.shape
                    img = cv2.resize(img, (w//resize_scale, h//resize_scale))
                    h, w = img.shape
                    if image_in < nimg * TRAIN_SIZE:
                        train_data.append(img)
                        train_labels.append(i)
                    elif image_in < nimg * (TRAIN_SIZE + TEST_SIZE):
                        test_data.append(img)
                        test_labels.append(i)
                    else:
                        validation_data.append(img)
                        validation_labels.append(i)

    #make train_labels
    train_labels = np.array(train_labels)

    train_data = np.array(train_data).reshape(len(train_labels), -1).astype('float32')/255

    test_labels = np.array(test_labels)
    test_data = np.array(test_data).reshape(len(test_labels), -1).astype('float32')/255

    validation_labels = np.array(validation_labels)
    validation_data = np.array(validation_data).reshape(len(validation_labels), -1).astype('float32')/255

    np.save("data_cache/train_data.npy", train_data)
    np.save("data_cache/train_labels.npy", train_labels)

    np.save("data_cache/validation_data.npy", validation_data)
    np.save("data_cache/validation_labels.npy", validation_labels)

    np.save("data_cache/test_data.npy", test_data)
    np.save("data_cache/test_labels.npy", test_labels)
else:
    train_data = np.load("data_cache/train_data.npy")
    train_labels = np.load("data_cache/train_labels.npy")

    test_data = np.load("data_cache/test_data.npy")
    test_labels = np.load("data_cache/test_labels.npy")

    validation_data = np.load("data_cache/validation_data.npy")
    validation_labels = np.load("data_cache/validation_labels.npy")

print("done constructing data")


train_labels_one_hot = to_categorical(train_labels)
test_labels_one_hot = to_categorical(test_labels)
validation_labels_one_hot = to_categorical(validation_labels)

print(train_data.shape)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(train_data.shape[-1],)))
model.add(Dense(512, activation='relu'))
model.add(Dense(len(patterns), activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_data, train_labels_one_hot, batch_size=256, epochs=2, verbose=1, 
                    validation_data=(validation_data, validation_labels_one_hot))

#TEST MODEL
[validation_loss, validation_acc] = model.evaluate(validation_data, validation_labels_one_hot)
print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(validation_loss, validation_acc))
model.save('test.model')
