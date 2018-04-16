import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

import torch.utils.data


TRAIN_DIR = "../data/tracking_images"

TRAIN_SIZE = .8
TEST_SIZE = .1
VAL_SIZE = .1

RESIZE_SCALE = 3
PATTERNS = ['cascade', '423', 'columns', 'twoInLH', 'twoInRh']

class PatternImageDataset(torch.utils.data.Dataset):
    def __init__(self, partition='TRAIN'):
        super(PatternImageDataset, self).__init__()
        self.images = []
        self.labels = []

        if partition == "TRAIN":
            data_file = "data_cache/train_data.npy"
            labels_file = "data_cache/train_labels.npy"
        elif partition.startswith("VALID"):
            data_file = "data_cache/validation_data.npy"
            labels_file = "data_cache/validation_labels.npy"
        else:
            data_file = "data_cache/test_data.npy"
            labels_file = "data_cache/test_labels.npy"

        if not os.path.exists(data_file):
            print("Preparing", partition,  "dataset...")
            for i, name in tqdm(enumerate(PATTERNS), total=len(PATTERNS)):
                for path in os.listdir(TRAIN_DIR):
                    if name in path:
                        full_path = os.path.join(TRAIN_DIR, path)
                        nimg = len(os.listdir(full_path))
                        if partition == "TRAIN":
                            r = range(int(nimg*TRAIN_SIZE))
                        elif partition == "TEST":
                            r = range(int(nimg*TRAIN_SIZE), int(nimg*(TRAIN_SIZE+TEST_SIZE)))
                        elif partition.startswith("VALID"):
                            r = range(int(nimg*(TRAIN_SIZE+TEST_SIZE)), nimg)
                        else:
                            raise ValueError("partition must must be TRAIN, TEST, or VALID")
                        for image_in in r:
                            img = cv2.imread(os.path.join(full_path, str(image_in) + ".png"), 0)
                            h, w = img.shape
                            img = cv2.resize(img, (w//RESIZE_SCALE, h//RESIZE_SCALE))
                            h, w = img.shape
                            self.images.append(img)
                            self.labels.append(i)

            #make train_labels
            self.labels = np.array(self.labels)
            self.images = np.array(self.images).astype('float32')/255

            np.save(data_file, self.images)
            np.save(labels_file, self.labels)
        else:
            self.images = np.load(data_file)
            self.labels = np.load(labels_file)


        self.images = torch.from_numpy(self.images)

        self.labels = torch.from_numpy(self.labels)
        self.labels = self.labels


    def __getitem__(self, i):
        return (self.images[i], self.labels[i])

    def __len__(self):
        return len(self.labels)

    def get_example_shape(self):
        return self.images[0].shape
