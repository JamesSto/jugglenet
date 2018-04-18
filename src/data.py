import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

import torch.utils.data


DATA_DIR = "../data/tracking_images"

RESIZE_SCALE = 3
PATTERNS = ['ss3', 'ss423', 'ss441', 'ss50505', 'ss531', 'columns', 'twoInLH', 'twoInRh']

class PatternImageDataset(torch.utils.data.Dataset):
    def __init__(self, partition='TRAIN'):
        super(PatternImageDataset, self).__init__()
        self.images = []
        self.labels = []

        if partition == "TRAIN":
            data_file = "data_cache/train_images.npy"
            labels_file = "data_cache/train_image_labels.npy"
        elif partition == "VALID":
            data_file = "data_cache/validation_images.npy"
            labels_file = "data_cache/validation_image_labels.npy"
        else:
            data_file = "data_cache/test_images.npy"
            labels_file = "data_cache/test_image_labels.npy"

        if not os.path.exists(data_file):
            print("Preparing", partition,  "dataset...")
            for i, name in tqdm(enumerate(PATTERNS), total=len(PATTERNS)):
                for path in os.listdir(DATA_DIR):
                    if name in path:
                        full_path = os.path.join(DATA_DIR, path, partition.lower())
                        imgs = [x for x in os.listdir(full_path) if os.path.splitext(x)[1] == '.png']
                        for image_name in imgs:
                            img_path = os.path.join(full_path, image_name)
                            img = self._process_image(img_path)
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

    def _process_image(self, img_path):
        img = cv2.imread(img_path, 0)
        h, w = img.shape
        img = cv2.resize(img, (w//RESIZE_SCALE, h//RESIZE_SCALE))
        h, w = img.shape
        img = np.expand_dims(img, 0)
        return img


    def __getitem__(self, i):
        return (self.images[i], self.labels[i])

    def __len__(self):
        return len(self.labels)

    def get_example_shape(self):
        return self.images[0].shape


class TrackingDataset(torch.utils.data.Dataset):
    def __init__(self, partition='TRAIN'):
        pass

if __name__ == "__main__":
    x = PatternImageDataset("TRAIN")
    print(x.get_example_shape())
