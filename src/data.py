import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

import torch.utils.data


DATA_DIR = "../data/tracking_images"

RESIZE_SCALE = 3
PATTERNS = ['ss3', 'ss423', 'ss441', 'ss50505', 'ss531', 'ss42', 'columns']

class PatternImageDataset(torch.utils.data.Dataset):
    def __init__(self, partition='TRAIN'):
        super(PatternImageDataset, self).__init__()
        self.images = []
        self.labels = []

        print("Preparing", partition, "dataset...")
        for i, name in tqdm(enumerate(PATTERNS), total=len(PATTERNS)):
            for path in os.listdir(DATA_DIR):
                if name in path:
                    full_path = os.path.join(DATA_DIR, path, partition.lower())
                    imgs = [x for x in os.listdir(full_path) if os.path.splitext(x)[1] == '.png']
                    for image_name in imgs:
                        img_path = os.path.join(full_path, image_name)
                        self.images.append(img_path)
                        self.labels.append(i)

        #make train_labels
        self.labels = np.array(self.labels)
        self.labels = torch.from_numpy(self.labels)

    def _process_image(self, img_path):
        img = cv2.imread(img_path, 0)
        h, w = img.shape
        img = cv2.resize(img, (w//RESIZE_SCALE, h//RESIZE_SCALE)).astype('float32')/255
        h, w = img.shape
        img = np.expand_dims(img, 0)
        return torch.from_numpy(img)


    def __getitem__(self, i):
        img = self._process_image(self.images[i])        
        return (img, self.labels[i])

    def __len__(self):
        return len(self.labels)

    def get_example_shape(self):
        return self[0][0].shape


class TrackingDataset(torch.utils.data.Dataset):
    def __init__(self, partition='TRAIN'):
        pass

if __name__ == "__main__":
    x = PatternImageDataset("TRAIN")
    print(x.get_example_shape())
