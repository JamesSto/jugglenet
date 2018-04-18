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
        for i, name in enumerate(PATTERNS):
            count = 0
            for path in os.listdir(DATA_DIR):
                if name + "_" in path:
                    full_path = os.path.join(DATA_DIR, path, partition.lower())
                    imgs = [x for x in os.listdir(full_path) if os.path.splitext(x)[1] == '.png']
                    for image_name in imgs:
                        img_path = os.path.join(full_path, image_name)
                        self.images.append((img_path, False))
                        self.labels.append(i)
                        # Append both the flipped and unflipped version to the training image dataset
                        if partition == "TRAIN":
                            self.images.append((img_path, True))
                            self.labels.append(i)
                        count += 1
            if partition == "TRAIN":
                print(PATTERNS[i] + ": ", count, "traing examples")


        #make train_labels
        self.labels = np.array(self.labels)
        self.labels = torch.from_numpy(self.labels)

    def _process_image(self, img_path, flip):
        img = cv2.imread(img_path, 0)
        h, w = img.shape
        img = cv2.resize(img, (w//RESIZE_SCALE, h//RESIZE_SCALE)).astype('float32')/255
        if flip:
            img = cv2.flip(img, 1)
        img = np.expand_dims(img, 0)
        return torch.from_numpy(img)


    def __getitem__(self, i):
        img = self._process_image(*self.images[i])        
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
    from random import shuffle
    s = list(range(len(x)))
    shuffle(s)
    for index in s:
        img, label = x[index]
        if label != PATTERNS.index("ss531"):
            continue
        cv2.imshow("531 image", np.expand_dims(np.squeeze(img.numpy()), 2))
        cv2.waitKey(100)
    print(x.get_example_shape())
