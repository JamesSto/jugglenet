import os
import json

import cv2
import numpy as np
import torch
from tqdm import tqdm

import torch.utils.data


DATA_DIR = "../data/tracking_images"

RESIZE_SCALE = 3
PATTERNS = ['ss3', 'ss423', 'ss441', 'ss50505', 'ss531', 'ss42', 'columns', 'box']

class PatternDataset(torch.utils.data.Dataset):
    def __init__(self, partition='TRAIN', use_images=True):
        super(PatternDataset, self).__init__()
        self.images = []
        self.tracks = []
        self.labels = []
        self.num_examples = []

        self.use_images = use_images

        print("Preparing", partition, "dataset...")
        for i, name in enumerate(PATTERNS):
            count = 0
            for path in os.listdir(DATA_DIR):
                if name + "_" in path:
                    full_path = os.path.join(DATA_DIR, path, partition.lower())
                    imgs = [x for x in os.listdir(full_path) if os.path.splitext(x)[1] == '.png']
                    tracks = [x for x in os.listdir(full_path) if os.path.splitext(x)[1] == '.npy']
                    assert(len(imgs) == len(tracks))
                    for image_name, track_name in zip(imgs, tracks):
                        self.labels.append(i)
                        if self.use_images:
                            img_path = os.path.join(full_path, image_name)
                            self.images.append((img_path, False))
                            # Append both the flipped and unflipped version to the training image dataset
                            if partition == "TRAIN":
                                self.images.append((img_path, True))
                                self.labels.append(i)
                                count += 1
                        else:
                            track_path = os.path.join(full_path, track_name)
                            self.tracks.append(track_path)
                        count += 1
            self.num_examples.append(count)
            if partition == "TRAIN":
                print(PATTERNS[i] + ": ", count, "training examples")


        #make train_labels
        self.labels = np.array(self.labels)
        self.labels = torch.from_numpy(self.labels)

        mean = sum(self.num_examples)/len(self.num_examples)
        self.weights = torch.Tensor([mean/x for x in self.num_examples])

    def _process_image(self, img_path, flip):
        img = cv2.imread(img_path, 0)
        h, w = img.shape
        img = cv2.resize(img, (w//RESIZE_SCALE, h//RESIZE_SCALE)).astype('float32')/255
        if flip:
            img = cv2.flip(img, 1)
        img = np.expand_dims(img, 0)
        return torch.from_numpy(img)

    def _process_track(self, track_path):
        track = np.load(track_path)
        return torch.from_numpy(track)

    def __getitem__(self, i):
        if self.use_images:
            img = self._process_image(*self.images[i])        
            return (img, self.labels[i])
        else:
            track = self._process_track(self.tracks[i])
            return track, self.labels[i]

    def __len__(self):
        return len(self.labels)

    def get_example_shape(self):
        return self[0][0].shape

    def get_weights(self):
        return self.weights


if __name__ == "__main__":
    x = PatternDataset("TRAIN", use_images=False)
    # from random import shuffle
    # s = list(range(len(x)))
    # shuffle(s)
    # for index in s:
    #     img, label = x[index]
    #     if label != PATTERNS.index("ss531"):
    #         continue
    #     cv2.imshow("531 image", np.expand_dims(np.squeeze(img.numpy()), 2))
    #     cv2.waitKey(100)
    print(x.get_example_shape())
    print(x.get_weights())
