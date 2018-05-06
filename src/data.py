import os
import json

import cv2
import numpy as np
import torch
from tqdm import tqdm

import torchvision.transforms as transforms
import torch.utils.data



TRAIN_SIZE = 0.8
VALID_SIZE = 0.1
TEST_SIZE = 0.1

IMAGE_DIR = "../data/tracking_images"
VIDEO_DIR = "../data/videos"

RESIZE_SCALE = 2
PATTERNS = ['ss3', 'ss423', 'ss441', 'ss50505', 'ss531', 'ss42', 'columns']

# For now, this dataset sucks.
# The videos are in different enough places that any reasonable model
# can get good accuracy presumably just by looking at the background
# Will need to collect more data to make this reasonable
class FrameDataset(torch.utils.data.Dataset):
    def __init__(self, partition='TRAIN'):
        self.examples = []
        self.labels = []
        self.num_examples = []

        for i, name in enumerate(PATTERNS):
            count = 0
            for path in os.listdir(VIDEO_DIR):
                if name + "_" in path:
                    vid_path = os.path.join(VIDEO_DIR, path)
                    vid_len = FrameDataset.get_frame_count(vid_path)
                    if partition == "TRAIN":
                        portion = list(range(vid_len))[:int(vid_len*TRAIN_SIZE)]
                    elif partition == "TEST":
                        portion = list(range(vid_len))[int(vid_len*TRAIN_SIZE):int(vid_len*(TRAIN_SIZE+TEST_SIZE))]
                    elif partition == "VALID":
                        portion = list(range(vid_len))[int(-vid_len*TEST_SIZE):]
                    else:
                        raise ValueError("invalid partition")
                    self.examples.extend(zip(portion, [vid_path]*len(portion)))
                    self.labels.extend([i]*len(portion))
                    count += len(portion)
            self.num_examples.append(count)

        self.labels = np.array(self.labels)
        self.labels = torch.from_numpy(self.labels)

        mean = sum(self.num_examples)/len(self.num_examples)
        self.weights = torch.Tensor([mean/x for x in self.num_examples])

    def get_weights(self):
        return self.weights

    def __getitem__(self, i):
        frame = FrameDataset.extract_frame(*self.examples[i])
        return frame, self.labels[i]

    def __len__(self):
        return len(self.examples)

    def get_example_shape(self):
        return self[0][0].shape

    @staticmethod
    def extract_frame(frame_index, video_file):
        cap = cv2.VideoCapture(video_file)
        cap.set(1, frame_index)
        _, frame = cap.read()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        frame = cv2.resize(frame, (224, 224)).T
        return normalize(torch.from_numpy(frame.astype('float32')/255))

    @staticmethod
    def get_frame_count(video_file):
        vid = cv2.VideoCapture(video_file)
        count = vid.get(cv2.CAP_PROP_FRAME_COUNT)
        if count > 0:
            return int(count)

        grabbed, _ = vid.read()
        count = 0
        while grabbed:
            count += 1
            grabbed, _ = vid.read()
        return count



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
            for path in os.listdir(IMAGE_DIR):
                if name + "_" in path:
                    full_path = os.path.join(IMAGE_DIR, path, partition.lower())
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
            res = self._process_image(*self.images[i])        
        else:
            res = self._process_track(self.tracks[i])

        return res, self.labels[i]

    def __len__(self):
        return len(self.labels)

    def get_example_shape(self):
        return self[0][0].shape

    def get_weights(self):
        return self.weights


if __name__ == "__main__":
    d = FrameDataset()
    print(d.get_example_shape())
