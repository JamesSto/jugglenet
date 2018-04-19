import os
import json
import numpy as np
import cv2

#this is how long the lines are
FEATURE_TRACK_LENGTH = 120
WHITE = (255, 255, 255)

TRAIN_SIZE = 0.8
VALID_SIZE = 0.1
TEST_SIZE = 0.1

STRIDE = 2

IMAGE_Shape = (800, 480, 3)

def make_pictures(frame_data, out_folder, show):
    count = 0
    for frame_start in range(0, len(frame_data)-FEATURE_TRACK_LENGTH, STRIDE):
        img = np.zeros(IMAGE_Shape, np.uint8)
        all_coords = []
        for frame_number in range(frame_start, frame_start+FEATURE_TRACK_LENGTH-1):
            frame = frame_data[frame_number]
            coords = list(zip(*[iter(frame)]*2))
            all_coords.append(coords)
            next_frame = frame_data[frame_number+1]
            next_coords = zip(*[iter(next_frame)]*2)

            for l1, l2 in zip(coords, next_coords):
                cv2.line(img, l1, l2, WHITE, 2)

        if show:
            cv2.imshow('img', img)
        cv2.imwrite(out_folder+"/" + str(count)+'.png', img)
        np.save(out_folder+"/" + str(count), np.array(all_coords))
        k = cv2.waitKey(1)
        count += 1

def construct_image_dataset(tracking_file, out_folder=None, show=False):
    with open(tracking_file, 'r') as f:
        data = f.readlines()[2:]
    data = [[int(x.strip()) for x in d.split(",")] for d in data]
    if len(data) * min(VALID_SIZE, TEST_SIZE) < FEATURE_TRACK_LENGTH:
        # print("Cannot create sufficient data for ", tracking_file, "- only", len(data), "frames")
        return
    else:
        print("Creating data for", tracking_file)

    if out_folder is None:
        out_folder, _ = os.path.splitext(tracking_file)
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    if not os.path.exists(os.path.join(out_folder, "train")):
        os.mkdir(os.path.join(out_folder, "train"))
        os.mkdir(os.path.join(out_folder, "test"))
        os.mkdir(os.path.join(out_folder, "valid"))
    
    make_pictures(data[:int(len(data)*TRAIN_SIZE)], os.path.join(out_folder, "train"), show)        
    make_pictures(data[int(len(data)*TRAIN_SIZE):int(len(data)*(TRAIN_SIZE+TEST_SIZE))], os.path.join(out_folder, "test"), show)
    make_pictures(data[int(len(data)*(TRAIN_SIZE+TEST_SIZE)):], os.path.join(out_folder, "valid"), show)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    tracking_data_dir = "../../data/tracking_data/"
    tracking_image_dir = "../../data/new_tracking_images/"
    for pattern in os.listdir(tracking_data_dir):
        out_dir, _ = os.path.splitext(pattern)
        if os.path.exists(tracking_image_dir + out_dir):
            continue
        construct_image_dataset(tracking_data_dir + pattern, tracking_image_dir + out_dir, False)
