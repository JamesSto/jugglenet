import cv2, numpy as np, pandas as pd, os, math
import numpy as np
import os
import pandas as pd
import sys
from tqdm import tqdm

#this is how long the lines are
feature_track_length = 120
WHITE = (255, 255, 255)

def make_pictures(tracking_file, out_folder=None, show=False):
    df = pd.read_csv(tracking_file)
    #start at frame 0
    frame_number = 0
    count = 0
    while frame_number < len(df.values):
        #create a new blank image
        img = np.zeros((800, 480, 3), np.uint8)
        #iterate through all the points (and draw lines between them)
        for i in range(feature_track_length):
            a = df.values[frame_number-i]
            a = (int(a[0]), int(a[1])), (int(a[2]), int(a[3])), (int(a[4]), int(a[5]))
            coords = a
            a = df.values[frame_number-1-i]
            a = (int(a[0]), int(a[1])), (int(a[2]), int(a[3])), (int(a[4]), int(a[5]))
            next_coords = a

            cv2.line(img, coords[0], next_coords[0], WHITE, 2)
            cv2.line(img, coords[1], next_coords[1], WHITE, 2)
            cv2.line(img, coords[2], next_coords[2], WHITE, 2)
            
        if show:
            cv2.imshow('img', img)
        if out_folder is None:
            out_folder, _ = os.path.splitext(tracking_file)
        if not os.path.exists(out_folder):
            os.mkdir(out_folder)
        cv2.imwrite(out_folder+"/" + str(count)+'.png', img)
        k = cv2.waitKey(1)
        if k == 27: 
            break
        frame_number += 2
        count += 1
    cv2.destroyAllWindows()


if __name__ == "__main__":
    for pattern in tqdm(os.listdir("../tracking_data")):
        print(pattern)
        out_folder, _ = os.path.splitext(pattern)
        if os.path.exists("../tracking_images/" + out_folder):
            continue
        make_pictures("../tracking_data/" + pattern, "../tracking_images/" + out_folder)
