import os
import pickle

import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.utils.data
from torch.autograd import Variable

from tqdm import tqdm

from basic_model import BasicNetwork
from data import PatternImageDataset, PATTERNS



NUM_EPOCHS = 10
BATCH_SIZE = 256
LEARNING_RATE = 0.00001

def train(model, epoch, train_loader, optimizer):
    model.train()

    for batch_idx, (imgs, labels) in enumerate(train_loader):
        imgs, labels = Variable(imgs), Variable(labels)
        optimizer.zero_grad()
        output = model(imgs)

        loss = torch.nn.functional.cross_entropy(output, labels)
        loss.backward()
        optimizer.step()

        preds = torch.max(output, dim=1)[1]
        accuracy = torch.sum(torch.eq(preds, labels).float())/len(labels)
        print("Batch", batch_idx, ": loss =", loss.data[0], "| Accuracy: ", accuracy.data[0])

def main():
    train_dataset = PatternImageDataset()
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=BATCH_SIZE,
                                               shuffle=True)
    model = BasicNetwork(train_dataset.get_example_shape(), len(PATTERNS))
    optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARNING_RATE)


    for i in range(NUM_EPOCHS):
        train(model, i, train_loader, optimizer)


if __name__ == "__main__":
    main()