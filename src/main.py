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
        print("Batch", batch_idx, ": loss =", round(loss.data[0], 3), "| Accuracy: ", round(accuracy.data[0], 3))

def eval(model, epoch, valid_loader):
    model.eval()

    correct_total = 0
    loss_total = 0
    num_batches = 0
    for batch_idx, (imgs, labels) in enumerate(valid_loader):
        imgs, labels = Variable(imgs), Variable(labels)
        output = model(imgs)
        loss = torch.nn.functional.cross_entropy(output, labels)
        loss_total += loss.data[0]

        preds = torch.max(output, dim=1)[1]
        correct_total += torch.sum(torch.eq(preds, labels).float())
        num_batches += 1

    accuracy = (correct_total / len(valid_loader.dataset)).data[0]
    loss = (loss_total / num_batches)
    print("\n*****************************************")
    print("EVAL EPOCH", epoch, ": ACCURACY:", round(accuracy, 3), "| LOSS:", round(loss, 3))
    print("*****************************************\n")


def main():
    train_dataset = PatternImageDataset()
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=BATCH_SIZE,
                                               shuffle=True)
    model = BasicNetwork(train_dataset.get_example_shape(), len(PATTERNS))

    valid_dataset = PatternImageDataset("VALID")
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True)

    optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARNING_RATE)


    for i in range(NUM_EPOCHS):
        train(model, i, train_loader, optimizer)
        eval(model, i, valid_loader)

if __name__ == "__main__":
    main()