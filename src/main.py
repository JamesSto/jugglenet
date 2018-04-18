import os
import pickle

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

import torch
import torch.utils.data
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from tqdm import tqdm

from models import BasicDenseNetwork, ConvolutionalNetwork
from data import PatternImageDataset, PATTERNS


NUM_EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.00005

def train(model, epoch, train_loader, optimizer, writer):
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
        global_step = batch_idx + epoch*math.ceil(len(train_loader.dataset)/train_loader.batch_size)
        writer.add_scalar("train/accuracy", accuracy, global_step)
        writer.add_scalar("train/loss", loss, global_step)

def eval(model, epoch, valid_loader, writer):
    model.eval()

    correct_total = 0
    loss_total = 0
    num_batches = 0
    for _, (imgs, labels) in enumerate(valid_loader):
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

    writer.add_scalar("valid/accuracy", accuracy, epoch)
    writer.add_scalar("valid/loss", loss, epoch)


def main():
    train_dataset = PatternImageDataset()
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=BATCH_SIZE,
                                               shuffle=True)
    model = BasicDenseNetwork(train_dataset.get_example_shape(), len(PATTERNS))
    # model = ConvolutionalNetwork(train_dataset.get_example_shape(), len(PATTERNS))


    valid_dataset = PatternImageDataset("VALID")
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True)

    optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARNING_RATE)
    writer = SummaryWriter()


    for i in range(NUM_EPOCHS):
        train(model, i, train_loader, optimizer, writer)
        eval(model, i, valid_loader, writer)

if __name__ == "__main__":
    main()
