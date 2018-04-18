import math
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.utils.data
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from models import BasicDenseNetwork, ConvolutionalNetwork
from data import PatternImageDataset, PATTERNS


NUM_EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.00003

def per_class_totals(output, labels, num_classes):
    _, output = torch.max(output, dim=1)
    corrects = torch.zeros(len(PATTERNS))
    totals = torch.zeros(len(PATTERNS))
    for i in range(num_classes):
        out = output[labels == i]
        if len(out.size()) < 1:
            out_total = 0
        else:
            out_total = out.size()[0]
        out_corrects = torch.nonzero(out == i)
        if len(out_corrects.size()) > 0:
            num_correct = out_corrects.size()[0]
        else:
            num_correct = 0
        corrects[i] = num_correct
        totals[i] = out_total

    return corrects, totals

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

        corrects, totals = per_class_totals(output, labels, len(PATTERNS))
        for pattern, correct, total in zip(PATTERNS, corrects, totals):
            if total > 0:
                writer.add_scalar("train/" + pattern + "_accuracy", correct/total, global_step)

def eval(model, epoch, valid_loader, writer):
    model.eval()

    correct_totals = torch.zeros(len(PATTERNS))
    class_totals = torch.zeros(len(PATTERNS))
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
        corrects, totals = per_class_totals(output, labels, len(PATTERNS))
        correct_totals += corrects
        class_totals += totals

    accuracy = (correct_total / len(valid_loader.dataset)).data[0]
    loss = (loss_total / num_batches)
    print("\n*****************************************")
    print("EVAL EPOCH", epoch, ": ACCURACY:", round(accuracy, 3), "| LOSS:", round(loss, 3))
    print("*****************************************\n")

    writer.add_scalar("valid/accuracy", accuracy, epoch)
    writer.add_scalar("valid/loss", loss, epoch)

    for pattern, correct, total in zip(PATTERNS, correct_totals, class_totals):
        if total > 0:
            writer.add_scalar("valid/" + pattern + "_accuracy", correct/total, epoch)


def main(model):
    train_dataset = PatternImageDataset()
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=BATCH_SIZE,
                                               shuffle=True)
    if model.lower().startswith("conv"):
        print("Using ConvolutionalNetwork")
        model = ConvolutionalNetwork(train_dataset.get_example_shape(), len(PATTERNS))
    elif model.lower().startswith("b"):
        print("Using FullyConnected Network")
        model = BasicDenseNetwork(train_dataset.get_example_shape(), len(PATTERNS))
    else:
        raise ValueError("Invalid model")


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
    parser = ArgumentParser()
    parser.add_argument("-m", default="conv")
    args = parser.parse_args()

    main(args.m)
