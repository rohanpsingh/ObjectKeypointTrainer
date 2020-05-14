from __future__ import print_function
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import torch
import numpy as np
import random
import argparse
import os
import shutil
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch import autograd
from torch.utils.data import DataLoader

from dataloader import ObjectKeypointDataset
from models.StackedHourGlass import StackedHourGlass


def train(net, train_data, criterion, optimizer, epoch):
    net.train()
    heatmaploss = criterion
    losses = []

    for b, (inputs, targets, meta) in enumerate(train_data):
        inputs  = torch.autograd.Variable(inputs.cuda())
        targets = torch.autograd.Variable(targets.cuda())
        out = net(inputs)
        hm_loss = heatmaploss(out[1], targets)
        loss = hm_loss

        losses.append(loss.data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return sum(losses)/len(losses)

def eval(net, valid_data, criterion):
    net.eval()
    heatmaploss = criterion
    error = []
    with torch.no_grad():
        for b, (inputs, targets, meta) in enumerate(valid_data):
            inputs = torch.autograd.Variable(inputs.cuda())
            targets = torch.autograd.Variable(targets.cuda())
            out = net(inputs)
            error.append(heatmaploss(out[1], targets).data)
    return sum(error)/len(error)


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", required=False, default=False)
    ap.add_argument("-e", "--epochs", required=False, default=1000)
    ap.add_argument("-d", "--dataset", required=True)
    ap.add_argument("-c", "--checkpoint", required=False, default=5)
    ap.add_argument("--batch_size", required=False, default=8)
    ap.add_argument("--num_keypts", required=False, default=9)
    ap.add_argument("--rate", required=False, default=2.5e-4)
    args = vars(ap.parse_args())
    
    outdir = args["out"]
    num_epochs = int(args["epochs"])
    dataset_path = args["dataset"]
    checkpt = int(args["checkpoint"])
    train_batch_size = int(args["batch_size"])
    valid_batch_size = int(args["batch_size"])
    num_feats = int(args["num_keypts"])
    learning_rate = float(args["rate"])

    print("Training for a total of {} epochs".format(num_epochs))
    print("at a learning rate of {}".format(learning_rate))
    print("Weights are saved after every {} epochs".format(checkpt))
    print("Train batch size: {}".format(train_batch_size))
    print("Valid batch size: {}".format(valid_batch_size))
    print("Total number of keypoints: {}".format(num_feats))

    #set manual seeds and defaults
    manualSeed = 0
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    cudnn.deterministic=True
    cudnn.benchmark = False
    torch.set_default_tensor_type(torch.FloatTensor)

    #set network model, loss criterion and optimizer
    net = StackedHourGlass(256, 2, 2, 4, num_feats).cuda()
    net = nn.DataParallel(net).cuda()
    criterion = nn.MSELoss(reduction='mean').cuda()
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate)

    print("Dataset path: {}".format(dataset_path))
    #set train and valid dataloaders
    train_set = ObjectKeypointDataset(os.path.join(dataset_path, "train.txt"), num_feats, 256, 64)
    train_data = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=6)
    print("train data size is: {} batches of batch size: {}".format(len(train_data), train_batch_size))

    valid_set = ObjectKeypointDataset(os.path.join(dataset_path, "valid.txt"), num_feats, 256, 64, is_train=False)
    valid_data = DataLoader(valid_set, batch_size=valid_batch_size, shuffle=True, num_workers=6)
    print("valid data size is: {} batches of batch size: {}".format(len(valid_data), valid_batch_size))

    if outdir:
        print("Writing to ", outdir)
        if os.path.isdir(outdir):
            shutil.rmtree(outdir)
        os.makedirs(os.path.join(outdir,'valids'))
        os.makedirs(os.path.join(outdir,'trains'))
        f = open(os.path.join(outdir,'log'), 'w')
        f.write("train log... \n")
        log = "train_batch_size: " + repr(train_batch_size) + " valid_batch_size: " + repr(valid_batch_size) + "\n" + \
              "train data size is: " + repr(len(train_data)) + " batches of batch size: " + repr(train_batch_size) + "\n" + \
              "valid data size is: " + repr(len(valid_data)) + " batches of batch size: " + repr(valid_batch_size) + "\n" + \
              "num_feats: " + repr(num_feats) + "\n" + \
              repr(optimizer)
        f.write(log + "\n")
        f.close()

    for t in range(num_epochs):
        train_loss = train(net, train_data, criterion, optimizer, t)
        valid_loss = eval(net, valid_data, criterion)
        log = "iters: " + repr(t) + " train_loss: " + repr(train_loss.item()) +  " valid_loss: " + repr(valid_loss.item()) + " lr: " + repr(optimizer.param_groups[0]['lr'])
        print(log)
        print("================")
        if outdir:
            f = open(os.path.join(outdir,'log'), 'a')
            f.write(log + "\n")
            f.close()
            if (t%checkpt==0):
                torch.save(net.state_dict(), os.path.join(outdir, "trained_" + str(int(t)) + ".model"))

if __name__=='__main__':
    main()
