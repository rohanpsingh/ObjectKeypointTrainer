import random
import argparse
import os
import shutil
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from dataloader import ObjectKeypointDataset
from models.StackedHourGlass import StackedHourGlass


def train(net, train_data, criterion, optimizer):
    net.train()
    heatmaploss = criterion
    losses = []

    for inputs, targets, _ in train_data:
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

def valid(net, valid_data, criterion):
    net.eval()
    heatmaploss = criterion
    error = []
    with torch.no_grad():
        for inputs, targets, _ in valid_data:
            inputs = torch.autograd.Variable(inputs.cuda())
            targets = torch.autograd.Variable(targets.cuda())
            out = net(inputs)
            error.append(heatmaploss(out[1], targets).data)
    return sum(error)/len(error)


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--outdir", required=False, default=False)
    ap.add_argument("-e", "--epochs", required=False, default=100, type=int)
    ap.add_argument("-d", "--dataset", required=True)
    ap.add_argument("-c", "--checkpoint", required=False, default=5, type=int)
    ap.add_argument("--num_keypts", required=True, type=int)
    ap.add_argument("--batch_size", required=False, default=8, type=int)
    ap.add_argument("--rate", required=False, default=2.5e-4, type=float)
    args = ap.parse_args()

    print("Training for a total of {} epochs".format(args.epochs))
    print("at a learning rate of {}".format(args.rate))
    print("Weights are saved after every {} epochs".format(args.checkpoint))
    print("Train batch size: {}".format(args.batch_size))
    print("Valid batch size: {}".format(args.batch_size))
    print("Total number of keypoints: {}".format(args.num_keypts))

    #set manual seeds and defaults
    manual_seed = 0
    print("Random Seed: ", manual_seed)
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    cudnn.deterministic=True
    cudnn.benchmark = False
    torch.set_default_tensor_type(torch.FloatTensor)

    #set network model, loss criterion and optimizer
    net = StackedHourGlass(256, 2, 2, 4, args.num_keypts).cuda()
    net = torch.nn.DataParallel(net).cuda()
    criterion = torch.nn.MSELoss(reduction='mean').cuda()
    optimizer = torch.optim.RMSprop(net.parameters(), lr=args.rate)

    print("Dataset path: {}".format(args.dataset))
    #set train and valid dataloaders
    train_set = ObjectKeypointDataset(os.path.join(args.dataset, "train.txt"), args.num_keypts, 256, 64)
    train_data = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=6)
    print("train data size is: {} batches of batch size: {}".format(len(train_data), args.batch_size))

    valid_set = ObjectKeypointDataset(os.path.join(args.dataset, "valid.txt"), args.num_keypts, 256, 64, is_train=False)
    valid_data = DataLoader(valid_set, batch_size=args.batch_size, shuffle=True, num_workers=6)
    print("valid data size is: {} batches of batch size: {}".format(len(valid_data), args.batch_size))

    if args.outdir:
        print("Writing to ", args.outdir)
        if os.path.isdir(args.outdir):
            shutil.rmtree(args.outdir)
        os.makedirs(os.path.join(args.outdir,'valids'))
        os.makedirs(os.path.join(args.outdir,'trains'))
        log = "args.batch_size: " + repr(args.batch_size) + " args.batch_size: " + repr(args.batch_size) + "\n" + \
              "train data size is: " + repr(len(train_data)) + " batches of batch size: " + repr(args.batch_size) + "\n" + \
              "valid data size is: " + repr(len(valid_data)) + " batches of batch size: " + repr(args.batch_size) + "\n" + \
              "args.num_keypts: " + repr(args.num_keypts) + "\n" + \
              repr(optimizer)
        with open(os.path.join(args.outdir,'log'), 'w') as f:
            f.write("train log... \n")
            f.write(log + "\n")

    for epoch in range(args.epochs):
        train_loss = train(net, train_data, criterion, optimizer)
        valid_loss = valid(net, valid_data, criterion)
        log = "iters: " + repr(epoch) + \
              " train_loss: " + repr(train_loss.item()) +  \
              " valid_loss: " + repr(valid_loss.item()) + \
              " lr: " + repr(optimizer.param_groups[0]['lr'])
        print(log)
        print("================")
        if args.outdir:
            f = open(os.path.join(args.outdir,'log'), 'a')
            f.write(log + "\n")
            f.close()
            if epoch%args.checkpoint==0:
                torch.save(net.state_dict(), os.path.join(args.outdir, "trained_" + str(int(epoch)) + ".model"))

if __name__=='__main__':
    main()
