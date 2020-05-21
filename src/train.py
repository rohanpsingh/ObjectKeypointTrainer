import random
import argparse
import os
import logging
import numpy as np

import torch
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

    #set up logger
    log_file_path = os.path.join(args.outdir, 'train.log') if args.outdir else 'tmp.log'
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)-6s %(levelname)-8s %(message)s',
                        datefmt='%d/%m/%Y %H:%M:%S',
                        filename=log_file_path,
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(message)s'))
    logging.getLogger('').addHandler(console)

    logging.info("Training for a total of %d epochs", args.epochs)
    logging.info("Weights are saved after every %d epochs", args.checkpoint)
    logging.info("Total number of keypoints: %d", args.num_keypts)
    logging.info("Dataset path: %s", args.dataset)

    #set manual seeds and defaults
    manual_seed = 0
    logging.info("Random Seed: %d", manual_seed)
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark = False
    torch.set_default_tensor_type(torch.FloatTensor)

    #set network model, loss criterion and optimizer
    net = StackedHourGlass(256, 2, 2, 4, args.num_keypts).cuda()
    net = torch.nn.DataParallel(net).cuda()
    criterion = torch.nn.MSELoss(reduction='mean').cuda()
    optimizer = torch.optim.RMSprop(net.parameters(), lr=args.rate)
    logging.info(repr(optimizer))

    #set train and valid dataloaders
    train_set = ObjectKeypointDataset(os.path.join(args.dataset, "train.txt"), args.num_keypts, 256, 64)
    train_data = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=6)
    logging.info("train data size is: %d batches of batch size: %d", len(train_data), args.batch_size)

    valid_set = ObjectKeypointDataset(os.path.join(args.dataset, "valid.txt"), args.num_keypts, 256, 64, is_train=False)
    valid_data = DataLoader(valid_set, batch_size=args.batch_size, shuffle=True, num_workers=6)
    logging.info("valid data size is: %d batches of batch size: %d", len(valid_data), args.batch_size)

    if args.outdir:
        #create output directory if required
        if not os.path.isdir(args.outdir):
            os.makedirs(args.outdir)
        logging.info("Saving weights to %s", args.outdir)
    else:
        logging.info("Output directory not provided. Weights will NOT be saved!!!!")

    for epoch in range(args.epochs):
        train_loss = train(net, train_data, criterion, optimizer)
        valid_loss = valid(net, valid_data, criterion)
        logging.info("iters: %d train_loss: %f valid_loss: %f lr: %f",
                     epoch,
                     train_loss.item(),
                     valid_loss.item(),
                     optimizer.param_groups[0]['lr'])
        if args.outdir and epoch%args.checkpoint==0:
            torch.save(net.state_dict(), os.path.join(args.outdir, "trained_" + str(int(epoch)) + ".model"))

if __name__=='__main__':
    main()
