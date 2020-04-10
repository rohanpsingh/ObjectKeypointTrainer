import argparse
import sys
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import re
import matplotlib.pylab as pylab


def parselogfile(f):
    lines = [line.rstrip("\n") for line in f.readlines()]
    iters = []
    train_loss = []
    valid_loss = []
    for line in lines:
        if len(line) == 0:
            continue
        parts = re.split(" ", line)
        if "iters:" in parts and "train_loss:" in parts and "valid_loss:" in parts:
            for i in range(len(parts)):
                if parts[i]=="iters:":
                    iters.append(int(parts[i+1]))
                if parts[i]=="train_loss:":
                    train_loss.append(float(parts[i+1]))
                if parts[i]=="valid_loss:":
                    valid_loss.append(float(parts[i+1]))
        else:
            continue
    return iters, train_loss, valid_loss

def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "log_file_1",
        help = "path to log file"
        )
    parser.add_argument(
        "log_file_2",
        help = "path to log file"
        )
    args = parser.parse_args()

    f = open(args.log_file_1)
    iters1, train1, valid1 = parselogfile(f)
    f = open(args.log_file_2)
    iters2, train2, valid2 = parselogfile(f)


    params = {'legend.fontsize': 'xx-large',
              'figure.figsize': (15, 5),
              'axes.labelsize': 'xx-large',
              'axes.titlesize':'xx-large',
              'xtick.labelsize':'xx-large',
              'ytick.labelsize':'xx-large'}
    pylab.rcParams.update(params)

    plt.plot(iters1[3:60], valid1[3:60], 'r', label="Loss = L_hm", linewidth=5.0)
    plt.plot(iters2[3:60], valid2[3:60], 'g', label="Loss = L_hm + L_mv", linewidth=5.0)

    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Validation loss', fontsize=20)
    plt.grid()
    plt.legend(loc='upper right')
    plt.show()


if __name__ == "__main__":
    main(sys.argv)
