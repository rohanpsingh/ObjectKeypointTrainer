import argparse
import sys
import matplotlib.pyplot as plt
import re



def parselogfile(logfile):
    with open(logfile) as f:
        lines = [line.rstrip("\n") for line in f.readlines()]
    iters = []
    train_loss = []
    valid_loss = []
    for line in lines:
        if len(line) == 0:
            continue
        parts = re.split(" ", line)
        if "iters:" in parts:
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
    args = parser.parse_args()

    iters, train, valid = parselogfile(args.log_file_1)

    plt.plot(iters, train, 'r', label='train')
    plt.plot(iters, valid, 'g', label='valid')
    plt.xlabel('iters')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(loc='upper right')
    plt.show()


if __name__ == "__main__":
    main(sys.argv)
