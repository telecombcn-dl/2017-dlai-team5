import csv
import os
import matplotlib.pyplot as plt
import numpy as np

from sys import argv, exit

usage = """Usage: {} file

Reads CSV file and outputs 2nd and 3rd column as plot (in PNG and in PDF) in
the working dir."""

if __name__ == "__main__":
    if len(argv) != 2:
        print(usage.format(argv[0]))
        exit(0)
    fn = argv[1]
    x = []
    y = []
    with open(fn, 'r') as f:
        reader = csv.reader(f)
        next(reader) # discard header
        for row in reader:
            x.append(float(row[1]))
            y.append(float(row[2]))
    N_avg = 75
    y_avg = np.convolve(y, np.ones(N_avg)/N_avg, mode='same')
    # long_series_avg = np.convolve(long_series, np.ones(N_avg)/N_avg, mode='same')
    plt.plot(x[:-N_avg], y[:-N_avg], '#d3ebd5')
    # plt.hold('on')
    plt.plot(x[:-N_avg], y_avg[:-N_avg], '#01597f', linewidth=2.0)
    plt.xlabel('#Episode')
    plt.ylabel('Reward')
    plt.grid('on')
    plt.savefig('out.pdf')
    plt.savefig('out.png')


