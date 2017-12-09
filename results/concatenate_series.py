import csv
import os
import matplotlib.pyplot as plt
import numpy as np

from sys import argv, exit

usage = """Usage: {} folder

Concatentates all the series present in folder (as CSV files) and outputs
a plot (in PNG and in PDF) in the working dir."""

if __name__ == "__main__":
    if len(argv) != 2:
        print(usage.format(argv[0]))
        exit(0)
    folder = argv[1]
    filenames = list(filter(lambda n: n.endswith('.csv'), os.listdir(folder)))
    print(filenames)
    series = []
    for fn in filenames:
        with open(fn, 'r') as f:
            s = []
            reader = csv.reader(f)
            reader.next() # discard header
            for row in reader:
                s.append(float(row[2]))
            series.append(s)
    long_series = []
    for t in zip(*series):
        long_series += list(t)
    episodes = range(1, 5*len(long_series)+1, 5)
    N_avg = 100
    long_series_avg = np.convolve(long_series, np.ones(N_avg)/N_avg, mode='same')
    plt.plot(episodes, long_series, '#d3ebd5')
    plt.hold('on')
    plt.plot(episodes[:-N_avg], long_series_avg[:-N_avg], '#01597f', linewidth=2.0)
    plt.xlabel('#Episode')
    plt.ylabel('Reward')
    plt.grid('on')
    plt.savefig('out.pdf')
    plt.savefig('out.png')


