import csv
import os
import mathplotlib.plt as plt

from sys import argv

assert len(argv) == 2

if __name__ == "__main__":
    folder = argv[1]
    files = filter(lambda f: os.listdir(folder))


