#! /usr/bin/env/ python3
# -*- coding: utf-8 -*-

from sys import argv, exit
import getopt
from core import Core
import numpy as np

def main(argv):
    try:
        opt_list, args = getopt.getopt(argv, ":h", ["file=", 'help'])
    except getopt.GetoptError:
        print("main.py: getopterror")
        exit(2)

    for opt, arg in opt_list:
        if opt == "-h":
            print("--file <input file>")
            exit(1)
        elif opt == "--file":
            Core(file_path=arg)
        else:
            exit(1)

if __name__ == '__main__':
    main(argv[1:])