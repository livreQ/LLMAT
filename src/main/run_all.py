#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: qichen@cs.toronto.ca
# Created Time : Sun Jul 21 01:47:25 2024
# File Name: run_all.py
# Description:
"""

from src.main.args import parse_args
import argparse
from src.main.run_benchmarks import run_mat, run_toy


def main(args):

    if args.benchmark == 'mat':
        run_mat(args)
    elif args.benchmark == 'toy':
        run_toy(args)
    else:
        raise (NotImplementedError)


if __name__ == '__main__':

    args = parse_args()
    main(args)
