#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: qichen@cs.toronto.ca
# Created Time : Wed Jan 15 03:25:43 2025
# File Name: node.py
# Description:
"""

import json
import numpy as np
import math
import operator


class Node:
    obj_counter = 0

    # If a leave holds >= SPLIT_THRESH, we split into two new nodes.

    def __init__(self, parent=None, reset_id=False, tau=None, clf=None, is_splitable=False, threshold=0.3):
        # Note: every node is initialized as a leaf
        self.v_bar = float('inf')
        self.n = 0
        self.tau = tau
        self.uct = 0
        self.clf = clf
        self.threshold = threshold

        #insert curt into the kids of parent
        self.parent = parent
        self.kids = []  # 0:good, 1:bad
        self.dataset = None
        self.is_splitable = is_splitable

        if reset_id:
            Node.obj_counter = 0

        self.id = Node.obj_counter

        #data for good and bad kids, respectively
        Node.obj_counter += 1

    def update_kids(self, good_kid, bad_kid):
        assert len(self.kids) == 0
        self.kids.append(good_kid)
        self.kids.append(bad_kid)
        # assert self.kids[0].classifier.get_mean() > self.kids[1].classifier.get_mean()

    def is_good_kid(self):
        if self.parent is not None:
            if self.parent.kids[0] == self:
                return True
            else:
                return False
        else:
            return False

    def is_leaf(self):
        if len(self.kids) == 0:
            return True
        else:
            return False

    def visit(self):
        self.n += 1

    def update_dataset(self, dataset):
        assert len(dataset) > 0
        if self.dataset is not None:
            self.dataset.drop(self.dataset.index, inplace=True)
        self.dataset = dataset
        self.n = len(self.dataset)
        if self.n <= 2:
            self.is_splitable = False
        # print(self.dataset["targets"])
        self.v_bar = self.dataset["targets"].mean()
        self.v_var = self.dataset["targets"].var(ddof=0)

    def clear_data(self):
        if self.dataset is not None:
            self.dataset.drop(self.dataset.index, inplace=True)

    def get_name(self):
        # state is a list of jsons
        return "node" + str(self.id)

    def pad_str_to_8chars(self, ins, total):
        if len(ins) <= total:
            ins += ' ' * (total - len(ins))
            return ins
        else:
            return ins[0:total]

    def get_rand_sample(self):
        if len(self.dataset) > 0:
            rand_idx = np.random.randint(0, self.n)
            return self.dataset.loc[rand_idx]
        else:
            return None

    def get_parent_str(self):
        return self.parent.get_name()

    def __str__(self):
        name = self.get_name()
        name = self.pad_str_to_8chars(name, 7)
        name += (self.pad_str_to_8chars('is good:' + str(self.is_good_kid()), 15))
        name += (self.pad_str_to_8chars('is leaf:' + str(self.is_leaf()), 15))

        val = 0
        name += (self.pad_str_to_8chars(' val:{0:.4f}   '.format(round(self.get_vbar(), 3)), 20))
        name += (self.pad_str_to_8chars(' uct:{0:.4f}   '.format(round(self.get_uct(), 3)), 20))

        name += self.pad_str_to_8chars('sp/n:' + "/" + str(self.n), 15)

        parent = '----'
        if self.parent is not None:
            parent = self.parent.get_name()
        parent = self.pad_str_to_8chars(parent, 10)

        name += (' parent:' + parent)

        kids = ''
        kid = ''
        for k in self.kids:
            kid = self.pad_str_to_8chars(k.get_name(), 10)
            kids += kid
        name += (' kids:' + kids)

        return name

    def get_uct(self, lmbda=10):
        """
        As in LA-MCTS
        """
        if self.parent == None:
            return float(-1000000)
        if self.n == 0:
            return float(-1000000)
        return self.v_bar + 2 * lmbda * math.sqrt(2 * np.power(self.parent.n, 0.5) / self.n)

    def get_uct2(self, lmbda=10):
        """
           mean + 2 * lambda * sqrt(variance)
        """
        if self.parent == None:
            return float(-1000000)
        if self.n == 0:
            return float(-1000000)
        return self.v_bar + 2 * lmbda * math.sqrt(self.v_var)

    def get_vbar(self):
        return self.v_bar

    def get_n(self):
        return self.n
