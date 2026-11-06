#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: qichen@cs.toronto.ca
# Created Time : 2025-11-25
# File Name: __init__.py
# Description: TOY benchmarks module for testing with synthetic functions (Levy, Ackley)
"""

from .toy_benchmarks import ToyBench, Levy, Ackley

__all__ = ['ToyBench', 'Levy', 'Ackley']
