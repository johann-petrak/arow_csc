# Andreas Vlachos, 2013:
# export PYTHONPATH="hvector/build/lib.linux-x86_64-2.7/:$PYTHONPATH"
import sys
sys.path.append("hvector") 
from _mycollections import mydefaultdict
from mydouble import mydouble, counts
import cPickle as pickle
import gzip
from operator import itemgetter

import random
import math
import numpy


class Prediction(object):
    """
    A prediction (?)
    """
    def __init__(self):
        self.label2score = {}
        self.score = float("-inf")
        self.label = None
        self.featureValueWeights = []
        self.label2prob = {}
        self.entropy = 0.0


