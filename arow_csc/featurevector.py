import sys
sys.path.append("hvector") 
from _mycollections import mydefaultdict
from mydouble import mydouble, counts
from collections import defaultdict

def make_featurevector_old(oldvec=None):
    if(oldvec): return mydefaultdict(mydouble,oldvec)
    else: return mydefaultdict(mydouble)

def make_featurevector_new(oldvec=None):
    if(oldvec): return defaultdict(float,oldvec)
    else: return defaultdict(float)

def dot_old(fv1, fv2):
    return fv1.dot(fv2)

def iaddc_old(fv1, fv2, val):
    return fv1.iaddc(fv2, val)

def dot_new(A, B):
    keys = set(A).intersection(B)
    sum = 0.0
    for k in keys:
        sum = sum + A[k] * B[k]
    return sum

def iaddc_new(addedTo, addedFrom, alpha):
    for k in addedFrom:
        addedTo[k] += alpha * addedFrom[k]

dot = dot_old
iaddc = iaddc_old
make_featurevector = make_featurevector_old