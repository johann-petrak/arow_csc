from collections import defaultdict

## This is just to check the interface used

class FeatureVectorTmp1(object):

    def __init__(self,oldvec=None):
        if(oldvec): self.fv = defaultdict(float, oldvec.fv)
        else: self.fv = defaultdict(float)

    @staticmethod
    def create(oldvec=None):
        return FeatureVectorTmp1(oldvec)

    @staticmethod
    def dot(A, B):
        keys = set(A.fv).intersection(B.fv)
        sum = 0.0
        for k in keys:
            sum = sum + A.fv[k] * B.fv[k]
        return sum

    @staticmethod
    def iaddc(addedTo, addedFrom, alpha):
        for k in addedFrom.fv:
            addedTo.fv[k] += alpha * addedFrom.fv[k]


    def __contains__(self, item):
        return item in self.fv

    def __getitem__(self, item):
        return self.fv.__getitem__(item)

    def __setitem__(self, key, value):
        return self.fv.__setitem__(key,value)

    def __iter__(self):
        return self.fv.__iter__()