from collections import defaultdict


class FeatureVectorDefaultdict:

    @staticmethod
    def create(oldvec=None):
        if(oldvec): return defaultdict(float,oldvec)
        else: return defaultdict(float)

    @staticmethod
    def dot(A, B):
        keys = set(A).intersection(B)
        sum = 0.0
        for k in keys:
            sum = sum + A[k] * B[k]
        return sum

    @staticmethod
    def iaddc(addedTo, addedFrom, alpha):
        for k in addedFrom:
            addedTo[k] += alpha * addedFrom[k]