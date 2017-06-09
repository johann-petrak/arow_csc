import sys
sys.path.append("hvector") 
from _mycollections import mydefaultdict
from mydouble import mydouble, counts

class FeatureVectorHvector(object):

    @staticmethod
    def create(oldvec=None):
        if(oldvec): return mydefaultdict(mydouble,oldvec)
        else: return mydefaultdict(mydouble)

    @staticmethod
    def dot(fv1, fv2):
        return fv1.dot(fv2)

    @staticmethod
    def iaddc(fv1, fv2, val):
        return fv1.iaddc(fv2, val)