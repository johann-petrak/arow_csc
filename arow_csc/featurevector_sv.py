from sparsevectors import SparseVector

class FeatureVectorSV(object):

    @staticmethod
    def create(oldvec=None):
        if(oldvec): return SparseVector(oldvec)
        else: return SparseVector()

    @staticmethod
    def dot(A, B):
        return A.dot(B)

    @staticmethod
    def iaddc(addedTo, addedFrom, alpha):
        return addedTo.iaddc(addedFrom, alpha)

