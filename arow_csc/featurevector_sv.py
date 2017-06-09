from sparse_vector import SparseVector

class FeatureVectorSV(object):

    @staticmethod
    def create(oldvec=None):
        if(oldvec): return SparseVector(oldvec)
        else: return SparseVector(1)

    @staticmethod
    def dot(A, B):
        sum = 0.0
        for i in A.indices:
            bval = B[i]
            if bval != 0.0:
                sum = sum + A[i] * bval
        return sum

    @staticmethod
    def iaddc(addedTo, addedFrom, alpha):
        for i in addedFrom.indices:
            toval = addedTo[i]
            if toval != 0.0:
                addedTo[i] += alpha * addedFrom[i]
            else:
                addedTo[i] = alpha * addedFrom[i]

    #@staticmethod
    #def set(fv, el, val):
    #    fv[el] = val