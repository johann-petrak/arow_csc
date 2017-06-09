
class FeatureVectorDict(object):

    @staticmethod
    def create(oldvec=None):
        if(oldvec): return dict(oldvec)
        else: return dict()

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
            if k in addedTo:
                addedTo[k] += alpha * addedFrom[k]
            else:
                addedTo[k] = alpha * addedFrom[k]

    @staticmethod
    def set(fv, el, val):
        fv[el] = val