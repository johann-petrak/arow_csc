from __future__ import print_function
implementation = "hvector"
needimport = True
_impl = None

class FeatureVector:
    @staticmethod
    def setimplementation(impl):
        global implementation
        implementation = impl

    @staticmethod
    def create(oldvec=None):
        global needimport, _impl, implementation
        if needimport:
            needimport = False
            if implementation == "hvector":
                print("Importing hvector implementation")
                from .featurevector_hvector import FeatureVectorHvector
                _impl = FeatureVectorHvector
            elif implementation == "defaultdict":
                print("Importing defaultdict implementation")
                from .featurevector_defaultdict import FeatureVectorDefaultdict
                _impl = FeatureVectorDefaultdict
        return _impl.create(oldvec)

    @staticmethod
    def dot(fv1, fv2):
        return _impl.dot(fv1, fv2)

    @staticmethod
    def iaddc(fv1, fv2, val):
        return _impl.iaddc(fv1, fv2, val)
