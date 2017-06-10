from __future__ import print_function
import sys
implementation = "sv"
needimport = True
_impl = None

class FeatureVector(object):
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
            elif implementation == "dict":
                print("Importing dict implementation")
                from .featurevector_dict import FeatureVectorDict
                _impl = FeatureVectorDict
            elif implementation == "sv":
                print("Importing sv implementation")
                from .featurevector_sv import FeatureVectorSV
                _impl = FeatureVectorSV
            elif implementation == "tmp1":
                print("Importing tmp1 implementation")
                from .featurevector_tmp1 import FeatureVectorTmp1
                _impl = FeatureVectorTmp1
            else:
                sys.exit("Not a known sparse vector implementation: ",implementation)
        return _impl.create(oldvec)

    @staticmethod
    def dot(fv1, fv2):
        return _impl.dot(fv1, fv2)

    @staticmethod
    def iaddc(fv1, fv2, val):
        return _impl.iaddc(fv1, fv2, val)

