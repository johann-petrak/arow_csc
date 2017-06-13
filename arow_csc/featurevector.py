from __future__ import print_function
import sys
implementation = "sv"
implementations = ["hvector", "defaultdict", "dict", "sv", "tmp1"]
needimport = True
_impl = None



class FeatureVector(object):
    @staticmethod
    def setimplementation(impl):
        global implementation, needimport
        implementation = impl
        needimport = True

    @staticmethod
    def getimplimentations():
        global implementations
        return implementations

    # Activate the currently configured implementation, if necessary. If impl is specified,
    # force loading/activating it.
    @staticmethod
    def activateimplementation(impl=None):
        global needimport, _impl, implementation
        if not impl:
            impl = implementation
        else:
            needimport = True
        active = True
        try:
            if needimport:
                if impl == "hvector":
                    print("Importing hvector implementation")
                    from .featurevector_hvector import FeatureVectorHvector
                    _impl = FeatureVectorHvector
                elif impl == "defaultdict":
                    print("Importing defaultdict implementation")
                    from .featurevector_defaultdict import FeatureVectorDefaultdict
                    _impl = FeatureVectorDefaultdict
                elif impl == "dict":
                    print("Importing dict implementation")
                    from .featurevector_dict import FeatureVectorDict
                    _impl = FeatureVectorDict
                elif impl == "sv":
                    print("Importing sv implementation")
                    from .featurevector_sv import FeatureVectorSV
                    _impl = FeatureVectorSV
                elif impl == "tmp1":
                    print("Importing tmp1 implementation")
                    from .featurevector_tmp1 import FeatureVectorTmp1
                    _impl = FeatureVectorTmp1
                else:
                    active = False
                    raise "Not a known implemention: "+impl
                needimport = False
        except:
            active = False
            needimport = True
        return active

    @staticmethod
    def create(oldvec=None):
        global needimport, _impl, implementation
        if needimport:
            have = FeatureVector.activateimplementation()
            if not have:
                raise Exception("Not available or not a known sparse vector implementation: ", implementation)
        return _impl.create(oldvec)

    @staticmethod
    def dot(fv1, fv2):
        return _impl.dot(fv1, fv2)

    @staticmethod
    def iaddc(fv1, fv2, val):
        return _impl.iaddc(fv1, fv2, val)

