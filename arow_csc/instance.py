from __future__ import print_function
import sys
from .featurevector import FeatureVector
import re

re_cc = re.compile("([0-9]+):([0-9\.E+-]+)")
re_feat = re.compile("([^\s|:]+):([0-9\.E+-]+)")
re_ws = re.compile("\s+")

class Instance(object):
    """
    An data instance to be used with AROW. Each instance is composed of a 
    feature vector (a dict or Huang-style sparse vector) and a dictionary
    of costs (where the labels should be encoded).
    """

    def __init__(self, feat_vector, costs=None):
        self.featureVector = FeatureVector.create(feat_vector)
        self.costs = costs
        if self.costs != None:
            self._normalize_costs()

    def _normalize_costs(self):
        """
        Normalize the costs by setting the lowest one to zero and the rest
        as increments over zero. 
        """
        min_cost = float("inf")
        self.maxCost = float("-inf")
        self.worstLabels = []
        self.correctLabels = []
        for label, cost in self.costs.items():
            if cost < min_cost:
                min_cost = cost
                self.correctLabels = [label]
            elif cost == min_cost:
                self.correctLabels.append(label)
            if cost > self.maxCost:
                self.maxCost = cost
                self.worstLabels = [label]
            elif cost == self.maxCost:
                self.worstLabels.append(label)
        if min_cost > 0:
            for label in self.costs:
                self.costs[label] -= min_cost
            self.maxCost -= min_cost

    def __str__(self):
        costs_list = [str(label) + ':' + str(self.costs[label]) for label in self.costs]
        feat_list = [str(feat) + ':' + str(self.featureVector[feat]) for feat in self.featureVector]
        return ','.join(costs_list) + '\t' + ' '.join(feat_list)

    @staticmethod
    def removeHapaxLegomena(instances):
        """
        Hapax Legomena are features that appear only once in the whole
        dataset. This static method remove these features from the
        dataset.
        """
        print("Counting features")
        #feature2counts = mydefaultdict(mydouble)
        feature2counts = FeatureVector.create()
        for instance in instances:
            for element in instance.featureVector:
                feature2counts[element] += 1
        print(len(feature2counts))
        print("Removing hapax legomena")
        newInstances = []
        for instance in instances:
            #newFeatureVector = mydefaultdict(mydouble)
            newFeatureVector = FeatureVector.create()
            for element in instance.featureVector:
                # if this feature was encountered more than once
                if feature2counts[element] > 1:
                    newFeatureVector[element] = instance.featureVector[element]
            newInstances.append(Instance(newFeatureVector, instance.costs))
        return newInstances

    @staticmethod
    def instance_from_svm_input(svm_input):
        """
        Generate an Instance from a line of SVMLight input (string).
        This so far only supports lines with a target of -1 or +1 and creates costs of 0.0 or 1.0 for those.
        """
        feat_vec = FeatureVector.create()
        costs = {}
        splitted = svm_input.split()
        target = splitted[0].strip()
        if target == "-1":
            costs["neg"] = 0.0
            costs["pos"] = 1.0
        elif target == "+1" or target == "1":
            costs["neg"] = 1.0
            costs["pos"] = 0.0
        else:
            sys.exit("Unexpected target value for SVM input line:",svm_input)
        for elem in splitted[1:]:
            fid, val = elem.split(':')
            value = float(val)
            if value != 0.0:
                dim = int(fid)
                feat_vec[dim] = float(val)
        return Instance(feat_vec, costs)

    @staticmethod
    def instance_from_vw(line,nclasses=None):
        """
        Generate an Instance from a line of  vowpal wabbit like (simple) format.
        This understands per-instance class costs but so far does not allow importance fields and
        only supports training format where the class/cost pairs must be present.
       :param string: one line in vowpal wabbit format
        :param nclasses: the number of classes to expect, if not specified all classes must be present in each line
        :return: an Instance
        """
        global re_feat, re_cc
        line = line.strip()
        tinfo, finfo = line.split("|", 1)
        costs = {}
        fv = FeatureVector.create()
        ## in the target information, only consider integer-colon-float
        parts = re_ws.split(tinfo)
        for part in parts:
            m = re_cc.match(part)
            if m:
                cid, cost = m.groups()
                costs[str(cid)] = float(cost)
        ## NOTE: we ignore name spaces for now and expect there are none in the input!!!
        parts = re_ws.split(finfo.strip())
        for part in parts:
            m = re_feat.match(part)
            if m:
                fname, fval = m.groups()
            else:
                fname = part
                fval = 1.0
            fval = float(fval)
            if fval != 0.0:
                # print("adding feature ", fname, "=", fval)
                fv[str(fname)] = fval
        return Instance(fv,costs)