#!/usr/bin/env python
from __future__ import print_function
import sys
from arow_csc import AROW, Instance, Prediction
import random
import numpy as np

if __name__ == "__main__":

    random.seed(13)           
    np.random.seed(13)
    dataLines = open(sys.argv[1]).readlines()

    instances = []
    classifier_p = AROW()
    print("Reading the data")
    for line in dataLines:
        instance = Instance.instance_from_svm_input(line)
        instances.append(instance)

    ##random.shuffle(instances)
    #instances = instances[:100]
    # Keep some instances to check the performance
    testingInstances = instances[int(len(instances) * 0.75) + 1:]
    trainingInstances = instances[:int(len(instances) * 0.75)]

    print("training data:",str(len(trainingInstances)), "instances")
    #trainingInstances = Instance.removeHapaxLegomena(trainingInstances)
    #classifier_p.train(trainingInstances, True, True, 10, 0.1, False)
    
    # the penultimate parameter is True for AROW, false for PA
    # the last parameter can be set to True if probabilities are needed.
    classifier_p = AROW.trainOpt(trainingInstances, 10, [0.01, 0.1, 1.0, 10, 100], 0.1, True, False)

    cost = classifier_p.batchPredict(testingInstances)
    avgCost = float(cost)/len(testingInstances)
    print("Avg Cost per instance",str(avgCost),"on",str(len(testingInstances)),"testing instances")

    #avgRatio = classifier_p.batchPredict(testingInstances, True)
    #print "entropy sums: " + str(avgRatio)

    # Save the parameters:
    #print "saving"
    #classifier_p.save(sys.argv[1] + ".arow")    
    #print "done"
    # load again:
    #classifier_new = AROW()
    #print "loading model"
    #classifier_new.load(sys.argv[1] + ".arow")
    #print "done"

    #avgRatio = classifier_new.batchPredict(testingInstances, True)
    #print "entropy sums: " + str(avgRatio)
