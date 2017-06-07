#!/usr/bin/env python
#from __future__ import print_function
import sys
from arow_csc import AROW, Instance, Prediction
sys.path.append("hvector")
from _mycollections import mydefaultdict
from mydouble import mydouble, counts
import random
import numpy as np

if __name__ == "__main__":

    random.seed(13)           
    np.random.seed(13)
    dataLines = open(sys.argv[1]).readlines()

    instances = []
    classifier_p = AROW()
    print "Reading the data"
    for line in dataLines:
        details = line.split()
        costs = {}
        featureVector = mydefaultdict(mydouble)
        
        if details[0] == "-1":
            costs["neg"] = 0
            costs["pos"] = 1
        elif details[0] == "+1":
            costs["neg"] = 1
            costs["pos"] = 0

        for feature in details[1:]:
            featureID, featureVal = feature.split(":")
            featureVector[featureID] = float(featureVal)
            #featureVector["dummy"+str(len(instances))] = 1.0
            #featureVector["dummy2"+str(len(instances))] = 1.0
            #featureVector["dummy3"+str(len(instances))] = 1.0
        instances.append(Instance(featureVector, costs))
        #print instances[-1].costs

    random.shuffle(instances)
    #instances = instances[:100]
    # Keep some instances to check the performance
    testingInstances = instances[int(len(instances) * 0.75) + 1:]
    trainingInstances = instances[:int(len(instances) * 0.75)]

    print "training data: " + str(len(trainingInstances)) + " instances"
    #trainingInstances = Instance.removeHapaxLegomena(trainingInstances)
    #classifier_p.train(trainingInstances, True, True, 10, 0.1, False)
    
    # the penultimate parameter is True for AROW, false for PA
    # the last parameter can be set to True if probabilities are needed.
    classifier_p = AROW.trainOpt(trainingInstances, 10, [0.01, 0.1, 1.0, 10, 100], 0.1, True, False)

    cost = classifier_p.batchPredict(testingInstances)
    avgCost = float(cost)/len(testingInstances)
    print "Avg Cost per instance " + str(avgCost) + " on " + str(len(testingInstances)) + " testing instances"

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
