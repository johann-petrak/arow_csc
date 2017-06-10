from __future__ import print_function
import sys
from .featurevector import FeatureVector
import pickle
import gzip
from operator import itemgetter
from .prediction import Prediction

import random
import math
import numpy


class AROW(object):
    """
    An AROW classifier. It has one weight vector for each label in
    the dataset.
    """

    # index to use in the sparse feature vector for the bias
    BIASINDEX = "BIAS"

    def __init__(self):
        self.probabilities = False
        self.currentWeightVectors = {}
        self.currentVarianceVectors = {}
        self.debug = False

    def predict(self, instance, verbose=False, probabilities=False):
        """
        Predict the label for an instance using the current weight vector.
        """
        instance.featureVector[self.BIASINDEX] = 1.0 # Always add bias
        prediction = Prediction()
        for label, weightVector in self.currentWeightVectors.items():
            score = FeatureVector.dot(instance.featureVector, weightVector)
            prediction.label2score[label] = score
            if score > prediction.score:
                prediction.score = score
                prediction.label = label
        if verbose:
            self._add_info(instance, prediction)
        if probabilities:
            # if we have probabilistic training
            self._calc_probs(instance, prediction)
        return prediction

    def _add_info(self, instance, prediction):
        """
        Add verbosity info to the prediction.
        """
        for feature in instance.featureVector:
            # keep the feature weights for the predicted label
            prediction.featureValueWeights.append([feature, instance.featureVector[feature], self.currentWeightVectors[prediction.label][feature]])
            # order them from the most positive to the most negative
            prediction.featureValueWeights = sorted(prediction.featureValueWeights, key=itemgetter(2))

    def _calc_probs(self, instance, prediction):
        """
        Add probability info to the prediction.
        """
        if self.probabilities:
            probPredictions ={}
            for label in self.probWeightVectors[0].keys():
                # smoothing the probabilities with add 0.01 of 1 out of the vectors
                probPredictions[label] = 0.01/len(self.probWeightVectors)
            # for each of the weight vectors obtained get its prediction
            for probWeightVector in self.probWeightVectors:
                maxScore = float("-inf")
                maxLabel = None
                for label, weightVector in probWeightVector.items():
                    score = FeatureVector.dot(instance.featureVector,weightVector)
                    if score > maxScore:
                        maxScore = score
                        maxLabel = label
                # so the winning label adds one vote
                probPredictions[maxLabel] += 1

            # now let's normalize:
            for label, score in probPredictions.items():
                prediction.label2prob[label] = float(score)/len(self.probWeightVectors)

            # Also compute the entropy:
            for prob in prediction.label2prob.values():
                if prob > 0:
                    prediction.entropy -= prob * math.log(prob, 2)
            # normalize it:
            prediction.entropy /= math.log(len(prediction.label2prob),2)
        else:
            print("Need to obtain weight samples for probability estimates first")

    def batchPredict(self, instances, probabilities=False):
        """
        This is just used to optimize the params
        if probabilities is True we return the ratio for the average entropies, otherwise the loss
        """
        totalCost = 0
        sumCorrectEntropies = 0
        sumIncorrectEntropies = 0
        sumLogProbCorrect = 0
        totalCorrects = 0
        totalIncorrects = 0
        sumEntropies = 0
        for instance in instances:
            prediction = self.predict(instance, False, probabilities)
            # This is without probabilities, with probabilities we want the average entropy*cost 
            if probabilities:
                if instance.costs[prediction.label] == 0:
                    sumLogProbCorrect -= math.log(prediction.label2prob[prediction.label],2)
                    totalCorrects += instance.maxCost
                    sumEntropies += instance.maxCost*prediction.entropy
                    sumCorrectEntropies += instance.maxCost*prediction.entropy
                else:
                    maxCorrectProb = 0.0
                    for correctLabel in instance.correctLabels:
                        if prediction.label2prob[correctLabel] > maxCorrectProb:
                            maxCorrectProb = prediction.label2prob[correctLabel]
                    #if maxCorrectProb > 0.0:
                    sumLogProbCorrect -= math.log(maxCorrectProb, 2)
                    #else:
                    #    sumLogProbCorrect = float("inf")
                    totalIncorrects += instance.maxCost
                    sumEntropies += instance.maxCost*(1-prediction.entropy)
                    sumIncorrectEntropies += instance.maxCost*prediction.entropy                    
            else:
                # no probs, just keep track of the cost incurred
                #!!!1
                if prediction.label not in instance.costs:
                    instance.costs[prediction.label] = 1.0
                if instance.costs[prediction.label] > 0:
                    totalCost += instance.costs[prediction.label]

        if probabilities:
            avgCorrectEntropy = sumCorrectEntropies/float(totalCorrects)
            print(avgCorrectEntropy)
            avgIncorrectEntropy = sumIncorrectEntropies/float(totalIncorrects)
            print(avgIncorrectEntropy)
            print(sumLogProbCorrect)
            return sumLogProbCorrect
        else:
            return totalCost

    def _initialize_vectors(self, instances, averaging, rounds, adapt):
        """
        Initialize the weight vectors in the beginning of training.
        We have one variance and one weight vector per class.
        """
        self.currentWeightVectors = {} 
        if adapt:
            self.currentVarianceVectors = {}
        if averaging:
            averagedWeightVectors = {}
            updatesLeft = rounds * len(instances)
        for label in instances[0].costs:
            self.currentWeightVectors[label] = FeatureVector.create()
            # remember: this is sparse in the sense that everething that doesn't have a value is 1
            # everytime we to do something with it, remember to add 1
            if adapt:
                self.currentVarianceVectors[label] = {}
            # keep the averaged weight vector
            if averaging:
                averagedWeightVectors[label] = FeatureVector.create()
        return averagedWeightVectors, updatesLeft

    def _update_parameters(self, instance, prediction, averaging, adapt, param,
                           averagedWeightVectors, updatesLeft):
        """
        Update the weights and return the total number of errors.
        """
        # first we need to get the score for the correct answer
        # if the instance has more than one correct answer then pick the min
        minCorrectLabelScore = float("inf")
        minCorrectLabel = None
        for label in instance.correctLabels:
            #!!!2
            if label not in self.currentWeightVectors:
                self.currentWeightVectors[label] = FeatureVector.create()
            score = FeatureVector.dot(instance.featureVector, self.currentWeightVectors[label])
            if score < minCorrectLabelScore:
                minCorrectLabelScore = score
                minCorrectLabel = label

        if self.debug: print("DEBUG: 1")
        # the loss is the scaled margin loss also used by Mejer and Crammer 2010
        loss = prediction.score - minCorrectLabelScore  + math.sqrt(instance.costs[prediction.label])
        if adapt:
            # Calculate the confidence values
            # first for the predicted label
            zVectorPredicted = FeatureVector.create()
            zVectorMinCorrect = FeatureVector.create()
            for feature in instance.featureVector:
                if self.debug: print("DEBUG: feature ",feature)
                # the variance is either some value that is in the dict or just 1
                if feature in self.currentVarianceVectors[prediction.label]:
                    zVectorPredicted[feature] = instance.featureVector[feature] * self.currentVarianceVectors[prediction.label][feature]
                else:
                    zVectorPredicted[feature] = instance.featureVector[feature]
                # then for the minCorrect:
                #!!!3
                if self.debug: print("DEBUG: point !!3 ")
                if minCorrectLabel not in self.currentVarianceVectors:
                    self.currentVarianceVectors[minCorrectLabel] = FeatureVector.create()
                if self.debug: print("DEBUG: point !!3a ")
                if feature in self.currentVarianceVectors[minCorrectLabel]:
                    zVectorMinCorrect[feature] = instance.featureVector[feature] * self.currentVarianceVectors[minCorrectLabel][feature]
                else:
                    zVectorMinCorrect[feature] = instance.featureVector[feature]
            if self.debug: print("DEBUG: before dot, vecs: ",zVectorPredicted,instance.featureVector,zVectorMinCorrect,instance.featureVector)
            confidence = FeatureVector.dot(zVectorPredicted,instance.featureVector) + \
                         FeatureVector.dot(zVectorMinCorrect,instance.featureVector)
            beta = 1.0 / (confidence + param)
            alpha = loss * beta

            # update the current weight vectors
            FeatureVector.iaddc(self.currentWeightVectors[prediction.label],zVectorPredicted, -alpha)
            FeatureVector.iaddc(self.currentWeightVectors[minCorrectLabel],zVectorMinCorrect, alpha)
            if averaging:
                FeatureVector.iaddc(averagedWeightVectors[prediction.label],zVectorPredicted, -alpha * updatesLeft)
                #!!!4
                if minCorrectLabel not in averagedWeightVectors:
                    averagedWeightVectors[minCorrectLabel] = FeatureVector.create()
                FeatureVector.iaddc(averagedWeightVectors[minCorrectLabel],zVectorMinCorrect, alpha * updatesLeft)
        else:
            # the squared norm is twice the square of the features since they are the same per class 
            norm = 2 * FeatureVector.dot(instance.featureVector,instance.featureVector)
            factor = loss / (norm + 1.0 / (2 * param))
            FeatureVector.iaddc(self.currentWeightVectors[prediction.label],instance.featureVector, -factor)
            FeatureVector.iaddc(self.currentWeightVectors[minCorrectLabel],instance.featureVector, factor)
            if averaging:
                FeatureVector.iaddc(averagedWeightVectors[prediction.label],instance.featureVector, -factor * updatesLeft)
                #!!!9
                if minCorrectLabel not in averagedWeightVectors:
                    averagedWeightVectors[minCorrectLabel] = FeatureVector.create()
                FeatureVector.iaddc(averagedWeightVectors[minCorrectLabel],instance.featureVector, factor * updatesLeft)
        if self.debug: print("DEBUG: 2")
        if adapt:
            # update the diagonal covariance
            #for feature in instance.featureVector.iterkeys():
            for feature in instance.featureVector:
                # for the predicted
                if feature in self.currentVarianceVectors[prediction.label]:
                    self.currentVarianceVectors[prediction.label][feature] -= beta * pow(zVectorPredicted[feature], 2)
                else:
                    # Never updated this covariance before, add 1
                    self.currentVarianceVectors[prediction.label][feature] = 1 - beta * pow(zVectorPredicted[feature], 2)
                # for the minCorrect
                if feature in self.currentVarianceVectors[minCorrectLabel]:
                    self.currentVarianceVectors[minCorrectLabel][feature] -= beta * pow(zVectorMinCorrect[feature], 2)
                else:
                    # Never updated this covariance before, add 1
                    self.currentVarianceVectors[minCorrectLabel][feature] = 1 - beta * pow(zVectorMinCorrect[feature], 2)


    #!!!!!!!!!!!!
    # this whole method was in Makis' code, not sure where it is used
    def init(self,costs,adapt=True):
        #!!!!!!!!!!!!!!!
        self.currentWeightVectors = {}
        if adapt:
            self.currentVarianceVectors = {}
        for label in costs:
            self.currentWeightVectors[label] = FeatureVector.create()
            # remember: this is sparse in the sense that everething that doesn't have a value is 1
            # everytime we to do something with it, remember to add 1
            if adapt:
                self.currentVarianceVectors[label] = {}

    def train(self, instances, averaging=True, shuffling=False, rounds=10, param=1, adapt=True):
        """
        Train the classifier. If adapt is False then we have PA-II with
        prediction-based updates. If adapt is True then we have AROW.
        The param value is only used in AROW, not in PA-II.
        """
        # This is a bit nasty, averagedWeightVectors will be None if
        # averaging is False. Setting it as an instance attribute
        # might be better.
        averagedWeightVectors, updatesLeft = self._initialize_vectors(instances, averaging, rounds, adapt)
        if self.debug: print("DEBUG train: initialized vectors")
        for r in range(rounds):
            if shuffling:
                random.shuffle(instances)
            errorsInRound = 0
            costInRound = 0
            for instance in instances:
                if self.debug: print("DEBUG train: processing instance", instance)
                prediction = self.predict(instance)
                if self.debug: print("DEBUG train: got prediction", prediction)
                # so if the prediction was incorrect
                # we are no longer large margin, since we are using the loss from the cost-sensitive PA
                #!!!5
                if prediction.label not in instance.costs:
                    if self.debug: print("DEBUG train: not in costs, set for ",prediction.label)
                    instance.costs[prediction.label] = 1.0
                if instance.costs[prediction.label] > 0:
                    if self.debug: print("DEBUG train: is in costs, add for ",prediction.label," cost is ",instance.costs[prediction.label])
                    errorsInRound += 1
                    costInRound += instance.costs[prediction.label]
                    if self.debug: print("DEBUG updating parms")
                    self._update_parameters(instance, prediction, averaging, adapt, param,
                                            averagedWeightVectors, updatesLeft)
                if averaging:
                    updatesLeft-=1
            print("Training error rate in round " + str(r) + " : " + str(float(errorsInRound) / len(instances)))
	    
        if averaging:
            for label in self.currentWeightVectors:
                self.currentWeightVectors[label] = FeatureVector.create()
                FeatureVector.iaddc(self.currentWeightVectors[label],averagedWeightVectors[label],1.0 / float(rounds * len(instances)))

        # Compute the final training error:
        finalTrainingErrors = 0
        finalTrainingCost = 0
        for instance in instances:
            prediction = self.predict(instance)
            #!!!6
            if prediction.label not in instance.costs:
                instance.costs[prediction.label] = 1.0
            if instance.costs[prediction.label] > 0:
                finalTrainingErrors +=1
                finalTrainingCost += instance.costs[prediction.label]

        finalTrainingErrorRate = float(finalTrainingErrors)/len(instances)
        print("Final training error rate=" + str(finalTrainingErrorRate))
        print("Final training cost=" + str(finalTrainingCost))

        return finalTrainingCost

    def probGeneration(self, scale=1.0, noWeightVectors=100):
        # initialize the weight vectors
        print("Generating samples for the weight vectors to obtain probability estimates")
        self.probWeightVectors = []
        for i in range(noWeightVectors):
            self.probWeightVectors.append({})
            for label in self.currentWeightVectors:
                self.probWeightVectors[i][label] = FeatureVector.create()

        for label in self.currentWeightVectors:
            # We are ignoring features that never got their weight set 
            for feature in self.currentWeightVectors[label]:
                # note that if the weight was updated, then the variance must have been updated too, i.e. we shouldn't have 0s
                weights = numpy.random.normal(self.currentWeightVectors[label][feature], scale * self.currentVarianceVectors[label][feature], noWeightVectors)
                # we got the samples, now let's put them in the right places
                for i,weight in enumerate(weights):
                    self.probWeightVectors[i][label][feature] = weight
                
        print("done")
        self.probabilities = True

    # train by optimizing the c parametr
    @staticmethod
    def trainOpt(instances, rounds = 10, paramValues=[0.01, 0.1, 1.0, 10, 100], heldout=0.2, adapt=True, optimizeProbs=False):
        print("Training with " + str(len(instances)) + " instances")

        # this value will be kept if nothing seems to work better
        bestParam = 1
        lowestCost = float("inf")
        bestClassifier = None
        trainingInstances = instances[:int(len(instances) * (1-heldout))]
        testingInstances = instances[int(len(instances) * (1-heldout)) + 1:]
        for param in paramValues:
            print("Training with param="+ str(param) + " on " + str(len(trainingInstances)) + " instances")
            # Keep the weight vectors produced in each round
            classifier = AROW()
            classifier.train(trainingInstances, True, True, rounds, param, adapt)
            print("testing on " + str(len(testingInstances)) + " instances")
            # Test on the dev for the weight vector produced in each round
            devCost = classifier.batchPredict(testingInstances)
            print("Dev cost:" + str(devCost) + " avg cost per instance " + str(devCost/float(len(testingInstances))))

            if devCost < lowestCost:
                bestParam = param
                lowestCost = devCost
                bestClassifier = classifier

        # OK, now we got the best C, so it's time to train the final model with it
        # Do the probs
        # So we need to pick a value between 
        if optimizeProbs:
            print("optimizing the scale parameter for probability estimation")
            bestScale = 1.0
            lowestEntropy = float("inf")
            steps = 20
            for i in range(steps):
                scale = 1.0 - float(i)/steps
                print("scale= " +  str(scale))
                bestClassifier.probGeneration(scale)
                entropy = bestClassifier.batchPredict(testingInstances, True)
                print("entropy sums: " + str(entropy))
                
                if entropy < lowestEntropy:
                    bestScale = scale
                    lowestEntropy = entropy
        
        
        # Now train the final model:
        print("Training with param="+ str(bestParam) + " on all the data")

        finalClassifier = AROW()
        finalClassifier.train(instances, True, True, rounds, bestParam, adapt)
        if optimizeProbs:
            print("Adding weight samples for probability estimates with scale " + str(bestScale))
            finalClassifier.probGeneration(bestScale)

        return finalClassifier
        
    # save function for the parameters:
    def save(self, filename):
        # prepare for pickling
        pickleDict = {}
        for label in self.currentWeightVectors:
            pickleDict[label] = {}
            for feature in self.currentWeightVectors[label]:
                pickleDict[label][feature] = self.currentWeightVectors[label][feature]
        with open(filename,'wb') as handle:
            pickle.dump(pickleDict, handle)
        # Check if there are samples for probability estimates to save
        if self.probabilities:
            pickleDictProbVectors = []
            for sample in self.probWeightVectors:
                label2vector = {}
                for label, vector in sample.items():
                    label2vector[label] = {}
                    for feature in vector:
                        label2vector[label][feature] = vector[feature]
                pickleDictProbVectors.append(label2vector)
            probVectorFile = gzip.open(filename + "_probVectors.gz", "wb")
            pickle.dump(pickleDictProbVectors, probVectorFile, -1)
            probVectorFile.close()
        # this is just for debugging, doesn't need to be loaded as it is not used for prediction
        # Only the non-one variances are added
        pickleDictVar = {}
        for label in self.currentVarianceVectors:
            pickleDictVar[label] = {}
            for feature in self.currentVarianceVectors[label]:
                pickleDictVar[label][feature] = self.currentVarianceVectors[label][feature]
        with open(filename+"_variances", 'wb') as handle:
                pickle.dump(pickleDictVar, handle)


    # load a model from a file:
    def load(self, filename):
        model_weights = open(filename, 'r')
        weightVectors = pickle.load(model_weights)
        model_weights.close()
        for label, weightVector in weightVectors.items():
            self.currentWeightVectors[label] = FeatureVector.create(weightVector)

        try:
            with gzip.open(filename + "_probVectors.gz", "rb") as probFile:
                print("loading probabilities")
                pickleDictProbVectors = pickle.load(probFile)
                self.probWeightVectors = []
                for sample in pickleDictProbVectors:
                    label2Vectors = {}
                    for label,vector in sample.items():
                        label2Vectors[label] = FeatureVector.create(vector)
                    self.probWeightVectors.append(label2Vectors)

                probFile.close()
                self.probabilities = True
        except IOError:
            print('No weight vectors for probability estimates')
            self.probabilities = False
        
