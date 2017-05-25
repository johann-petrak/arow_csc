from __future__ import print_function
import arow
import unittest

class InstanceTests(unittest.TestCase):

    def test_instance_1(self):
        data = "-1 1:0.1 2:0.5 9:0.1"
        inst = arow.instance_from_svm_input(data)
        print("Instance 1 is: ", inst)
        d2 = "+1 1:0.3 2:0.5 20:1.0"
        i2 = arow.instance_from_svm_input(d2)
        print("Instance 2 is: ",i2)
        print("dot=",arow.dot(inst.featureVector,i2.featureVector))
        arow.iaddc(inst.featureVector,i2.featureVector,-0.2)
        print("Instance 1 now:",inst)


class AROWTests(unittest.TestCase):

    def test_arow_1(self):
        dataset = ["-1 1:0.1 2:0.5 9:0.1",
                   "+1 1:0.6 2:0.2 8:0.2",
                   "-1 1:0.1 2:0.6 8:0.3",
                   "+1 1:0.4 2:0.7 9:0.4",
               ]
        data = [arow.instance_from_svm_input(d) for d in dataset]
        cl = arow.AROW()
        print([cl.predict(d).label for d in data])
        print([d.costs for d in data])
        print("TRAINING")
        cl.train(data,shuffling=False)
        
        print("Predictions:",[cl.predict(d, verbose=True).label for d in data])
        print("Weights:",[cl.predict(d, verbose=True).featureValueWeights for d in data])
        print("Costs:",[d.costs for d in data])
        
            
        

if __name__ == "__main__":
    unittest.main()
