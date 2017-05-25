from __future__ import print_function
import arow
import unittest


class DetailedTest(unittest.TestCase):
        dataset = ["-1 1:0.1 2:0.5 9:0.1",
                   "+1 1:0.6 2:0.2 8:0.2",
                   "-1 1:0.1 2:0.6 8:0.3",
                   "+1 1:0.4 2:0.7 9:0.4",
               ]
        data = [arow.instance_from_svm_input(d) for d in dataset]
        cl = arow.AROW()
        #averagedVectors, left = cl._initialize_vectors(data,True,3,True)
        #print("avgVecs=",averagedVectors,"left=",left) 
	# iter 1
        #for instance in data:
        #  prediction = cl.predict(instance,verbose=True)
        #  print("Prediction=",prediction.label)
        cl.train(data,True,False,2,1,False)
        for instance in data:
          prediction = cl.predict(instance,verbose=True)
          print("Prediction=",prediction.label)
        

if __name__ == "__main__":
    unittest.main()
