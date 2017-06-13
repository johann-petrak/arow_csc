from __future__ import print_function
from arow_csc import AROW, Instance, Prediction 
from arow_csc import Instance, FeatureVector
import unittest
import sys

class FeatureVectorTests(unittest.TestCase):

    def test_fv_checkimpls(self):
        for impl in FeatureVector.getimplimentations():
            print("Activating ", impl, ": ", FeatureVector.activateimplementation(impl), file=sys.stderr)

    def test_fv_1(self):
        for impl in FeatureVector.getimplimentations():
            if FeatureVector.activateimplementation(impl):
                fv = FeatureVector.create()

class InstanceTests(unittest.TestCase):

    def test_instance_1(self):
        for impl in FeatureVector.getimplimentations():
            if FeatureVector.activateimplementation(impl):
                data = "-1 1:0.1 2:0.5 9:0.1"
                inst = Instance.instance_from_svm_input(data)
                c = inst.costs
                fv = inst.featureVector
                self.assertEqual(len(c), 2)
                self.assertEqual(len(fv), 3)
                self.assertEqual(c["pos"], 1.0)

    def test_instance_2(self):
        for impl in FeatureVector.getimplimentations():
            if FeatureVector.activateimplementation(impl):
                data = "-1 40:0.064018 75:0.064018 89:0.064018 97:0.064018 102:0.064018 103:0.064018 114:0.064018 171:0.064018 174:0.064018 177:0.064018 188:0.064018 217:0.064018 262:0.064018 294:0.064018 348:0.064018 450:0.064018 480:0.064018 562:0.064018 943:0.064018 951:0.064018 967:0.064018 975:0.064018 1015:0.064018 1170:0.064018 1246:0.064018 1273:0.064018 1770:0.064018 1792:0.064018 1977:0.064018 2024:0.064018 2055:0.064018 2121:0.064018 2200:0.064018 2509:0.064018 2789:0.064018 3014:0.064018 3274:0.064018 3277:0.064018 3539:0.064018 3546:0.064018 3688:0.064018 3721:0.064018 3764:0.064018 4182:0.064018 4269:0.064018 4354:0.064018 4355:0.064018 4356:0.064018 4357:0.064018 4358:0.064018 4424:0.064018 4787:0.064018 4966:0.064018 4967:0.064018 4969:0.064018 5452:0.064018 5588:0.064018 5802:0.064018 5912:0.064018 5914:0.064018 6298:0.064018 6346:0.064018 6436:0.064018 6440:0.064018 6842:0.064018 7146:0.064018 7237:0.064018 7812:0.064018 8384:0.064018 8943:0.064018 8944:0.064018 9562:0.064018 9563:0.064018 9564:0.064018 9565:0.064018 9566:0.064018 10321:0.064018 11059:0.064018 11166:0.064018 12597:0.064018 12675:0.064018 17974:0.064018 18114:0.064018 20369:0.064018 22708:0.064018 25597:0.064018 26500:0.064018 27025:0.064018 28489:0.064018 28510:0.064018 28565:0.064018 29408:0.064018 31157:0.064018 32404:0.064018 38193:0.064018 45346:0.064018 45347:0.064018 45348:0.064018 52381:0.064018 52597:0.064018 53618:0.064018 68684:0.064018 70088:0.064018 70196:0.064018 72498:0.064018 80268:0.064018 80565:0.064018 84690:0.064018 84796:0.064018 92964:0.064018 100980:0.064018 103791:0.064018 106826:0.064018 108110:0.064018 127196:0.064018 127197:0.064018 140724:0.064018 140725:0.064018 162445:0.064018 175635:0.064018 258980:0.064018 277106:0.064018 277107:0.064018 286715:0.064018 297021:0.064018 304217:0.064018 316414:0.064018 384115:0.064018 390550:0.064018 405318:0.064018 418574:0.064018 458738:0.064018 540582:0.064018 577956:0.064018 582029:0.064018 584161:0.064018 619159:0.064018 828333:0.064018 864855:0.064018 868104:0.064018 886796:0.064018 886797:0.064018 973546:0.064018 1021795:0.064018 1049921:0.064018 1065554:0.064018 1065555:0.064018 1071389:0.064018 1105682:0.064018 1133331:0.064018 1134400:0.064018 1167450:0.064018 1169182:0.064018 1175501:0.064018 1175502:0.064018 1175503:0.064018 1177875:0.064018 1178363:0.064018 1178366:0.064018 1185827:0.064018 1185828:0.064018 1188190:0.064018 1188191:0.064018 1222152:0.064018 1229681:0.064018 1229682:0.064018 1229683:0.064018 1229684:0.064018 1229685:0.064018 1229686:0.064018 1229687:0.064018 1229688:0.064018 1229689:0.064018 1229690:0.064018 1229691:0.064018 1229692:0.064018 1229693:0.064018 1229694:0.064018 1229695:0.064018 1229696:0.064018 1229697:0.064018 1229698:0.064018 1229699:0.064018 1229700:0.064018 1229701:0.064018 1229702:0.064018 1229703:0.064018 1229704:0.064018 1229705:0.064018 1229706:0.064018 1229707:0.064018 1229708:0.064018 1229709:0.064018 1229710:0.064018 1229711:0.064018 1229712:0.064018 1229713:0.064018 1229714:0.064018 1229715:0.064018 1229716:0.064018 1229717:0.064018 1229718:0.064018 1229719:0.064018 1229720:0.064018 1229721:0.064018 1229722:0.064018 1229723:0.064018 1229724:0.064018 1229725:0.064018 1229726:0.064018 1229727:0.064018 1229728:0.064018 1229729:0.064018 1229730:0.064018 1229731:0.064018 1229732:0.064018 1229733:0.064018 1229734:0.064018 1229735:0.064018 1229736:0.064018 1229737:0.064018 1229738:0.064018 1229739:0.064018 1229740:0.064018 1229741:0.064018 1229742:0.064018 1229743:0.064018 1229744:0.064018 1229745:0.064018 1229746:0.064018 1229747:0.064018 1229748:0.064018 1229749:0.064018 1229750:0.064018 1229751:0.064018 1229752:0.064018 1229753:0.064018 1229754:0.064018 1229755:0.064018 1229756:0.064018 1229757:0.064018 1229758:0.064018 1229759:0.064018 1229760:0.064018"
                inst = Instance.instance_from_svm_input(data)
                costs = inst.costs
                fv = inst.featureVector
                self.assertIsNotNone(costs)
                self.assertIsNotNone(fv)
                self.assertEqual(len(costs), 2)
                self.assertEqual(len(fv), 244)

    def test_instance_3(self):
        for impl in FeatureVector.getimplimentations():
            if FeatureVector.activateimplementation(impl):
                data = "1:2.2 2:0 3:1.1 | f1:2.0   f2 f3:-12.1 f4:0.0 "
                inst = Instance.instance_from_vw(data)
                costs = inst.costs
                fv = inst.featureVector
                print("fv=", fv)
                self.assertIsNotNone(costs)
                self.assertIsNotNone(fv)
                self.assertEqual(len(costs), 3)
                self.assertEqual(len(fv), 3)
                self.assertEqual(costs["1"], 2.2)
                self.assertEqual(costs["2"], 0.0)
                self.assertEqual(costs["3"], 1.1)
                self.assertEqual(fv["f1"], 2.0)
                self.assertEqual(fv["f2"], 1.0)
                self.assertEqual(fv["f3"], -12.1)
                self.assertEqual("f4" in fv, False)
                # NOTE: this assert with defaultdict will insert f4 into the dictionary, so the previous assert would then fail!
                # NOTE: this will give a key error for traditional dictionary-based implementations!
                if impl == "hvector" or impl == "sv":
                    self.assertEqual(fv["f4"], 0.0)

class AROWTests(unittest.TestCase):

    def test_arow_1(self):
        for impl in FeatureVector.getimplimentations():
            if FeatureVector.activateimplementation(impl):
                dataset = ["-1 1:0.1 2:0.5 9:0.1",
                           "+1 1:0.6 2:0.2 8:0.2",
                           "-1 1:0.1 2:0.6 8:0.3",
                           "+1 1:0.4 2:0.7 9:0.4",
                           ]
                print("DEBUG: creating instances")
                data = [Instance.instance_from_svm_input(d) for d in dataset]
                cl = AROW()
                print("DEBUG before predict 1")
                print([cl.predict(d).label for d in data])
                print([d.costs for d in data])
                print("DEBUG before train")
                cl.train(data)

                print([cl.predict(d, verbose=True).label for d in data])
                print([cl.predict(d, verbose=True).featureValueWeights for d in data])
                print([d.costs for d in data])
        
            
        

if __name__ == "__main__":
    unittest.main()
