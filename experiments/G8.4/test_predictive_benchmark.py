import unittest
import numpy as np
from predictive_benchmark import auc, ap, ece, metrics, scaler, transform

class PredictiveBenchmarkTests(unittest.TestCase):
    def test_auc_rankings_and_ties(self):
        y=np.array([0,0,1,1])
        self.assertEqual(auc(y,np.array([.1,.2,.8,.9])),1.)
        self.assertEqual(auc(y,np.array([.9,.8,.2,.1])),0.)
        self.assertEqual(auc(np.array([0,1]),np.array([.5,.5])),.5)
        self.assertIsNone(auc(np.array([1,1]),np.array([.2,.8])))
    def test_average_precision_and_single_class(self):
        self.assertEqual(ap(np.array([0,1]),np.array([.1,.9])),1.)
        self.assertIsNone(ap(np.array([0,0]),np.array([.1,.9])))
    def test_ece_and_fixed_threshold(self):
        self.assertGreaterEqual(ece(np.array([0,1]),np.array([.1,.9])),0.)
        self.assertEqual(metrics(np.array([0,1]),np.array([.1,.9]))["threshold"],.5)
    def test_scaler_uses_only_training_data(self):
        train=np.array([[1.,2.],[3.,4.]])
        params=scaler(train); test=transform(np.array([[1000.,-1000.]]),params)
        self.assertEqual(params[0][0],2.)
        self.assertGreater(abs(test[0,0]),100.)
if __name__=="__main__": unittest.main()
