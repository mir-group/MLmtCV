import unittest
import numpy as np

from nncv.data_loader import *


class TestDataLoader(unittest.TestCase):

    _xyz = np.vstack([np.random.uniform(0, 1, 100), np.random.uniform(
        0, 2, 100), np.random.uniform(0, 3, 100)])
    _xyz = _xyz.T
    _label = np.hstack([np.zeros(30), np.ones(70)])
    _pe = np.hstack([0, np.random.uniform(0, 0.49, 19), np.random.uniform(
        0.5, 1, 9), 1.0, 1.0, np.random.uniform(1, 1.999, 39), np.random.uniform(2, 2.99, 29), 3.0])
    data = {'xyz': _xyz, 'label': _label, 'pe': _pe}
    np.savez("test.npz", xyz=_xyz, label=_label, pe=_pe)

    def test_weight_by_hist(self):
        w = weight_by_hist(self._pe, 3)
        self.assertEqual(1/w[0], 29, "wrong hist weighting")
        self.assertEqual(1/w[31], 41, "wrong hist weighting")
        self.assertEqual(1/w[71], 30, "wrong hist weighting")

    def test_weight_by_label(self):
        w, class_dict = weight_by_label(self._label)
        self.assertEqual(len(class_dict), 2, "wrong class recognition")
        self.assertIn(0, class_dict, "wrong class recognition")
        self.assertIn(1, class_dict, "wrong class recognition")
        self.assertEqual(w[0], 1/float(30), "wrong class weighting")
        self.assertEqual(w[-1], 1/float(70), "wrong class weighting")

    def test_define_weight(self):
        w = define_weight(label=self._label, pe=self._pe, ngrid=2)
        self.assertEqual(1/w[0], float(30*20), "wrong weighting")
        self.assertEqual(1/w[31], float(40*70), "wrong weighting")
        self.assertEqual(1/w[71], float(30*70), "wrong weighting")

    def test_normalize_data_bound(self):
        newpe, mean, std = normalize_data_bound(np.array(range(30)))
        self.assertEqual(mean, 14.5, "wrong mean")
        self.assertEqual(std, 29, "wrong std")

    def test_normalize_data_std(self):
        newpe, mean, std = normalize_data_std(np.ones(30))
        self.assertEqual(mean, 1.0, "wrong mean")
        self.assertEqual(std, 0.0, "wrong std")

    def test_dataloader(self):
        data = Data_Loader(filename="test.npz",
                           shuffle=True,
                           input_label=['xyz'],
                           target_label=['label', 'pe'],
                           n_sample=50,
                           batch_size=10)
