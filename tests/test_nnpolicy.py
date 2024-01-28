import numpy as np
import unittest

import sys
sys.path.append('src')
from nnpolicy import NNPolicy

class TestNNPolicy(unittest.TestCase):
    def test_output(self):

        tol = 1e-5
        nn = NNPolicy()

        # Far away
        obs = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 50.0, 50.0, 50.0, 0.0, 0.0, 0.0, 700.0, 0.0, 0.0])
        actual, _ = nn.next(obs)
        expected = np.array([0.24648535,  0.79983073,  0.7557439 , -1.0397155 , -1.0362167])

        self.assertTrue(np.allclose(actual, expected, atol=tol),
                        "Arrays are not almost equal: {} != {}".format(actual, expected))

        # Far away and movement
        obs = np.array([1.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 50.0, 50.0, 50.0, 0.0, 0.0, -20.0, 200.0, 0.0, 0.0])
        actual, _ = nn.next(obs)
        expected = np.array([0.9548137, 1.0729082, 0.0108611, -0.18440264, -1.3507757])
        self.assertTrue(np.allclose(actual, expected, atol=tol),
                        "Arrays are not almost equal: {} != {}".format(actual, expected))


if __name__ == '__main__':
    unittest.main()

