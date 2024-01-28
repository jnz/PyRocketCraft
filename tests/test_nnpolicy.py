"""
  Unit Test for the Neural Network.

  (c) Jan Zwiener (jan@zwiener.org)
"""

import unittest
import numpy as np

import sys
sys.path.append('src') # expected to be run in root directory
from nnpolicy import NNPolicy

class TestNNPolicy(unittest.TestCase):
    """
    Test class for NNPolicy.
    """

    def test_output(self):
        """
         Test the output of the neural network for a reasonable response.
        """

        tol = 1e-3 # numerical tolerance
        nn = NNPolicy()

        # Test 1
        # ------
        # Generate response for a given state
        obs = np.array([1.0, 0.0, 0.0, 0.0, # quaternion
                        0.0, 0.0, 0.0, # angular rates
                        50.0, 50.0, 50.0, # position (m) ENU
                        0.0, 0.0, 0.0, # velocity (m/s) ENU
                        700.0, 0.0, 0.0]) # thrust, thrust vectors alpha, beta
        actual, _ = nn.next(obs)
        expected = np.array([0.24648535, 0.79983073,
                             0.7557439, -1.0397155, -1.0362167])

        self.assertTrue(np.allclose(actual, expected, atol=tol),
                        "Not almost equal: {} != {}".format(actual, expected))

        # Test 2
        # ------
        # More dynamic state
        obs = np.array([1.0, 0.0, 0.0, 0.0,
                        0.2, 0.0, 0.0, # some body rotation
                        50.0, 50.0, 50.0,
                        0.0, 0.0, -20.0,  # falling down
                        200.0, 0.0, 0.0]) # low thrust
        actual, _ = nn.next(obs)
        expected = np.array([0.9548137, 1.0729082,
                             0.0108611, -0.18440264, -1.3507757])
        self.assertTrue(np.allclose(actual, expected, atol=tol),
                        "Not almost equal: {} != {}".format(actual, expected))


if __name__ == '__main__':
    unittest.main()
