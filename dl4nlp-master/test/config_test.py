__author__ = 'roger'

import unittest
import sys
sys.path.append("../")
from src.config import *


def test_optimizer_config():
    config = OptimizerConfig("../conf/brae.conf")
    print config.name
    print config.param


class MyTestCase(unittest.TestCase):
    def test_something(self):
        test_optimizer_config()
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
