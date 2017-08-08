import sys

sys.path.append('../')
from src.embedding import WordEmbedding

__author__ = 'roger'

import unittest


class MyTestCase(unittest.TestCase):
    def test_something(self):
        bin_word_map = WordEmbedding.load_word2vec_word_map("text.bin",
                                                            binary=True,
                                                            unicode_errors='replace')
        embedding = WordEmbedding(bin_word_map, filename="text.bin",
                                  unicode_errors='replace')
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
