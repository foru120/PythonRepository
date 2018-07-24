import unittest

class TestAddition(unittest.TestCase):
    def setUp(self):
        print('Setting up the test')

    def tearDown(self):
        print('Tearing down the test')

    def test_twoPlusTwo(self):
        total = 2 + 2
        self.assertEqual(4, total)

if __name__=='__main__':
    unittest.main()