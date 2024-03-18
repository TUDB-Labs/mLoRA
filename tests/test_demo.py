import unittest


class DemoTestCase(unittest.TestCase):
    def test_demo(self):
        self.assertEqual(1 + 2, 3)


if __name__ == '__main__':
    unittest.main()
