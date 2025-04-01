import unittest

import nozzleActivation

class TestNozzleActivation(unittest.TestCase):


    def test_generate_hex_number(self):
        self.assertEqual(nozzleActivation.generate_hex_number("00000000000000000000000000000000"), "00000000")
        self.assertEqual(nozzleActivation.generate_hex_number("11111111111111111111111111111111"), "ffffffff")
        self.assertEqual(nozzleActivation.generate_hex_number("10101010101010101010101010101010"), "aaaaaaaa")
        self.assertEqual(nozzleActivation.generate_hex_number("01010101010101010101010101010101"), "55555555")
        self.assertEqual(nozzleActivation.generate_hex_number("00110011001100110011001100110011"), "33333333")




if __name__ == '__main__':
    unittest.main()