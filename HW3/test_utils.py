import unittest
import torch 
import torch_testing as tt
from utils import SimpleVAE


class TestShape(unittest.TestCase):
    def test_one(self):
        model = SimpleVAE(8)
        a = torch.zeros([2, 2], dtype=torch.float32)
        b, _ = model.encode(a)
        shape =  (2, 8)
        self.assertEqual(shape, tuple(b.size()))

if __name__ == '__main__':
    unittest.main()
