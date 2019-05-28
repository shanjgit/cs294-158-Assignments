import pickle
import tensorflow as tf
import numpy as np
from utils import *


class ResNetTest(tf.test.TestCase):
    with open('./mnist-hw1.pkl', 'rb') as fp:
        data = pickle.load(fp)
        
    def test_shape_batch_3(self):
        test_data = tf.Variable(self.data['train'][:3],dtype=tf.float32)
        test_data_4 = tf.Variable(self.data['train'][:4],dtype=tf.float32)
        res = ResNetBlock(channels=3)
        out = res(test_data, debug=False)
        out_4 = res(test_data_4, debug=False)
        self.assertAllEqual(np.array([3,28,28,3]),tf.shape(out))
        self.assertAllEqual(np.array([4,28,28,3]),tf.shape(out_4))

    def test_shape_batch_1(self):
        test_data_1 = tf.Variable(self.data['train'][:1],dtype=tf.float32)
        res = ResNetBlock(channels=3)
        out_1 = res(test_data_1, debug=False)
        self.assertAllEqual(np.array([1,28,28,3]),tf.shape(out_1))
    
if __name__ == "__main__":
    tf.test.main()
    
