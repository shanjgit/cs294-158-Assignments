import pickle
import tensorflow as tf
from tensorflow import keras
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


class SeqModelTest(tf.test.TestCase):
    with open('./mnist-hw1.pkl', 'rb') as fp:
        data = pickle.load(fp)
    model = keras.Sequential([PixelCNNLayer(128, 3*4, input_shape=(28,28,3))])
        
    def test_shape_batch_3(self):
        test_data = tf.Variable(self.data['train'][:3],dtype=tf.float32)
        test_data_4 = tf.Variable(self.data['train'][:4],dtype=tf.float32)
        out = self.model(test_data)
        out_4 = self.model(test_data_4)
        self.assertAllEqual(np.array([3,28,28,12]),tf.shape(out))
        self.assertAllEqual(np.array([4,28,28,12]),tf.shape(out_4))

    def test_shape_batch_1(self):
        test_data_1 = tf.Variable(self.data['train'][:1],dtype=tf.float32)
        out_1 = self.model(test_data_1)
        self.assertAllEqual(np.array([1,28,28,12]),tf.shape(out_1))
        

class NLLTest(tf.test.TestCase):
    with open('./mnist-hw1.pkl', 'rb') as fp:
        data = pickle.load(fp)
    data = data['train'].astype(np.float32)
    # model = PixelCNN(128, 3*4)
        
    def test_nll(self):
        test_data = tf.Variable(self.data[:1],dtype=tf.float32)
        logits = tf.Variable(np.zeros((1,28,28,3,4)),dtype=tf.float32)
        out = nll(logits,test_data)
        ideal = -tf.math.log(0.25)
        print(ideal)
        print(out)
        self.assertAlmostEqual(out,ideal,delta=1e-5)
    
if __name__ == "__main__":
    tf.test.main()
    
