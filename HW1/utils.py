import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
import tensorflow_probability as tfp


class MaskedConv2D(layers.Conv2D):
    
    def __init__(self, filters, kernel_size, mask_type='B',**kwargs):
        super(MaskedConv2D, self).__init__(filters, kernel_size, **kwargs)
        if mask_type.lower() not in ('a','b'):
            raise ValueError('mask type not in (A,B)')
        self.mask_type = mask_type
        self.mask = None
    
    def __make_mask(self, size, type_A):
        m = np.zeros((size, size), dtype=np.float32)
        m[:size//2, :] = 1
        m[size//2, :size//2] = 1
        if not type_A:
            m[size//2, size//2] = 1
        return m

    def build(self, input_shape):
        super(MaskedConv2D, self).build(input_shape)
        print('Use customized mask layer')

        # Create a numpy array of ones in the shape of our convolution weights.
        self.mask = self.__make_mask(self.kernel_size[0], self.mask_type=='A')
        # Convert the numpy mask into a tensor mask.
        self.mask = tf.Variable(self.mask[:,:,np.newaxis, np.newaxis], trainable=False, dtype=tf.float32)
        self.kernel = self.kernel * self.mask

    def call(self, x):
        print()
        return super(MaskedConv2D, self).call(x)

    def get_config(self):
        # Add the mask type property to the config.
        return dict(list(super().get_config().items()) + list({'mask': self.mask_type}.items()))