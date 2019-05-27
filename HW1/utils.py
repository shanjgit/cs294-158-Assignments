import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
import tensorflow_probability as tfp


class MaskedConv2D(layers.Conv2D):
    
    def __init__(self, filters: int, kernel_size: int, mask_type='B',**kwargs):
        '''Implementation of vernila PixelCNN from https://arxiv.org/abs/1601.06759
        Args:
            filters: `int`, number of output filters
            kernel_size: `int`, size of CNN kernel
  			mask_type: A `str` character, representing the type of mask kernel, either `A` or `B`, where `A`
                       means that the mask kernel exclude the current pixel and the other includes the current pixel.
        
        '''
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
        #rint('Use customized mask layer')

        # Create a numpy array of ones in the shape of our convolution weights.
        self.mask = self.__make_mask(self.kernel_size[0], self.mask_type=='A')
        # Convert the numpy mask into a tensor mask.
        self.mask = tf.Variable(self.mask[:,:,np.newaxis, np.newaxis], trainable=False, dtype=tf.float32)
        
        
    @tf.function
    def call(self, x):
        self.kernel = self.kernel * self.mask
        return super(MaskedConv2D, self).call(x)

    def get_config(self):
        # Add the mask type property to the config.
        return dict(list(super().get_config().items()) + list({'mask': self.mask_type}.items()))

    

class ResNetBlock(layers.Layer):

    def __init__(self, channels=128):
        super(ResNetBlock, self).__init__()
        self.channels = channels
        self.conv1x1_1 = layers.Conv2D(self.channels//2, (1,1), activation='relu', padding='same')
        self.conv1x1_2 = layers.Conv2D(self.channels, (1,1), padding='same')
        self.conv_masked = MaskedConv2D(self.channels//2, 3, mask_type='B', activation='relu', 
                                                                                padding='same')
    @tf.function    
    def call(self, inputs, debug=False):
        if debug: tf.print('input',inputs.shape)
        res = self.conv1x1_1(inputs)
        if debug: tf.print('conv1.1',res.shape)
        res = self.conv_masked(res)
        if debug: tf.print('masked',res.shape)
        res = self.conv1x1_2(res)
        if debug: tf.print('conv1.2',res.shape)
        return tf.nn.relu(inputs + res)

    
class PixelCNN(keras.Model):

    def __init__(self, channels, final_channels):
        super(PixelCNN, self).__init__()
        self.res_blocks = [ResNetBlock(channels) for _ in range(12)]
        self.mask_2 = MaskedConv2D(channels, 3, mask_type='B', activation='relu', padding='same')
        self.mask_1 = MaskedConv2D(channels, 7, mask_type='A', activation='relu', padding='same')
        self.conv1x1_1 = layers.Conv2D(channels, (1,1), activation='relu', padding='same')
        self.conv1x1_2 = layers.Conv2D(final_channels, (1,1), padding='same')
    
    @tf.function
    def call(self, inputs, training=None):
        x = self.mask_1(inputs)
        for res in self.res_blocks:
            x = res(x)
        x = self.mask_2(x)
        x = self.conv1x1_1(x)
        x = self.conv1x1_2(x)
        return x
        
        
         
