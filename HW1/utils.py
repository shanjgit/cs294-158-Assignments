import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Activation, Reshape
from tensorflow.keras.layers import LayerNormalization as LayerNorm
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.python.eager import context
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
# imports for backwards namespace compatibility
# pylint: disable=unused-import
from tensorflow.python.keras.layers.pooling import AveragePooling1D
from tensorflow.python.keras.layers.pooling import AveragePooling2D
from tensorflow.python.keras.layers.pooling import AveragePooling3D
from tensorflow.python.keras.layers.pooling import MaxPooling1D
from tensorflow.python.keras.layers.pooling import MaxPooling2D
from tensorflow.python.keras.layers.pooling import MaxPooling3D
# pylint: enable=unused-import
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.util.tf_export import keras_export
import tensorflow_probability as tfp
from matplotlib import pyplot as plt


class MaskedConv2D(layers.Conv2D):
    
    def __init__(self, filters: int, kernel_size: int, mask_type='B',**kwargs):
        '''Implementation of venilla PixelCNN, see https://arxiv.org/abs/1601.06759
        Args:
            filters: `int`, number of output filters
            kernel_size: `int`, size of the CNN filter
  			mask_type: A `str` character, representing the type of mask kernel, either `A` or `B`, where `A`
                       means that the mask kernel excludes the current pixel and the other includes it.
        
        '''
        super(MaskedConv2D, self).__init__(filters, kernel_size, **kwargs)
        if mask_type.lower() not in ('a','b'):
            raise ValueError('mask type unknown')
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

        # Create a tf tensor of ones in the shape of our convolution weights.
        # ,then multiply this tensor with self.kernel
        m = self.__make_mask(self.kernel_size[0], self.mask_type=='A')
        
        # Convert the numpy mask into a tensor mask.
        self.mask = tf.Variable(m[:,:,np.newaxis, np.newaxis], trainable=False, dtype=tf.float32)
        #self.kernel = self.kernel * self.mask
            
    @tf.function
    def call(self, inputs):
        outputs = self._convolution_op(inputs, self.kernel*self.mask)
        if self.use_bias:
            if self.data_format == 'channels_first':
                if self.rank == 1:
                    bias = array_ops.reshape(self.bias, (1, self.filters, 1))
                    outputs += bias
                else:
                    outputs = nn.bias_add(outputs, self.bias, data_format='NCHW')
            else:
                outputs = nn.bias_add(outputs, self.bias, data_format='NHWC')
            
        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def get_config(self):
        # Add the mask type property to the config.
        return dict(list(super().get_config().items()) + list({'mask': self.mask_type}.items()))

    

class ResNetBlock(layers.Layer):

    def __init__(self, channels=128):
        super(ResNetBlock, self).__init__()
        self.channels = channels
        self.conv1x1_1 = layers.Conv2D(self.channels//2, (1,1), padding='same',use_bias=False)
        self.conv1x1_2 = layers.Conv2D(self.channels, (1,1), padding='same')
        self.conv_masked = MaskedConv2D(self.channels//2, 3, mask_type='B', activation='relu', padding='same')
        self.ln = LayerNorm(scale=False)
    
    @tf.function
    def call(self, inputs, debug=False):
        if debug: tf.print('input',inputs.shape)
        res = self.conv1x1_1(inputs)
        if debug: tf.print('conv1.1',res.shape)
        res = self.ln(res)
        res = Activation('relu')(res)
        res = self.conv_masked(res)
        if debug: tf.print('masked',res.shape)
        res = self.conv1x1_2(res)
        if debug: tf.print('conv1.2',res.shape)
        return tf.nn.relu(inputs + res)

    
class PixelCNN(keras.Model):

    def __init__(self, channels, final_channels=3*4, output_made = False):
        super(PixelCNN, self).__init__()
        self.output_made = output_made
        self.res_blocks = [ResNetBlock(channels) for _ in range(12)]
        self.mask_2 = MaskedConv2D(channels, 3, mask_type='B', activation='relu', padding='same')
        self.mask_1 = MaskedConv2D(channels, 7, mask_type='A', activation='relu', 
                                   padding='same')
        
        self.conv1x1_1 = layers.Conv2D(channels, (1,1), padding='same',use_bias=False)
        self.conv1x1_2 = layers.Conv2D(final_channels, (1,1),padding='same')
        self.ln_1 = LayerNorm(scale=False)
    
    @tf.function
    def call(self, inputs, training=None):
        x = self.mask_1(inputs)
        for res in self.res_blocks:
            x = res(x)
        x = self.mask_2(x)
        x = self.conv1x1_1(x)
        x = self.ln_1(x)
        x = Activation('relu')(x)
        x = self.conv1x1_2(x)
        
        x = Reshape((28,28,3,4))(x)
        # tf.print(x.shape)
        # x = Activation('softmax')(x)
        # x = tf.nn.softmax(x, axis=-1)
        return x
        

class PixelCNNLayer(layers.Layer):

    def __init__(self, channels, final_channels=3*4,**kwargs):
        super(PixelCNNLayer, self).__init__(**kwargs)
        self.res_blocks = [ResNetBlock(channels) for _ in range(12)]
        self.mask_2 = MaskedConv2D(channels, 3, mask_type='B', activation='relu', padding='same')
        self.mask_1 = MaskedConv2D(channels, 7, mask_type='A', activation='relu', 
                                   padding='same')
        
        self.conv1x1_1 = layers.Conv2D(channels, (1,1), padding='same',use_bias=False)
        self.conv1x1_2 = layers.Conv2D(final_channels, (1,1),padding='same')
        self.ln_1 = LayerNorm(scale=False)
    
    @tf.function
    def call(self, inputs):
        x = self.mask_1(inputs)
        for res in self.res_blocks:
            x = res(x)
        x = self.mask_2(x)
        x = self.conv1x1_1(x)
        x = self.ln_1(x)
        x = Activation('relu')(x)
        x = self.conv1x1_2(x)
        x = Activation('relu')(x)
        return x        
        
def make_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)
    return None

def nll(logits, x):
    loss = tf.nn.softmax_cross_entropy_with_logits(tf.one_hot(tf.cast(x, dtype=tf.uint8), depth=4),
                                                  logits, axis=-1) 
    return tf.cast(tf.reduce_mean(loss), tf.float32)


def plot_training_history(title, label, val_history, train_history, train_marker='.', val_marker='.', labels=None):
    """utility function for plotting training history"""
    plt.title(title)
    plt.xlabel(label)
    train_plots = train_history
    val_plots = val_history
    num_train = len(train_plots)
    for i in range(num_train):
        label='train_loss'
        if labels is not None:
            label += str(labels[i])
        plt.plot(train_plots[i], train_marker, label=label)
    label='val_loss'
    if labels is not None:
        label += str(labels[0])
    plt.plot(val_plots, val_marker, label=label)
    plt.legend(loc='lower center', ncol=num_train+1) 
    
    
def sample_image(batch_size, model):
    image = np.random.choice(4, size=(batch_size, 28, 28, 3))
    for i in range(28):
        for j in range(28):
            for k in range(3):
                prob_output =  model(tf.Variable(image, dtype=tf.float32, trainable=False)).numpy()
                prob_output = prob_output.reshape((batch_size,28,28,3,-1))
                
                # print(prob_output.shape)
                for b in range(batch_size):
                    # if k == 0 and b ==0: 
                    print(f'i:{i}, j:{j}, k:{k}')
                    print(prob_output[b,i,j,k])
                    image[b, i, j, k] = np.random.choice(4, p=prob_output[b, i, j, k])
            
    return image
