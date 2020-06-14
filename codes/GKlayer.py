import numpy as np
import keras
import tensorflow as tf
from keras import backend as K
from keras.layers import Layer
from keras import activations,initializers,regularizers,constraints


class GKlayer(Layer):
# Group knock_off layer
# group_stru is a dummy variable (p(num of var) by q(num of group)) to represent the group structure of variables 
# input_shape is (2*p,1)
# output_dim is (num_group)
    def __init__(self, group_stru,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None, **kwargs):
        self.group_stru=group_stru
        self.num_var=group_stru.shape[0]
        self.num_group=group_stru.shape[1]
        self.output_dim = group_stru.shape[1]
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        super(GKlayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1],1),
                                      initializer=keras.initializers.glorot_normal(seed=None),
                                      regularizer=self.kernel_regularizer,
                                      trainable=True)
        super(GKlayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        tmp=tf.ones((1,self.num_group),dtype=tf.float32)  # 1 by q
        W = K.dot(self.kernel, tmp)   # 2p by q
        group_stru2=np.concatenate((self.group_stru ,self.group_stru ),axis=0)   # 2p by q 
        W = W * group_stru2    # 2p by q 
        return K.dot(x, W)  # n by q

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_group)
        
        
        

class Glayer(Layer):
# Group layer
# group_stru is a dummy variable (p(num of var) by q(num of group)) to represent the group structure of variables
# the variables in the same group is connected to neuro 
# input_shape is (p,)
# output_dim is (q,)
    def __init__(self, group_stru,
                 activation=None,
                 use_bias=False,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None, **kwargs):
        self.group_stru=group_stru
        self.num_var=group_stru.shape[0]
        self.num_group=group_stru.shape[1]
        self.output_dim = group_stru.shape[1]
        self.group_stru=group_stru
        self.num_var=group_stru.shape[0]
        self.num_group=group_stru.shape[1]
        self.output_dim = group_stru.shape[1]
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        super(Glayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1],1),
                                      initializer=keras.initializers.glorot_normal(seed=None),
                                      trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.num_group,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        super(Glayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        tmp=tf.ones((1,self.num_group),dtype=tf.float32) # 1 by g
        W = K.dot(self.kernel, tmp)  # p by g
        W = W * self.group_stru 
        output = K.dot(x, W)
        if self.use_bias:
            #output = K.bias_add(output, self.bias, data_format='channels_last')
            output = tf.nn.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output
 

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)