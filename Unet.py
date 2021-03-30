import os
import sys
import random
import warnings

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dropout, Lambda, Conv2D, Conv2DTranspose, BatchNormalization, Activation, MaxPooling2D, DepthwiseConv2D, SeparableConv2D, add, MaxPooling2D, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow as tf

class UNet(object):
    """
    A UNet for spectral enhancement.
    """
    def __init__(
        self, n_inputs, kernel_size, drop_rate_encoder, drop_rate_decoder, upsample=False, seed=42):

        # Placeholders for input, output and training
#         self.X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
        self.X = tf.Variable(tf.zeros(shape=(None, n_inputs)), name="X")
        if upsample is False:
#             self.y = tf.placeholder(tf.float32, shape=(None, n_inputs), name="y")
            self.y = tf.Variable(tf.zeros(shape=(None, n_inputs)), name="y")
        elif upsample is True:
            self.y = tf.Variable(tf.zeros(shape=(None, n_inputs*2)), name="y")
#         self.training = tf.placeholder_with_default(False, shape=(), name='training')
        self.training = tf.Variable(tf.zeros(shape=()), name='training')
        
        X_reshape = tf.reshape(self.X, shape=(-1, n_inputs, 1, 1))

        self.he_init = tensorflow.keras.initializers.VarianceScaling(seed=seed)
        
        # Make a prediction
        self.pred_3dshape = self.make_unet(X_reshape, kernel_size, drop_rate_encoder, drop_rate_decoder, upsample=upsample)
        if upsample is False:
            self.pred = tf.reshape(self.pred_3dshape, shape=[-1, n_inputs], name='pred')
        elif upsample is True:
            self.pred = tf.reshape(self.pred_3dshape, shape=[-1, n_inputs*2], name='pred')
            
        # Calculate mean squared error
        with tf.name_scope("loss"):
            self.loss = tf.math.reduce_mean(tf.square(self.perd - self.y), name="mse")
#             self.loss = tf.reduce_mean(tf.square(self.pred - self.y), name="mse")

    
    def conv_batch_act_drop_pool(input_, kernel_size, n_filters, drop_rate_encoder, drop_rate_decoder, name,
                                 drop=True,pool=True, path='encoder', activation='relu'):
        net = input_
    
        for i, F in enumerate(n_filters):
            net = SeparableConv2D(
            filters=F,
            kernel_size=kernel_size,
            strides=(1,1),
            padding='same',
            activation=None,
            kernel_initializer=self.he_init,
            name='conv_{}'.format(i+1))(net)

            net = BatchNormalization(name='bn_{}'.format(i+1))(net)

            net = Activation(activation, name='elu{}_{}'.format(name, i+1))(net)
            if drop is True:
                if path is 'encoder':
                    net = Dropout(drop_rate_encoder)(net)
                if path is 'decoder':
                    net = Dropout(drop_rate_decoder)(net)
                    
        if pool is False:
            return net

        pool = MaxPooling2D((2, 1), strides=2, name="pool_{}")(net)

        return net, pool

    def upconv_2D(self, tensor, n_filter, name, activation='relu'):

        tensor = Conv2DTranspose(
        filters=n_filter,
        kernel_size=2,
        strides=2,
        kernel_initializer=self.he_init,
        name='upsample_{}'.format(name))(tensor)

        tensor = BatchNormalization(name='bn_{}'.formant(name))(tensor)

        tensor = Activation(name='elu{}'.format(name))(tensor)
        
        return tensor
#         return Dropout(drrate)(tensor)

    def upconv_concat(self, inputA, input_B, n_filter, name):

        up_conv = upconv_2D(inputA, n_filter, name)

        return concatenate(
        [up_conv, input_B], name='concat_{}'.format(name))

    def make_unet(self, net, kernel_size, do_en, do_de, nk1=32, nk2=64, nk3=128, nk4=256, upsample=False, ):
    # def unet_model(height, width, chaanels, dr_conv, dr_upconv, x, y):    


        conv1, pool1 = self.conv_batch_act_drop_pool(net, kernel_size, [nk1, nk1, nk1, nk1], do_en, do_de, drop=True, name=1) 
        conv2, pool2 = self.conv_batch_act_drop_pool(pool1, kernel_size, [nk2, nk2, nk2, nk2], do_en, do_de, drop=True, name=2) # 16x8
        conv3, pool3 = self.conv_batch_act_drop_pool(pool2, kernel_size, [nk3, nk3, nk3, nk3], do_en, do_de, drop=True, name=3) # 8x4
        conv4, pool4 = self.conv_batch_act_drop_pool(pool3, kernel_size, [nk4, nk4, nk4, nk4], do_en, do_de, drop=True, pool=False, name=4) # 4x2
        
        up1 = self.upconv_concat(conv4, conv3, nk3, name=1)
        conv5 = self.conv_batch_act_drop_pool(up1, kernel_size, [nk3, nk3, nk3, nk3], do_en, do_de, drop=True, pool=False, path='decoder', name=5)
        up2 = self.upconv_concat(conv5, conv2, nk2, name=2)
        conv6 = self.conv_batch_act_drop_pool(up2, kernel_size, [nk2, nk2, nk2, nk2], do_en, do_de, drop=True, pool=False, path='decoder', name=6)
        up3 = self.upconv_concat(conv6, conv1, nk1, name=3)
        conv7 = self.conv_batch_act_drop_pool(up3, kernel_size, [nk1, nk1, nk1, nk1], do_en, do_de, drop=True, pool=False, path='decoder', name=7)
        if upsample is False:
            pred_3dshape=Conv2D(filters=1, kernel_size=(1,1), padding='same', activation=None, kernel_initializer='he_normal', name='final')(conv7)
        elif upsample is True:
            up4 = self.upconv_2D(conv7, nk1, name=4)
            conv8 = self.conv_batch_act_drop_pool(up4, kernel_size, [nk1, nk1, nk1, nk1], do_en, do_de, drop=True, pool=False, path='decoder', name=8)
            
            pred_3dshape=Conv2D(filters=1, kernel_size=(1,1), padding='same', activation=None, kernel_initializer=self.he_init, name='final')(conv8)
            model = Model(net, pred_3dshape)
            model.compile(optimizer=Adam(lr=0.001, decay=0.0001), loss='mse')
        
        return model
    


    
#         model = Model(net, pred_3dshape)
#         model.compile(optimizer=Adam(lr=))
        
#         return model
        
#         from tensorflow.keras.utils import multi_gpu_model
#         parallel_model = multi_gpu_model(model, gpus=2)

#         parallel_model.compile(optimizer=Adam(lr=0.001, decay=0.0001),loss=losses)
        
#         return parallel_model