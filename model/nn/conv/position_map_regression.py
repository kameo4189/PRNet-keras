from tensorflow.keras.layers import Input
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Add
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import tensorflow as tf
from configure import config

def weighted_tf_mean_squared_error(yTrue, yPred):
    loss_ = tf.compat.v1.losses.mean_squared_error(yTrue, yPred, 
        config.UV_WEIGHT_LOSS_MASK[None,:,:,None])
    return loss_

def weighted_tf_root_mean_squared_error(yTrue, yPred):
    loss_ = weighted_tf_mean_squared_error(yTrue, yPred)
    loss_ = K.sqrt(loss_)
    return loss_

def weighted_mean_square_error(yTrue, yPred):
    loss_ = K.square(yTrue - yPred)
    loss_ *= config.UV_WEIGHT_LOSS_MASK[None,:,:,None]
    loss_ = K.mean(loss_)
    return loss_

loss_funtions = {
    "wtmse": ["weighted_tf_mean_squared_error", weighted_tf_mean_squared_error],
    "wtrmse": ["weighted_tf_root_mean_squared_error", weighted_tf_root_mean_squared_error],
    "wmse": ["weighted_mean_square_error", weighted_mean_square_error]
}

def batchNormBlock(x, chanDim, scale=True, name=None):
    x = BatchNormalization(axis=chanDim, scale=scale, name=name)(x)
    return x

class PRN:
    @staticmethod
    def convBlock(x, K, kX, kY, stride, chanDim, padding="same", 
        activation="relu", use_bias=False, normalizer_fn=batchNormBlock,
        regularizer_fn=l2, reg=0.0002, name=None):
        
        # if a layer name was supplied, prepend it
        (convName, bnName, activName) = (None, None, None) 
        if name is not None:
            convName = name + "_conv"
            bnName = name + "_bn"
            activName = name + "_activ"

        x = Conv2D(K, (kX, kY), strides=stride, padding=padding,
            activation=None, use_bias=use_bias, 
            kernel_regularizer=regularizer_fn(reg), name=convName)(x)
        if (normalizer_fn is not None):  
            x = normalizer_fn(x, chanDim, name=bnName)
        if (activation is not None):
            x = Activation(activation, name=activName)(x)
        return x

    @staticmethod
    def convTransposeBlock(x, K, kX, kY, stride, chanDim, padding="same", 
        activation="relu", use_bias=False, normalizer_fn=batchNormBlock,
        regularizer_fn=l2, reg=0.0002, name=None):
        
        # if a layer name was supplied, prepend it
        (convTransName, bnName, activName) = (None, None, None) 
        if name is not None:
            convTransName = name + "_convt"
            bnName = name + "_bn"
            activName = name + "_activ"

        x = Conv2DTranspose(K, (kX, kY), strides=stride, padding=padding,
            activation=None, use_bias=use_bias, 
            kernel_regularizer=regularizer_fn(reg), name=convTransName)(x)
        if (normalizer_fn is not None):  
            x = normalizer_fn(x, chanDim, name=bnName)
        if (activation is not None):
            x = Activation(activation, name=activName)(x)
        return x

    @staticmethod
    def resBlock(x, K, kX, kY, stride, chanDim, activation="relu", 
        normalizer_fn=batchNormBlock, name=None):
        assert K%2==0 #num_outputs must be divided by channel_factor(2 here)

        # if a layer name was supplied, prepend it
        (shortcutName, cnvBlockName, addName, bnName, activName) = (None, None, None, None, None) 
        if name is not None:
            shortcutName = name + "_shortcut"
            cnvBlockName = name + "_convBlock-{0}"
            addName = name + "_add"
            bnName = name + "_bn"
            activName = name + "_activ"

        shortcut = x
        if stride != 1 or x.shape[3] != K:
            shortcut = PRN.convBlock(shortcut, K, 1, 1, stride, chanDim, 
                activation=None, normalizer_fn=None, name=shortcutName)

        x = PRN.convBlock(x, int(K/2), 1, 1, 1, chanDim, name=cnvBlockName.format(0))
        x = PRN.convBlock(x, int(K/2), kX, kY, stride, chanDim, name=cnvBlockName.format(1))
        x = PRN.convBlock(x, K, 1, 1, 1, chanDim, activation=None, 
            normalizer_fn=None, name=cnvBlockName.format(2))
        
        x = Add(name=addName)([x, shortcut])
        if (normalizer_fn is not None):  
            x = normalizer_fn(x, chanDim, name=bnName)
        if (activation is not None):
            x = Activation(activation, name=activName)(x)

        return x

    @staticmethod
    def build(width, height, depth):
        # initialize the input shape to be "channels last" and the
        # channels dimension itself
        inputShape = (height, width, depth)
        chanDim = -1

        # if we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        # set the input
        inputs = Input(shape=inputShape)

        size = 16
        # x: s x s x 3
        se = PRN.convBlock(inputs, size, 4, 4, 1, chanDim, name="res_256x256x16")  # 256 x 256 x 16
        se = PRN.resBlock(se, size * 2, 4, 4, 2, chanDim, name="res_128x128x32_1")  # 128 x 128 x 32
        se = PRN.resBlock(se, size * 2, 4, 4, 1, chanDim, name="res_128x128x32_2")  # 128 x 128 x 32
        se = PRN.resBlock(se, size * 4, 4, 4, 2, chanDim, name="res_64x64x64_1")  # 64 x 64 x 64
        se = PRN.resBlock(se, size * 4, 4, 4, 1, chanDim, name="res_64x64x64_2")  # 64 x 64 x 64
        se = PRN.resBlock(se, size * 8, 4, 4, 2, chanDim, name="res_32x32x128_1")  # 32 x 32 x 128
        se = PRN.resBlock(se, size * 8, 4, 4, 1, chanDim, name="res_32x32x128_2")  # 32 x 32 x 128
        se = PRN.resBlock(se, size * 16, 4, 4, 2, chanDim, name="res_16x16x256_1")  # 16 x 16 x 256
        se = PRN.resBlock(se, size * 16, 4, 4, 1, chanDim, name="res_16x16x256_2")  # 16 x 16 x 256
        se = PRN.resBlock(se, size * 32, 4, 4, 2, chanDim, name="res_8x8x512_1")  # 8 x 8 x 512
        se = PRN.resBlock(se, size * 32, 4, 4, 1, chanDim, name="res_8x8x512_2")  # 8 x 8 x 512

        pd = PRN.convTransposeBlock(se, size * 32, 4, 4, 1, chanDim, name="convt_8x8x512") # 8 x 8 x 512 
        pd = PRN.convTransposeBlock(pd, size * 16, 4, 4, 2, chanDim, name="convt_16x16x256_1") # 16 x 16 x 256
        pd = PRN.convTransposeBlock(pd, size * 16, 4, 4, 1, chanDim, name="convt_16x16x256_2") # 16 x 16 x 256
        pd = PRN.convTransposeBlock(pd, size * 16, 4, 4, 1, chanDim, name="convt_16x16x256_3") # 16 x 16 x 256
        pd = PRN.convTransposeBlock(pd, size * 8, 4, 4, 2, chanDim, name="convt_32x32x128_1") # 32 x 32 x 128
        pd = PRN.convTransposeBlock(pd, size * 8, 4, 4, 1, chanDim, name="convt_32x32x128_2") # 32 x 32 x 128
        pd = PRN.convTransposeBlock(pd, size * 8, 4, 4, 1, chanDim, name="convt_32x32x128_3") # 32 x 32 x 128
        pd = PRN.convTransposeBlock(pd, size * 4, 4, 4, 2, chanDim, name="convt_64x64x64_1") # 64 x 64 x 64
        pd = PRN.convTransposeBlock(pd, size * 4, 4, 4, 1, chanDim, name="convt_64x64x64_2") # 64 x 64 x 64
        pd = PRN.convTransposeBlock(pd, size * 4, 4, 4, 1, chanDim, name="convt_64x64x64_3") # 64 x 64 x 64

        pd = PRN.convTransposeBlock(pd, size * 2, 4, 4, 2, chanDim, name="convt_128x128x32_1") # 128 x 128 x 32
        pd = PRN.convTransposeBlock(pd, size * 2, 4, 4, 1, chanDim, name="convt_128x128x32_2") # 128 x 128 x 32
        pd = PRN.convTransposeBlock(pd, size, 4, 4, 2, chanDim, name="convt_256x256x16_1") # 256 x 256 x 16
        pd = PRN.convTransposeBlock(pd, size, 4, 4, 1, chanDim, name="convt_256x256x16_2") # 256 x 256 x 16

        pd = PRN.convTransposeBlock(pd, 3, 4, 4, 1, chanDim, name="convt_256x256x3_1") # 256 x 256 x 3
        pd = PRN.convTransposeBlock(pd, 3, 4, 4, 1, chanDim, name="convt_256x256x3_2") # 256 x 256 x 3
        pos = PRN.convTransposeBlock(pd, 3, 4, 4, 1, chanDim, activation="sigmoid", name="convt_256x256x3_s") # 256 x 256 x 3

        # create the model
        model = Model(inputs, pos, name="prn")

        # return the constructed network architecture
        return model
