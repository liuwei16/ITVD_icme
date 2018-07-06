# -*- coding: utf-8 -*-
'''ResNet50 model for Keras.
# Reference:
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
Adapted from code contributed by BigMoyan.
'''

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from keras.layers import *
from keras import backend as K

from .RoiPoolingConv import RoiPoolingConv
from .FixedBatchNormalization import FixedBatchNormalization
import numpy as np


def identity_block(input_tensor, kernel_size, filters, stage, block, trainable=True):

    nb_filter1, nb_filter2, nb_filter3 = filters
    
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, (1, 1), name=conv_name_base + '2a', trainable=trainable)(input_tensor)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b', trainable=trainable)(x)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, (1, 1), name=conv_name_base + '2c', trainable=trainable)(x)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)
    return x

def identity_block_td(input_tensor, kernel_size, filters, stage, block, trainable=True):

    # identity block time distributed

    nb_filter1, nb_filter2, nb_filter3 = filters
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = TimeDistributed(Convolution2D(nb_filter1, (1, 1), trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '2a')(input_tensor)
    x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Convolution2D(nb_filter2, (kernel_size, kernel_size), trainable=trainable, kernel_initializer='normal',padding='same'), name=conv_name_base + '2b')(x)
    x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Convolution2D(nb_filter3, (1, 1), trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '2c')(x)
    x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2c')(x)

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)

    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), trainable=True):

    nb_filter1, nb_filter2, nb_filter3 = filters
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, (1, 1), strides=strides, name=conv_name_base + '2a', trainable=trainable)(input_tensor)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b', trainable=trainable)(x)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, (1, 1), name=conv_name_base + '2c', trainable=trainable)(x)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Convolution2D(nb_filter3, (1, 1), strides=strides, name=conv_name_base + '1', trainable=trainable)(input_tensor)
    shortcut = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

def conv_block_td(input_tensor, kernel_size, filters, stage, block, input_shape, strides=(2, 2), trainable=True):

    # conv block time distributed

    nb_filter1, nb_filter2, nb_filter3 = filters
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = TimeDistributed(Convolution2D(nb_filter1, (1, 1), strides=strides, trainable=trainable, kernel_initializer='normal'), input_shape=input_shape, name=conv_name_base + '2a')(input_tensor)
    x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Convolution2D(nb_filter2, (kernel_size, kernel_size), padding='same', trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '2b')(x)
    x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Convolution2D(nb_filter3, (1, 1), kernel_initializer='normal'), name=conv_name_base + '2c', trainable=trainable)(x)
    x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2c')(x)

    shortcut = TimeDistributed(Convolution2D(nb_filter3, (1, 1), strides=strides, trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '1')(input_tensor)
    shortcut = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '1')(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

def transform_layer(input_x,strides=1, nb_filter=4, trainable=True,layer_name='transform', input_shape=None):
    x = TimeDistributed(
        Convolution2D(nb_filter, (1, 1), strides=strides, trainable=trainable, kernel_initializer='glorot_normal'),
        name=layer_name + '_conv1', input_shape=input_shape)(input_x)
    x = TimeDistributed(FixedBatchNormalization(axis=3), name=layer_name + '_bn1')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(
        Convolution2D(nb_filter, (3, 3), strides=strides,padding='same', trainable=trainable, kernel_initializer='glorot_normal'),
        name=layer_name + '_conv2')(x)
    x = TimeDistributed(FixedBatchNormalization(axis=3), name=layer_name + '_bn2')(x)
    x = Activation('relu')(x)
    return x

def split_layer(input_x, stride=1, bottleneck=4, cardinality=32, layer_name='split', input_shape = None):
    layers_split = list()
    for i in range(cardinality):
        splits = transform_layer(input_x, strides=stride, nb_filter=bottleneck,layer_name=layer_name+'_transf'+str(i),input_shape=input_shape)
        layers_split.append(splits)
    return Concatenate(axis=-1, name=layer_name+'_conc')(layers_split)

def trainsition_layer(input_x,  out_dim=512, layer_name = 'transition', trainable=True):
    x = TimeDistributed(
        Convolution2D(out_dim, (1, 1), strides=1,  trainable=trainable, kernel_initializer='glorot_normal'),
        name=layer_name + '_conv')(input_x)
    x = TimeDistributed(FixedBatchNormalization(axis=3), name=layer_name + '_bn')(x)
    # x = Activation('relu')(x)
    return  x

def residual_layer(input_x, out_dim=512, layer_name='resnext', input_shape=None):
    x = split_layer(input_x, stride=1, bottleneck=4, cardinality=32, layer_name=layer_name+'_split', input_shape=input_shape)
    x = trainsition_layer(x, out_dim=out_dim, layer_name=layer_name+'_transition')

    shortcut = TimeDistributed(
        Convolution2D(out_dim, (1, 1), strides=1, trainable=True, kernel_initializer='glorot_normal'),
        name=layer_name + '_sc')(input_x)
    shortcut = TimeDistributed(FixedBatchNormalization(axis=3), name=layer_name + '_scbn')(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

def classifier_layers(x, input_shape):

    x = residual_layer(x, out_dim=512, layer_name='resnext1', input_shape=input_shape)
    x = residual_layer(x, out_dim=1024, layer_name='resnext2', input_shape=input_shape)
    x = TimeDistributed(AveragePooling2D((7, 7)), name='avg_pool')(x)

    return x

def nn_base(input_tensor=None, trainable=False):
    img_input = input_tensor
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    x = ZeroPadding2D((3, 3))(img_input)
    x = Convolution2D(64, (7, 7), strides=(2, 2), name='conv1', trainable = False)(x)
    x = FixedBatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    # print('conv1: ', x._keras_shape[1:])
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), trainable = False)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', trainable = False)
    stage2 = identity_block(x, 3, [64, 64, 256], stage=2, block='c', trainable = False)
    # print('stage2: ', stage2._keras_shape[1:])
    x = conv_block(stage2, 3, [128, 128, 512], stage=3, block='a', trainable = trainable)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', trainable = trainable)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', trainable = trainable)
    stage3 = identity_block(x, 3, [128, 128, 512], stage=3, block='d', trainable = trainable)
    # print('stage3: ', stage3._keras_shape[1:])
    x = conv_block(stage3, 3, [256, 256, 1024], stage=4, block='a', trainable = trainable)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b', trainable = trainable)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c', trainable = trainable)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d', trainable = trainable)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e', trainable = trainable)
    stage4 = identity_block(x, 3, [256, 256, 1024], stage=4, block='f', trainable = trainable)
    # print('stage4: ', stage4._keras_shape[1:])
    x = conv_block(stage4, 3, [512, 512, 2048], stage=5, block='a', trainable=trainable)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', trainable=trainable)
    stage5 = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', trainable=trainable)
    # print('stage5: ', stage5._keras_shape[1:])

    predictor_sizes = np.array([stage3._keras_shape[1:3],
                                stage4._keras_shape[1:3],
                                stage5._keras_shape[1:3],
                                np.ceil(np.array(stage5._keras_shape[1:3]) / 2)])
    return [stage3, stage4, stage5], predictor_sizes

def befn_base(input,num_anchors,name,trainable=True):
    x = Convolution2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='glorot_normal',
                      name=name+'_rpn_conv1')(input)
    x_class = Convolution2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='glorot_normal',
                            name=name+'_rpn_class',trainable=trainable)(x)
    x_class_reshape = Reshape((-1, 1), name=name+'_class_reshape')(x_class)
    x_regr = Convolution2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='glorot_normal', name=name+'_rpn_regress',trainable=trainable)(x)
    x_regr_reshape = Reshape((-1,4), name=name+'_regress_reshape')(x_regr)
    return x_class_reshape, x_regr_reshape

def bfen(base_layers, num_anchors, trainable=True):

    P5 = Convolution2D(256,(1, 1), strides=1, padding='same',kernel_initializer='glorot_normal',
                               name='P5', trainable=trainable)(base_layers[2])
    P6 = Convolution2D(256, (3, 3), strides=2, padding='same',activation='relu', kernel_initializer='glorot_normal',
                       name='P6',trainable=trainable)(base_layers[2])
    stage4_rd = Convolution2D(256, (1, 1), strides=1, padding='same',kernel_initializer='glorot_normal',
                               name='stage4_reduced',trainable=trainable)(base_layers[1])
    P5_up = Deconvolution2D(256, kernel_size=4, strides=2,padding='same',
                            kernel_initializer='glorot_normal',
                            name='P5_up', trainable=trainable)(P5)
    if P5_up._keras_shape[1:]!=stage4_rd._keras_shape[1:]:
        P5_up = Cropping2D(cropping=((0,1),(0,0)),data_format='channels_last',
                           name='P5_up_crop')(P5_up)
    P4_pre = Add(name='P4_pre')([P5_up, stage4_rd])
    P4 = Convolution2D(256, (3, 3), strides=1, padding='same', kernel_initializer='glorot_normal',
                       name='P4', trainable=trainable)(P4_pre)


    stage3_rd = Convolution2D(256, (1, 1), strides=1, padding='same',kernel_initializer='glorot_normal',
                               name='stage3_reduced', trainable=trainable)(base_layers[0])
    P4_up = Deconvolution2D(256, kernel_size=4, strides=2, padding='same',
                            kernel_initializer='glorot_normal',
                            name='P4_up', trainable=trainable)(P4)
    if P4_up._keras_shape[1:]!=stage3_rd._keras_shape[1:]:
        P4_up = Cropping2D(cropping=((0,1),(0,0)),data_format='channels_last',
                           name='P4_up_crop')(P4_up)
    P3_pre = Add(name='P3_pre')([P4_up, stage3_rd])
    P3 = Convolution2D(256, (3, 3), strides=1, padding='same', kernel_initializer='glorot_normal',
                       name='P3', trainable=trainable)(P3_pre)

    P3_cls, P3_regr = befn_base(P3, num_anchors[0],name='pred0',trainable=trainable)
    P4_cls, P4_regr = befn_base(P4, num_anchors[1], name='pred1', trainable=trainable)
    P5_cls, P5_regr = befn_base(P5, num_anchors[2], name='pred2', trainable=trainable)
    P6_cls, P6_regr = befn_base(P6, num_anchors[3], name='pred3', trainable=trainable)

    y_cls = Concatenate(axis=1, name='mbox_cls')([P3_cls, P4_cls, P5_cls, P6_cls])
    y_regr = Concatenate(axis=1, name='mbox_regr')([P3_regr, P4_regr, P5_regr, P6_regr])

    return [y_cls, y_regr, P3]

def slpn(base_layer, input_rois, num_rois, nb_classes=2, trainable=True):

    pooling_regions = 7
    input_shape = (num_rois,7,7,256)
    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layer, input_rois])

    out = classifier_layers(out_roi_pool, input_shape=input_shape)
    out = TimeDistributed(Flatten())(out)

    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='glorot_normal',trainable=trainable), name='dense_class')(out)
    out_regr = TimeDistributed(Dense(4, activation='linear', kernel_initializer='glorot_normal',trainable=trainable), name='dense_regress')(out)
    return [out_class, out_regr]