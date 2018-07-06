from __future__ import division
import random
import os
import numpy as np
import cPickle
from keras.callbacks import ModelCheckpoint,  ReduceLROnPlateau
from keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model
from keras_itvd import config, data_generators
from keras_itvd import losses as losses
from keras_itvd import net_itvd as nn

# pass the settings from the command line, and persist them in the config object
C = config.Config()
os.environ["CUDA_VISIBLE_DEVICES"] = C.gpu_ids
batchsize = 8

# get the training data
cache_path = 'data/cache/detrac/train'
train_data = []
for data in sorted(os.listdir(cache_path)):
    cache_file = os.path.join(cache_path,data)
    with open(cache_file, 'rb') as fid:
        img_data = cPickle.load(fid)
	train_data += img_data
num_imgs_train = len(train_data)
random.shuffle(train_data)
print 'num of training samples: {}'.format(num_imgs_train)

# get the val data
cache_path = 'data/cache/detrac/val'
val_data = []
for data in sorted(os.listdir(cache_path)):
    cache_file = os.path.join(cache_path,data)
    with open(cache_file, 'rb') as fid:
        img_data = cPickle.load(fid)
	val_data += img_data
num_imgs_val = len(val_data)
print 'num of val samples: {}'.format(num_imgs_val)

img_input = Input(shape=(C.random_crop[0], C.random_crop[1], 3))
# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers, feat_map_sizes = nn.nn_base(img_input, trainable=True)

# get default anchors and define data generator
anchors, num_anchors = data_generators.get_anchors(img_height=C.random_crop[0],img_width=C.random_crop[1],
                                                   feat_map_sizes=feat_map_sizes.astype(np.int),
                                                   anchor_box_scales=C.anchor_box_scales,
                                                   anchor_ratios=C.anchor_ratios)
data_gen_train = data_generators.get_target(anchors,train_data, C, batchsize=batchsize, num_rois=C.num_rois, mode='train', data_out=False)
data_gen_val = data_generators.get_target(anchors,val_data, C, batchsize=batchsize, num_rois=C.num_rois, mode='test',data_out=False)

# define the BFEN, built on the base layers
bfen = nn.bfen(shared_layers, num_anchors)
model_befn = Model(img_input, bfen[:2])

weight_path = 'data/models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
model_befn.load_weights(weight_path, by_name=True)
print 'load weights from {}'.format(weight_path)

out_path = './output/valmodels/bfen'
if not os.path.exists(out_path):
    os.mkdir(out_path)
res_file = os.path.join(out_path,'records.txt')
init_lr = 1e-4
optimizer = Adam(lr=init_lr)
model_befn.compile(optimizer=optimizer, loss=[losses.rpn_loss_cls(), losses.rpn_loss_regr()],sample_weight_mode=None, metrics={'mbox_cls':'accuracy'})

lrplateau = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.5,
                              patience=2,
                              verbose=2,
                              mode='auto',
                              epsilon=0.00000001,
                              cooldown=0,
                              min_lr=1e-7)
num_epochs = 10
records = np.zeros((num_epochs, 4))
callback = [ModelCheckpoint(os.path.join(out_path,'e{epoch:02d}_va{val_mbox_cls_acc:.5f}_vl{val_loss:.5f}.hdf5'),
                                                           monitor='val_loss',
                                                           verbose=2,
                                                           save_best_only=False,
                                                           save_weights_only=True,
                                                           mode='auto',
                                                           period=1),
                                           # LearningRateScheduler(lr_schedule),
                                           # EarlyStopping(monitor='val_acc',
                                           #               min_delta=0.0001,
                                           #               patience=2),
                                            lrplateau]
                                           # TensorBoard(log_dir='./logs')]
history = model_befn.fit_generator(generator = data_gen_train,
                              steps_per_epoch = np.ceil(num_imgs_train/batchsize),
                              epochs = num_epochs,
                              verbose=2,
                              callbacks = callback,
                              validation_data = data_gen_val,
                              validation_steps = np.ceil(num_imgs_val/batchsize)
                              )
records[:,0] = np.asarray(history.history['loss'])
records[:,1] = np.asarray(history.history['mbox_cls_acc'])
records[:,2] = np.asarray(history.history['val_loss'])
records[:,3] = np.asarray(history.history['val_mbox_cls_acc'])

np.savetxt(res_file,np.array(records), fmt='%.6f')

print('Training complete, exiting.')
