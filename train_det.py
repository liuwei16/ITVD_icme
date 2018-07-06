from __future__ import division
import os
import time
import numpy as np
import cPickle
from keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model
from keras.utils import generic_utils
from keras_itvd import config, data_generators, bbox_process
from keras_itvd import losses as losses
from keras_itvd import net_itvd as nn

C = config.Config()
os.environ["CUDA_VISIBLE_DEVICES"] = C.gpu_ids
batchsize = 1

# get the training data
cache_path = 'data/cache/detrac/train'
train_data = []
for data in sorted(os.listdir(cache_path)):
    cache_file = os.path.join(cache_path,data)
    with open(cache_file, 'rb') as fid:
        img_data = cPickle.load(fid)
	train_data += img_data
num_imgs = len(train_data)
print 'num of training samples: {}'.format(num_imgs)

img_input = Input(shape=(C.random_crop[0], C.random_crop[1], 3))
roi_input = Input(shape=(None, 4))
# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers, feat_map_sizes = nn.nn_base(img_input, trainable=True)
# get default anchors and define data generator
anchors, num_anchors = data_generators.get_anchors(img_height=C.random_crop[0],img_width=C.random_crop[1],
                                                   feat_map_sizes=feat_map_sizes.astype(np.int),
                                                   anchor_box_scales=C.anchor_box_scales,
                                                   anchor_ratios=C.anchor_ratios)
data_gen_train = data_generators.get_target(anchors,train_data, C, batchsize=batchsize, num_rois=C.num_rois, mode='train', data_out=True)

# define the BEFN, built on the base layers
bfen = nn.bfen(shared_layers, num_anchors, trainable=True)
model_befn = Model(img_input, bfen[:2])

slpn = nn.slpn(bfen[2], roi_input, C.num_rois, nb_classes=2, trainable=True)
model_slpn = Model([img_input, roi_input], slpn)
model_all = Model([img_input, roi_input], bfen[:2] + slpn)

weight_path = 'data/models/bfen_val.hdf5'
model_all.load_weights(weight_path, by_name=True)
print 'load weights from {}'.format(weight_path)

init_lr_befn = 1e-5
init_lr_slpn = 1e-4
optimizer_befn = Adam(lr=init_lr_befn)
optimizer_slpn = Adam(lr=init_lr_slpn)
model_befn.compile(optimizer=optimizer_befn, loss=[losses.rpn_loss_cls_focal, losses.rpn_loss_regr()])
model_slpn.compile(optimizer=optimizer_slpn, loss=[losses.class_loss_cls_focal, losses.class_loss_regr(1)])

out_path = './output/valmodels/det'
if not os.path.exists(out_path):
    os.mkdir(out_path)
res_file = os.path.join(out_path,'records.txt')

epoch_length = num_imgs
num_epochs = C.num_epochs
iter_num = 0
add_epoch = 0
losses = np.zeros((epoch_length, 4))
rpn_accuracy_rpn_monitor = []
rpn_accuracy_for_epoch = []
start_time = time.time()

best_loss = np.Inf
best_cls_acc = 0.5
print('Starting training with learning rate:BEFN--{} SLPN--{}'.format(init_lr_befn, init_lr_slpn))
curr_loss_r, loss_befn_cls_r, loss_befn_regr_r, loss_slpn_cls_r, loss_slpn_regr_r, num_box = [],[],[],[],[],[]
vis = True
for epoch_num in range(num_epochs):
	progbar = generic_utils.Progbar(epoch_length)
	print('Epoch {}/{}'.format(epoch_num + 1+add_epoch, num_epochs+add_epoch))
	while True:
		try:
			X, Y, img_data = next(data_gen_train)

			loss_befn = model_befn.train_on_batch(X, Y)

			P_befn = model_befn.predict_on_batch(X)

			R = bbox_process.get_proposal(anchors,P_befn[0], P_befn[1], C, overlap_thresh=0.7,
										  pre_nms_topN=C.train_pre_nms_topN, post_nms_topN=C.train_post_nms_topN)
			X2, Y1, Y2 = bbox_process.get_target_det(R, img_data, C)

			if X2 is None or X2.shape[1]<C.num_rois:
				continue
			pos_samples = np.where(Y1[0, :, -1] == 0)[0]
			rpn_accuracy_for_epoch.append(len(pos_samples))

			loss_slpn = model_slpn.train_on_batch([X, X2], [Y1, Y2])

			losses[iter_num, 0] = loss_befn[1]
			losses[iter_num, 1] = loss_befn[2]

			losses[iter_num, 2] = loss_slpn[1]
			losses[iter_num, 3] = loss_slpn[2]

			iter_num += 1
			if iter_num%100 == 0:
				progbar.update(iter_num, [('befn_cls', np.mean(losses[:iter_num, 0])), ('befn_regr', np.mean(losses[:iter_num, 1])),
										  ('slpn_cls', np.mean(losses[:iter_num, 2])), ('slpn_regr', np.mean(losses[:iter_num, 3]))])

			if iter_num == epoch_length:
				loss_befn_cls = np.mean(losses[:, 0])
				loss_befn_regr = np.mean(losses[:, 1])
				loss_slpn_cls = np.mean(losses[:, 2])
				loss_slpn_regr = np.mean(losses[:, 3])
				mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
				rpn_accuracy_for_epoch = []
				curr_loss = loss_befn_cls + loss_befn_regr + loss_slpn_cls + loss_slpn_regr

				curr_loss_r.append(curr_loss)
				loss_befn_cls_r.append(loss_befn_cls)
				loss_befn_regr_r.append(loss_befn_regr)
				loss_slpn_cls_r.append(loss_slpn_cls)
				loss_slpn_regr_r.append(loss_slpn_regr)
				num_box.append(mean_overlapping_bboxes)

				print('Mean number of bbx from RPN overlapping ground truth: {}'.format(mean_overlapping_bboxes))
				print('Total Loss: {}'.format(curr_loss))
				print('Elapsed time: {}'.format(time.time() - start_time))

				iter_num = 0
				start_time = time.time()

				if curr_loss < best_loss:
					print('Total loss decreased from {} to {}, saving weights'.format(best_loss,curr_loss))
					best_loss = curr_loss
				model_all.save_weights(
				os.path.join(out_path, 'resnet_e{}_l{}.hdf5'.format(epoch_num + 1 + add_epoch, curr_loss)))
				break

		except Exception as e:
			print('Exception: {}'.format(e))
			continue
	records = np.concatenate((np.asarray(curr_loss_r).reshape((-1, 1)),
							  np.asarray(loss_befn_cls_r).reshape((-1, 1)),
							  np.asarray(loss_befn_regr_r).reshape((-1, 1)),
							  np.asarray(loss_slpn_cls_r).reshape((-1, 1)),
							  np.asarray(loss_slpn_regr_r).reshape((-1, 1)),
							  np.asarray(num_box).reshape((-1, 1))),
							 axis=-1)
	np.savetxt(res_file, np.array(records), fmt='%.4f')
print('Training complete, exiting.')
