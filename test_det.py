from __future__ import division
import os
import cv2
import numpy as np
import cPickle
import time
from keras_itvd import config, data_generators, bbox_process
from keras.layers import Input
from keras.models import Model
from keras_itvd import net_itvd as nn

os.environ["CUDA_VISIBLE_DEVICES"] = '2'
C = config.Config()
C.num_rois = int(300)
C.random_crop = (540, 960)
C.gpu_id = '0'
cache_path = 'data/cache/detrac/val'
test_seqs = []
for data in sorted(os.listdir(cache_path)):
    cache_file = os.path.join(cache_path,data)
    with open(cache_file, 'rb') as fid:
        img_data = cPickle.load(fid)
    test_seqs.append(img_data)

img_input = Input(shape=(540, 960, 3))
roi_input = Input(shape=(None, 4))
# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers, feat_map_sizes = nn.nn_base(img_input, trainable=False)
# get default anchors and define data generator
anchors, num_anchors = data_generators.get_anchors(img_height=C.random_crop[0],img_width=C.random_crop[1],
                                                   feat_map_sizes=feat_map_sizes.astype(np.int),
                                                   anchor_box_scales=C.anchor_box_scales,
                                                   anchor_ratios=C.anchor_ratios)
# define the RPN, built on the base layers
bfen = nn.bfen(shared_layers, num_anchors)
model_bfen = Model(img_input, bfen)

feat_layer = Input(shape=bfen[2]._keras_shape[1:])
slpn = nn.slpn(feat_layer, roi_input, C.num_rois, nb_classes=2, trainable=False)
model_slpn = Model([feat_layer, roi_input], slpn)

weight_path = 'data/models/det_val.hdf5'
model_bfen.load_weights(weight_path, by_name=True)
model_slpn.load_weights(weight_path, by_name=True)
print('Loading cls weights from {}'.format(weight_path))

output_path = 'output/valresults/det/'
if not os.path.exists(output_path):
    os.mkdir(output_path)

start_time = time.time()
count = 0
for s in range(len(test_seqs)):
	num_imgs, count = len(test_seqs[s]), count+1
	seq_name = sorted(os.listdir(cache_path))[s]
	print '{} seq {} has {} images'.format(count, seq_name, num_imgs)
	res_file = os.path.join(output_path, seq_name.rstrip(seq_name.split('_')[-1])+'Det_vehiclenet.txt')
	res_all = []
	for f in range(len(test_seqs[s])):

		filepath = test_seqs[s][f]['filepath']
		frame_number = int(filepath.split('/')[-1].split('.')[0][3:])
		img = cv2.imread(filepath)
		X = bbox_process.format_img(img, C)
		[Y1, Y2,featmap] = model_bfen.predict(X)
		R = bbox_process.get_proposal(anchors, Y1, Y2, C, overlap_thresh=0.7,pre_nms_topN=C.test_pre_nms_topN, post_nms_topN=C.num_rois)
		ROIs = bbox_process.proposal_post_process(R, roi_stride=8)

		[P_cls, P_regr] = model_slpn.predict([featmap, ROIs])
		new_boxes, new_probs = bbox_process.slpn_pred(ROIs, P_cls, P_regr, C, 0.1, nms_thresh=0.7,roi_stride=8)

		if len(new_boxes)!= 0:
			new_boxes[:,2], new_boxes[:,3] = new_boxes[:,2]-new_boxes[:,0], new_boxes[:,3]-new_boxes[:,1]
			f_res = np.repeat(frame_number,len(new_boxes),axis=0).reshape((-1,1))
			idxs = np.asarray(range(1,len(new_boxes)+1)).reshape((-1,1))
			res = np.concatenate((f_res,idxs, new_boxes, new_probs),axis=-1).tolist()
			res_all += res
	np.savetxt(res_file,np.array(res_all), fmt='%.8f')
print time.time()-start_time