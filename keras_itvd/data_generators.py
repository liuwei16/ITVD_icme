from __future__ import absolute_import
# import numpy as np
# import cv2
import random
from . import data_augment
from .utils.cython_bbox import bbox_overlaps
from .utils.bbox import box_op
from .bbox_transform import *

# get fast anchor tatget
def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """

    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr
def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors
def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors
def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

def get_anchors(img_width, img_height, feat_map_sizes, anchor_box_scales, anchor_ratios):
	downscale = np.asarray([[8],[16],[32],[64]])
	ancs= []
	num_anchors = np.zeros((len(downscale)),dtype=np.int)
	for layer in range(len(downscale)):
		anchor_scales = anchor_box_scales[layer] / downscale[layer]
		base_anchor = np.array([1, 1, downscale[layer], downscale[layer]]) - 1
		ratio_anchors = _ratio_enum(base_anchor, anchor_ratios[layer])
		anchors = np.vstack([_scale_enum(ratio_anchors[i, :], anchor_scales)
							 for i in xrange(ratio_anchors.shape[0])])
		num_anchors[layer] = len(anchors)

		output_width, output_height = feat_map_sizes[layer][1], feat_map_sizes[layer][0]

		shift_x = np.arange(output_width) * downscale[layer]
		shift_y = np.arange(output_height) * downscale[layer]
		shift_x, shift_y = np.meshgrid(shift_x, shift_y)
		shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
							shift_x.ravel(), shift_y.ravel())).transpose()
		all_anchors = np.expand_dims(anchors, axis=0) + np.expand_dims(shifts, axis=0).transpose((1, 0, 2))
		all_anchors = np.reshape(all_anchors, (-1, 4))
		# only keep anchors inside the image
		all_anchors[:,[0,1]][all_anchors[:,[0,1]] < 0] = 0
		all_anchors[:, 0][all_anchors[:, 0] < 0] = 0
		all_anchors[:, 1][all_anchors[:, 1] < 0] = 0
		all_anchors[:, 2][all_anchors[:, 2] >= img_width] = img_width - 1
		all_anchors[:, 3][all_anchors[:, 3] >= img_height] = img_height - 1

		all_anchors = np.concatenate((all_anchors, np.ones((all_anchors.shape[0], 1))), axis=-1)
		ancs.append(all_anchors)
	return np.concatenate(ancs,axis=0), num_anchors

def calc_rpn_multilayer(C, img_data, anchors, num_rois = 256):
	all_anchors = np.copy(anchors)
	num_bboxes = len(img_data['bboxes'])
	gta = img_data['bboxes']
	ignoreareas = img_data['ignoreareas']
	# calculate the valid anchors (without thoses in the ignore areas and outside the image)
	if len(ignoreareas)>0:
		ignore_overlap = box_op(np.ascontiguousarray(all_anchors[:,:4], dtype=np.float),
									   np.ascontiguousarray(ignoreareas, dtype=np.float))
		ignore_sum  = np.sum(ignore_overlap, axis=1)
		all_anchors[ignore_sum>0.7,-1] = 0
	valid_idxs = np.where(all_anchors[:,-1]==1)[0]

	# initialise empty output objectives
	y_rpn_overlap = np.zeros((all_anchors.shape[0], 1))
	y_is_box_valid = np.zeros((all_anchors.shape[0], 1))
	y_rpn_regr = np.zeros((all_anchors.shape[0], 4))

	valid_anchors = all_anchors[valid_idxs,:]
	valid_rpn_overlap = np.zeros((valid_anchors.shape[0], 1))
	valid_is_box_valid = np.zeros((valid_anchors.shape[0], 1))
	valid_rpn_regr = np.zeros((valid_anchors.shape[0], 4))
	if num_bboxes>0:
		valid_overlap = bbox_overlaps(np.ascontiguousarray(valid_anchors, dtype=np.float),
									  np.ascontiguousarray(gta, dtype=np.float))
		# find every anchor close to which bbox
		argmax_overlaps = valid_overlap.argmax(axis=1)
		max_overlaps = valid_overlap[np.arange(len(valid_idxs)), argmax_overlaps]
		# find which anchor closest to every bbox
		gt_argmax_overlaps = valid_overlap.argmax(axis=0)
		gt_max_overlaps = valid_overlap[gt_argmax_overlaps, np.arange(num_bboxes)]
		gt_argmax_overlaps = np.where(valid_overlap == gt_max_overlaps)[0]
		valid_rpn_overlap[gt_argmax_overlaps] = 1
		valid_rpn_overlap[max_overlaps>C.rpn_max_overlap] = 1
		for i in range(len(gta)):
			inds = valid_overlap[:,i].ravel().argsort()[-3:]
			valid_rpn_overlap[inds] = 1
		# get positives labels
		fg_inds = np.where(valid_rpn_overlap == 1)[0]
		if len(fg_inds)>num_rois/2:
			able_inds = np.random.choice(fg_inds, size=num_rois/2, replace=False)
			valid_is_box_valid[able_inds] = 1
		else:
			valid_is_box_valid[fg_inds] = 1
		# get  positives reress
		fg_inds = np.where(valid_is_box_valid == 1)[0]
		anchor_box = valid_anchors[fg_inds,:4]
		gt_box = gta[argmax_overlaps[fg_inds], :]
		# compute regression targets
		valid_rpn_regr[fg_inds, :] = bbox_transform(anchor_box, gt_box)

		bg_inds = np.where((max_overlaps < C.rpn_min_overlap) & (valid_is_box_valid.reshape((-1)) == 0))[0]
		if len(bg_inds)>num_rois-np.sum(valid_is_box_valid==1):
			able_inds = np.random.choice(bg_inds, size=num_rois-np.sum(valid_is_box_valid==1), replace=False)
			valid_is_box_valid[able_inds] = 1
		else:
			valid_is_box_valid[bg_inds] = 1

		# transform to the original overlap and validbox
		y_rpn_overlap[valid_idxs, :] = valid_rpn_overlap
		y_is_box_valid[valid_idxs, :] = valid_is_box_valid
		y_rpn_regr[valid_idxs, :] = valid_rpn_regr
	y_rpn_cls = np.expand_dims(np.concatenate([y_is_box_valid, y_rpn_overlap], axis=1),axis=0)
	y_rpn_regr = np.expand_dims(np.concatenate([np.repeat(y_rpn_overlap, 4, axis=1), y_rpn_regr], axis=1),axis=0)

	return y_rpn_cls, y_rpn_regr

def get_target(anchors, all_img_data, C,batchsize = 32, num_rois=256, mode='test',data_out=True):
	current = 0
	while True:
		x_img_batch, y_cls_batch, y_regr_batch, img_data_batch = [], [], [] ,[]
		if current>=len(all_img_data)-batchsize:
			random.shuffle(all_img_data)
			current = 0
		for img_data in all_img_data[current:current+batchsize]:
			try:
				if mode=='train':
					img_data, x_img = data_augment.augment(img_data, C, augment=True)
				else:
					img_data, x_img = data_augment.augment(img_data, C, augment=False)
				y_rpn_cls, y_rpn_regr = calc_rpn_multilayer(C, img_data, anchors, num_rois=num_rois)
				x_img = x_img.astype(np.float32)
				x_img[:, :, 0] -= C.img_channel_mean[0]
				x_img[:, :, 1] -= C.img_channel_mean[1]
				x_img[:, :, 2] -= C.img_channel_mean[2]
				x_img = np.expand_dims(x_img, axis=0)

				x_img_batch.append(x_img)
				y_cls_batch.append(y_rpn_cls)
				y_regr_batch.append(y_rpn_regr)
				img_data_batch.append(img_data)
			except Exception as e:
				print 'get_batch_gt:',e
		x_img_batch = np.concatenate(np.array(x_img_batch),axis=0)
		y_cls_batch = np.concatenate(np.array(y_cls_batch), axis=0)
		y_regr_batch = np.concatenate(np.array(y_regr_batch), axis=0)
		current += batchsize

		if data_out:
			yield np.copy(x_img_batch), [np.copy(y_cls_batch), np.copy(y_regr_batch)], img_data_batch
		else:
			yield np.copy(x_img_batch), [np.copy(y_cls_batch), np.copy(y_regr_batch)]