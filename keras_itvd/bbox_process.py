from __future__ import division
import numpy as np
from .utils.cython_bbox import bbox_overlaps
from bbox_transform import bbox_transform_inv, bbox_transform,clip_boxes
from nms_wrapper import nms

def format_img(img, C):
	""" formats the image channels based on config """
	img = img.astype(np.float32)
	img[:, :, 0] -= C.img_channel_mean[0]
	img[:, :, 1] -= C.img_channel_mean[1]
	img[:, :, 2] -= C.img_channel_mean[2]

	img = np.expand_dims(img, axis=0)
	return img

def filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep

def compute_targets(ex_rois, gt_rois, classifier_regr_std,std):
    """Compute bounding-box regression targets for an image."""
    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4
    targets = bbox_transform(ex_rois, gt_rois)
	# Optionally normalize targets by a precomputed mean and stdev
    if std:
		targets = targets/np.array(classifier_regr_std)
    return targets

def get_target_det(R, img_data, C, roi_stride=8):

	gta = img_data['bboxes']
	if len(gta)==0:
		return None,None,None,None
	R = np.vstack((R, gta))
	overlaps = bbox_overlaps(np.ascontiguousarray(R, dtype=np.float),
									   np.ascontiguousarray(gta, dtype=np.float))
	# find every roi close to which bbox
	gt_assignment = overlaps.argmax(axis=1)
	max_overlaps = overlaps.max(axis=1)
	# find foreground ROIs
	fg_inds = np.where(max_overlaps>=C.classifier_positive_overlap)[0]

	fg_rois_per_image = np.round(0.5* C.num_rois)
	fg_rois_per_this_image = min(fg_rois_per_image, fg_inds.size)
	if fg_inds.size>0:
		fg_inds = np.random.choice(fg_inds, size=int(fg_rois_per_this_image),replace=False)

	# find background ROIs
	bg_inds_op = np.where((max_overlaps< C.classifier_max_overlap)&
					   (max_overlaps>=C.classifier_min_overlap))[0]
	bg_inds_nop = np.where(max_overlaps < C.classifier_min_overlap)[0]
	bg_rois_per_this_image = C.num_rois - fg_rois_per_this_image
	bg_rois_nop_per_this_image = np.round(0.5* bg_rois_per_this_image)
	bg_rois_op_per_this_image = bg_rois_per_this_image-bg_rois_nop_per_this_image
	if bg_inds_op.size>0:
		try:
			bg_inds_op = np.random.choice(bg_inds_op, size=int(bg_rois_op_per_this_image),replace=False)
		except:
			bg_inds_op = np.random.choice(bg_inds_op, size=int(bg_rois_op_per_this_image),replace=True)
	if bg_inds_nop.size>0:
		try:
			bg_inds_nop = np.random.choice(bg_inds_nop, size=int(bg_rois_nop_per_this_image),replace=False)
		except:
			bg_inds_nop = np.random.choice(bg_inds_nop, size=int(bg_rois_nop_per_this_image),replace=True)
	keep_inds = np.concatenate((fg_inds, bg_inds_op, bg_inds_nop))

	# IoUs = max_overlaps[keep_inds]
	rois = R[keep_inds]
	box_target_data = np.zeros_like(rois)
	box_target_data[:fg_inds.size, :] = compute_targets(R[fg_inds], gta[gt_assignment[fg_inds]],
														C.classifier_regr_std, std = True)
	rois[:,2], rois[:,3] = rois[:,2]-rois[:,0], rois[:,3]-rois[:,1]

	x_roi = rois.reshape((-1,4))
	y_class_regr_coords = box_target_data
	y_class_regr_label = np.zeros_like(y_class_regr_coords)
	y_class_regr_label[:fg_inds.size, :] = 1

	y_class_num = np.zeros((keep_inds.size,2))
	y_class_num[:fg_inds.size, 0] = 1
	y_class_num[fg_inds.size:, 1] = 1

	X = np.array(np.round(x_roi/roi_stride))
	if len(np.where(X[:,2]<=0)[0])>0 or len(np.where(X[:,3]<=0)[0])>0:
		X[np.where(X[:,2]<=0)[0],2] = 1
		X[np.where(X[:,3]<=0)[0],3] = 1

	Y1 = np.array(y_class_num)
	Y2 = np.concatenate([np.array(y_class_regr_label),np.array(y_class_regr_coords)],axis=1)

	# return np.expand_dims(X, axis=0), np.expand_dims(Y1, axis=0), np.expand_dims(Y2, axis=0), IoUs
	return np.expand_dims(X, axis=0), np.expand_dims(Y1, axis=0), np.expand_dims(Y2, axis=0)

def proposal_post_process(rois, roi_stride=8):
	# for classifier the box should be x y w h
	rois[:,2], rois[:,3] = rois[:,2]-rois[:,0], rois[:,3]-rois[:,1]
	X = np.array(np.round(rois/roi_stride))
	if len(np.where(X[:,2]<=0)[0])>0 or len(np.where(X[:,3]<=0)[0])>0:
		X[np.where(X[:,2]<=0)[0],2] = 1
		X[np.where(X[:,3]<=0)[0],3] = 1
	return np.expand_dims(X, axis=0)

def slpn_pred(ROIs, P_cls, P_regr, C, bbox_thresh=0.1, nms_thresh=0.3,roi_stride=8):
	# classifier output the box of x y w h and downscaled
	scores = np.squeeze(P_cls[:,:,0], axis=0)
	regr = np.squeeze(P_regr, axis=0)
	rois = np.squeeze(ROIs, axis=0)

	keep = np.where(scores>=bbox_thresh)[0]
	if len(keep)==0:
		return [], []

	rois[:, 2] += rois[:, 0]
	rois[:, 3] += rois[:, 1]
	rois = rois[keep]*roi_stride
	scores = scores[keep]
	regr = regr[keep]*np.array(C.classifier_regr_std).astype(dtype=np.float32)
	# regr = regr[keep]
	pred_boxes = bbox_transform_inv(rois, regr)
	pred_boxes = clip_boxes(pred_boxes, [C.random_crop[0],C.random_crop[1]])

	keep = np.where((pred_boxes[:,2]-pred_boxes[:,0]>=3)&
					(pred_boxes[:,3]-pred_boxes[:,1]>=3))[0]
	pred_boxes = pred_boxes[keep]
	scores = scores[keep].reshape((-1,1))

	keep = nms(np.hstack((pred_boxes, scores)), nms_thresh, usegpu=False, gpu_id=0)
	pred_boxes = pred_boxes[keep]
	scores = scores[keep]

	return pred_boxes, scores

def get_proposal(all_anchors, cls_layer, regr_layer, C, overlap_thresh=0.7,pre_nms_topN=1000,post_nms_topN=300, roi_stride=8):
	A = np.copy(all_anchors[:,:4])
	scores = cls_layer.reshape((-1,1))
	bbox_deltas = regr_layer.reshape((-1,4))
	proposals = bbox_transform_inv(A, bbox_deltas)
	proposals = clip_boxes(proposals, [C.random_crop[0],C.random_crop[1]])
	keep = filter_boxes(proposals, roi_stride)
	proposals = proposals[keep,:]
	scores = scores[keep]
	order = scores.ravel().argsort()[::-1]
	order = order[:pre_nms_topN]
	proposals =  proposals[order,:]
	scores = scores[order]
	keep = nms(np.hstack((proposals, scores)), overlap_thresh, usegpu=False, gpu_id=0)
	keep = keep[:post_nms_topN]
	proposals = proposals[keep,:]
	return proposals