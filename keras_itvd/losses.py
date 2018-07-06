from keras import backend as K
from keras.objectives import categorical_crossentropy

if K.image_dim_ordering() == 'tf':
	import tensorflow as tf

lambda_rpn_regr = 1.0
lambda_rpn_class = 1.0
lambda_cls_regr = 1.0
lambda_cls_class = 1.0
epsilon = 1e-4

def rpn_loss_regr_multilayer(num_anchors):
	def rpn_loss_regr_fixed_num(y_true, y_pred):
		loss = 0
		for i in range(len(y_true)):
			x = y_true[i][:, :, :, 4 * num_anchors[i]:] - y_pred[i]
			x_abs = K.abs(x)
			x_bool = K.cast(K.less_equal(x_abs, 1.0), tf.float32)
			loss += lambda_rpn_regr * K.sum(
			y_true[i][:, :, :, :4 * num_anchors[i]] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[i][:, :, :, :4 * num_anchors[i]])
		return loss
	return rpn_loss_regr_fixed_num

def rpn_loss_cls_multilayer(num_anchors):
	def rpn_loss_cls_fixed_num(y_true, y_pred):
		loss = 0
		for i in range(len(y_true)):
			loss += lambda_rpn_class * K.sum(y_true[i][:, :, :, :num_anchors[i]] * K.binary_crossentropy(y_pred[i][:, :, :, :], y_true[i][:, :, :, num_anchors[i]:])) \
					/ K.sum(epsilon + y_true[i][:, :, :, :num_anchors[i]])
		return loss
	return rpn_loss_cls_fixed_num

def rpn_loss_regr():
	def rpn_loss_regr_fixed_num(y_true, y_pred):
		x = y_true[:, :, 4:] - y_pred
		x_abs = K.abs(x)
		x_bool = K.cast(K.less_equal(x_abs, 1.0), tf.float32)
		return lambda_rpn_regr * K.sum(
			y_true[:,:, :4] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(
			epsilon + y_true[:, :, :4])
	return rpn_loss_regr_fixed_num

def rpn_loss_cls():
	def rpn_loss_cls_fixed_num(y_true, y_pred):
		return lambda_rpn_class * K.sum(y_true[:,:,:1] * K.binary_crossentropy(y_pred[:,:,:],y_true[:,:,1:])) / K.sum(epsilon + y_true[:,:,:1])
	return rpn_loss_cls_fixed_num

# apply focal loss for rpn cls loss
def rpn_loss_cls_focal(y_true, y_pred):
	classification_loss = tf.reduce_sum(y_true[:,:,:1] * K.binary_crossentropy(y_pred[:, :, :], y_true[:, :, 1:]), axis=-1)
	positives = y_true[:,:,1]
	negatives = y_true[:,:,0]-y_true[:,:,1]
	# firstly we compute the focal weight
	foreground_alpha = positives*tf.constant(0.25)
	background_alpha = negatives*tf.constant(0.75)
	foreground_weight = foreground_alpha*(tf.constant(1.0)-y_pred[:,:,0])**tf.constant(2.0)
	background_weight = background_alpha*y_pred[:,:,0]**tf.constant(2.0)
	focal_weight = foreground_weight+background_weight

	assigned_boxes = tf.reduce_sum(positives)
	class_loss = tf.reduce_sum(classification_loss * focal_weight, axis=-1)/tf.maximum(1.0, assigned_boxes)
	return class_loss

def class_loss_regr(num_classes):
	def class_loss_regr_fixed_num(y_true, y_pred):
		x = y_true[:, :, 4*num_classes:] - y_pred
		x_abs = K.abs(x)
		x_bool = K.cast(K.less_equal(x_abs, 1.0), 'float32')
		return lambda_cls_regr * K.sum(y_true[:, :, :4*num_classes] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :, :4*num_classes])
	return class_loss_regr_fixed_num

def class_loss_cls(y_true, y_pred):
	return lambda_cls_class * K.mean(categorical_crossentropy(y_true[0, :, :], y_pred[0, :, :]))

# apply focal loss for classifier loss
def class_loss_cls_focal(y_true, y_pred):
	classification_loss = -tf.reduce_sum(y_true * tf.log(y_pred+1e-10), axis=-1)
	positives = y_true[:, :, 0]
	negatives = y_true[:, :, 1]
	# firstly we compute the focal weight
	foreground_alpha = positives * tf.constant(0.25)
	background_alpha = negatives * tf.constant(0.75)
	foreground_weight = foreground_alpha * (tf.constant(1.0) - y_pred[:, :, 0]) ** tf.constant(2.0)
	background_weight = background_alpha * (tf.constant(1.0) - y_pred[:, :, 1]) ** tf.constant(2.0)
	focal_weight = foreground_weight + background_weight

	assigned_boxes = tf.reduce_sum(positives)
	class_loss = tf.reduce_sum(classification_loss * focal_weight, axis=-1) / tf.maximum(1.0, assigned_boxes)
	return class_loss
