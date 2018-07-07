
class Config:

	def __init__(self):
		self.gpu_ids = '0'
		self.num_epochs = 10
		self.PNW = True

		# setting for data augmentation
		self.use_horizontal_flips = True
		self.brightness = (0.5, 2, 0.5)
		self.translate = ((0, 30), (0, 30), 0.5)
		self.scale = (0.8, 1.3, 0.5)
		self.random_crop = (448, 768, 1, 5)
		self.in_thre = 0.3

		# setting for scales
		self.anchor_box_scales = [[16],[32,64],[128],[256]]
		self.anchor_ratios = [[1], [1], [0.5, 1, 2], [0.5, 1, 2]]

		# image channel-wise mean to subtract, the order is BGR
		self.img_channel_mean = [103.939, 116.779, 123.68]
		# scaling the stdev
		self.classifier_regr_std = [0.1, 0.1, 0.2, 0.2]
		# num of selected rois
		self.num_rois = 256

		# overlaps for bfen
		self.rpn_min_overlap = 0.3
		self.rpn_max_overlap = 0.5

		# overlaps for classifier ROIs
		self.classifier_min_overlap = 0.1
		self.classifier_max_overlap = 0.5
		self.classifier_positive_overlap = 0.5

		# setting for num of proposals
		self.train_pre_nms_topN = 12000
		self.train_post_nms_topN = 800
		self.test_pre_nms_topN = 6000
		self.test_post_nms_topN = 300
