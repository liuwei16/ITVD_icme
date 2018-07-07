# Improving Tiny Vehicle Detection in Complex Scenes
This is the Keras implementation of our paper accepted in ICME 2018.
## Introduction
In this paper, we propose a deep network for accu-
rate vehicle detection, with the main idea of using a relatively large feature map for proposal generation, and keeping ROI feature’s spatial layout to represent and detect tiny vehicles. Even with only 100 proposals, the resulting proposal network achieves an encouraging recall over 99%. Furthermore, unlike a common practice which flatten features after ROI pooling, we argue
that for a better detection of tiny vehicles, the spatial layout of the ROI features should be preserved and fully integrated. Accordingly, we use a multi-path light-weight processing chain to effectively integrate ROI features, while preserving the spatial layouts. Experiments done on the challenging DETRAC vehicle detection benchmark show that the proposed
method largely improves a competitive baseline (ResNet50
based Faster RCNN) by 16.5% mAP. For more details, please refer to our [paper](https://github.com/liuwei16/ITVD_icme/blob/master/docs/2018ICME-ITVD.pdf). 

<img align="right" src="https://github.com/liuwei16/ITVD_icme/blob/master/docs/itvd.png">


### Dependencies

* Python 2.7
* Numpy
* Tensorflow 1.x
* Keras 2.0.6
* OpenCV

## Contents
1. [Installation](#installation)
2. [Preparation](#preparation)
3. [Models](#models)
4. [Training](#training)
5. [Test](#test)

### Installation
1. Get the code. We will call the cloned directory as '$ITVD_icme'.
```
  git clone https://github.com/liuwei16/ITVD_icme.git
```
2. Install the requirments.
```
  pip install -r requirements.txt
```

### Preparation
1. Download the dataset.
We trained and tested our model on the recent [DETRAC](http://detrac-db.rit.albany.edu) vehicle detection dataset, you should firstly download the datasets. By default, we assume the dataset is stored in '$ITVD_icme/data/detrac/'.

2. Dataset preparation.
Follow the [./generate_data.py](https://github.com/liuwei16/ITVD_icme/blob/master/generate_data.py) to create the cache files for training, validation and test. By default, we assume the cache files is stored in '$ITVD_icme/data/cache/detrac/'.

3. Download the initialized models.
We use the backbone [ResNet-50](https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5) in our experiments. By default, we assume the weight files is stored in '$ITVD_icme/data/models/'.

### Models
We have provided the models that are trained from training subset and training+validation subsets. To help reproduce the results in our paper,
1. For validation set: [det_val.hdf5](https://pan.baidu.com/s/1WPo6dUfMchV_EECmG_VZgw)
2. For test set: [det_test.hdf5](https://pan.baidu.com/s/1WPo6dUfMchV_EECmG_VZgw)

### Training
Optionally, you can set the training parameters in [./keras_itvd/config.py](https://github.com/liuwei16/ITVD_icme/blob/master/keras_itvd/config.py). For the ablation experiments, models are trained on the validation subset. For the results submitted to the benchmark, models are trained on the validation+test subsets.

1. Train the proposal generation network -- BFEN.
Follow the [./train_bfen.py](https://github.com/liuwei16/ITVD_icme/blob/master/train_bfen.py) to train the BFEN. By default, the output weight files will be saved in '$ITVD_icme/output/valmodels/bfen/'.

2. Train the detction network.
Follow the [./train_det.py](https://github.com/liuwei16/ITVD_icme/blob/master/train_det.py) to train the detection. By default, the output weight files will be saved in '$ITVD_icme/output/valmodels/bfen/'. Optionally, you can jointly train the whole network by setting the self.PNW = False in [./keras_itvd/config.py](https://github.com/liuwei16/ITVD_icme/blob/master/keras_itvd/config.py). By default, the whole network is initialized from the pretrained BEFN, which is corresponding to the Proposal Network Warm-up (PNW) strategy introduced in the paper, we find this strategy is helpful for improvement as demostrated in the experiments. We also provid the weight files of the pretrained BEFN:
(1) Trained on training set: [bfen_val.hdf5](https://pan.baidu.com/s/1WPo6dUfMchV_EECmG_VZgw)
(2) Trained on training+validation set: [bfen_test.hdf5](https://pan.baidu.com/s/1WPo6dUfMchV_EECmG_VZgw)

### Test
Follow the [./test_det.py](https://github.com/liuwei16/ITVD_icme/blob/master/test_det.py) to get the detection results. By default, the output .txt files will be saved in '$ITVD_icme/output/valresults/det/'.

## Citation
If you think our work is useful in your research, please consider citing:
```
@inproceedings{liu2018improving,
  title={Improving Tiny Vehicle Detection in Complex Scenes},
  author={Wei Liu, Shengcai Liao, Weidong Hu, Xuezhi Liang, Yan Zhang},
  booktitle={IEEE International Conference on Multimedia and Expo (ICME)},
  year={2018}
}
```







