from __future__ import division
import cv2
import numpy as np
import copy
import matplotlib.pyplot as plt
# da functions
def _translate(image, horizontal=(0,40), vertical=(0,10)):
    '''
    Randomly translate the input image horizontally and vertically.

    Arguments:
        image (array-like): The image to be translated.
        horizontal (int tuple, optinal): A 2-tuple `(min, max)` with the minimum
            and maximum horizontal translation. A random translation value will
            be picked from a uniform distribution over [min, max].
        vertical (int tuple, optional): Analog to `horizontal`.

    Returns:
        The translated image and the horzontal and vertical shift values.
    '''
    rows,cols,ch = image.shape

    x = np.random.randint(horizontal[0], horizontal[1]+1)
    y = np.random.randint(vertical[0], vertical[1]+1)
    x_shift = np.random.choice([-x, x])
    y_shift = np.random.choice([-y, y])

    M = np.float32([[1,0,x_shift],[0,1,y_shift]])
    return cv2.warpAffine(image, M, (cols, rows)), x_shift, y_shift

def _brightness(image, min=0.5, max=2.0):
    '''
    Randomly change the brightness of the input image.

    Protected against overflow.
    '''
    hsv = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)

    random_br = np.random.uniform(min,max)

    #To protect against overflow: Calculate a mask for all pixels
    #where adjustment of the brightness would exceed the maximum
    #brightness value and set the value to the maximum at those pixels.
    mask = hsv[:,:,2] * random_br > 255
    v_channel = np.where(mask, 255, hsv[:,:,2] * random_br)
    hsv[:,:,2] = v_channel

    return cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)

def _scale(image, min=0.9, max=1.1):
    '''
    Scale the input image by a random factor picked from a uniform distribution
    over [min, max].

    Returns:
        The scaled image, the associated warp matrix, and the scaling value.
    '''

    rows,cols,ch = image.shape

    #Randomly select a scaling factor from the range passed.
    scale = np.random.uniform(min, max)

    M = cv2.getRotationMatrix2D((cols/2,rows/2), 0, scale)
    return cv2.warpAffine(image, M, (cols, rows)), M, scale

def augment(img_data, c, augment=True):
    assert 'filepath' in img_data
    assert 'bboxes' in img_data
    img_data_aug = copy.deepcopy(img_data)
    img = cv2.imread(img_data_aug['filepath'])
    img_height, img_width = img.shape[:2]
    if augment:
        # random brightness
        if c.brightness and np.random.randint(0, 2) == 0:
            img = _brightness(img, min=c.brightness[0], max=c.brightness[1])
        # random horizontal flip
        if c.use_horizontal_flips and np.random.randint(0, 2) == 0:
            img = cv2.flip(img, 1)
            if len(img_data_aug['bboxes']) > 0:
                img_data_aug['bboxes'][:, [0, 2]] = img_width - img_data_aug['bboxes'][:, [2, 0]]
            if len(img_data_aug['ignoreareas']) > 0:
                img_data_aug['ignoreareas'][:, [0, 2]] = img_width - img_data_aug['ignoreareas'][:, [2, 0]]
        # random translate
        if c.translate and np.random.randint(0, 2) == 0:
            img, xshift, yshift = _translate(img, c.translate[0], c.translate[1])
            # adjust the box
            if len(img_data_aug['bboxes']) > 0:
                img_data_aug['bboxes'][:, [0, 2]] += xshift
                img_data_aug['bboxes'][:, [1, 3]] += yshift
                before_limiting = copy.deepcopy(img_data_aug['bboxes'])
                x_coords = img_data_aug['bboxes'][:, [0, 2]]
                x_coords[x_coords >= img_width] = img_width - 1
                x_coords[x_coords < 0] = 0
                img_data_aug['bboxes'][:, [0, 2]] = x_coords
                y_coords = img_data_aug['bboxes'][:, [1, 3]]
                y_coords[y_coords >= img_height] = img_height - 1
                y_coords[y_coords < 0] = 0
                img_data_aug['bboxes'][:, [1, 3]] = y_coords
                before_area = (before_limiting[:,2]-before_limiting[:,0])*(before_limiting[:,3]-before_limiting[:,1])
                after_area =  (img_data_aug['bboxes'][:,2]-img_data_aug['bboxes'][:,0])*(img_data_aug['bboxes'][:,3]-img_data_aug['bboxes'][:,1])
                img_data_aug['bboxes'] = img_data_aug['bboxes'][after_area >= c.in_thre* before_area]
            if len(img_data_aug['ignoreareas']) > 0:
                img_data_aug['ignoreareas'][:, [0, 2]] += xshift
                img_data_aug['ignoreareas'][:, [1, 3]] += yshift

                x_coords = img_data_aug['ignoreareas'][:, [0, 2]]
                x_coords[x_coords >= img_width] = img_width - 1
                x_coords[x_coords < 0] = 0
                img_data_aug['ignoreareas'][:, [0, 2]] = x_coords
                y_coords = img_data_aug['ignoreareas'][:, [1, 3]]
                y_coords[y_coords >= img_height] = img_height - 1
                y_coords[y_coords < 0] = 0
                img_data_aug['ignoreareas'][:, [1, 3]] = y_coords

                after_area = (img_data_aug['ignoreareas'][:, 2] - img_data_aug['ignoreareas'][:, 0]) * (img_data_aug['ignoreareas'][:, 3] - img_data_aug['ignoreareas'][:, 1])
                img_data_aug['ignoreareas'] = img_data_aug['ignoreareas'][after_area >10]

        # random scale
        if c.scale and np.random.randint(0, 2) == 0:
            img, M, scale_factor = _scale(img, c.scale[0], c.scale[1])
            # Adjust the box coordinates
            if len(img_data_aug['bboxes']) > 0:
                toplefts = np.array([img_data_aug['bboxes'][:, 0], img_data_aug['bboxes'][:, 1],
                                     np.ones(img_data_aug['bboxes'].shape[0])])
                bottomrights = np.array([img_data_aug['bboxes'][:, 2], img_data_aug['bboxes'][:, 3],
                                         np.ones(img_data_aug['bboxes'].shape[0])])
                new_toplefts = (np.dot(M, toplefts)).T
                new_bottomrights = (np.dot(M, bottomrights)).T
                img_data_aug['bboxes'][:, [0, 1]] = np.round(new_toplefts).astype(np.int64)
                img_data_aug['bboxes'][:, [2, 3]] = np.round(new_bottomrights).astype(np.int64)
                w, h = img_data_aug['bboxes'][:, 2] - img_data_aug['bboxes'][:, 0], img_data_aug['bboxes'][:,
                                                                                    3] - img_data_aug['bboxes'][
                                                                                         :, 1]
                img_data_aug['bboxes'] = img_data_aug['bboxes'][(w > 10) & (h > 10)]
                if scale_factor > 1:
                    before_limiting = copy.deepcopy(img_data_aug['bboxes'])
                    x_coords = img_data_aug['bboxes'][:, [0, 2]]
                    x_coords[x_coords >= img_width] = img_width - 1
                    x_coords[x_coords < 0] = 0
                    img_data_aug['bboxes'][:, [0, 2]] = x_coords
                    y_coords = img_data_aug['bboxes'][:, [1, 3]]
                    y_coords[y_coords >= img_height] = img_height - 1
                    y_coords[y_coords < 0] = 0
                    img_data_aug['bboxes'][:, [1, 3]] = y_coords
                    before_area = (before_limiting[:, 2] - before_limiting[:, 0]) * (
                    before_limiting[:, 3] - before_limiting[:, 1])
                    after_area = (img_data_aug['bboxes'][:, 2] - img_data_aug['bboxes'][:, 0]) * (
                    img_data_aug['bboxes'][:, 3] - img_data_aug['bboxes'][:, 1])
                    img_data_aug['bboxes'] = img_data_aug['bboxes'][after_area >= c.in_thre * before_area]

            if len(img_data_aug['ignoreareas']) > 0:
                toplefts = np.array([img_data_aug['ignoreareas'][:, 0], img_data_aug['ignoreareas'][:, 1],
                                     np.ones(img_data_aug['ignoreareas'].shape[0])])
                bottomrights = np.array([img_data_aug['ignoreareas'][:, 2], img_data_aug['ignoreareas'][:, 3],
                                         np.ones(img_data_aug['ignoreareas'].shape[0])])
                new_toplefts = (np.dot(M, toplefts)).T
                new_bottomrights = (np.dot(M, bottomrights)).T
                img_data_aug['ignoreareas'][:, [0, 1]] = np.round(new_toplefts).astype(np.int64)
                img_data_aug['ignoreareas'][:, [2, 3]] = np.round(new_bottomrights).astype(np.int64)
                if scale_factor > 1:
                    x_coords = img_data_aug['ignoreareas'][:, [0, 2]]
                    x_coords[x_coords >= img_width] = img_width - 1
                    x_coords[x_coords < 0] = 0
                    img_data_aug['ignoreareas'][:, [0, 2]] = x_coords
                    y_coords = img_data_aug['ignoreareas'][:, [1, 3]]
                    y_coords[y_coords >= img_height] = img_height - 1
                    y_coords[y_coords < 0] = 0
                    img_data_aug['ignoreareas'][:, [1, 3]] = y_coords
                    after_area = (img_data_aug['ignoreareas'][:, 2] - img_data_aug['ignoreareas'][:, 0]) * (
                    img_data_aug['ignoreareas'][:, 3] - img_data_aug['ignoreareas'][:, 1])
                    img_data_aug['ignoreareas'] = img_data_aug['ignoreareas'][after_area > 10]
        # random crop a patch
        y_range = img_height - c.random_crop[0]
        x_range = img_width - c.random_crop[1]
        min_1_object_fulfilled = False
        trial_counter = 0
        while (not min_1_object_fulfilled) and (trial_counter < c.random_crop[3]):
            crop_ymin = np.random.randint(0, y_range + 1)
            crop_xmin = np.random.randint(0, x_range + 1)
            patch_X = np.copy(
                img[crop_ymin:crop_ymin + c.random_crop[0], crop_xmin:crop_xmin + c.random_crop[1]])
            if len(img_data_aug['ignoreareas']) > 0:
                boxes = copy.deepcopy(img_data_aug['ignoreareas'])
                boxes[:, [0, 2]] -= crop_xmin
                boxes[:, [1, 3]] -= crop_ymin
                y_coords = boxes[:, [1, 3]]
                y_coords[y_coords < 0] = 0
                y_coords[y_coords >= c.random_crop[0]] = c.random_crop[0] - 1
                boxes[:, [1, 3]] = y_coords
                x_coords = boxes[:, [0, 2]]
                x_coords[x_coords < 0] = 0
                x_coords[x_coords >= c.random_crop[1]] = c.random_crop[1] - 1
                boxes[:, [0, 2]] = x_coords
                after_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                boxes = boxes[after_area > 10]
                img_data_aug['ignoreareas'] = boxes
            if len(img_data_aug['bboxes']) > 0:
                boxes = copy.deepcopy(img_data_aug['bboxes'])
                boxes[:, [0, 2]] -= crop_xmin
                boxes[:, [1, 3]] -= crop_ymin
                before_limiting = copy.deepcopy(boxes)

                y_coords = boxes[:, [1, 3]]
                y_coords[y_coords < 0] = 0
                y_coords[y_coords >= c.random_crop[0]] = c.random_crop[0] - 1
                boxes[:, [1, 3]] = y_coords

                x_coords = boxes[:, [0, 2]]
                x_coords[x_coords < 0] = 0
                x_coords[x_coords >= c.random_crop[1]] = c.random_crop[1] - 1
                boxes[:, [0, 2]] = x_coords

                before_area = (before_limiting[:, 2] - before_limiting[:, 0]) * (
                before_limiting[:, 3] - before_limiting[:, 1])
                after_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                boxes = boxes[after_area >= c.in_thre * before_area]
                trial_counter += 1
                if len(boxes) > 0:
                    min_1_object_fulfilled = True
                    img = patch_X
                    img_data_aug['bboxes'] = boxes
                elif trial_counter >= c.random_crop[3]:
                    img = patch_X
                    img_data_aug['bboxes'] = []

            else:
                img = patch_X
                break
        img_height = c.random_crop[0]
        img_width = c.random_crop[1]

        # gt = img_data_aug['bboxes']
        # ig = img_data_aug['ignoreareas']
        # plt.imshow(img, interpolation='bicubic')
        # plt.close()
        # for i in range(len(gt)):
        #     (x1, y1, x2, y2) = gt[i, :]
        #     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # for i in range(len(ig)):
        #     (x1, y1, x2, y2) = ig[i, :]
        #     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # plt.imshow(img, interpolation='bicubic')
    else:
        # random crop a  patch
        img_height = c.random_crop[0]
        img_width = c.random_crop[1]
        crop_ymin = int((img_height -c.random_crop[0]) / 2)
        crop_xmin = int((img_width - c.random_crop[1]) / 2)
        patch_X = np.copy(img[crop_ymin:crop_ymin + c.random_crop[0], crop_xmin:crop_xmin + c.random_crop[1]])
        img = patch_X
        if len(img_data_aug['bboxes']) > 0:
            boxes = copy.deepcopy(img_data_aug['bboxes'])
            boxes[:, [0, 2]] -= crop_xmin
            boxes[:, [1, 3]] -= crop_ymin
            before_limiting = copy.deepcopy(boxes)

            y_coords = boxes[:, [1, 3]]
            y_coords[y_coords < 0] = 0
            y_coords[y_coords >= c.random_crop[0]] = c.random_crop[0] - 1
            boxes[:, [1, 3]] = y_coords

            x_coords = boxes[:, [0, 2]]
            x_coords[x_coords < 0] = 0
            x_coords[x_coords >= c.random_crop[1]] = c.random_crop[1] - 1
            boxes[:, [0, 2]] = x_coords

            before_area = (before_limiting[:, 2] - before_limiting[:, 0]) * (
                before_limiting[:, 3] - before_limiting[:, 1])
            after_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            boxes = boxes[after_area >= c.in_thre * before_area]
            img_data_aug['bboxes'] = boxes
    img_data_aug['width'] = img_width
    img_data_aug['height'] = img_height
    return img_data_aug, img
