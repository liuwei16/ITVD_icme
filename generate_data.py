import os
import cPickle
import numpy as np
from keras_itvd.utils.bbox import box_op
import matplotlib.pyplot as plt
import cv2
root_dir = 'detrac'
all_img_path = os.path.join(root_dir, 'trainval')
all_anno_path = os.path.join(root_dir, 'annotations')
all_ignore_path = os.path.join(root_dir, 'anno')
types =['train']

cols, rows = 960, 540
for type in types:
    img_path = os.path.join(all_img_path, type)
    anno_path = os.path.join(all_anno_path, type)
    res_path = os.path.join('cache/detrac', 'train')
    if not os.path.exists(res_path):
        os.mkdir(res_path)
    for seq in sorted(os.listdir(img_path)):
        seq_name = seq.rstrip('_'+seq.split('_')[-1])
        seq_ignore_path = os.path.join(all_ignore_path, seq_name, 'ignored_region')
        with open(seq_ignore_path, 'rb') as fid:
            lines = fid.readlines()
        seq_ignoreareas = []
        for i in range(2,len(lines)):
            rec = lines[i].strip().split(' ')
            rec = np.asarray([int(float(rec[0])),int(float(rec[1])),int(float(rec[2])+float(rec[0])), int(float(rec[3])+float(rec[1]))])
            seq_ignoreareas.append(rec)
        cache_file = os.path.join(res_path, seq)
        seq_img_path = os.path.join(img_path, seq)
        seq_anno_path = os.path.join(anno_path, seq)
        if os.path.exists(cache_file):
            print cache_file, 'already generated'
        imgs = sorted(os.listdir(seq_img_path))
        image_data = []
        for index in imgs:
            if int(index.split('.')[0][3:])%5 !=0:
                continue
            annotation = {}
            img = os.path.join(seq_img_path,index)
            gt = os.path.join(seq_anno_path,index.split('.')[0])
            annotation['filepath'] = img
            annotation['ignoreareas'] = np.array(seq_ignoreareas)
            boxes = []
            if os.path.exists(gt):
                with open(gt, 'r') as f:
                    for line in f:
                        line_split = line.strip().split(' ')
                        if len(line_split)<4:
                            continue
                        x1, y1, x2, y2 = int(float(line_split[0])), int(float(line_split[1])), \
                         int(float(line_split[2])+float(line_split[0])), int(float(line_split[3])+float(line_split[1]))
                        x1 = max(x1,1)
                        y1 = max(y1,1)
                        x2 = min(x2,cols-1)
                        y2 = min(y2,rows-1)
                        box = np.asarray([x1, y1, x2, y2])
                        boxes.append(box)
            boxes = np.array(boxes)
            # img = cv2.imread(annotation['filepath'])
            # for i in range(len(annotation['ignoreareas'])):
		     #    cv2.rectangle(img,(annotation['ignoreareas'][i,0], annotation['ignoreareas'][i,1]), (annotation['ignoreareas'][i,2], annotation['ignoreareas'][i,3]), (0, 0, 255),3)
            # for i in range(len(boxes)):
		     #    cv2.rectangle(img,(int(boxes[i,0]), int(boxes[i,1])), (int(boxes[i,2]), int(boxes[i,3])), (255, 0, 0),2)
		     #    plt.imshow(img, interpolation='bicubic')
            if len(boxes)==0:
                continue
            if len(annotation['ignoreareas'])>0:
                boxig_overlap = box_op(np.ascontiguousarray(boxes, dtype=np.float),
                                       np.ascontiguousarray(annotation['ignoreareas'], dtype=np.float))
                ignore_sum  = np.sum(boxig_overlap, axis=1)
                boxes = boxes[ignore_sum<0.5, :]
            annotation['bboxes'] = boxes
            image_data.append(annotation)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(image_data, fid, cPickle.HIGHEST_PROTOCOL)
        print '{}, {}'.format(type,seq)

