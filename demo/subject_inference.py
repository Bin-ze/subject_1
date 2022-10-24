# -*- coding: utf-8 -*-
# __author__:bin_ze
# 10/22/22 6:33 PM

import collections
import os
import time
import sys
import logging
import argparse

import mmcv
import cv2
from mmdet.apis import (inference_detector, init_detector)

import numpy as np

'''
参考：https://github.com/ultralytics/yolov5/blob/master/detect.py
实现了包括推理，画框，可视化的流程
支持单独调用，以及直接调用类自身完成
'''


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

colors = Colors()

class Inference:
    def __init__(self, config, checkpoint, save_path, device, conf):
        """
        Initialize model
        Returns: model
        """

        self.detection_names = ('dibiao', 'changdeng', 'zhuozi', 'shikuai', 'zhangaiwu', 'menkan', 'yizi', 'zhuankuai')
        self.segmentation_names = ('caodi', 'shuinidi', 'mubandi', 'louti', 'xiepo', 'shadi', 'eluanshidi')

        self.mask_colors = [
            np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            for _ in range(len(self.segmentation_names) + 1)
        ]

        self.save_path = save_path

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # Parse
        detection_config, segmentation_config = config
        detection_checkpoint, segmentation_checkpoint = checkpoint
        # instance model
        self.detection_model = init_detector(detection_config, detection_checkpoint, device=device)
        self.segmentation_model = init_detector(segmentation_config, segmentation_checkpoint, device=device)

        self.window = 'seg_det_res'
        self.conf = conf
        self.prefix = ['jpg', 'jpeg', 'png']  # input prefix

    def process_image(self, input_image):

        if isinstance(input_image, str):
            self.img_path = input_image.split("/")[-1]
            assert self.img_path.split('.')[-1] in self.prefix, 'input is not image!!!'
            self.im = mmcv.imread(input_image)
        elif isinstance(input_image, np.ndarray):
            self.im = input_image

        result_det = inference_detector(self.detection_model, self.im)
        result_seg = inference_detector(self.segmentation_model, self.im)

        return result_det, result_seg

    def format_det(self, result):

        bboxes = np.vstack(result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(result)
        ]
        labels = np.concatenate(labels)

        inds = bboxes[:, -1] >= self.conf

        bboxes, conf = bboxes[inds, :], bboxes[inds, -1]

        labels = labels[inds]

        return bboxes, labels, conf

    def format_seg(self, result):
        mask = []
        bboxes = np.vstack(result[0])
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(result[0])
        ]
        labels = np.concatenate(labels)
        inds = bboxes[:, -1] >= self.conf
        bboxes = bboxes[inds, :]
        labels = labels[inds]

        if len(labels):
            for k, v in dict(collections.Counter(labels)).items():
                for j in range(v):
                    mask.append(result[1][k][j])
        # plot
        for index, mask_ in enumerate(mask):
            color_mask = self.mask_colors[labels[index]]
            mask_ = mask_.astype(bool)
            self.im[mask_] = self.im[mask_] * 0.4 + color_mask * 0.6

        return mask, labels, bboxes

    def plot_seg(self, mask, labels):
        for index, mask_ in enumerate(mask):
            color_mask = self.mask_colors[labels[index]]
            mask_ = mask_.astype(bool)
            self.im[mask_] = self.im[mask_] * 0.4 + color_mask * 0.6

        return

    def plot_bbox(self, bboxes, labels, conf):

        for j, box in enumerate(bboxes):
            cls = labels[j]
            color = colors(cls)
            cls = self.detection_names[cls]
            label = f'{cls} {conf[j]:.1f}'
            self.box_label(box, label, color=color)

        return

    def show_or_save(self, view_img=False, imwrite=True):

        if view_img:
            cv2.namedWindow(self.window)
            cv2.imshow(self.window, self.im)
            cv2.waitKey(1)  # 1 millisecond
        if imwrite:
            cv2.imwrite('{}/{}'.format(self.save_path, self.img_path), self.im)

        return

    def box_label(self, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        """
        :param box:
        :param label:
        :param color:
        :param txt_color:
        :return:
        """
        # 先画框
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(self.im, p1, p2, color, thickness=2, lineType=cv2.LINE_AA)
        # 画框
        if label:
            w, h = cv2.getTextSize(label, 0, fontScale=1, thickness=1)[0]  # text width, height
            outside = p1[1] - h - 3 >= 0  # label fits outside box
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(self.im, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, 1, txt_color,
                        thickness=1, lineType=cv2.LINE_AA)

    def __call__(self, input_image, plot=True):
        result_det, result_seg = self.process_image(input_image)

        # subject_2
        det_bboxes, det_labels, det_conf = self.format_det(result_det)
        # subject_3
        seg_mask, seg_labels, seg_bboxes = self.format_seg(result_seg)

        if plot:
            self.plot_seg(seg_mask, seg_labels)
            self.plot_bbox(det_bboxes, det_labels, det_conf)
            self.show_or_save(view_img=False, imwrite=False)

        return (det_bboxes, det_labels, det_conf), (seg_mask, seg_labels, seg_bboxes), self.im


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        stream=sys.stdout,
                        format="%(asctime)s | %(filename)s:%(lineno)d | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_folder', type=str, default='/mnt/data/guozebin/subject_1/subject_1/data/detection/Annotations', help='infer annotations folder')
    parser.add_argument('--img_folder', type=str, default='/mnt/data/guozebin/subject_1/subject_1/data/detection/JPEGImages', help='infer img path')
    parser.add_argument('--detection_config', type=str, default='/mnt/data/guozebin/subject_1/subject_1/work_dirs/yolox_tiny_8x8_300e_coco/yolox_tiny_8x8_300e_coco.py', help='model config path')
    parser.add_argument('--segmentation_config', type=str, default='/mnt/data/guozebin/subject_1/subject_1/configs/_sugject_1/yolact_r50_1x8_coco.py', help='model config path')
    parser.add_argument('--detection_checkpoint', type=str, default='/mnt/data/guozebin/subject_1/subject_1/work_dirs/yolox_tiny_8x8_300e_coco/epoch_30.pth', help='use infer model path')
    parser.add_argument('--segmentation_checkpoint', type=str, default='/mnt/data/guozebin/subject_1/subject_1/work_dirs/yolact_r50_1x8_coco/epoch_55.pth', help='use infer model path')
    parser.add_argument('--save_path', type=str, default='/mnt/data/guozebin/subject_1/subject_1/infer_result', help='infer result save path')
    parser.add_argument('--device', type=str, default='cuda:2', help='device')
    parser.add_argument('--conf', type=float, default=0.3, help='confidence')
    args = parser.parse_args()
    logging.info(args)

    # init Colors
    colors = Colors()

    # instantiate inference class
    inf = Inference(config=[args.detection_config, args.segmentation_config], checkpoint=[args.detection_checkpoint, args.segmentation_checkpoint], save_path=args.save_path,
                    device=args.device, conf=args.conf)


    # example
    path = os.listdir(args.annotation_folder)
    img_paths = []
    for i in path:
        if 'xml' not in i: continue
        img_paths.append(args.img_folder + '/' + i.replace('xml', 'jpg'))

    # loop run
    for img in img_paths:
        whole_img = img
        t1 = time.time()
        inf(input_image=whole_img, plot=True)
        print('inference_time：', time.time() - t1)
