#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : 推理阶段启动入口


import sys
import logging
import argparse

from interface import Inference
from interface.infer_service import InferService


def start_service(detection_config, segmentation_config, detection_checkpoint, segmentation_checkpoint, save_path, device, service_port, conf,
                  service_config=None):
    if not service_config:
        service_config = {
            "service_route": "autotable/predict",
            "service_port": service_port,
            "app_name": "{}_{}".format("autotable", service_port)
        }

    infer_interface = Inference(config=[detection_config, segmentation_config], checkpoint=[detection_checkpoint, segmentation_checkpoint], save_path=args.save_path,
                    device=device, conf=conf)
    infer_service = InferService(infer_interface=infer_interface, service_config=service_config, work_dir=save_path)
    infer_service.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        stream=sys.stdout,
                        format="%(asctime)s | %(filename)s:%(lineno)d | %(levelname)s | %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument('--detection_config', type=str,
                        default='/mnt/data/guozebin/subject_1/subject_1/work_dirs/yolox_tiny_8x8_300e_coco/yolox_tiny_8x8_300e_coco.py',
                        help='model config path')
    parser.add_argument('--segmentation_config', type=str,
                        default='/mnt/data/guozebin/subject_1/subject_1/configs/_sugject_1/yolact_r50_1x8_coco.py',
                        help='model config path')
    parser.add_argument('--detection_checkpoint', type=str,
                        default='/mnt/data/guozebin/subject_1/subject_1/work_dirs/yolox_tiny_8x8_300e_coco/best_bbox_mAP_epoch_49.pth',
                        help='use infer model path')
    parser.add_argument('--segmentation_checkpoint', type=str,
                        default='/mnt/data/guozebin/subject_1/subject_1/work_dirs/yolact_r50_1x8_coco/epoch_55.pth',
                        help='use infer model path')
    parser.add_argument('--save_path', type=str, default='/mnt/data/guozebin/subject_1/subject_1/infer_result',
                        help='infer result save path')
    parser.add_argument('--conf', type=float, default=0.3, help='confidence')
    parser.add_argument("--device", type=str, default='cuda:1')
    parser.add_argument("--service_port", type=int, default=8000)

    args = parser.parse_args()
    logging.info("args: {}".format(args))

    start_service(args.detection_config, args.segmentation_config, args.detection_checkpoint, args.segmentation_checkpoint, args.save_path,
                  args.device, args.service_port, args.conf,
                  service_config=None)
