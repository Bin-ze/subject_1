# -*- coding: utf-8 -*-
# __author__:bin_ze
# 10/23/22 9:59 AM

import pyrealsense2 as rs
from subject_inference import *



def detects_video(pipe, hole_filling, colorizer):

    frames = pipe.wait_for_frames()

    color_frame = frames.get_color_frame()
    # if not color_frame:
    #     continue

    im0s = np.asanyarray(color_frame.get_data())#.transpose(2,0,1)
    # colorizer = rs.colorizer()
    align = rs.align(rs.stream.color)
    frames = align.process(frames)
    # 获取对齐更新后的深度图
    aligned_depth_frame = frames.get_depth_frame()
    filled_depth = hole_filling.process(aligned_depth_frame)
    colorized_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())
    # colorized_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())
    # depth = np.asanyarray(aligned_depth_frame.get_data())
    depth = np.asanyarray(filled_depth.get_data())
    return im0s, depth, colorized_depth, frames.timestamp


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        stream=sys.stdout,
                        format="%(asctime)s | %(filename)s:%(lineno)d | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_folder', type=str, default='/mnt/data/guozebin/subject_1/subject_1/data/detection/Annotations', help='infer img path')
    parser.add_argument('--detection_config', type=str, default='/mnt/data/guozebin/subject_1/subject_1/work_dirs/yolox_tiny_8x8_300e_coco/yolox_tiny_8x8_300e_coco.py', help='model config path')
    parser.add_argument('--segmentation_config', type=str, default='/mnt/data/guozebin/subject_1/subject_1/configs/_sugject_1/yolact_r50_1x8_coco.py', help='model config path')
    parser.add_argument('--detection_checkpoint', type=str, default='/mnt/data/guozebin/subject_1/subject_1/work_dirs/yolox_tiny_8x8_300e_coco/epoch_30.pth', help='use infer model path')
    parser.add_argument('--segmentation_checkpoint', type=str, default='/mnt/data/guozebin/subject_1/subject_1/work_dirs/yolact_r50_1x8_coco/epoch_27.pth', help='use infer model path')
    parser.add_argument('--save_path', type=str, default='/mnt/data/guozebin/subject_1/subject_1/demo/infer_result', help='infer result save path')
    parser.add_argument('--device', type=str, default='cuda:2', help='device')
    parser.add_argument('--conf', type=float, default=0.3, help='confidence')
    args = parser.parse_args()
    logging.info(args)

    # instantiate inference class
    inf = Inference(config=[args.detection_config, args.segmentation_config], checkpoint=[args.detection_checkpoint, args.segmentation_checkpoint], save_path=args.save_path,
                    device=args.device, conf=args.conf)


    # init realsense
    pipe = rs.pipeline()
    cfg = rs.config()
    rs.config.enable_device_from_file(cfg, "/home/nuist/2022-1-4/20211230_175522.bag", repeat_playback=False)
    cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, 848, 480, rs.format.rgb8, 30)
    pipe.start(cfg)
    device =pipe.get_active_profile().get_device()
    playback=device.as_playback()
    playback.set_real_time(False)

    colorizer = rs.colorizer()
    hole_filling = rs.hole_filling_filter()

    while True:
            im0s, depth, colorized_depth, timestamp = detects_video(pipe, hole_filling, colorizer)
            inf(input_image=im0s, plot=True)


