# -*- coding: utf-8 -*-
# __author__:bin_ze
# 10/22/22 6:33 PM
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import json
import logging
import argparse
import torch
import collections
from mmcv.runner import wrap_fp16_model

# subject 3 import
from subject_3.predicts.constant import action_list
from subject_3.PathPlanning.Search_based_Planning.Search_2D.ARAstar import *
from  subject_3.demo import ActionPredict

# subject 2 import
import time
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

        self.action_list = {'上楼梯': 'UpStairs', '下楼梯': 'DownStairs', '绕行': 'Detour', '直行': 'GoStraight',
                            '左转': 'TurnLeft', '右转': 'TurnRight','侧移': 'SideWay', '转身': 'TurnBack', '坐下': 'SitDown',
                            '站起': 'StandUp', '跨越': 'StepOver', '上坡': 'UpHill', '下坡': 'DownHill'}

        self.obstacle = ('changdeng', 'zhuozi', 'shikuai', 'zhangaiwu', 'menkan', 'yizi', 'zhuankuai')
        
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
        fp16 =False
        if fp16:
            wrap_fp16_model(self.detection_model)
            wrap_fp16_model(self.segmentation_model)
        self.window = 'seg_det_res'
        self.conf = conf
        self.prefix = ['jpg', 'jpeg', 'png']  # input prefix

        # subject 2
        path_att_all = '../subject_2/Knowledge/Attribute_triplet_dict_new_t8.json'
        path_Text_triplet = '../subject_2/Knowledge/Text_visualization_triplet3.json'
        path_Scene_triplet = '../subject_2/Knowledge/Scene_triplet-3.json'
        self.data_att = self.sub2_load_json(path_att_all)
        self.data_triplet = self.sub2_load_json(path_Text_triplet)
        self.data_scene = self.sub2_load_json(path_Scene_triplet)
        self.detection_names_subject2 = ('地标', '长凳', '桌子', '小石头', '柱子', '木门槛', '椅子', '砖头')
        self.segmentation_names_subject2 = ('草地', '水泥地', '木板地', '楼梯', '斜坡', '沙地', '鹅卵石地面')

        # subject_3 model
        self.subject_3_model = ActionPredict().to(device)
        self.device = device
        #self.need_list = []

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

    def sub2_load_json(self, path_json):
        f_json = open(path_json)
        data_json = json.load(f_json)
        return data_json

    def sub2_attribute_retrieve(self, object_look):
        # 属性
        if object_look == '鹅卵石地面1':
            object_look2 = '鹅卵石地面'
        if object_look == '椅子1':
            object_look2 = '椅子'
        else:
            object_look2 = object_look
        # 属性
        if object_look2 in self.data_att.keys():
            attribute_value = self.data_att[object_look2]
        else:
            attribute_value = []

        # # 常识三元组
        if object_look2 in self.data_triplet.keys():
            triplet_value = self.data_triplet[object_look2]
        else:
            triplet_value = []
        # 场景三元组
        if object_look in self.data_scene.keys():
            scene_value = self.data_scene[object_look]
        else:
            scene_value = []
        return attribute_value, triplet_value, scene_value

    def sub2_callback5(self, data):

        if data and data != []:
            data_object = data
            class_name_list = data_object['类别']
            class_name_list2 = {}
            index = 0
            for str_obj in class_name_list:
                if str_obj not in class_name_list2.keys():  # 移除重复的
                    # 物体在原始列表中的index
                    class_name_list2[str_obj] = index
                index = index + 1
            class_name_list3 = []  #
            for str_name in class_name_list2.keys():
                class_name_list3.extend([str_name])  # 非重复列表
            object_all = []
            for str_obj in class_name_list3:
                attribute_value, triplet_value, scene_value = self.sub2_attribute_retrieve(str_obj)  # 返回列表

                attribute_value1 = []

                # 合并三种知识
                hello_str = []
                hello_str.extend(attribute_value)
                # attribute_附加
                if attribute_value1 != []:
                    hello_str.extend(attribute_value1)

                hello_str.extend(triplet_value)
                if str_obj != '水泥地':
                    hello_str.extend(scene_value)
                object_all.extend(hello_str)
            return object_all

    def sub2_listener(self, data_dict):  # 收到数据

        # 根据课题一，获取到对应的三种知识
        object_all = self.sub2_callback5(data_dict)
        # 可视化
        object_all = str(object_all)
        js_write = '<html> <head> <meta charset="UTF-8"><meta http-equiv="refresh" content="1"><script type="text/javascript" src="http://code.jquery.com/jquery-2.1.1.min.js"></script> <script type="text/javascript" src="https://cdn.bootcdn.net/ajax/libs/echarts/5.1.2/echarts.min.js"></script><script type="text/javascript" src="./roslibjs/build/roslib.min.js"></script> <script type="text/javascript" src="jquery-1.9.1.min.js"></script> <script src="https://apps.bdimg.com/libs/jquery/2.1.4/jquery.min.js"></script><style>.main { width: 100%; height: 100%; margin: 0 auto}</style></head><body><div id="main" class="main"></div><script type="text/javascript">function readTextFile(file){ var rawFile = new XMLHttpRequest(); rawFile.open("GET", file, false); rawFile.onreadystatechange = function () { if(rawFile.readyState === 4) { if(rawFile.status === 200 || rawFile.status == 0) {       var allText = rawFile. responseText;  alert(allText);}} }; rawFile.send(null);  }  var data =' + object_all + ';var obj_head_list_0 = [\'路边椅子\', \'花池\', \'楼梯\', \'斜坡\', \'小石头\', \'草地\', \'沙地\', \'水泥地\', \'木门槛\', \'鹅卵石地面\', \'砖头\', \'柱子\']; var attr_tail_list_1 = [\'场景\', \'障碍物\', \'行走\', \'跨越\', \'绕行\', \'踩踏\', \'台阶\', \'15cm\', \'坚实\', \'宽阔\', \'双脚站立时平衡\', \'坐下\', \'四面均可\', \'扶手\', \'靠背\', \'双脚触地\', \'移动\', \'一般\', \'地形\', \'较坚实\', \'较狭窄\', \'29.75cm\', \'平整\', \'较滑\', \'<=3\', \'小障碍\', \'26cm\', \'地面\', \'24.5cm\', \'凹凸\', \'易凹陷\', \'35cm\', \'抗滑\', \'0.8cm\', \'20.5cm\', \'狭窄\', \'大障碍\', \'向上\', \'向下\', \'右方\', \'左方\'];  var common_tail_list_2 = [\'门槛\', \'矩形\', \'材质\', \'障碍\', \'阻隔\', \'建筑\', \'长方体\', \'装饰\', \'承重\', \'椅子\', \'公共设施\', \'长椅\', \'坐下\', \'休息\', \'美观\', \'坚硬\', \'路边\', \'马路\', \'区域\', \'构造物\', \'疏导\', \'美化\', \'遮挡\', \'可绕过\', \'公园\', \'草地\', \'户外\', \'楼梯\', \'地貌\', \'地形\', \'步行\', \'上坡\', \'下坡\', \'单步上坡\', \'单步下坡\', \'倾斜\', \'可上坡\', \'可下坡\', \'岩石\', \'凝结物\', \'建造\', \'铺垫\', \'水泥地\', \'地面\', \'植被\', \'路面变化\', \'柔软\', \'潮湿\', \'不平整\', \'绿色\', \'吸水\', \'可行走\', \'草坪\', \'干燥\', \'黄色\', \'流动\', \'粗糙\', \'沙土地\', \'鹅卵石地\', \'单步行走\', \'灰色\', \'水泥地面\', \'转弯\', \'动作\', \'向左\', \'站立\', \'稳定\', \'不稳定\', \'向左转\', \'向右\', \'向右转\', \'变速（加减速）\', \'快速\', \'慢速\', \'上楼梯\', \'升高\', \'在楼上\', \'抬腿\', \'在楼下\', \'下楼梯\', \'降低\', \'在坡上\', \'斜坡\', \'在坡下\', \'跨越\', \'小石头\', \'坐着\', \'行走\', \'站起\', \'方向变换\', \'转向\', \'静止\', \'停下\', \'单步跨越\', \'单步上楼梯\', \'单步下楼梯\', \'坐具\', \'坐\', \'落座\', \'上楼\', \'下楼\', \'交通\', \'连接\', \'可上楼\', \'可下楼\', \'指挥\', \'摔倒\', \'走路\', \'室内\', \'石头墩子\']; var categories = []; for (var i = 0; i < 3; i++) { categories[i] = {   name: \'类目\' + i }; var categories = []; for (var i = 0; i < 3; i++) { categories[i] = { name: \'类目\' + i };} var nodeAndLinks = genNodeData(data); renderChart(nodeAndLinks[0], nodeAndLinks[1]);} function genNodeData(input_message) { info_string1 = ""; info_string2 = ""; var messages = input_message;	 var a = ""; var key_ind = []; var name_ind = []; var category_ind = [];  var head_ind = []; var tail_ind = []; var relation_ind = []; let nodes = []; let links = []; for (k = 0; k < messages.length; k++) { message = messages[k]; head_ind[head_ind.length] = message.head; tail_ind[tail_ind.length] = message.tail; relation_ind[relation_ind.length] = message.relation; for (i = 0; i < 2; i++) { if (name_ind.findIndex(value => value === Object.values(message)[i]) === -1) { key_ind[key_ind.length] = Math.random().toString(16);  name_ind[name_ind.length] = Object.values(message)[i]; if (!(obj_head_list_0.findIndex(value => value === Object.values(message)[i]) === -1)) { category_ind[category_ind.length] = 0 } else { if (!(common_tail_list_2.findIndex(value => value === Object.values(message)[i]) === -1)) { category_ind[category_ind.length] = 2 } else {category_ind[category_ind.length] = 1}  } } } }  for (let i = 0; i < key_ind.length; i++) { let color = getColorByCategory(category_ind[i]); nodes.push({  id: key_ind[i], name: name_ind[i], symbolSize: category_ind[i] === 0 ? 90 : 50,  category: category_ind[i],   itemStyle: { color: color  } }) } for (let i = 0; i < head_ind.length; i++) { let color = getColorByNodeId(nodes, key_ind[name_ind.findIndex(value => value === tail_ind[i])]); links.push({ source: key_ind[name_ind.findIndex(value => value === head_ind[i])], target: key_ind[name_ind.findIndex(value => value === tail_ind[i])],  name: relation_ind[i], lineStyle: { color: color } }) } return [nodes, links]} function getColorByCategory(category) {  if (category === 0) { return \'#5470C6\' } else if (category === 1) { return \'#FAC858\' } else { return \'#9FE080\'}} function getColorByNodeId(nodes, nodeId) { return nodes.filter(r => r.id === nodeId)[0].itemStyle.color } function renderChart(nodeData, linkData) { var chartDom = document.getElementById(\'main\'); var myChart = echarts.init(chartDom); var option; option = { title: { text: \'\'}, tooltip: { formatter: function (x) { return x.data.label; } }, toolbox: { show: true, feature: { mark: { show: true}, restore: { show: true }, saveAsImage: { show: true}} }, legend: [{ data: categories.map(function (a) { return a.name;  }) }], force:{   repulsion: 2500,  friction: 0.3, edgeLength: [10,50] }, series: [{  name: \'echart\', zoom: 0.3, scaleLimit: { min: 0.1, max: 3, }, type: \'graph\',  layout: \'force\',  symbolSize: 40,  roam: true,   edgeSymbol: [\'circle\', \'arrow\'],  edgeSymbolSize: [2, 15],    animationDuration: 0.001,   animationEasingUpdate: "quinticInOut",     darkMode: \'auto\', animation: "auto",  animationDurationUpdate: 50, animationThreshold: 2000, progressiveThreshold: 3000,  progressive: 400,   hoverLayerThreshold: 3000, stateAnimation: { duration: 300, easing: \'cubinOut\' }, edgeLabel: {  normal: { textStyle: { fontSize: 20} } }, force: {repulsion: 2500, edgeLength: [10, 50] }, draggable: true,  lineStyle: { normal: { width: 5,  color: \'source\', curveness: 0.3} }, emphasis: { focus: \'adjacency\',lineStyle: { width: 10 } }, edgeLabel: { normal: { show: true, formatter: function (x) {return x.data.name; }}  }, label: { normal: { show: true, textStyle: {} }  }, data: nodeData,  links: linkData, categories: [\'0\', \'1\', \'2\'] }] }; option && myChart.setOption(option); }</script></body></html>'
        f2_text = open('sub2_json_1025.html', 'w', encoding="utf8")
        f2_text.write(js_write)
        f2_text.close()
        return object_all

    def formot_subject_3(self, det_bboxes, det_labels):
         det_dict = {}
         subject_2_dict = {}
         if len(det_bboxes):
             for bbox, label in zip(det_bboxes, det_labels):
                 det_dict[self.detection_names[label]] = [[int(n) for n in bbox[:4].tolist()]]
                 subject_2_dict[self.detection_names_subject2[label]] = [[int(n) for n in bbox[:4].tolist()]]

         return det_dict, subject_2_dict

    def formot_seg(self, seg_bboxes, seg_labels):
        data_dict = {}
        if len(seg_labels):
            for bbox, label in zip(seg_bboxes, seg_labels):
                data_dict[self.segmentation_names_subject2[label]] = [[int(n) for n in bbox[:4].tolist()]]

        return data_dict

    def formot_subject_2(self, det_dict, seg_dict):
        data_dict = collections.defaultdict(list)
        det_dict.update(seg_dict)
        for k, v in det_dict.items():
            data_dict['类别'].append(k)
            data_dict['框坐标'].append(v[0])

        return dict(data_dict)

    def subject_3_infer(self, det_dict):

        im = self.im.copy()
        im = im.transpose(2, 0, 1)

        im = torch.tensor(im).unsqueeze(0).to(self.device).float()
        im /= 255.0

        y = self.subject_3_model(im)
        y[0][0][8] -= 10
        y[0][0][9] -= 10
        y[0][0][3] *= 6
        y = torch.softmax(y, dim=2)
        y = torch.multinomial(y.squeeze(), 1)
        direction = action_list[y.item()]
        # model output
        ##add logic judge #need task 2 result
        # last knowledge is three tuple [{'relation':,'tail':,}]
        Env = env.Env()
        ## map,set for obstacle #need task1 result
        ### det_dict {classname:bboxes[[x,x,y,y]...],...}
        ### self.obstacle = set(name1,name2...)
        #############
        obs = set()
        for k in det_dict:
            if k in self.obstacle:
                for bbox in det_dict[k]:
                    xmin, ymin, xmax, ymax = bbox
                    for i in range(xmin // 10, xmax // 10):
                        for j in range(ymin // 10, ymax // 10):
                            if (i, j) != (32, 5):
                                obs.add((i, j))
        ####
        ##########
        # debug data map start goal
        s_start = (32, 5)
        s_goal = None
        if direction in ['上楼梯', '下楼梯'] and '楼梯' in det_dict:
            x_mean = []
            y_max = 0
            for bbox in det_dict['楼梯']:
                xmin, ymin, xmax, ymax = bbox
                x_mean.append((xmin + xmax) // 2)
                y_max = max(ymax, y_max)
            for i in range(x_mean // 10 - 5, x_mean // 10 + 5):
                for j in range(y_max // 10 - 5, y_max // 10):
                    if (i, j) not in obs:
                        s_goal = (i, j)
                        break
                if s_goal is not None:
                    break

        if direction == '左转':
            for i in range(11, 21):
                for j in range(38, 48):
                    if (i, j) not in obs:
                        s_goal = (i, j)
                        break
                if s_goal is not None:
                    break

        elif direction == '右转':
            for i in range(43, 53):
                for j in range(38, 48):
                    if (i, j) not in obs:
                        s_goal = (i, j)
                        break
                if s_goal is not None:
                    break
        else:
            for i in range(27, 37):
                for j in range(38, 48):
                    if (i, j) not in obs:
                        s_goal = (i, j)
                        break
                if s_goal is not None:
                    break

        if s_goal is None:
            s_goal = (32, 43)
        ######
        Env.obs = obs
        arastar = AraStar(s_start, s_goal, 2.5, "euclidean")
        arastar.Env = Env
        arastar.obs = Env.obs

        plot = plotting.Plotting(s_start, s_goal)
        plot.env = Env
        plot.obs = Env.obs

        try:
            direction = self.action_list[direction]
            path, visited = arastar.searching()
            plot.animation_ara_star(path, visited, "action predict{}".format(direction), './output.jpg')
        except:
            direction = self.action_list['绕行']
            plot.animation_ara_star(None, None, "action predict{}".format(direction), './output.jpg', True)

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

        # format
        det_dict, subject_2_dict = self.formot_subject_3(det_bboxes, det_labels)
        seg_dict = self.formot_seg(seg_bboxes, seg_labels)

        # project 2 run
        data_dict = self.formot_subject_2(subject_2_dict, seg_dict)
        #object_all = self.sub2_listener(data_dict)

        #self.need_list.append(det_dict)
        # project 3 run
        #self.subject_3_infer(det_dict)

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
    parser.add_argument('--detection_checkpoint', type=str, default='/mnt/data/guozebin/subject_1/subject_1/work_dirs/yolox_tiny_8x8_300e_coco/best_bbox_mAP_epoch_49.pth', help='use infer model path')
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
    t = time.time()
    for img in img_paths[:100]:
        whole_img = img
        t1 = time.time()
        inf(input_image=whole_img, plot=False)
        print('inference_time：', time.time() - t1)
    print('total', time.time() - t)