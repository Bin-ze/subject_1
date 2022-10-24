# @Author  : zhangyouyuan (zyyzhang@fuzhi.ai)
# @Desc    :
import yaml
import os
import logging
import json
import datetime
import numpy as np
import random
import torch
import torch.backends.cudnn as cudnn
import xmltodict
import xml.etree.ElementTree as ET
from collections import OrderedDict
from torch.utils import model_zoo
from urllib.parse import urlparse
import os
import random
import sys
import shutil
random.seed(2)
__all__ = [
    'format_to_color',
    'check_file_exists',
    'create_not_exist_path',
    'search_file',
    'read_xml',
    'write_yaml_file',
    'write_json_file',
    'write_txt_file',
    'JsonCommonEncoder',
    'json_encode',
    'json_decode',
    'load_pretrain_from_url_or_local',
    'mv_or_add_chr_param',
    'set_seed',
    'AverageMeter',
    'get_all_category_from_voc'
]


# image
def format_to_color(image):
    """
    将灰度图、4通道alpha图转为三通道彩色图
    :param image:
    :return:
    """
    image_shape = image.shape

    if len(image_shape) == 2:
        image = np.expand_dims(image, axis=2)
        image = np.tile(image, reps=(1, 1, 3))
    elif len(image_shape) == 3:
        if image_shape[-1] < 3:
            image = np.stack([image[:, :, 0], image[:, :, 0], image[:, :, 0]], axis=2)
        elif image_shape[-1] > 3:
            image = image[:, :, :3]
    return image


# file
def check_file_exists(file_abspath):
    if not os.path.exists(file_abspath):
        raise FileNotFoundError("file: {} not exist".format(file_abspath))


def create_not_exist_path(file_path):
    """
    输入可为path或者文件
    因此，如果是path，如果文件夹名包含点，则必须跟上斜杠/
    :param file_path:
    :return:
    """
    if not os.path.exists(file_path):
        file_name = os.path.basename(file_path)
        if len(file_name.split('.')) > 1 and not file_path.endswith('/'):  # if file or dir ?
            file_path = os.path.abspath(os.path.dirname(file_path))
        logging.info("path: {} not exist, to create it".format(file_path))
        os.makedirs(file_path, exist_ok=True)


def search_file(search_path, suffix=None, quiet=False):
    """
    搜索指定目录下的符合指定后缀的文件
    :param quiet: bool, default False
               是否只执行，不打印输出
    :param search_path: str
               搜索的文件夹路径
    :param suffix: str or list, default ''
               指定后缀, 可以是单个后缀，也可以是多个后缀
    :return: list, 所有搜索到的文件的路径组成的list
    """

    # check suffix
    suffix = suffix if suffix else ''
    suffix = suffix if isinstance(suffix, list) else [suffix]

    if not quiet:
        print('search path -> {}, suffix -> "{}"'.format(search_path, ' '.join(suffix)))

    all_file_path = []
    for root_dir, folder_name, file_names in os.walk(search_path):
        root_dir = root_dir.replace('\\', '/')
        if not quiet:
            print(root_dir)
        for file_name in file_names:
            file_suffix = os.path.splitext(file_name)[-1]
            if suffix[0] and (file_suffix not in suffix):
                continue
            file_name = file_name.replace('\\', '/')
            file_path = root_dir + '/' + file_name
            all_file_path.append(file_path)

    if not quiet:
        print('searched file num -> {}'.format(len(all_file_path)))
    return all_file_path


def read_xml(xml_path):
    with open(xml_path, mode='r', encoding='utf-8') as file_r:
        xml_dict = xmltodict.parse(file_r.read())
    return xml_dict


def write_json_file(json_dict, json_save_path, ensure_ascii=False, indent=4):
    if not json_save_path.endswith('.json'):
        raise ValueError('json save path need suffix .json ...')

    create_not_exist_path(json_save_path)
    with open(json_save_path, mode='w', encoding='utf-8') as file_w:
        file_w.write(json.dumps(json_dict, cls=JsonCommonEncoder,
                                ensure_ascii=ensure_ascii, indent=indent))

    print(f'save {json_save_path} success ...')


def write_txt_file(json_dict, txt_save_path, ensure_ascii=False):
    if not txt_save_path.endswith('.txt'):
        raise ValueError('json save path need suffix .txt ...')

    create_not_exist_path(txt_save_path)
    with open(txt_save_path, mode='w', encoding='utf-8') as file_w:
        file_w.write(json.dumps(json_dict, cls=JsonCommonEncoder,
                                ensure_ascii=ensure_ascii, indent=4))

    print(f'save {txt_save_path} success ...')


def write_yaml_file(yaml_dict, yaml_save_path, ensure_ascii=False):
    if not yaml_save_path.endswith('.yaml'):
        raise ValueError('json save path need suffix .yaml ...')

    create_not_exist_path(yaml_save_path)
    with open(yaml_save_path, mode='w', encoding='utf-8') as file_w:
        file_w.write(yaml.safe_dump(yaml_dict, indent=4))

    print(f'save {yaml_save_path} success ...')


# json
class JsonCommonEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            return obj.strftime("%Y-%m-%d %H:%M:%S")
        elif isinstance(obj, datetime.date):
            return obj.strftime("%Y-%m-%d")
        elif isinstance(obj, datetime.time):
            return obj.strftime("%H:%M:%S")
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return round(float(obj), 4)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (tuple, list)):
            new_obj = []
            for i in obj:
                new_obj.append(self.default(i))
            return new_obj
        elif isinstance(obj, bytes):
            return str(obj, encoding='utf-8')
        elif isinstance(obj, str):
            return obj
        else:
            return json.JSONEncoder.default(self, obj)


def json_encode(data, default="{}"):
    try:
        json_str = json.dumps(data, cls=JsonCommonEncoder, ensure_ascii=False,
                              indent=4)
    except Exception as exp:
        json_str = default
    return json_str


def json_decode(data, default={}):
    try:
        try:
            obj = json.loads(data)
        except Exception as exp:
            obj = eval(data)
    except Exception as exp:
        obj = default
    return obj


# model
def load_pretrain_from_url_or_local(url, model_dir):
    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    filepath = os.path.join(model_dir, filename)
    if os.path.exists(filepath):
        sd = torch.load(filepath, map_location='cpu')
    else:
        sd = model_zoo.load_url(url=url, model_dir=model_dir, map_location='cpu')
    return sd


def mv_or_add_chr_param(params, param_correct_type='keep', key_word='model.'):
    new_state_dict = OrderedDict()
    for key, value in params.items():
        if param_correct_type == 'keep':
            new_param_name = key
        elif param_correct_type == 'add':
            new_param_name = key_word + key
        elif param_correct_type == 'sub':
            new_param_name = key.replace(key_word, '')
        else:
            new_param_name = key
        new_state_dict[new_param_name] = value

    return new_state_dict


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    cudnn.benchmark = False
    cudnn.deterministic = True


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# dataset
def get_all_category_from_voc(path):
    all_xml_path = search_file(path, suffix='.xml')
    all_category = set()
    for xml_path in all_xml_path:
        xml_dict = read_xml(xml_path)
        objects = xml_dict['annotation'].get('object', [])
        if not isinstance(objects, (list, tuple)):
            objects = [objects]
        if not objects:
            continue
        category = [obj['name'] for obj in objects]
        all_category.update(category)

    return all_category
def _find_class(dataset_path):
    xml_path=dataset_path+'/train/annotations'
    CLASS=[]
    for xml in os.listdir(xml_path):
        xml_file =os.path.join(xml_path,xml)
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for obj in root.findall('object'):
            name =obj.find('name').text
            if name in CLASS:continue
            else:
                CLASS.append(name)
    num_classes=len(CLASS)
    CLASS=tuple(CLASS)
    return CLASS,num_classes
def mkdir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    if not os.path.exists(path):
        os.makedirs(path)
def datset_trans(dataset_path):
    root_path = dataset_path
    traintxtsavepath = root_path + '/train/ImageSets/Main'
    valtxtsavepath = root_path + '/valid/ImageSets/Main'
    testtxtsavepath = root_path + '/test/ImageSets/Main'
    train_xmlpath = root_path + '/train/annotations'
    vaild_xmlpath = root_path + '/valid/annotations'
    test_xmlpath = root_path + '/test/annotations'
    mkdir(traintxtsavepath)
    mkdir(valtxtsavepath)
    mkdir(testtxtsavepath)
    #train vaild test set
    train=os.listdir(train_xmlpath)
    vaild=os.listdir(vaild_xmlpath)
    test=os.listdir(test_xmlpath)

    #dataset info
    total=len(train)+len(vaild)+len(test)
    print("total size:", total)
    print("train size:", len(train))
    print("val size:", len(vaild))
    print("test size:", len(test))

    ftest = open(testtxtsavepath + '/test.txt', 'w')
    ftrain = open(traintxtsavepath + '/train.txt', 'w')
    fval = open(valtxtsavepath + '/val.txt', 'w')

    for tr in train:
        if tr[-4:]!='.xml':continue
        name=tr[:-4]+'\n'
        ftrain.write(name)
    for val in vaild:
        if val[-4:]!='.xml':continue
        name=val[:-4]+'\n'
        fval.write(name)
    for ts in test:
        if ts[-4:]!='.xml':continue
        name=ts[:-4]+'\n'
        ftest.write(name)

    ftrain.close()
    fval.close()
    ftest.close()
