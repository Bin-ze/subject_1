# @Author  : zhangyouyuan (zyyzhang@fuzhi.ai)
# @Desc    :
import imageio
import os
import cv2
from utils.service_tools import get_host_ip, HttpClient
from utils.image_io import encode_image_base64, decode_base64_image
import time
from utils.utils import JsonCommonEncoder
annotation_folder = '/mnt/data/guozebin/subject_1/subject_1/data/detection/Annotations'
img_folder = '/mnt/data/guozebin/subject_1/subject_1/data/detection/JPEGImages'

def main_func():
    url = f'http://{get_host_ip()}:8000/autotable/predict'
    path = os.listdir(annotation_folder)
    img_paths = []
    for i in path:
        if 'xml' not in i: continue
        img_paths.append(img_folder + '/' + i.replace('xml', 'jpg'))

    for img in img_paths[:100]:
        save_path = img.split('/')[-1]
        t1 = time.time()
        image = cv2.imread(img)
        image = encode_image_base64(image)
        post_data = {'content': image}
        client = HttpClient(url)
        result = client.post(json=post_data)
        predict_decode = decode_base64_image(result['data'][0]['visual_result'])
        print(time.time() - t1)
        cv2.imwrite('/mnt/data/guozebin/subject_1/subject_1/infer_result/' + save_path, predict_decode)

    print('finished ...')


if __name__ == '__main__':
    main_func()
