# @Author  : zhangyouyuan (zyyzhang@fuzhi.ai)
# @Desc    :
import imageio
import cv2
from utils.service_tools import get_host_ip, HttpClient
from utils.image_io import encode_image_base64, decode_base64_image
import time
from utils.utils import JsonCommonEncoder

def main_func():
    image_path = '/mnt/data/guozebin/subject_1/subject_1/data/detection/JPEGImages/0000_color.jpg'
    save_path = '1.jpg'
    url = f'http://{get_host_ip()}:8000/autotable/predict'
    for i in range(100):
        t1 = time.time()
        image = cv2.imread(image_path)
        image = encode_image_base64(image)
        post_data = {'content': image}
        client = HttpClient(url)
        result = client.post(json=post_data)
        predict_decode = decode_base64_image(result['data'][0]['visual_result'])
        print(time.time() - t1)
        #imageio.imwrite(save_path, predict_decode)

        print('finished ...')


if __name__ == '__main__':
    main_func()
