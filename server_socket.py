import gzip
import struct
import time
from socket import *
from demo.subject_inference import Inference
import logging
import sys
import argparse
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
def encode(imgBytes):
    imgIO = BytesIO(imgBytes)
    img = Image.open(imgIO)
    img = np.array(img)

    return  cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# init model
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        stream=sys.stdout,
                        format="%(asctime)s | %(filename)s:%(lineno)d | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument('--detection_config', type=str, default='work_dirs/yolox_tiny_8x8_300e_coco/yolox_tiny_8x8_300e_coco.py', help='model config path')
    parser.add_argument('--segmentation_config', type=str, default='configs/_sugject_1/yolact_r50_1x8_coco.py', help='model config path')
    parser.add_argument('--detection_checkpoint', type=str, default='work_dirs/yolox_tiny_8x8_300e_coco/best_bbox_mAP_epoch_49.pth', help='use infer model path')
    parser.add_argument('--segmentation_checkpoint', type=str, default='work_dirs/yolact_r50_1x8_coco/epoch_55.pth', help='use infer model path')
    parser.add_argument('--save_path', type=str, default='./infer_result', help='infer result save path')
    parser.add_argument('--device', type=str, default='cuda:2', help='device')
    parser.add_argument('--conf', type=float, default=0.3, help='confidence')
    args = parser.parse_args()
    logging.info(args)


    # instantiate inference class
    inf = Inference(config=[args.detection_config, args.segmentation_config], checkpoint=[args.detection_checkpoint, args.segmentation_checkpoint], save_path=args.save_path,
                    device=args.device, conf=args.conf)
    #创建套接字
    tcp_server = socket(AF_INET,SOCK_STREAM)
    #绑定ip，port
    #这里ip默认本机
    address = ('192.168.2.43', 8080)
    tcp_server.bind(address)
    #多少个客户端可以连接
    tcp_server.listen(128)
    print('start server')
    #使用listen将其变为被动的，这样就可以接收别人的链接了
    a = 0
    while True:
        # 创建接收
        t1 = time.time()
        a += 1
        # 如果有新的客户端来链接服务器，那么就产生一个新的套接字专门为这个客户端服务
        client_socket, clientAddr = tcp_server.accept()
        #接收对方发送过来的数据
        headSize = struct.calcsize('l') # 计算文件头长度
        head =  client_socket.recv(headSize) # 接收文件头
        if head:
            imgSize = struct.unpack('l', head)[0] # 获取图像长度，类型：int
            recvSize = 0 # 接收到的数据长度
            imgBytesZ = b'' # 接收到的数据
            while True:
                if imgSize - recvSize > 0:
                    imgBuf = client_socket.recv(10240)
                    recvSize += len(imgBuf)
                else:
                    break
                imgBytesZ += imgBuf
            imgBytes = gzip.decompress(imgBytesZ) # 解压数据
            img = encode(imgBytes)
            inf(input_image=img, plot=True)
            print('infer time:', time.time() - t1)
            print(a)
        #发送数据给客户端
        #send_data = client_socket.send("2222".encode("utf-8"))
        #关闭套接字
        #关闭为这个客户端服务的套接字，就意味着为不能再为这个客户端服务了
        #如果还需要服务，只能再次重新连
    client_socket.close()
