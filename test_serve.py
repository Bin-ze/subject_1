# -*- coding: utf-8 -*-
# __author__:bin_ze
# 10/24/22 2:30 PM

from socket import *
import cv2
from utils.image_io import encode_image_base64, decode_base64_image
from utils.utils import JsonCommonEncoder
json = JsonCommonEncoder()
def main():
    # 1.创建套接字
    tcp_socket = socket(AF_INET,SOCK_STREAM)
    # 2.准备连接服务器，建立连接
    serve_ip = "192.168.2.43"
    serve_port = 8088  #端口，比如8000
    tcp_socket.connect((serve_ip, serve_port))  # 连接服务器，建立连接,参数是元组形式

    #准备需要传送的数据
    img = cv2.imread('/mnt/data/guozebin/subject_1/subject_1/data/detection/JPEGImages/0000_color.jpg')
    image = encode_image_base64(img)
    tcp_socket.send(image)
    #从服务器接收数据
    #注意这个1024byte，大小根据需求自己设置
    from_server_msg = tcp_socket.recv(10240000)
    #加上.decode("gbk")可以解决乱码
    print(from_server_msg.decode("utf-8"))
    #关闭连接
    tcp_socket.close()

if __name__ == '__main__':
    main()