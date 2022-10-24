import gzip
import struct
import time
from socket import *
import json
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
def Save(imgBytes):
    imgIO = BytesIO(imgBytes)
    img = Image.open(imgIO)
    img = img.convert('YCbCr') # 转换成YCbCr格式
    img.save('{:.2f}.jpeg'.format(time.time()))
#创建套接字
tcp_server = socket(AF_INET,SOCK_STREAM)
#绑定ip，port
#这里ip默认本机
address = ('192.168.2.43',8085)
tcp_server.bind(address)
#多少个客户端可以连接
tcp_server.listen(128)
#使用listen将其变为被动的，这样就可以接收别人的链接了
while True:
    # 创建接收
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
        #Save(imgBytes) # 保存图像
    #发送数据给客户端
    send_data = client_socket.send("2222".encode("utf-8"))
    #关闭套接字
    #关闭为这个客户端服务的套接字，就意味着为不能再为这个客户端服务了
    #如果还需要服务，只能再次重新连
client_socket.close()