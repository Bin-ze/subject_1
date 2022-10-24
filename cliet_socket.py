import base64
import gzip
import struct
import time
from io import BytesIO
import os
from socket import *
from PIL import Image

annotation_folder = '/mnt/data/guozebin/subject_1/subject_1/data/detection/Annotations'
img_folder = '/mnt/data/guozebin/subject_1/subject_1/data/detection/JPEGImages'
img_paths = []
path = os.listdir(annotation_folder)
for i in path:
    if 'xml' not in i: continue
    img_paths.append(img_folder + '/' + i.replace('xml', 'jpg'))

for img in img_paths[:100]:
    t1 = time.time()
    # 1.创建套接字
    tcp_socket = socket(AF_INET,SOCK_STREAM)
    # 2.准备连接服务器，建立连接
    serve_ip = "192.168.2.43"
    serve_port = 8085  #端口，比如8000
    tcp_socket.connect((serve_ip, serve_port))  # 连接服务器，建立连接,参数是元组形式

    #准备需要传送的数据
    img = Image.open(img)
    imgIO = BytesIO()  # 创建文件对象，类型：io.BytesIO
    img.save(imgIO, 'JPEG')  # 以JPEG格式存储，减少数据大小

    imgIOZ = BytesIO()  # 创建文件对象，类型：io.BytesIO
    imgIOZ.write(gzip.compress(imgIO.getvalue()))  # 压缩原图并存入文件对象

    imgBytes = imgIOZ.getvalue()  # 图像的字节流，类型：bytes
    print(len(imgBytes))  # 显示字节流长度

    imgSize = len(imgBytes)  # 图像大小（字节流长度），类型：int
    head = struct.pack('l', imgSize)  # 构造文件头信息，内容是图像大小（字节流长度），类型：bytes
    tcp_socket.send(head)  # 发送文件头

    imgIOZ.seek(0, 0)  # 从开头读取图片
    while True:
        imgBuf = imgIOZ.read(10240)  # self.bufSize大小的图片，类型：bytes
        if not imgBuf:  # 传输完成退出循环
            break
        tcp_socket.send(imgBuf)  # 发送self.bufSize大小的图片

    #从服务器接收数据
    #注意这个1024byte，大小根据需求自己设置
    # from_server_msg = tcp_socket.recv(10240)
    # #加上.decode("gbk")可以解决乱码
    # print(from_server_msg.decode("utf-8"))
    # #关闭连接
    # #tcp_socket.close()
    # print(time.time()-t1)
    # tcp_socket.close()