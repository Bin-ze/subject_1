# -*- coding: utf-8 -*-
# __author__:bin_ze
# 10/24/22 2:09 PM

from socket import  *
from utils.image_io import encode_image_base64, decode_base64_image

#创建套接字
tcp_server = socket(AF_INET,SOCK_STREAM)
#绑定ip，port
#这里ip默认本机
address = ('192.168.2.43', 8088)
tcp_server.bind(address)
# 启动被动连接
#多少个客户端可以连接
tcp_server.listen(128)
#使用socket创建的套接字默认的属性是主动的
#使用listen将其变为被动的，这样就可以接收别人的链接了

# 创建接收
# 如果有新的客户端来链接服务器，那么就产生一个新的套接字专门为这个客户端服务
client_socket, clientAddr = tcp_server.accept()
# client_socket用来为这个客户端服务，相当于的tcp_server套接字的代理
# tcp_server_socket就可以省下来专门等待其他新客户端的链接
# 这里clientAddr存放的就是连接服务器的客户端地址
#接收对方发送过来的数据
from_client_msg = client_socket.recv(10240000)#接收1024给字节,这里recv接收的不再是元组，区别UDP

image = decode_base64_image(from_client_msg)
print("接收的数据：", image)
#发送数据给客户端
send_data = client_socket.send("2222".encode("utf-8"))
#关闭套接字
#关闭为这个客户端服务的套接字，就意味着为不能再为这个客户端服务了
#如果还需要服务，只能再次重新连
client_socket.close()