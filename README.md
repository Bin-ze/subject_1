# 环境配置
bash env.sh

# 运行项目代码


> cd demo

> python subject_inference.py

**注释： subject_inference.py里Infer类实现了检测分割的所有所需接口，可以根据需要进行结果返回，打印，画图，保存等操作，在运行时，请载入相应的配置文件地址，以及模型权重地址,调用相应接口展示结果，后续需要根据其他课题组的需要拓展对应的接口。**

# 运行步骤：
## step 1 :启动服务
> python server_socket.py

## step2 :传输数据 :这部分代码需要放在板卡上，注意请修改服务代码以及这个客户端代码的ip以及端口号，保证可以传输数据。
> python cliet_socket.py

# 权重下载
## 检测：
> wget https://extremevision-js-userfile.oss-cn-hangzhou.aliyuncs.com/user-28826-files/134e8b9d-bc53-44df-bb0f-2c659dadea71/det.pth
## 分割：
> wget https://extremevision-js-userfile.oss-cn-hangzhou.aliyuncs.com/user-28826-files/55e7f905-a0b4-4678-9524-f819e333660b/seg.pth

# 数据集下载
> https://extremevision-js-userfile.oss-cn-hangzhou.aliyuncs.com/user-28826-files/26498ebe-895d-453f-9e68-4729b8a1721b/subject_data.tar.gz

**注释：之前的数据集有些问题，已经处理妥当，需要验证结果请下载数据集，并在图片上进行推理.**