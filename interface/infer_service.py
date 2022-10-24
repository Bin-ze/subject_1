from typing import Dict, Union
import time
import base64
import os
import logging
from drpc.web import HttpServer

from utils.utils import JsonCommonEncoder
from utils.service_tools import get_host_ip
from utils.image_io import decode_base64_image, encode_image_base64


class InferService(object):

    def __init__(self, infer_interface, service_config=None, client_max_size=1024*1024*100, work_dir='', **kwargs):
        self.work_dir = work_dir
        if not work_dir:
            self.work_dir = os.path.dirname(os.path.abspath(__file__))
        self.infer_interface = infer_interface
        self.client_max_size = client_max_size

        # config
        self.service_config = service_config
        self.service_port = self.service_config["service_port"]
        self.service_route = self.service_config["service_route"]
        self.host_ip = get_host_ip()
        if not service_config:
            self.service_config = {
                "service_route": "autotable/predict",
                "service_port": 8086,
                "app_name": "{}_{}".format("autotable", 8086)
            }
        assert (not self.service_config["service_port"] <= 0 and self.service_config["service_port"] <= 65535)

        logging.info(f'service_ip: {self.host_ip}')
        logging.info(f'service_port: {self.service_port}')
        logging.info(f'service_url: {self.service_route}')

        # encoder
        self.json_encoder = JsonCommonEncoder()

    async def predict(self, content):
        """
        HTTP: /autotable/predict POST
        """
        predicts = []
        if isinstance(content, (list, tuple)):
            for image in content:
                predicts.append(self.infer_single_sample(image))
        else:
            predicts.append(self.infer_single_sample(content))

        return predicts

    def infer_single_sample(self, content):
        image = decode_base64_image(content)
        #visual_result = image
        _, _, visual_result = self.infer_interface(input_image=image, plot=True)
        #infer_data = self.json_encoder.default(infer_data)
        # encode
        visual_result = encode_image_base64(visual_result, to_str=True)
        result = {
            # 'infer_data': infer_data,
            # 'visual_data': visual_data,
            'visual_result': visual_result
        }
        return result

    def run(self):
        httpserver = HttpServer()
        httpserver.register(self)
        httpserver.run(port=self.service_port, app_config={'client_max_size': self.client_max_size})
