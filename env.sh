conda create --name openmmlab python=3.8 -y
conda activate openmmlab
conda install pytorch torchvision -c pytorch
pip install -U openmim
mim install mmcv-full
pip install -v -e .
#pip install pyrealsense2
pip install drpc-0.1.6-py3-none-any.whl