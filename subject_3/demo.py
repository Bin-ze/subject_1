from subject_3.predicts.models import RULSTM
from torchvision.models.resnet import resnet50
import torch.nn as nn
import torch
from subject_3.predicts.constant import action_list
from subject_3.PathPlanning.Search_based_Planning.Search_2D.ARAstar import *
import time


def backbone_init():
    res50 = resnet50(pretrained=True)
    ms = list(res50.children())[:-1]
    backbone = nn.Sequential(*ms)
    backbone.eval()
    return backbone

def model_init():
    num_class = 13
    model = RULSTM(num_class, 2048, 1024,
                   0.2, sequence_completion=True)
    checkpoint = torch.load('../subject_3/models/best.pth.tar')['state_dict']
    model.load_state_dict(checkpoint)
    model.eval()
    return model


class ActionPredict(nn.Module):
    def __init__(self):
        super(ActionPredict, self).__init__()
        self.backbone = backbone_init()
        self.model = model_init()
    def forward(self,x):
        N = x.shape[0]
        h =self.backbone(x)
        h = h.view(1,N,2048)
        y= self.model(h)
        return y

def knowledge_judge(direction,knowledge):
    lt_tail =None
    for syz in knowledge:
        if syz['relation'] == '楼梯行走方向':
            lt_tail = syz['tail']
    if lt_tail is not None:
        if direction not in ['左转','右转']:
            if lt_tail == '向上':
                direction = '上楼梯'
            elif lt_tail == '向下':
                direction = '下楼梯'
            else:
                print('tail error lt_tail is',lt_tail,',不是向上或向下')
        else:
            pass
    return direction


if __name__ == '__main__':

    device = 'cuda'
    model = ActionPredict().to(device)
    x = torch.randn(2,3,480,640).to(device)
    # [N,C,H,W] RGB input image N only 1-2 pic for fps
    y = model(x)
    print(y.shape)
    start = time.time()
    for i in range(100):
        ###
        y = model(x)
        direction  = action_list[torch.argmax(y).item()]
        #model output
        ##add logic judge #need task 2 result
        # last knowledge is three tuple [{'relation':,'tail':,}]
        knowledge = []
        #######
        direction = knowledge_judge(direction,knowledge)

        Env = env.Env()

        ## map,set for obstacle #need task1 result
        ### det_dict {classname:bboxes[[x,x,y,y]...],...}
        ### obsname_set = set(name1,name2...)
        det_dict ={}
        obsname_set = set()
        #############
        obs = set()
        for k in det_dict:
            if k in obsname_set:
                for bbox in det_dict[k]:
                    xmin,ymin,xmax,ymax = bbox
                    for i in range(xmin//10,xmax//10):
                        for j in range(ymin//10,ymax//10):
                            if (i,j)!=(32,5):
                                obs.add((i,j))
        ####
        ##########
        # debug data map start goal
        s_start = (32, 5)
        s_goal = None
        if direction in ['上楼梯','下楼梯'] and '楼梯' in det_dict:
            x_mean = []
            y_max = 0
            for bbox in det_dict['楼梯']:
                xmin,ymin,xmax,ymax=bbox
                x_mean.append((xmin+xmax)//2)
                y_max = max(ymax,y_max)
            for i in range(x_mean//10-5,x_mean//10+5):
                for j in range(y_max//10-5,y_max//10):
                    if (i,j) not in obs:
                        s_goal = (i, j)
                        break
                if s_goal is not None:
                    break
        if direction == '左转':
            for i in range(11,21):
                for j in range(38,48):
                    if (i,j) not in obs:
                        s_goal = (i, j)
                        break
                if s_goal is not None:
                    break
        elif direction == '右转':
            for i in range(43,53):
                for j in range(38,48):
                    if (i,j) not in obs:
                        s_goal = (i, j)
                        break
                if s_goal is not None:
                    break
        else:
            for i in range(27,37):
                for j in range(38,48):
                    if (i,j) not in obs:
                        s_goal = (i, j)
                        break
                if s_goal is not None:
                    break
        if s_goal is None:
            s_goal = (32,43)
        ######
        Env.obs = obs
        arastar = AraStar(s_start, s_goal, 2.5, "euclidean")
        arastar.Env = Env
        arastar.obs = Env.obs

        plot = plotting.Plotting(s_start, s_goal)
        plot.env = Env
        plot.obs = Env.obs
        path, visited = arastar.searching()
        ## output path planning
        plot.animation_ara_star(path, visited, "行为预测{}".format(direction),'./output.jpg')
    print(100.0/(time.time()-start))