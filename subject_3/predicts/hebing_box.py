import numpy as np


def dist(x1, y1, x2, y2):
    distance = np.sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1))
    return distance

def rect_distance(x1, y1, x1b, y1b, x2, y2, x2b, y2b):
    """
    计算两个矩形框的距离
    input：两个矩形框，分别左上角和右下角坐标
    return：像素距离
    """
    left = x2b < x1
    right = x1b < x2
    bottom = y2b < y1
    top = y1b < y2
    if top and left:
        return dist(x1, y1b, x2b, y2)
    elif left and bottom:
        return dist(x1, y1, x2b, y2b)
    elif bottom and right:
        return dist(x1b, y1, x2, y2b)
    elif right and top:
        return dist(x1b, y1b, x2, y2)
    elif left:
        return x1 - x2b
    elif right:
        return x2 - x1b
    elif bottom:
        return y1 - y2b
    elif top:
        return y2 - y1b
    else:  # rectangles intersect
        return 0.
def two2one(x1, y1, x1b, y1b, x2, y2, x2b, y2b):
    """
    将两个矩形框，变成一个更大的矩形框
    input：两个矩形框，分别左上角和右下角坐标
    return：融合后矩形框左上角和右下角坐标
    """
    x = min(x1, x2)
    y = min(y1, y2)
    xb = max(x1b, x2b)
    yb = max(y1b, y2b)
    return x, y, xb, yb

def box_select_self(boxes1):
    """
    多box，最终融合距离近的，留下新的，或未被融合的
    input：多box的列表，例如：[[12,23,45,56],[36,25,45,63],[30,25,60,35]]
    return：新的boxes，这里面返回的结果是这样的，被合并的box会置为[]，最终返回的，可能是这样[[],[],[50,23,65,50]]
    """
    ## fisrt boxes add, boxes minus
    if len(boxes1) > 0:
        for bi in range(len(boxes1)):
            for bj in range(len(boxes1)):
                if bi != bj:
                    if len(boxes1[bi]) == 4 and len(boxes1[bj]) == 4:
                        x1, y1, x1b, y1b = int(boxes1[bi][0]), int(boxes1[bi][1]), int(boxes1[bi][2]), int(boxes1[bi][3])
                        x2, y2, x2b, y2b = int(boxes1[bj][0]), int(boxes1[bj][1]), int(boxes1[bj][2]), int(boxes1[bj][3])
                        dis = rect_distance(x1, y1, x1b, y1b, x2, y2, x2b, y2b)
                        if dis < 1:
                            boxes1[bj][0], boxes1[bj][1], boxes1[bj][2], boxes1[bj][3] = two2one(x1, y1, x1b, y1b, x2, y2, x2b, y2b)
                            boxes1[bi] = []
    return boxes1