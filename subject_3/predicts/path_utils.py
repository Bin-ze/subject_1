import numpy as np
import operator

def inbox_x(box,x):
    return (x>box[0]) and (x<box[2])


def inbox_y(box,y):
    return (y>box[1]) and (y<box[3])

def judge_stuck(start,graph,boxes):
    x = int(start[0])
    y = int(start[1])
    stuck = False
    stuck_box = None
    if x >= graph.shape[0]:
        x = graph.shape[0] - 1
    elif x < 0:
        x = 0

    if x >= graph.shape[1]:
        x = graph.shape[1] - 1

    elif x < 0:
        x = 0

    # print('start',start)
    if graph[x][y][0]:
        for box in boxes:
            if inbox_x(box,x) and inbox_y(box,y):
                stuck = True
                stuck_box = box
                break
    # print(stuck_box)
    return stuck,stuck_box


def get_new_start(start,end,stuck_box,mode):
    dmin_x = min(np.abs(start[0]-stuck_box[0]),np.abs(start[0]-stuck_box[2]))
    dmin_y = min(np.abs(start[1] - stuck_box[1]), np.abs(start[1] - stuck_box[3]))
    # print(mode)
    if dmin_x <= dmin_y:
        # if end[1]>= (stuck_box[1]+stuck_box[3])/2:
        if mode:
            new_start = (start[0],stuck_box[3])
        else:
            new_start = (start[0], stuck_box[1])
    else:
        # if end[0]>= (stuck_box[0]+stuck_box[2])/2:
        if mode:
            new_start = (stuck_box[2],start[1])
        else:
            new_start = (stuck_box[0], start[1])
    # print(new_start)
    return new_start

def judge_update(node):
    update = False
    if node.finish:
        if node.best_path is None:
            update = True
        elif (node.suc) and (not node.best_suc):
            update = True
        elif node.length < node.best_path_length:
            update = True
    return update


def update_nodes(best_path,best_path_suc,best_path_length,node):
    node.best_path =best_path
    node.best_path_suc = best_path_suc
    node.best_path_length = best_path_length
    if node.last_node is not None:
        update_nodes(best_path,best_path_suc,best_path_length,node.last_node)

class PointNode():

    def __init__(self,last_node,pos,finish=False,suc=False):
        self.last_node = last_node
        self.pos = pos
        self.pos2 = None
        self.finish = finish
        self.suc = suc
        self.length = None
        self.path = None
        self.best_path_suc = None
        self.best_path = None
        self.best_path_length = None

    def p2p(self,start,shape,mode):
        start = np.array(start)
        end = np.array(self.end)
        dis = np.linalg.norm(start - end)
        dx = self.step * (end[0] - start[0]) / dis
        dy = self.step * (end[1] - start[1]) / dis
        Finish = False
        suc = False
        out_loop = False
        while not out_loop:
            start[0] = start[0] + dx
            start[1] = start[1] + dy

            bianjie =False
            if start[0] >= shape[0]:
                start[0] = shape[0] - 1
                bianjie = True
            elif start[0] < 0:
                start[0] = 0
                bianjie = True
            if start[1] >= shape[1]:
                start[1] = shape[1] - 1
                bianjie = True
            elif start[1] < 0:
                start[1] = 0
                bianjie = True

            if bianjie:
                new_start =(int(start[0] - dx),int(start[1] - dy))
                break

            stuck, stuck_box = judge_stuck(start, self.graph, self.boxes)
            if stuck:
                # 回退
                start[0] = start[0] - dx
                start[1] = start[1] - dy
                self.pos2 = (start[0],start[1])
                new_start = get_new_start(start, end, stuck_box,mode)

                out_loop = True
            elif dx * (end[0] - start[0]) <= 0:
                out_loop = True
                new_start = (end[0], end[1])
            else:
                pass

        if operator.eq(self.end,new_start):
            Finish = True
            suc = True
        if (new_start[0] in [0,self.shape[0]-1]) or (new_start[1] in [0,self.shape[1]-1]):
            suc = False
            Finish =True
        # print(new_start)
        node = PointNode(self,new_start)
        node.finish=Finish
        node.suc=suc
        path = []
        buffer_node = node
        return_start =False
        while not return_start:
            if buffer_node.pos2 is not None:
                path.append(buffer_node.pos2)
            path.append(buffer_node.pos)
            if buffer_node.last_node is None:
                return_start = True
            else:
                buffer_node = buffer_node.last_node
        points = list(reversed(path))
        cleared_points = [points[0]]
        for p_id in range(len(points) - 1):
            if np.abs(points[p_id + 1][0] - points[p_id][0]) + np.abs(points[p_id + 1][1] - points[p_id][1]) > self.step:
                cleared_points.append(points[p_id + 1])
        length = 0
        for l_num in range(len(cleared_points)-1):
            dis = np.linalg.norm(np.array(cleared_points[l_num+1]) -np.array(cleared_points[l_num]))
            length += dis
        if len(points) > 20:
            node.finish = True
        if node.finish:
            node.path = cleared_points
            node.length = length


        return node
    def search(self,shape,boxes,end,step=10):
        self.step = step
        self.shape = shape
        self.boxes = boxes
        self.end =end
        graph = np.zeros(shape)
        for box in boxes:
            graph[box[0]:box[2], box[1]:box[3]] = 1
        self.graph = graph
        node1 = self.p2p(self.pos,shape,mode = True)
        node2 = self.p2p(self.pos,shape,mode = False)
        # print(self.best_path)
        if judge_update(node1):
            update_nodes(node1.path,node1.suc,node1.length,node1)
        if judge_update(node2):
            update_nodes(node2.path,node2.suc,node2.length,node2)
        # print(node1.step)
        if not node1.finish:
            node1.search(shape, boxes, end)
        if not node2.finish:
            node2.search(shape, boxes, end)



if __name__ == "__main__":
    import cv2

    def trans(point):
        return (point[1], point[0])
    start = (328, 240)
    end = (120, 50)
    img = cv2.imread('./test.jpg')
    # print(img.shape)
    boxes = [[190, 100, 225, 165], [105, 170, 130, 215], [135, 285, 170, 325], [190, 400, 250, 470],
             [285, 145, 325, 190]]
    # boxes = [[190,0,225,260],[105,170,130,215],[135,285,170,325],[190,400,250,470],[285,145,325,190]]
    # xmin,ymin,xmax,ymax
    boxes = np.array(boxes)
    # tuple,np,tuple,tuple


    start_node =PointNode(None,start)
    start_node.search(img.shape, boxes, end)
    shortest_path = start_node.best_path
    # print(shortest_path)
    for i in range(len(shortest_path)-1):
        img = cv2.line(img,trans(shortest_path[i]),trans(shortest_path[i+1]),color=(50,205,50),thickness=3)
    cv2.imshow('tt',img)
    cv2.imwrite('./results.jpg',img)
