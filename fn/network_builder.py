import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import measurements
from scipy.ndimage.morphology import generate_binary_structure
from skimage import measure,color
import networkx as nx
import scipy.ndimage as ndi

class segment_thin():
    def __init__():
        pass
    def get_largest_connect(img):
        i = img.copy()
        s = generate_binary_structure(2,2)
        label, number = measurements.label(img, structure = s)
        area = measurements.sum(img, label, index=range(label.max() + 1))
        areaImg = area[label]
        areaMax = areaImg.max()
        for y in range(i.shape[0]):
            for x in range(i.shape[1]):
                if areaImg[y,x] != areaMax:
                    i[y,x] = 0
                else:
                    i[y,x] = 1
        return i

    def Zhang_Suen_thin(img):
        '''
        source: http://blog.csdn.net/jia20003/article/details/52142992
        '''
        H, W, C = img.shape

        out = np.zeros((H, W), dtype=np.int)
        out[img[..., 0] > 0] = 1

        out = 1 - out
        while True:
            s1 = []
            s2 = []

            for y in range(1, H-1):
                for x in range(1, W-1):

                    if out[y, x] > 0:
                        continue

                    f1 = 0
                    if (out[y-1, x+1] - out[y-1, x]) == 1:
                        f1 += 1
                    if (out[y, x+1] - out[y-1, x+1]) == 1:
                        f1 += 1
                    if (out[y+1, x+1] - out[y, x+1]) == 1:
                        f1 += 1
                    if (out[y+1, x] - out[y+1,x+1]) == 1:
                        f1 += 1
                    if (out[y+1, x-1] - out[y+1, x]) == 1:
                        f1 += 1
                    if (out[y, x-1] - out[y+1, x-1]) == 1:
                        f1 += 1
                    if (out[y-1, x-1] - out[y, x-1]) == 1:
                        f1 += 1
                    if (out[y-1, x] - out[y-1, x-1]) == 1:
                        f1 += 1
                    if f1 != 1:
                        continue

                    f2 = np.sum(out[y-1:y+2, x-1:x+2])
                    if f2 < 2 or f2 > 6:
                        continue

                    if (out[y-1, x] + out[y, x+1] + out[y+1, x]) < 1 :
                        continue

                    if (out[y, x+1] + out[y+1, x] + out[y, x-1]) < 1 :
                        continue
                    s1.append([y, x])
            for v in s1:
                out[v[0], v[1]] = 1

            for y in range(1, H-1):
                for x in range(1, W-1):        

                    if out[y, x] > 0:
                        continue

                    f1 = 0
                    if (out[y-1, x+1] - out[y-1, x]) == 1:
                        f1 += 1
                    if (out[y, x+1] - out[y-1, x+1]) == 1:
                        f1 += 1
                    if (out[y+1, x+1] - out[y, x+1]) == 1:
                        f1 += 1
                    if (out[y+1, x] - out[y+1,x+1]) == 1:
                        f1 += 1
                    if (out[y+1, x-1] - out[y+1, x]) == 1:
                        f1 += 1
                    if (out[y, x-1] - out[y+1, x-1]) == 1:
                        f1 += 1
                    if (out[y-1, x-1] - out[y, x-1]) == 1:
                        f1 += 1
                    if (out[y-1, x] - out[y-1, x-1]) == 1:
                        f1 += 1
                    if f1 != 1:
                        continue

                    f2 = np.sum(out[y-1:y+2, x-1:x+2])
                    if f2 < 2 or f2 > 6:
                        continue

                    if (out[y-1, x] + out[y, x+1] + out[y, x-1]) < 1 :
                        continue

                    if (out[y-1, x] + out[y+1, x] + out[y, x-1]) < 1 :
                        continue                 
                    s2.append([y, x])
            for v in s2:
                out[v[0], v[1]] = 1

            if len(s1) < 1 and len(s2) < 1:
                break
        out = 1 - out
        out = out.astype(np.uint8) * 255
        return out

    def neighbours(x,y,image):
        img = image
        x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1
        return [ img[x_1][y],img[x_1][y1],img[x][y1],img[x1][y1],         
                img[x1][y], img[x1][y_1], img[x][y_1], img[x_1][y_1] ]    

    def transitions(neighbours):
        n = neighbours + neighbours[0:1]  
        return sum( (n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]) )  
     
    def zhangSuen(image):
        Image_Thinned = image.copy()  
        changing1 = changing2 = 1
        while changing1 or changing2:  
           
            changing1 = []
            rows, columns = Image_Thinned.shape
            for x in range(1, rows - 1):
                for y in range(1, columns - 1):
                    P2,P3,P4,P5,P6,P7,P8,P9 = n = segment_thin.neighbours(x, y, Image_Thinned)
                    if (Image_Thinned[x][y] == 1     and    
                        2 <= sum(n) <= 6   and    
                        segment_thin.transitions(n) == 1 and   
                        P2 * P4 * P6 == 0  and   
                        P4 * P6 * P8 == 0):        
                        changing1.append((x,y))
            for x, y in changing1: 
                Image_Thinned[x][y] = 0
            # Step 2
            changing2 = []
            for x in range(1, rows - 1):
                for y in range(1, columns - 1):
                    P2,P3,P4,P5,P6,P7,P8,P9 = n = segment_thin.neighbours(x, y, Image_Thinned)
                    if (Image_Thinned[x][y] == 1   and        
                        2 <= sum(n) <= 6  and      
                        segment_thin.transitions(n) == 1 and      
                        P2 * P4 * P8 == 0 and       
                        P2 * P6 * P8 == 0):           
                        changing2.append((x,y))    
            for x, y in changing2: 
                Image_Thinned[x][y] = 0
        return Image_Thinned
class node_find:
    def __init__(self, image):
        self.img = image
        self.conv3 = self.conv3_transition()
        self.conv5 = self.conv5_transition()
        self.conv7 = self.conv7_transition()
        self.yrange = image.shape[0]-1
        self.xrange = image.shape[1]-1
        self.blank = image.copy()
    def ridiculous_situation():
        s1 = np.array([1,0,1,0,1,0,0,0,1]).reshape(3,3)
        s2 = np.array([1,0,1,0,1,0,0,1,0]).reshape(3,3)
        s3 = np.array([1,0,1,0,1,0,1,0,0]).reshape(3,3)
        s4 = np.array([1,0,0,0,1,1,0,1,0]).reshape(3,3)
        s5 = np.array([1,0,0,0,1,1,1,0,0]).reshape(3,3)
        s6 = np.array([1,0,0,0,1,0,1,0,1]).reshape(3,3)
        s7 = np.array([0,1,0,0,1,1,0,1,0]).reshape(3,3)
        s8 = np.array([0,1,0,0,1,1,1,0,0]).reshape(3,3)
        s9 = np.array([0,1,0,1,1,1,0,0,0]).reshape(3,3)
        s10= np.array([0,1,0,0,1,0,1,0,1]).reshape(3,3)
        s11= np.array([0,1,0,1,1,0,0,0,1]).reshape(3,3)
        s12= np.array([0,1,0,1,1,0,0,1,0]).reshape(3,3)
        s13= np.array([0,0,1,0,1,0,1,0,1]).reshape(3,3)
        return [s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13]
    def kernel_check(self,i,j):
        situation = node_find.ridiculous_situation()
        rank = 1
        try:
            kernel = self.img[i-rank:i+rank+1,j-rank:j+rank+1]
            for i in range(13):
                if np.all(situation[i]==kernel):
                    return True
                else:
                    pass
            else:
                return False
        except IndexError:
            return False
    def lookup_by_brain(self):
        assert len(self.img.shape)==2
        a,b = self.img.shape
        run = self.img.copy()*0
        terminal_list = []      
        map_point_list = []     
        edge_list = []          
        node_list = []          
        for i in range(a):
            for j in range(b):
                if self.img[i,j] == 0:
                    pass
                else:
                    map_point_list.append((i,j))
        print('nodes =',len(map_point_list))
        rank = 0                         
        while map_point_list != []:
            rank += 1
            if rank == 1:
                for p in range(len(map_point_list)-1,-1,-1):
                    kernel = self.kernel(map_point_list[p][0],map_point_list[p][1],rank=rank)
                    if kernel == 3:
                        edge_list.append(map_point_list[p])
                        map_point_list.remove(map_point_list[p])

                    elif kernel == 2:
                        terminal_list.append(map_point_list[p])
                        map_point_list.remove(map_point_list[p])
                print('nodes =',len(map_point_list))
                for p in range(len(map_point_list)-1,-1,-1):
                    if self.kernel_check(map_point_list[p][0],map_point_list[p][1]) is False:
                        edge_list.append(map_point_list[p])
                        map_point_list.remove(map_point_list[p])
                    else:
                        node_list.append(map_point_list[p])
                        map_point_list.remove(map_point_list[p])
            else:
                break
        node_list = map_point_list + terminal_list
        img_new = self.draw(node_list)
        return img_new
    def kernel(self,i,j,rank):
        try:
            kernel_sum = np.sum(self.img[i-rank:i+rank+1,j-rank:j+rank+1])
        except IndexError:
            y_start,y_end,x_start,x_end = i-rank,i+rank+1,j-rank,j+rank+1
            if y_start < 0:
                y_start = 0
            if y_end > self.yrange:
                y_end = self.yrange
            if x_start < 0:
                x_start = 0
            if x_end >self.xrange:
                x_end = self.xrange
            kernel_sum = np.sum(self.img[y_start:y_end,x_start:x_end])
        return kernel_sum
    def conv3_kernel(self, i,j,rank):
        try:
            kernel_sum = np.sum(self.conv3[i-rank:i+rank+1,j-rank:j+rank+1])
        except IndexError:
            y_start,y_end,x_start,x_end = i-rank,i+rank+1,j-rank,j+rank+1
            if y_start < 0:
                y_start = 0
            if y_end > self.yrange:
                y_end = self.yrange
            if x_start < 0:
                x_start = 0
            if x_end >self.xrange:
                x_end = self.xrange
            kernel_sum = np.sum(self.conv3[y_start:y_end,x_start:x_end])
        return kernel_sum
    def conv5_kernel(self, i,j,rank):
        try:
            kernel_sum = np.sum(self.conv5[i-rank:i+rank+1,j-rank:j+rank+1])
        except IndexError:
            y_start,y_end,x_start,x_end = i-rank,i+rank+1,j-rank,j+rank+1
            if y_start < 0:
                y_start = 0
            if y_end > self.yrange:
                y_end = self.yrange
            if x_start < 0:
                x_start = 0
            if x_end >self.xrange:
                x_end = self.xrange
            kernel_sum = np.sum(self.conv5[y_start:y_end,x_start:x_end])
        return kernel_sum
    def is_terminal(self,kernel_sum):
        if kernel_sum == 2:
            return True
    def is_edge(self,kernel_sum):
        if kernel_sum == 3:
            return True
    def is_node(self,kernel_sum):
        if (kernel_sum >= 4) and (kernel_sum <= 9):
            return True
    def is_edge_r2(self,kernel_sum,rank):
        if kernel_sum < rank*200-100:
            return True
    def draw(self,node_list):
        new = self.blank.copy()
        for (i,j) in node_list:
            new[i,j] = 2
        return new
    def conv3_transition(self):
        assert len(self.img.shape)==2
        a,b = self.img.shape
        run = self.img.copy()*0
        for i in range(a):
            for j in range(b):
                if self.img[i,j] == 1:
                    kernel = self.kernel(i,j,rank=1)
                    run[i,j] = kernel
                else:
                    pass
        return run
    def conv5_transition(self):
        assert len(self.img.shape)==2
        a,b = self.img.shape
        run = self.img.copy()*0
        for i in range(a):
            for j in range(b):
                if self.img[i,j] == 1:
                    kernel = self.kernel(i,j,rank=2)
                    run[i,j] = kernel
                else:
                    pass
        return run
    def conv7_transition(self):
        assert len(self.img.shape)==2
        a,b = self.img.shape
        run = self.img.copy()*0
        for i in range(a):
            for j in range(b):
                if self.img[i,j] == 1:
                    kernel = self.kernel(i,j,rank=3)
                    run[i,j] = kernel
                else:
                    pass
        return run
    def lookup_onlyTerminals(self):
        assert len(self.img.shape)==2
        a,b = self.img.shape
        run = self.img.copy()*0
        terminal_list = []      
        map_point_list = []     
        edge_list = []          
        node_list = []          
        for i in range(a):
            for j in range(b):
                if self.img[i,j] == 0:
                    pass
                else:
                    map_point_list.append((i,j))
        print('nodes =',len(map_point_list))
        rank = 1
        for p in range(len(map_point_list)-1,-1,-1):
            kernel = self.kernel(map_point_list[p][0],map_point_list[p][1],rank=rank)
            if kernel == 3:

                edge_list.append(map_point_list[p])
                map_point_list.remove(map_point_list[p])

            elif kernel == 2:
                terminal_list.append(map_point_list[p])
                map_point_list.remove(map_point_list[p])
        print('nodes =',len(terminal_list))
        img_new = self.draw(terminal_list)
        return img_new

    def lookup(self):
        assert len(self.img.shape)==2
        a,b = self.img.shape
        run = self.img.copy()*0
        terminal_list = []      
        map_point_list = []     
        edge_list = []          
        node_list = []          
        for i in range(a):
            for j in range(b):
                if self.img[i,j] == 0:
                    pass
                else:
                    map_point_list.append((i,j))
        print('nodes =',len(map_point_list))
        rank = 0                        
        while rank <= 3:
            rank += 1
            if rank == 1:
                for p in range(len(map_point_list)-1,-1,-1):
                    kernel = self.kernel(map_point_list[p][0],map_point_list[p][1],rank=rank)
                    if kernel == 3:
                        edge_list.append(map_point_list[p])
                        map_point_list.remove(map_point_list[p])

                    elif kernel == 2:
                        terminal_list.append(map_point_list[p])
                        map_point_list.remove(map_point_list[p])
                print('nodes =',len(map_point_list))

            elif rank == 2:
                con_differ5_3 = self.conv5 - self.conv3
                con_differ7_5 = self.conv7 - self.conv5
                for i in range(len(map_point_list)-1,-1,-1):
                    kernel5_3 = con_differ5_3[map_point_list[i][0],map_point_list[i][1]]
                    kernel7_5 = con_differ7_5[map_point_list[i][0],map_point_list[i][1]]
                    if (kernel5_3 < 3) or (kernel7_5 < 3):
                        edge_list.append(map_point_list[i])
                        map_point_list.remove(map_point_list[i])
                print('nodes =',len(map_point_list))

            elif rank == 3:
                self.conv3[self.conv3==3] = 100
                for i in range(len(map_point_list)-1,-1,-1):
                    kernel = self.conv3_kernel(map_point_list[i][0],map_point_list[i][1],rank=rank)
                    if kernel < 300:
                        edge_list.append(map_point_list[i])
                        map_point_list.remove(map_point_list[i])
                print('nodes =',len(map_point_list))

            elif rank == 4:

                self.conv5[self.conv5<=7] = 100
                for i in range(len(map_point_list)-1,-1,-1):
                    kernel = self.conv3_kernel(map_point_list[i][0],map_point_list[i][1],rank=rank)
                    if kernel < 500:
                        edge_list.append(map_point_list[i])
                        map_point_list.remove(map_point_list[i])
                print('nodes =',len(map_point_list))

            else:
                break

        node_list = map_point_list + terminal_list
        img_new = self.draw(node_list)
        return img_new


    def get_center_list(self, marked):
        labels = measure.label(marked, connectivity=2)
        center = []
        center_int = []
        properties = measure.regionprops(labels)
        for prop in properties:
            center.append(prop.centroid)

        for (i,j) in center:
            center_int.append((int(i), int(j)))
        return center_int

    def get_edge_list(self, center_list, skeleton):
        point_list = []

        for i in range(skeleton.shape[0]):
            for j in range(skeleton.shape[1]):
                if skeleton[i,j] == 1:
                    point_list.append((i,j))

        edge_list = list(set(point_list)-set(center_list))
        return edge_list

    def add_weight(self, segment, skeleton, center_list, edge_list):
        weight = ndi.distance_transform_edt(segment)
        weight_list = []
        for (i,j) in center_list:
            point_weight = weight[i,j]
            weight_list.append((point_weight,2))
        for (i,j) in edge_list:
            edge_weight = weight[i,j]
            weight_list.append((edge_weight,1))
        weight_array = np.array(weight_list)

        center_array = np.array(center_list)
        edge_array = np.array(edge_list)
        point_array = np.concatenate((center_array,edge_array),axis=0)

        big_array = np.concatenate((point_array,weight_array),axis=1)
        
        order_array = np.linspace(1, big_array.shape[0], big_array.shape[0])
        order_array = np.expand_dims(order_array,axis=-1)

        structure_order = np.concatenate((order_array,big_array),axis=1)
        np.save("structure_and_order.npy",structure_order)

        return structure_order

    def get_connection_list(self, structure_order):
        use = structure_order[:,0:3]

        iter_range = use.shape[0]

        connection = []
        for i in range(iter_range-1):
            first_one = use[i,1:3]
            if i%100 == 0:
                print(i)
            for j in range(i+1,iter_range):
                second_one = use[j,1:3]
                distance = np.linalg.norm(first_one-second_one)
                if distance < 1.5:
                    connection.append((int(use[i,0]),int(use[j,0])))
        con_array = np.array(connection)
        np.save('con_relationship_check.npy',con_array)

        return con_array

class mc:
    '''
    In Morphology Calculation, the input must be coordinates sequence
    '''
    def __init__(self):
        pass
    def distance(node1,node2,power):
        x = np.array([node1[1]-node2[1], node1[0]-node2[0]])
        d = np.linalg.norm(x)
        if power == 1:
            return d
        if power == 2:
            return d**2
    def calLen(path):
        num = len(path)
        s = 0
        for i in range(num-1):
            s += mc.distance(path[i],path[i+1],1)
        return s
    def pixLen(path): 
        '''
        Calculate the length by counting pixels.
        --------
        path: list of coordinates sequence
        '''
        output = len(path)
        return output 
    def msrg(path): 
        '''
        Calculate the mean square radius of gyration.
        '''
        pathMatrix = np.array(path)
        fiberCenter = pathMatrix.mean(axis = 0)
        output = 0
        for pix in path:
            output += mc.distance(pix, fiberCenter, 2)
        output = output / len(path)
        return output
    def direction(path):
        start = path[0]
        end = path[-1]
        y = end[0]-start[0]
        x = end[1]-start[1]
        if x == 0:
            theta = 90
        else:
            theta = np.arctan(y/x) / np.pi * 180
        return theta
    def b_center(path): 
        '''
        Calculate the bary center.
        '''
        pathMatrix = np.array(path)
        fiberCenter = pathMatrix.mean(axis = 0)
        return fiberCenter.tolist()
    def mse2e(path): 
        '''
        Calculate the square of end to end distance.
        '''
        output = mc.distance(path[0], path[-1], 2)
        return output
    def e2e(path): 
        '''
        Calculate the end to end distance.
        '''
        output = mc.distance(path[0], path[-1], 1)
        return output
    def persLen(path, s): 
        '''
        Calculate the persistence length.
        The projection on the first chain segment.
        --------
        path: list of coordinates sequence
        s   : length of the custom chain segment
        '''
        if len(path) > s:
            lenCluster = []
            start = [path[s][0] - path[0][0], path[s][1] - path[0][1]]
            projectStart = (start[0] ** 2 + start[1] ** 2) ** 0.5
            full  = [path[-1][0] - path[0][0], path[-1][1] - path[0][1]]
            projection = (start[0] * full[0] + start[1] * full[1]) / projectStart
            return projection
        else:
            # It means the path is too short for slice.
            return 0
    def curvature(path, s):
        '''
        Calculate the curvature list.
        --------
        path: list of coordinates sequence
        s   : length of the custom chain segment
        '''
        if len(path) > 2*s:
            curvature = []
            for i in range(s, len(path) - s):
                a = path[i-s]
                b = path[i]
                c = path[i+s]
                x = mc.distance(a, b, 1)
                y = mc.distance(a, c, 1)
                z = mc.distance(b, c, 1)
                try:
                    r = x*y*z/(((x+y-z)*(x-y+z)*(y+z-x)*(x+y+z))**0.5)
                    cur = 1 / r
                except ZeroDivisionError:
                    cur = 0
                finally:
                    curvature.append(cur)
            return(np.mean(curvature), np.std(curvature))

        else:
            print('Failure for wrong interval.')
    def persistence(path):
        def len_solve(theta, s):
            return -s/np.log(theta)/2
        def theta(start_vector, end_vector):
            up = start_vector[1]*end_vector[1]+start_vector[0]*end_vector[0]
            down = ((start_vector[0]**2+start_vector[1]**2)**0.5)*((end_vector[0]**2+end_vector[1]**2)**0.5)
            return np.abs(up/down)
        
        cos_list = []
        
        if len(path) <= 10:
            return 0
        else:
            s0 = path[0]
            s1 = path[5]
            for i in range(len(path)-5):
                e0 = path[i]
                e1 = path[i+5]
                start_vector = [s0[0]-s1[0], s0[1]-s1[1]]
                end_vector = [e0[0]-e1[0], e0[1]-e1[1]]

                theta_ = theta(start_vector, end_vector)
                cos_list.append(theta_)
            cos_theta = np.mean(cos_list)
            per = len_solve(cos_theta, 5)
            return per

class network_builder():
    def __init__(self, nodes, edges, center_num):
        self.nodes = nodes
        self.edges = edges
        self.center_num = center_num

        self.node_order = nodes[:,0].astype('int').tolist()
        node_location = nodes[:,1:3]
        self.node_location = node_location[:,[1,0]].astype('int').tolist()
        self.node_weight = nodes[:,3].tolist()
        self.node_kind = nodes[:,4].astype('int').tolist()

        self.node_num = len(nodes)

        self.net = self.load_Graph()

        self.npos = dict(zip(self.node_order, self.node_location))
        self.nlabels = dict(zip(list(range(1,self.node_num+1)), list(range(1,self.node_num+1))))

        self.npos_con= dict(zip(self.node_order[:center_num], self.node_location[:center_num]))
        self.nlabels_con = dict(zip(list(range(1,center_num+1)), list(range(1,center_num+1))))

        self.paths = self.path_split()

    def load_Graph(self):
        net = nx.Graph()
        for i in range(self.node_num):
            net.add_node(i+1,
                node_weight = self.node_weight[i],
                node_kind = self.node_kind[i])
        net.add_edges_from(self.edges)
        return net

    def analysis(self):
        param = []
        for (i,j) in self.paths:
            path = nx.shortest_path(net,i,j)
            seq, weight = network_builder.path_to_seq(path)
            param.append([mc.pixLen(seq),
                          mc.calLen(seq),
                          mc.direction(seq),
                          mc.msrg(seq),
                          mc.e2e(seq),
                          np.mean(weight)])
        df = pd.DataFrame(np.array(param))
        df.columns = ['pixLen','calLen','theta','msrg','e2e','thick']
        return df

    def direction_on(self):
        param = []
        for (i,j) in self.paths:
            path = nx.shortest_path(net,i,j)
            seq, weight = path_to_seq(path)
            param.append([mc.calLen(seq),
                          mc.direction(seq)])
        df = pd.DataFrame(np.array(param))
        df.columns = ['calLen','theta']
        return df

    def path_to_seq(self, path):
        weight = []
        for i in path:
            weight.append(self.node[i][1])
            seq.append((self.npos[i]))
        return seq, weight

    def path_split(self):
        edges = []
        for i in range(1,self.center_num+1):
            print(i)
            for j in range(i, self.center_num-1):
                if nx.has_path(self.net, i, j) is True:
                    path = nx.shortest_path(self.net, i, j)
                    path_check = path[1:-1]
                    if len(path_check) >= 3:
                        if min(path_check) <= self.center_num:
                            pass
                        else:
                            edges.append((i,j))
        return edges