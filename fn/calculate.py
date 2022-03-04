import numpy as np
import pandas as pd
import collections
import networkx as nx
class morph_calculate:
    '''
    In morph_calculate, the input must be coordinates sequence
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
            s += morph_calculate.distance(path[i],path[i+1],1)
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
            output += morph_calculate.distance(pix, fiberCenter, 2)
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
        output = morph_calculate.distance(path[0], path[-1], 2)
        return output
    def e2e(path): 
        '''
        Calculate the end to end distance.
        '''
        output = morph_calculate.distance(path[0], path[-1], 1)
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
                x = morph_calculate.distance(a, b, 1)
                y = morph_calculate.distance(a, c, 1)
                z = morph_calculate.distance(b, c, 1)
                try:
                    r = x*y*z/(((x+y-z)*(x-y+z)*(y+z-x)*(x+y+z))**0.5)
                    cur = 1 / r
                except ZeroDivisionError:
                    cur = 0
                finally:
                    curvature.append(cur)
            return(np.mean(curvature), np.std(curvature))
            #return curvature
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

def analysis(net, node_in_net, npos, edge_list):
    param = []
    for (i,j) in edge_list:
        path = nx.shortest_path(net,i,j)
        seq, weight = path_to_seq(node_in_net, npos, path)
        param.append([morph_calculate.calLen(seq),
                      morph_calculate.direction(seq),
                      morph_calculate.msrg(seq),
                      morph_calculate.e2e(seq),
                      np.mean(weight)])
    df = pd.DataFrame(np.array(param))
    df.columns = ['calLen','theta','msrg','e2e','thick']
    return df
def path_to_seq(node_in_net, npos, path):
    seq = []
    weight = []
    for i in path:
        weight.append(node_in_net[i-1][1])
        seq.append((npos[i]))
    return seq, weight
def path_split(path_num, net):
    edges = []
    for i in range(1,path_num+1):
        for j in range(i, path_num-1):
            if nx.has_path(net, i, j) is True:
                path = nx.shortest_path(net, i, j)
                path_check = path[1:-1]
                if len(path_check) >= 3:
                    if min(path_check) <= path_num:
                        pass
                    else:
                        edges.append((i,j))
    return edges
def load_Graph(node_order,node_location,node_weight,node_kind,edges):
    net = nx.Graph()
    for i in range(len(node_order)):
        net.add_node(i+1,
                     node_weight=node_weight[i],
                     node_kind=node_kind[i])
    net.add_edges_from(edges)
    return net
def remove_redundant_nodes(net):
    degree_list = list(nx.degree(net))
    for (i, j) in degree_list:
        if j == 2:
            net.nodes()[i]['node_kind'] = 1
        elif j == 0:
            net.nodes()[i]['node_kind'] = 1
    return net
def zipped_list(net, node_in_net, npos, edge_list):
    zip_list = []
    order = 1
    for (i,j) in edge_list:
        path = nx.shortest_path(net, i, j)
        seq, weight = path_to_seq(node_in_net, npos, path)
        zip_list.append([order, (i,j), tuple(seq), 1])
        order += 1
    return zip_list
def share(zip_list, i, j):
    pair_i = zip_list[i-1][1]
    pair_j = zip_list[j-1][1]

    i_1, i_2 = pair_i[0], pair_i[1]
    j_1, j_2 = pair_j[0], pair_j[1]

    if i_1 == j_1:
        return True, zip_list[i-1][2], zip_list[j-1][2], 1
    elif i_1 == j_2:
        return True, zip_list[i-1][2], list(reversed(zip_list[j-1][2])), 1
    elif i_2 == j_1:
        return True, list(reversed(zip_list[i-1][2])), zip_list[j-1][2], 2
    elif i_2 == j_2:
        return True, list(reversed(zip_list[i-1][2])), list(reversed(zip_list[j-1][2])), 2
    else:
        return False, False, False, False
def seq_vector_direction(seq, slice_length=10):
    if len(seq) < slice_length:
        vector = [seq[-1][0] - seq[0][0], seq[-1][1] - seq[0][1]]
    else:
        vector = [seq[slice_length-1][0] - seq[0][0], seq[slice_length-1][1] - seq[0][1]]
    return vector
def theta(vector_i, vector_j):
    x1, y1 = vector_i[0], vector_i[1]
    x2, y2 = vector_j[0], vector_j[1]
    cos_theta = (x1*x2+y1*y2)/(((x1**2+y1**2)*(x2**2+y2**2))**0.5)
    return cos_theta
def order_fetch(zip_list):
    order_list = []
    for zip_seq in zip_list:
        if zip_seq[3] != 0:
            order_list.append(zip_seq[0])
    return order_list
def get_combine_list(zip_list):
    order_list = []
    combine_list = []
    for zip_seq in zip_list:
        if zip_seq[3] != 0:
            order_list.append(zip_seq[0])

    i = len(order_list)

    for a in range(i-2):
        for b in range(a+1, i-1):
            combine_list.append((order_list[a], order_list[b]))
    return combine_list
def flexible_determine(theta, len_i, len_j, cos_short, cos_long):
    if (len_i < 5) or (len_j < 5):
        return True
    elif (len_i < 10) or (len_j < 10):
        if theta < -cos_short or theta > cos_short:
            return True
        else:
            return False
    else:
        if theta < -cos_long or theta > cos_long:
            return True
        else:
            return False
def output_zip_list(zip_list):
    output_list = []
    for zip_seq in zip_list:
        if zip_seq[3] == 1:
            output_list.append(zip_seq)
    return output_list
def output_zip_index(zip_list):
    output_list = []
    for zip_seq in zip_list:
        output_list.append(zip_seq[0])
    return output_list
def edge_connect(net, npos, zip_list_in, cos):
    def pair_to_seq(net, npos, pair):
        path = nx.shortest_path(net, pair[0], pair[1])
        seq = []
        for i in path:
            seq.append(npos[i])
        return tuple(seq)
    zip_list = zip_list_in.copy()
    edge_conlist = []
    slice_length = 10
    # create the combination of path index
    combine_list = get_combine_list(zip_list)

    for (i,j) in combine_list:
        common, seq_i, seq_j, common_node = share(zip_list,i,j)
        # There is no common node.
        if common == False:
            pass
        # There is one or two common nodes
        else:
            vector_i = seq_vector_direction(seq_i,slice_length=slice_length)
            vector_j = seq_vector_direction(seq_j,slice_length=slice_length)
            cos_theta = theta(vector_i, vector_j)
            # If the angle is small, do simplification.
            if flexible_determine(cos_theta, len(seq_i), len(seq_j), cos[0], cos[1]) is True:
                # Check the number of nodes
                common_node_index = zip_list[i-1][1][common_node-1]
                set_1 = set([zip_list[i-1][1][0],
                             zip_list[i-1][1][-1],
                             zip_list[j-1][1][0],
                             zip_list[j-1][1][-1]])
                set_2 = set([zip_list[i-1][1][common_node-1]])
                new_pair = tuple(set_1 - set_2)
                # The normal condition, two shares one common node.
                if len(new_pair) == 2:
                    net.nodes()[common_node_index]['node_kind'] = 1
                    # Mark the 2 paths in zip_list.
                    zip_list[i-1][3] = 0
                    zip_list[j-1][3] = 0
                    # Collect the edge connections.
                    edge_conlist.append((i,j))
                # The two nodes of the two paths are same.
                elif len(new_pair) == 1:
                    len_i = len(zip_list[i-1][2])
                    len_j = len(zip_list[j-1][2])
                    if len_i <= len_j:
                        zip_list[j-1][3] = 0
                    else:
                        zip_list[i-1][3] = 0
    print(len(edge_conlist),'pair')
    return zip_list, edge_conlist
def remove_polyline(zip_list):
    check = []
    order = 1
    for zip_seq in zip_list:
        cnt_length = morph_calculate.calLen(zip_seq[2])
        e2e_length = morph_calculate.e2e(zip_seq[2])
        if (cnt_length - e2e_length * 1.2) > 0:
            pass
        else:
            check.append([order, zip_seq[1], zip_seq[2], zip_seq[3]])
            order += 1
    return check
def edge_net(zip_list_index,edge_conlist):
    Enet = nx.Graph()
    Enode_list = zip_list_index
    Eedge_list = edge_conlist
    Enet.add_nodes_from(Enode_list)
    Enet.add_edges_from(Eedge_list)
    return Enet
def node_fetch(zip_list, Enet):
    edge_cluster_list = sorted(nx.connected_components(Enet),key=len,reverse=True)
    map_cluster = []
    for edge_cluster in edge_cluster_list:
        edge_con_cluster = []
        for edge in list(edge_cluster):
            edge_con_cluster.append(zip_list[edge-1][1][0])
            edge_con_cluster.append(zip_list[edge-1][1][1])
        map_cluster.append(edge_con_cluster)
    return map_cluster
def get_curves(map_cluster):
    curve_pair = []
    for map_roi in map_cluster:
        counts = collections.Counter(map_roi).most_common()
        node1, times1 = counts[-2][0], counts[-2][1]
        node2, times2 = counts[-1][0], counts[-1][1]
        
        if (times1==1) and (times2==1):
            curve_pair.append((node1, node2))
        else:
            pass
    return curve_pair