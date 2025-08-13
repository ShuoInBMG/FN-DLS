import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import itertools
import pandas as pd
from tqdm.auto import tqdm
from scipy import ndimage

class network_edge_refine:
    def __init__(self, node_list, connection):
        
        # all points
        self.node_index = node_list[:,0]
        self.location = node_list[:,1:3]
        self.node_type = node_list[:,3]
        
        # only nodes
        self.num_of_true_node_index = len(np.where(self.node_type>1)[0])
        self.true_node_index = self.node_index[:self.num_of_true_node_index]
        self.true_node_location = self.location[:self.num_of_true_node_index]
        
        # short edges: connections of all points including edge points
        self.short_edges = connection.tolist()
        
        # build full-path graph with all points
        self.fullPathGraph = self.load_fullPathGraph()
        
        # initialize paths
        self.initPaths = []
        
        self.npos = self.get_location_dict()
    
    def get_location_dict(self):
        '''
        Collect locations as dictionary.
        '''
        update = {}
        for index, loc in zip(self.node_index, self.location):
            update[index] = (loc[1], loc[0])
        return update
    
    def load_fullPathGraph(self):
        '''
        Build the full-path graph.
        '''
        net = nx.Graph()
        for i in self.node_index:
            net.add_node(i, node_type = self.node_type[i-1])
        net.add_edges_from(self.short_edges)
        return net
    
    def find_initPaths(self):
        '''
        Find connections between true nodes.
        --------
        return: initial paths between true nodes
        '''
        totals = self.num_of_true_node_index*(self.num_of_true_node_index-1)//2
        loop = tqdm(total = totals)
        update_path = []
        print("Total nodes: %d"%self.num_of_true_node_index)
        for i in range(1, self.num_of_true_node_index):
            for j in range(i+1, self.num_of_true_node_index+1):
                loop.update(1)
                try:
                    path = nx.shortest_path(self.fullPathGraph, i, j)
                    if self.initPaths_continuity_check(path) == True:
                        update_path.append((i, j))
                    else:
                        pass
                except nx.NetworkXNoPath:
                    pass
        loop.close()
        self.initPaths = np.array(update_path)
                
    def initPaths_continuity_check(self, path):
        '''
        Check if there is other nodes in the way.
        --------
        path: the selected node.
        '''
        path_check = path[1:-1]
        # if any node exsists in the path, there will be zero or minus.
        other_node = [(x-self.num_of_true_node_index)>0 for x in path_check]
        res = np.all(other_node)
        if res == True:
            return True
        else:
            return False
    
    def path2coSeq(self, path):
        '''
        Acquire the coordinate sequence of the selected path.
        '''
        seq = []
        for index in path:
            seq.append(self.location[index-1].tolist())
        return seq
    
    def draw_Lp_map(self, Lp_list):
        
        Lp_image = np.zeros(shape=(906,905))
        
        for Lp_element in Lp_list:
            (node1, node2) = Lp_element[0]
            Lp = Lp_element[1]
            path = nx.shortest_path(self.fullPathGraph, node1, node2)
            coSeq = self.path2coSeq(path)
            
            if type(Lp) != np.float64:
                for loc in coSeq:
                    Lp_image[loc[0], loc[1]] = -1
            elif Lp > 200:
                for loc in coSeq:
                    Lp_image[loc[0], loc[1]] = -1
            else:
                for loc in coSeq:
                    Lp_image[loc[0], loc[1]] = np.log(Lp)
        return Lp_image
class assist_index:
    @staticmethod
    def distance(point1, point2):
        '''
        Calculate the distance between 2 points.
        --------
        point1, point2: the selected 2 points.
        '''
        x = np.array(point1) - np.array(point2)
        return np.linalg.norm(x)
    
    def length(path):
        '''
        Calculate the length of selected path. l2-norm
        --------
        path: the seleced path
        '''
        update = 0
        for i in range(len(path)-1):
            update += assist_index.distance(path[i], path[i+1])
        return update
    
    def sample(path, sample_interval = 11):
        if len(path) <= sample_interval:
            slice_ = np.array([0, np.round(len(path)/2), -1])
            sampled = path[(slice_,)]
        else:
            slice_ = np.arange(0, len(path), sample_interval)
            if (len(path) % sample_interval == 0):
                pass
            else:
                insert = np.array([-1])
                slice_ = np.vstack([slice_, insert])
        return path[(slice_,)]
    def center(path):
        '''
        Calculate the center of the path.
        '''
        return np.mean(path, axis=0)
    
    def msrg(path):
        '''
        Mean square radius of gyration.
        '''
        center = assist_index.center(path)
        update = 0
        for loc in path:
            update += (assist_index.distance(center, loc) ** 2)
        return update / len(path)
    
    def e2e(path):
        '''
        End to end distance.
        '''
        return assist_index.distance(path[0], path[-1])
        
        
    def yield_nodeSeq(path, sample_interval=5):
        '''
        Yield slice for curvature calculating.
        '''
        num = len(path)
        if (num < (2*sample_interval+1)):
            slice_ = np.array([0, np.round(num/2).astype("int"), -1])
            seq = [path[(slice_,)]]
        else:
            seq = []
            for i in range(sample_interval, num-sample_interval, sample_interval):
                slice_ = np.array([i-sample_interval, i, i+sample_interval])
                seq.append(path[(slice_,)])
            if ((num-1) % sample_interval) == 0:
                pass
            else:
                insert = np.array([-sample_interval*2-1, -sample_interval-1, -1])
                seq.append(path[(insert,)])
        return seq
            
    def get_radius(node1, node2, node3):
        '''
        Calculate the radius or curvature of the selected 3 nodes.
        '''
        a = assist_index.distance(node1, node2)
        b = assist_index.distance(node2, node3)
        c = assist_index.distance(node1, node3)
        p = (a+b+c)*0.5
        area = (p*(p-a)*(p-b)*(p-c))**0.5
        if area == 0:
            return "linear"
        else:
            radius = a*b*c/4/area
            return radius
    class mechanic:
        @staticmethod
        def Lp(node1, node2, node3):
            '''
            Calcuate the persistence length of the selected 3 nodes.
            '''
            R = assist_index.get_radius(node1, node2, node3)
            if R != "linear":
                L13 = assist_index.distance(node1, node3)
                cosTheta = -(L13**2 - 2*R**2)/(2*R**2)
                L = np.arccos(cosTheta)*R
                Lp_ = -L / (2*np.log(cosTheta))
                return Lp_
            else:
                return np.nan
class edge_opt:
    def __init__(self, node_list, path_list):
        
        # all points
        self.node_index = node_list[:,0]
        self.location = node_list[:,1:3]
        self.node_type = node_list[:,3]
        
        # only nodes, initialize
        self.num_of_true_node_index = len(np.where(self.node_type>1)[0])
        self.true_node_index = self.node_index[:self.num_of_true_node_index]
        self.true_node_location = self.location[:self.num_of_true_node_index]
        
        self.init_edges = np.array(path_list)
        self.Graph = self.load_Graph()
    
    def remove_edges(self, remove):
        '''
        Remove the wrong edges.
        '''
        before = self.init_edges.shape[0]
        before_ = self.init_edges.tolist()
        remove_ = remove.tolist()
        update = np.array([x for x in before_ if x not in remove_])
        self.init_edges = update
        after = self.init_edges.shape[0]
        print("%.f / %.f branch edges removed"%(before-after, before))
        
        nodes = list(set(self.init_edges.flatten().tolist()))
        self.node_type = np.ones(shape = self.node_type.shape)
        for node in nodes:
            self.node_type[node-1] = 2
    def update_property(self):
        '''
        Update the node information after changing node_type.
        '''
        self.num_of_true_node_index = len(np.where(self.node_type == 2)[0])
        self.true_node_index = self.node_index[np.where(self.node_type == 2)]
        self.true_node_location = self.location[np.where(self.node_type == 2)]
        
    def load_Graph(self):
        '''
        Build the graph with only nodes, so the paths are simplified.
        '''
        net = nx.Graph()
        for i in self.true_node_index:
            net.add_node(i, node_type = self.node_type[i-1])
        net.add_edges_from(self.init_edges)
        return net
    
    def opt_redundantNode(self):
        '''
        Remove redundant nodes from the graph.
        '''
        update_con_list_short = []
        edge_opt_before = self.init_edges.shape[0]
        for i in range(1, self.num_of_true_node_index+1):
            for j in range(i+1, self.num_of_true_node_index+1):
                node_i = self.location[i-1,:]
                node_j = self.location[j-1,:]
                distance_i2j = assist_index.distance(node_i, node_j)
                # if two nodes are next to each other, record them.
                if distance_i2j < 1.5:
                    update_con_list_short.append([i,j])
                else:
                    pass
        # build the redundant relationship
        Graph_short = nx.Graph()
        for i in self.true_node_index:
            Graph_short.add_node(i, node_type = self.node_type[i-1])
        Graph_short.add_edges_from(update_con_list_short)

        # find connective subgraph
        for sub_Graph_short in sorted(nx.connected_components(Graph_short), key=len, reverse=True):
            sub_Graph_node_list = list(sub_Graph_short)
            # when the node cluster is able to be simplified
            if len(sub_Graph_node_list) > 1:
                # only keep the first node
                node_keep = sub_Graph_node_list[0]
                node_remove_list = sub_Graph_node_list[1:]
                # for every node needed to be removed
                for node_remove in node_remove_list:
                    # search its location and replace with the node kept
                    self.init_edges[np.where(self.init_edges==node_remove)] = node_keep
                    # change the property from "node" into "edge"
                    self.node_type[node_remove-1] = 1
        # remove double node-kept edges
        self.init_edges = self.init_edges[np.where(self.init_edges[:,0] != self.init_edges[:,1])]

        for i in range(len(self.init_edges)):
            if self.init_edges[i,0] > self.init_edges[i,1]:
                self.init_edges[i] = np.array([self.init_edges[i,1],self.init_edges[i,0]])
        
        self.init_edges = np.unique(self.init_edges, axis=0)
        
        
        opt_before = self.num_of_true_node_index
        self.update_property()
        opt_after = self.num_of_true_node_index
        edge_opt_after = self.init_edges.shape[0]
        print("%.f / %.f nodes were removed during opt_redundantNode"%(opt_before-opt_after, opt_before))
        print("%.f / %.f edges were removed during opt_redundantNode"%(edge_opt_before-edge_opt_after, edge_opt_before))
        self.Graph = self.load_Graph()
    
    def opt_nodeDegree(self):
        '''
        Optimize the nodes with degree == 2.
        '''
        degree_list = list(nx.degree(self.Graph))
        for (i, j) in degree_list:
            if j == 2:
                self.node_type = 1
                
                # update the edge, remove the redundant
                self.update_remove_2degreeNodeEdge(i)
                
            elif j == 0:
                self.node_type = 0
        
        # update the node type
        self.update_node_property()
        
        #self.update_Graph
    def update_remove_2degreeNodeEdge(self, node):
        '''
        If the node degree is 2, this node exsists in the path_list twice.
        '''
        node_location = np.where(self.init_edges==node)
    
    def update_image(self):
        '''
        Update the image with new node_type.
        --------
        node_type: the list of nodes types
        '''
        blank_image = np.zeros(shape=(896,895))
        for (idx, loc) in zip(self.node_type, self.location):
            blank_image[loc[0],loc[1]] = idx
        plt.imshow(blank_image)
        
    def path2coSeq(self, path):
        '''
        Acquire the coordinate sequence of the selected path.
        '''
        seq = []
        for index in path:
            seq.append(self.location[index-1].tolist())
        return np.array(seq)
    
    def edges_Lp(self,sample_interval, Graph):
        '''
        Calculate persistence length for every edges.
        '''
        output = []
        for [i,j] in self.init_edges:
            path = np.array(nx.shortest_path(Graph, i, j))
            nodeSeq = assist_index.yield_nodeSeq(path, sample_interval)
            
            update = []
            for Seq in nodeSeq:
                coSeq = self.path2coSeq(Seq)
                Lp = assist_index.mechanic.Lp(coSeq[0], coSeq[1], coSeq[2])
                update.append(Lp)
            Lp_min = np.nanmin(update)
            #print("Edge(%d, %d): length: %d, Lp: %.2f"%(i,j,len(path), Lp_min))
            output.append([(i,j), Lp_min])
        return output
    
    def edges_len(self, Graph):
        '''
        Calculate length, msrg, e2e for every edges.
        '''
        output = []
        for [i,j] in self.init_edges:
            path = np.array(nx.shortest_path(Graph, i, j))
            coSeq = self.path2coSeq(path)
            
            Seq_len = assist_index.length(coSeq)
            Seq_msrg = assist_index.msrg(coSeq)
            Seq_e2e = assist_index.e2e(coSeq)
            
            output.append([(i,j), Seq_len, Seq_msrg, Seq_e2e])
        return output
class edge_merge:
    @staticmethod
    def degree_check(full_graph, simple_graph, bunch):
        '''
        Get the nodes with wrong branches.
        --------
        full_graph: short connections
        simple_graph: long connections
        '''
        short_degree = np.array(list(nx.degree(full_graph, nbunch=bunch)))
        long_degree = np.array(list(nx.degree(simple_graph, nbunch=bunch)))
        
        long_degree[:,1] = long_degree[:,1] - short_degree[:,1]
        update = long_degree[np.where(long_degree[:,1] > 0)]
        
        return update[:,0]
    
    def find_pairs(node, simple_graph):
        '''
        Generate pairs for selected node.
        '''
        neighbor = list(nx.neighbors(simple_graph, node))
        for neighbor_ in neighbor:
            if node < neighbor_:
                yield (node, neighbor_)
            else:
                yield (neighbor_, node)
    def branch_determine(path1, path2):
        '''
        Conditions for 2 branch paths.
        '''
        # share nodes
        share = [x for x in path1 if x in path2]
        
        # if 4 more nodes are same, it is needless branch.
        if len(share) >= 4:
            if len(path1) >= len(path2):
                return "path1"
            else:
                return "path2"
        else:
            return "True branch"

    def branch_opt(branch_nodes, full_graph, simple_graph):
        '''
        Optimize the branches.
        --------
        branch_nodes: from degree check
        full_graph: used to find path        
        simple_graph: used to generate edges
        '''
        # record the edges to be removed
        remove_edges = []
        # check every node with extra edges
        for branch_node in branch_nodes:
            edges = []
            # collect edges related to the selected branch_node
            for edge in edge_merge.find_pairs(branch_node, simple_graph):
                edges.append(edge)
            # collect paths of these edges.
            paths = [nx.shortest_path(full_graph, edge[0], edge[1]) for edge in edges]
            # combinations path1 - path2
            for path1, path2 in itertools.combinations(paths, 2):
                res = edge_merge.branch_determine(path1, path2)
                if res == "True branch":
                    pass
                elif res == "path1":
                    remove_edges.append((path1[0], path1[-1]))
                elif res == "path2":
                    remove_edges.append((path2[0], path2[-1]))
        return np.array(remove_edges)
class dual_opt:
    def __init__(self, simpleGraph, fullGraph, locations):
        self.normal_network = simpleGraph
        self.full_graph = fullGraph
        self.normal_edges = list(simpleGraph.edges())
        self.normal_nodes = list(simpleGraph.nodes())
        self.normal_edges_index = self.get_edge_index()

        self.location = locations
        # 折叠阈值，大于160的并股视为连续
        self.sinous_threshold = 160
        # 需要修改的边关系
        self.correction = self.look_up_nodes()
    
    # 组建对偶图谱
    def build_dual_graph(self):
        self.dual_nodes = list(range(len(self.normal_edges)))
        self.dual_edges = self.collect_dual_edges()

        Dual = nx.Graph()
        Dual.add_nodes_from(self.dual_nodes)
        Dual.add_edges_from(self.dual_edges)
        self.dual_graph = Dual

    def get_edge_index(self):
        edge_dict = {}
        edge_index = 1
        for edge in self.normal_edges:
            edge_dict[edge_index] = edge
            edge_index += 1
        return edge_dict

    def path2coSeq(self, path):
        seq = []
        for index in path:
            seq.append(self.location[index-1].tolist())
        return seq

    def look_up_nodes(self):
        path_add_list  = []
        # 对所有的节点进行循环
        for i in self.normal_nodes:
            # 先数邻接点
            adj_nodes = list(self.normal_network[i])
            # 如果邻接点数量小于2，则说明为孤立点或端点，跳过不做处理
            if len(adj_nodes) < 2:
                pass
            # 当邻接点数量大于2时
            else:
                # 这些是需要在本个节点优化中需要考虑的边的另一端,制作判断的组合
                clusters = list(itertools.combinations(adj_nodes, 2))
                # 算出两两组合的纤维夹角
                cluster_res = self.look_up_adj_edges(i, clusters)
                
                while len(cluster_res.shape) == 2:
                    if np.max(cluster_res[:,2]) >= 160:
                        max_theta = cluster_res[np.argmax(cluster_res[:,2])]
                        setA = set(adj_nodes)
                        setB = set([max_theta[0], max_theta[1]])
                        adj_nodes = list(setA - setB)

                        path_added = np.array([[i, max_theta[0]], [i, max_theta[1]]])
                        path_add_list.append(path_added)

                        clusters = list(itertools.combinations(adj_nodes, 2))
                        cluster_res = self.look_up_adj_edges(i, clusters)
                    else:
                        break
        path_add_list = np.array(path_add_list)
        path_add_list = np.unique(path_add_list, axis=0)
        return path_add_list

    def look_up_adj_edges(self, i, clusters):
        cluster_res = []
        for cluster in clusters:
            path_1 = nx.shortest_path(self.full_graph, i, cluster[0])
            coSeq_1 = self.path2coSeq(path_1)[:5]

            path_2 = nx.shortest_path(self.full_graph, i, cluster[1])
            coSeq_2 = self.path2coSeq(path_2)[:5]

            vector_1 = np.array(coSeq_1[-1]) - np.array(coSeq_1[0])
            vector_2 = np.array(coSeq_2[-1]) - np.array(coSeq_2[0])

            cost = vector_1.dot(vector_2) / (np.sqrt(vector_1.dot(vector_1)) * np.sqrt(vector_2.dot(vector_2)))

            cluster_res.append([cluster[0],cluster[1], np.around(np.rad2deg(np.arccos(cost)),0)])
        
        return np.array(cluster_res)

    def path2coSeq(self, path):
        seq = []
        for index in path:
            seq.append(self.location[index-1].tolist())
        return seq

    def collect_dual_edges(self):
        # 需要self.normal_edges格式为list
        # self.correction的格式无所谓，主要是能一次性索引到底部的数值就可以
        res_list = []
        for c_ in self.correction:
            c_list = []
            for c in c_:
                c1 = np.sort(c)
                index = self.normal_edges.index((c1[0], c1[1]))
                c_list.append(index)
            res_list.append(c_list)
        return res_list

    def cluster2fiber(self):
        edges = []
        for cluster in list(nx.connected_components(self.dual_graph)):
            node_lf = []
            for edge in cluster:
                node = list(self.normal_network.edges)[edge]
                node_lf.append(node[0])
                node_lf.append(node[1])

            dic = {}
            for key in node_lf:
                dic[key] = dic.get(key,0) + 1

            edge_merge = np.array(list(dic.keys()))[np.array(list(dic.values())) == 1]
            edges.append([edge_merge[0],edge_merge[1]])
        self.opt_edges = edges

class network_graph:
    def __init__(self):
        self.edges = np.array(np.load("Len_list_cut.npy",allow_pickle=True)[:,0].tolist())
        self.nodes_location = np.load("node_list_cut.npy")[:,1:3]
        self.nodes = self.get_nodes()
        
        self.length = np.load("Len_list_cut.npy",allow_pickle=True)[:,1]
        self.msrg = np.load("Len_list_cut.npy",allow_pickle=True)[:,2]
        self.e2e = np.load("Len_list_cut.npy",allow_pickle=True)[:,3]
        self.lp = np.load("Lp_list_cut.npy",allow_pickle=True)[:,1:]
        
        self.node_Graph = self.build_node_Graph()
        self.node_pos = self.get_nodes_pos()
        
        self.node_npos_dual = self.get_dual_pos()
        self.node_list_dual = self.get_dual_node()
        self.edge_dual = self.get_dual_edge()
        
        self.image = np.load("img_skeleton_cut.npy").astype("int")
        
    def get_nodes(self):
        '''
        Collect nodes with 1 edge at least.
        '''
        nodes = self.edges.flatten()
        nodes_set = set(list(nodes))
        return np.array(list(nodes_set))
    
    def get_nodes_pos(self):
        '''
        Collect nodes position.
        '''
        pos = {}
        for node, loc in zip(self.nodes, self.nodes_location):
            pos[node] = loc.tolist()
        return pos
    
    def build_node_Graph(self):
        '''
        Build graph.
        '''
        net = nx.Graph()
        for node_index in self.nodes:
            net.add_node(node_index)
        for edge,length_,msrg_,e2e_,lp_ in zip(self.edges,self.length,self.msrg,self.e2e,self.lp):
            net.add_edge(edge[0], edge[1], 
                         length = length_,
                         msrg = msrg_,
                         e2e = e2e_,
                         lp = lp_)
        return net
            
    def get_dual_pos(self):
        '''
        Get mid of e2e as the pos of dual node.
        '''
        pos = {}
        i = 0
        for edge in self.edges:
            node1, node2 = edge[0], edge[1]
            loc1, loc2 = self.nodes_location[node1-1], self.nodes_location[node2-1]
            locs = ((loc1 + loc2) *0.5).tolist()
            pos[i] = [locs[1],locs[0]]
            i += 1
        return pos   
    
    def get_dual_node(self):
        '''
        Get index for edges.
        '''
        num = self.edges.shape[0]
        return np.arange(num)
    
    def get_dual_edge(self):
        '''
        Get 
        '''
        num = self.edges.shape[0]
        update = []
        # one edge and edges left
        for i in range(num-1):
            edge_inCheck = self.edges[i]
            edge_inList = self.edges[i+1:]
            # two nodes of the edge
            node1, node2 = edge_inCheck[0], edge_inCheck[1]
            index1 = np.where(edge_inList==node1)[0]
            index2 = np.where(edge_inList==node2)[0]
            index = np.hstack([index1,index2])+i+1
            for j in index:
                update.append((i, j))
        return np.array(update)
    
    def build_edge_Graph(self):
        '''
        Build dual graph.
        '''
        node_npos_dual = self.node_npos_dual
        node_list_dual = self.node_list_dual.tolist()
        edge_dual = self.edge_dual.tolist()
        
        net = nx.Graph()
        for node_index_dual in node_list_dual:
            net.add_node(node_index_dual)
        for edge,length_,msrg_,e2e_,lp_ in zip(edge_dual,self.length,self.msrg,self.e2e,self.lp):
            net.add_edge(edge[0], edge[1], 
                         length = length_,
                         msrg = msrg_,
                         e2e = e2e_,
                         lp = lp_)
        return net
    
    def draw_Graph(self):
        plt.imshow(self.image, cmap="binary")
        plt.scatter(self.nodes_location[:,1],self.nodes_location[:,0])
def get_labels(num):
    labels = {}
    for i in num:
        labels[i] = i
    return labels
def get_path_list(fullGraph,simpleGraph):
    update = []
    for [i, j] in list(simpleGraph.edges):
        path = list(nx.shortest_path(fullGraph, i, j))
        update.append(path)
    return update

class small_branch_remove:
    def __init__(self, opt_graph:nx.Graph, npos:dict):
        self.opt_graph = opt_graph
        self.npos = npos
        self.find_degree_one_nodes()
        self.edge_length = self.get_edge_length(edges = self.degree_one_edges)
    def find_degree_one_nodes(self):
        self.degree_one_nodes = [node for node, degree in self.opt_graph.degree() if degree == 1]
        self.degree_one_edges = [(node, neighbor) for node in self.degree_one_nodes for neighbor in self.opt_graph.neighbors(node)]
    def get_edge_length(self, edges):
        length = []
        for node, neighbor in edges:
            node_pos = np.array(self.npos[node])
            neighbor_pos = np.array(self.npos[neighbor])
            length.append(np.linalg.norm(node_pos - neighbor_pos))
        return length
    def find_redundantNodes(self, branchThresh:float):
        delete_nodes = []
        delete_edges = []
        for i, length in enumerate(self.edge_length):
            if length >= branchThresh:
                pass
            else:
                node = self.degree_one_edges[i][0]
                neighbor = self.degree_one_edges[i][1]
                if nx.degree(self.opt_graph, neighbor) > 1:
                    delete_nodes.append(node)
                    delete_edges.append(self.degree_one_edges[i])
        return delete_nodes, delete_edges
    def return_opt_graph(self, delete_edges):
        self.new_opt_graph = self.opt_graph.copy()
        for edge in delete_edges:
            self.new_opt_graph.remove_edge(edge[0], edge[1])
        print(f"Small branches removed: {len(delete_edges)}")

class multi_strand_discover:
    def __init__(self, fullGraph:nx.Graph, simpleGraph:nx.Graph, npos:dict, mask):
        self.fullGraph = fullGraph
        self.simpleGraph = simpleGraph
        self.npos = npos
        self.mask = mask
        self.mask_dist = self.return_distance_img()
        self.path_list = self.index2coord_transform()
    
    def return_distance_img(self):
        return ndimage.distance_transform_edt(self.mask)
    
    def calculate_thickness(self, one_path:list):
        thick = []
        for (y, x) in one_path:
            thick.append(self.mask_dist[y,x])
        return np.mean(np.array(thick))
    
    def calculate_all_thickness(self):
        thicks = []
        for path in self.path_list:
            thicks.append(self.calculate_all_thickness(path))
        return thicks

    def index2coord_transform(self):
        path_list = []
        for i, edge in enumerate(self.simpleGraph.edges):
            index_path = nx.shortest_path(self.fullGraph, edge[0], edge[1])
            coord_path = [self.npos[x] for x in index_path]
            path_list.append(coord_path)
        return path_list
class multi_strand_discover:
    def __init__(self, fullGraph:nx.Graph, simpleGraph:nx.Graph, npos:dict, mask, max_display = 1):
        self.fullGraph = fullGraph
        self.simpleGraph = simpleGraph
        self.npos = npos
        self.mask = mask
        self.max_display = max_display
        self.mask_dist = self.return_distance_img()
        self.path_list = self.index2coord_transform()
    
    def return_distance_img(self):
        return ndimage.distance_transform_edt(self.mask)
    
    def calculate_thickness(self, one_path:list):
        thick = []
        for (x, y) in one_path:
            thick.append(self.mask_dist[y,x])
        return np.mean(np.array(thick))
    
    def calculate_all_thickness(self):
        self.thicks = []
        for path in self.path_list:
            self.thicks.append(self.calculate_thickness(path))
        return self.thicks

    def index2coord_transform(self):
        path_list = []
        for i, edge in enumerate(self.simpleGraph.edges):
            index_path = nx.shortest_path(self.fullGraph, edge[0], edge[1])
            coord_path = [self.npos[x] for x in index_path]
            path_list.append(coord_path)
        return path_list

    def determine_bunch(self, bunchThresh):
        self.bunch_pair = []
        median_thick = np.median(np.array(self.thicks))
        for msd_edge, msd_thick in zip(self.simpleGraph.edges, self.thicks):
            bunch = msd_thick / median_thick
            if bunch > bunchThresh:
                self.bunch_pair.append([msd_edge, int(bunch)])
        print(f"{len(self.bunch_pair)} bunch pair found based on bunchTresh={bunchThresh}")
    def generate_displacement(self):
        random_numbers = np.random.uniform(0, 1, 2)
        sign = np.random.choice([-1, 1])
        random_numbers = random_numbers * sign
        sorted_numbers = np.sort(random_numbers)
        return sorted_numbers*self.max_display
    def generate_path(self, edge):
        nodeA, nodeB = edge[0], edge[1]
        original_path = nx.shortest_path(self.fullGraph, nodeA, nodeB)
        original_path = [self.npos[x] for x in original_path]
        direction_vector = np.array(original_path[-1]) - np.array(original_path[0])
        normal_vector = np.array([-direction_vector[1], direction_vector[0]])
        normal_vector = normal_vector / np.linalg.norm(normal_vector)
        displacement_range = self.generate_displacement()
        new_path = []
        for point in original_path:
            # 对于起点和终点，坐标保持不变
            if point in [original_path[0], original_path[-1]]:
                new_path.append(point)
            else:
                # 对于其他点，根据法向量和随机位移生成新的坐标
                displacement = np.random.uniform(*displacement_range)
                new_point = point + displacement * normal_vector
                new_path.append(new_point.tolist())
        return new_path
    def generate_bunch(self, mode = "auto"):
        '''
        "auto": generate = bunch
        "ony": generate = 1
        '''
        self.add_bunch_edge = []
        loop = tqdm(total = len(self.bunch_pair))
        for edge, bunch in self.bunch_pair:
            loop.update(1)
            new_path = self.generate_path(edge)
            self.add_bunch_edge.append(new_path)
            if mode == "auto":
                for i in range(bunch):
                    self.add_bunch_edge.append(self.generate_path(edge))
            elif mode == "only":
                pass
            elif mode == "rough":
                if bunch >= 2:
                    self.add_bunch_edge.append(self.generate_path(edge))
            elif type(mode) == int:
                for i in range(mode):
                    self.add_bunch_edge.append(self.generate_path(edge))
            else:
                print(f"Invalid mode: {mode}")
        loop.close()
        print(f"{len(self.add_bunch_edge)} generated")

    def smooth_coordinate(self, coordinates, iters):
        for _ in range(iters):
            new_coordinates = coordinates.copy()
            for i in range(1, len(coordinates)-1):
                new_coordinates[i] = (coordinates[i-1] + coordinates[i] + coordinates[i+1]) / 3
                coordinates = new_coordinates
        return coordinates
    
    def draw_network(self, iters):
        plt.figure(figsize=(10,10),dpi=300)
        for edge in self.simpleGraph.edges():
            index_path = nx.shortest_path(self.fullGraph, edge[0], edge[1])
            coord_path = np.array([self.npos[x] for x in index_path])
            coord_path = self.smooth_coordinate(coord_path, iters = iters[0])
            raw_plotx, raw_ploty = zip(*coord_path)
            #plt.plot(raw_plotx, raw_ploty, color = "steelblue",linewidth=2)
            #plt.scatter(raw_plotx, raw_ploty, color = "steelblue")
            plt.scatter(coord_path[0][0], coord_path[0][1], color="maroon")
            plt.scatter(coord_path[-1][0], coord_path[-1][1], color="maroon")
        for edge in self.add_bunch_edge:
            edge = np.array(edge)
            edge = self.smooth_coordinate(edge, iters = iters[1])
            raw_plotx, raw_ploty = zip(*edge)
            plt.plot(raw_plotx, raw_ploty, color = "orchid",linewidth=1.5)
            #plt.scatter(raw_plotx, raw_ploty, color = "orchid")
        plt.axis("equal")
        plt.axis("off")
        plt.gca().invert_yaxis()
        plt.show()

class lmp_write:
    def __init__(self, msd_main:multi_strand_discover, bunch, msd_bunch:multi_strand_discover):
        self.msd_main = msd_main
        self.bunch = bunch
        if bunch == True:
            self.msd_bunch = msd_bunch

        self.msd_main.calculate_all_thickness()
        self.msd_main.determine_bunch(bunchThresh = 0)
        self.msd_main.generate_bunch(mode="only")

        if bunch == True:
            self.msd_bunch.calculate_all_thickness()
            self.msd_bunch.determine_bunch(bunchThresh = 1)
            self.msd_bunch.generate_bunch(mode=2)

        self.node_with_index = self.collect_node()

    def smooth_coordinate(self, coordinates, iters):
        for _ in range(iters):
            new_coordinates = coordinates.copy()
            for i in range(1, len(coordinates)-1):
                new_coordinates[i] = (coordinates[i-1] + coordinates[i] + coordinates[i+1]) / 3
                coordinates = new_coordinates
        return coordinates

    def collect_node(self):
        node = set()
        for edge in self.msd_main.add_bunch_edge:
            node.add(tuple(edge[0]))
            node.add(tuple(edge[-1]))
        node_list = [list(point) for point in node]
        self.node_num = len(node_list)
        node_with_index = {i+1:point for i, point in enumerate(node_list)}
        return node_with_index

    def find_node_index(self, point):
        for index, endpoint in self.node_with_index.items():
            if (endpoint == point).all():
                return index

    def collect_atom_bond_information(self):
        self.atom_information = []
        self.bond_information = []
        atom_index = self.node_num  # 使用前先+1
        bond_index = 0              # 使用前先+1

        # 先加入已经确定好的端点
        for index, endpoint in self.node_with_index.items():
            # index, molecule-id, type, x, y, z
            self.atom_information.append([index, 1, 1, endpoint[0], endpoint[1], 0])
        # 2写入主干路径
        for edge in self.msd_main.add_bunch_edge:
            edge = np.array(edge)
            edge = self.smooth_coordinate(edge, iters = 1)
            # 存储路径上的点的索引
            edge_node_index = []
            edge_node_index.append(self.find_node_index(edge[0])) # 写入起点
            # 写入新的点坐标
            for node in edge[1:-1]:
                atom_index += 1
                self.atom_information.append([atom_index, 1, 1, node[0], node[1], 0])
                edge_node_index.append(atom_index)
            edge_node_index.append(self.find_node_index(edge[-1])) # 写入终点
            # 写入边关系
            for node_index in range(len(edge_node_index)-1):
                bond_index += 1
                self.bond_information.append([bond_index, 1, edge_node_index[node_index], edge_node_index[node_index+1]])
        # 3写入并股路径
        if self.bunch == True:
            for edge in self.msd_bunch.add_bunch_edge:
                edge = np.array(edge)
                edge = self.smooth_coordinate(edge, iters = 1)
                # 存储路径上的点的索引
                edge_node_index = []
                edge_node_index.append(self.find_node_index(edge[0])) # 写入起点
                # 写入新的点坐标
                for node in edge[1:-1]:
                    atom_index += 1
                    self.atom_information.append([atom_index, 1, 1, node[0], node[1], 0])
                    edge_node_index.append(atom_index)
                edge_node_index.append(self.find_node_index(edge[-1])) # 写入终点
                # 写入边关系
                for node_index in range(len(edge_node_index)-1):
                    bond_index += 1
                    self.bond_information.append([bond_index, 1, edge_node_index[node_index], edge_node_index[node_index+1]])

        self.atom_nums = atom_index
        self.bond_nums = bond_index

    def write_lmp_datafile(self, path, box_range = [-5, 305, -5, 605, -5, 5]):
        with open(path, "w", newline='\r\n') as f:
            f.write("#LAMMPS data file\n\n")
            f.write(f"{self.atom_nums} atoms\n")
            f.write(f"{self.bond_nums} bonds\n\n")
            f.write("1 atom types\n")
            f.write("1 bond types\n\n")
            f.write(f"{box_range[0]:.2f} {box_range[1]:.2f} xlo xhi\n")
            f.write(f"{box_range[2]:.2f} {box_range[3]:.2f} ylo yhi\n")
            f.write(f"{box_range[4]:.2f} {box_range[5]:.2f} zlo zhi\n\n")
            f.write("Atoms\n\n")
            for atom in self.atom_information:
                fwrite = f"{atom[0]} {atom[1]} {atom[2]} {atom[3]:.2f} {atom[4]:.2f} {atom[5]:.2f}\n"
                f.write(fwrite)
            f.write("\n")
            f.write("Bonds\n\n")
            for bond in self.bond_information:
                fwrite = f"{bond[0]} {bond[1]} {bond[2]} {bond[3]}\n"
                f.write(fwrite)