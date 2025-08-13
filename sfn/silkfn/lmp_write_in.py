import numpy as np
from scipy import ndimage
from tqdm.auto import tqdm
import networkx as nx
class multi_strand_discover:
    def __init__(self, 
                 fullGraph, 
                 simpleGraph, 
                 npos, 
                 mask, 
                 mode = "auto",
                 max_display = 1):
        self.fullGraph = fullGraph
        self.simpleGraph = simpleGraph
        self.npos = npos
        self.mask = mask
        self.mode = mode
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
        median_thick = np.mean(np.array(self.thicks))
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
    def generate_bunch(self):
        '''
        "auto": generate = bunch
        "ony": generate = 1
        '''
        self.add_bunch_edge = []
        if self.mode == False:
            pass
        else:
            loop = tqdm(total = len(self.bunch_pair))
            for edge, bunch in self.bunch_pair:
                loop.update(1)
                new_path = self.generate_path(edge)
                self.add_bunch_edge.append(new_path)
                if self.mode == "auto":
                    for i in range(bunch):
                        self.add_bunch_edge.append(self.generate_path(edge))
                elif self.mode == "only":
                    pass
                elif self.mode == "rough":
                    if bunch >= 2:
                        self.add_bunch_edge.append(self.generate_path(edge))
                elif type(self.mode) == int:
                    for i in range(self.mode-1):
                        self.add_bunch_edge.append(self.generate_path(edge))
                else:
                    print(f"Invalid mode: {self.mode}")
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
            #plt.scatter(coord_path[0][0], coord_path[0][1], color="maroon")
            #plt.scatter(coord_path[-1][0], coord_path[-1][1], color="maroon")
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
    def __init__(self, msd_main, bunch, msd_bunch):
        self.msd_main = msd_main
        self.bunch = bunch
        if bunch == True:
            self.msd_bunch = msd_bunch

        self.msd_main.calculate_all_thickness()
        self.msd_main.determine_bunch(bunchThresh = 0)
        self.msd_main.generate_bunch()

        if bunch == True:
            self.msd_bunch.calculate_all_thickness()
            self.msd_bunch.determine_bunch(bunchThresh = 1)
            self.msd_bunch.generate_bunch()

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