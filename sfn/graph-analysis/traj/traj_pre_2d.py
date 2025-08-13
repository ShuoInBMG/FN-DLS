import numpy as np
from tqdm.auto import tqdm

class xyz_traj_handle:
    def __init__(self, xyz_file_path, timestep):
        self.read_xyz_file(xyz_file_path)
        self.timestep = timestep
    def read_xyz_file(self, xyz_file_path):
        with open(xyz_file_path, 'r') as f:
            self.xyz_file = f.readlines()
            f.close()
    
        self.atom_nums = int(self.xyz_file[3][:-1])
        self.slice_len = self.atom_nums + 9
        self.frames = int(len(self.xyz_file) / self.slice_len)
    def collect_numbers(self, sentence):
        number_sentence = sentence[:-1].split()
        number_sentence = [float(x) for x in number_sentence]
        return number_sentence
    def detect_box_bounds(self, bound_slice):
        print(bound_slice)
        dimensions = []
        for line in bound_slice:
            start, end = map(float, line.split())
            dimensions.append((start, end))
        return dimensions
    def denormalize_coordinates(self, matrix, dimensions):
        for i in range(2, 5):
            start, end = dimensions[i-2]
            matrix[:, i] = matrix[:, i] * (end - start) + start
        return matrix
    def extract_frames(self):
        # seperate frames according to timestep
        frame_matrix = []
        for timestep in range(0, self.frames, self.timestep):
            timestep_start = timestep * self.slice_len
            timestep_end = timestep_start + self.slice_len
            bound_slice = self.xyz_file[timestep_start+5 : timestep_start+8]
            dimensions = self.detect_box_bounds(bound_slice)
            slices = self.xyz_file[timestep_start:timestep_end]
            coord_matrix = []
            for coords in slices[9:]:
                coord_matrix.append(self.collect_numbers(coords))
            sorted_coords = sorted(coord_matrix, key = lambda x:x[0])
            sorted_coords = np.array(sorted_coords)
            sorted_coords = sorted_coords[sorted_coords[:,1] != 3]
            sorted_coords_de = self.denormalize_coordinates(sorted_coords, dimensions)
            frame_matrix.append(sorted_coords_de)
        self.frame_arr = np.array(frame_matrix)
        del frame_matrix
        del self.xyz_file
        print(f"Traj: frame nums {self.frames//self.timestep}")
class xyz_data_handle:
    def __init__(self, data_file_path):
        self.read_data_file_path(data_file_path)

        self.atom_nums = int(self.xyz_file[2][:-7])
        self.bond_nums = int(self.xyz_file[3][:-7])
    
    def read_data_file_path(self, data_file_path):
        with open(data_file_path, 'r') as f:
            self.xyz_file = f.readlines()
            f.close()

    def collect_numbers(self, sentence):
        number_sentence = sentence[:-1].split()
        number_sentence = [float(x) for x in number_sentence]
        return number_sentence
    
    def extract_informations(self):
        self.read_atom_information()
        self.read_bond_information()
    
    def read_atom_information(self):
        atom_information = []
        for line in range(14, 14 + self.atom_nums):
            atom = self.collect_numbers(self.xyz_file[line])
            atom_information.append(atom)
        atom_information = np.array(atom_information)
        self.atom_information = atom_information[:,[3,4]]
        print(f"Traj: atom nums {self.atom_information.shape[0]}")
    
    def read_bond_information(self):
        bond_information = []
        for line in range(17 + self.atom_nums, 17 + self.atom_nums + self.bond_nums):
            bond = self.collect_numbers(self.xyz_file[line])
            bond_information.append(bond)
        bond_information = np.array(bond_information)
        self.bond_information = bond_information[:,[2,3]]
        self.bond_information = self.bond_information - 1  # node index start from 0, not 1 like lammps
        print(f"Traj: bond nums {self.bond_information.shape[0]}")

class traj_multistrand:
    def __init__(self, traj_frame_arr, bond_info):
        self.traj_frame_arr = traj_frame_arr  # Shape: (num_timesteps, num_nodes, 2)
        self.bond_info = bond_info  # Shape: (num_edges, 2)
        self.main_node, self.edge_node = self.get_node_degree_three()
        self.get_all_if_pair()

        self.traj_frame_nums = self.traj_frame_arr.shape[0]

    def get_node_degree_three(self):
        unique_elements, counts = np.unique(self.bond_info, return_counts=True)
        main_node = unique_elements[counts >= 3]
        main_node = main_node.astype(int)
        edge_node = unique_elements[counts <= 2]
        edge_node = edge_node.astype(int)
        return main_node, edge_node
    
    def check_pair(self, pair):
        for row in self.bond_info:
            if np.array_equal(row, pair):
                return True
        return False
    
    def get_if_pair(self, node):
        if_pair = []
        for edge_node in self.edge_node:
            # 整理大小顺序
            if node < edge_node:
                pair = (node, edge_node)
            else:
                pair = (edge_node, node)
            # 如果这对不属于键对
            if not self.check_pair(pair):
                if_pair.append(pair)
        return if_pair
    
    def get_all_if_pair(self):
        self.if_pair_all = {}
        for node in tqdm(self.edge_node):
            if_pair = self.get_if_pair(node)
            self.if_pair_all[node] = if_pair
    
    def calculate_strand_length_for_this_timestep(self, frame_slice_in_frame_arr):
        collect_min_length = []
        for if_pairs in tqdm(self.if_pair_all.values()):
            min_length = 10
            for pair in if_pairs:   # if_pairs = [(1,2), (0,2)]
                node_a = frame_slice_in_frame_arr[pair[0]]
                node_b = frame_slice_in_frame_arr[pair[1]]
                length = np.linalg.norm(node_a - node_b)
                if length < min_length:
                    min_length = length
            collect_min_length.append(min_length)
        collect_min_length = np.array(collect_min_length)
        mean_length = np.mean(collect_min_length)
        return mean_length
    
    def calculate_strand_length_for_all_timestep(self):
        collect_mean_length = []
        for i in range(self.traj_frame_nums):
            mean_length = self.calculate_strand_length_for_this_timestep()
            collect_mean_length.append(mean_length)
        return collect_mean_length

                    



            

