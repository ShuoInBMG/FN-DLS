import numpy as np

class traj_multistrand:
    def __init__(self, 
                 traj_frame_arr, 
                 bond_info,
                 path_info):
        self.traj_frame_arr = traj_frame_arr  # Shape: (num_timesteps, num_nodes, 2)
        self.bond_info = bond_info.astype(int)  # Shape: (num_edges, 2)
        self.main_node, self.edge_node = self.get_node_degree_three()
        self.traj_frame_nums = self.traj_frame_arr.shape[0]
        self.path_info = path_info
    def get_node_degree_three(self):
        unique_elements, counts = np.unique(self.bond_info, return_counts=True)
        main_node = unique_elements[counts >= 3]
        main_node = main_node.astype(int)
        edge_node = unique_elements[counts <= 2]
        edge_node = edge_node.astype(int)
        return main_node, edge_node
    def collect_main_frame_arr(self):
        timestep_index = np.arange(self.traj_frame_nums)[:, None, None]
        main_node = self.main_node[None,:,None]-1
        xylocation_index = np.arange(2)[None, None, :]

        return self.traj_frame_arr[timestep_index, main_node, xylocation_index]

    def collect_main_distance(self):
        main_location = self.collect_main_frame_arr()

        distances = np.sqrt(np.sum((main_location[1:] - main_location[:-1])**2, axis=-1))
        total_distances = np.mean(distances, axis=1)
        return total_distances

def group_connected_paths(connected_paths):
    grouped_paths = []
    current_group = [connected_paths[0]]

    for i in range(1, len(connected_paths)):
        if connected_paths[i][0] == current_group[0][0] and connected_paths[i][-1] == current_group[0][-1] and len(connected_paths[i]) == len(current_group[0]):
            current_group.append(connected_paths[i])
        else:
            if len(current_group) > 1:
                grouped_paths.append(current_group)
            current_group = [connected_paths[i]]

    if len(current_group) > 1:
        grouped_paths.append(current_group)

    return grouped_paths

class detect_motion:
    def __init__(self, full_path_list, frame_arr, total_node):
        self.full_path_list = full_path_list  # use multistrand paths
        self.frame_arr = frame_arr
        self.timesteps = frame_arr.shape[0]

        collect_informations = []
        collect_directions = []
        collect_distance = []
        for time in range(self.timesteps):
            coorslice = self.frame_arr[time, :, :]  # extract coordinates at each timestep
            shift, direction, distance = self.calculate_information_at_one_timestep(coorslice)   # calculate shift
            collect_informations.append(shift)
            collect_directions.append(direction)
            collect_distance.append(distance)
        
        self.time_shift_differ = self._detect_differ_of_shift(collect_informations)
        self.time_shift_direct = collect_directions[:-1]
        self.time_shift_distance = self._detect_differ_of_distance(collect_distance)
        # time_shift_differ shape (time-1, )
        self.slip_count_sequence = []  # 时间序列
        self.tost_count_sequence = []  # 时间序列
        self.fixd_count_sequence = []  # 时间序列
        for time in range(self.timesteps-1):
            slip_count_at_one_time = [] # 组列表
            tost_count_at_one_time = [] # 组列表
            # 在每个时间步内对187组并股计算，每一组遍历是shifts和direction的组合
            for shifts, direction in zip(self.time_shift_differ[time], self.time_shift_direct[time]):
                for shift in shifts:
                    # shift的形状应该为(length, x-y 2)
                    slip_count = self._detect_projection(shift, direction)
                    slip_count_at_one_time += slip_count  # 里面是展平的标记编号
            self.slip_count_sequence.append(slip_count_at_one_time)
            for distances in self.time_shift_distance[time]:
                for distance in distances:
                    to_count = self._detect_disassemble(distance)
                    tost_count_at_one_time += to_count
            self.tost_count_sequence.append(tost_count_at_one_time)
            
            self.fixd_count_sequence.append([1 if x==0 and y==0 else 0 for x,y in zip(slip_count_at_one_time, tost_count_at_one_time)])

        if total_node > 0:
            self.results = []
            for time in range(self.timesteps-1):
                assert len(self.slip_count_sequence[time]) == total_node
                assert len(self.tost_count_sequence[time]) == total_node
                assert len(self.fixd_count_sequence[time]) == total_node
                self.results.append([self.slip_count_sequence[time].count(1)/total_node,
                                     self.tost_count_sequence[time].count(1)/total_node,
                                     self.tost_count_sequence[time].count(-1)/total_node,
                                     self.fixd_count_sequence[time].count(0)/total_node])
            self.results = np.array(self.results)
    def calculate_information_at_one_timestep(self, coorslice):
        collect_shift_at_one_time = []
        collect_direction_at_one_time = []
        collect_distance_at_one_time = []
        for path_list in self.full_path_list:
            coord = []
            for path in path_list:
                coord.append([coorslice[i-1] for i in path])
            coord = np.array(coord) # shape (num of fibers, length, 2 x-y)
            shift, direction, distance= self._detect_shift(coord)  # 对每个单独的路径
            collect_shift_at_one_time.append(shift)
            collect_direction_at_one_time.append(direction)
            collect_distance_at_one_time.append(distance)  # 对一束并股
        return collect_shift_at_one_time, collect_direction_at_one_time, collect_distance_at_one_time
    
    def _detect_shift(self, coord):
        coord_shift = []
        distance_shift = []
        standard_fiber = coord[0]
        strand_fibers = coord[1:]
        direction = self._detect_direction(standard_fiber)
        for strand_fiber in strand_fibers:
            # standard_fiber  (length, 2 x-y)
            # strand_fiber (length, 2 x-y)
            shift_diff_vector = strand_fiber - standard_fiber
            distance_shift.append(np.linalg.norm(shift_diff_vector, axis=1))
            coord_shift.append(shift_diff_vector)
        return np.array(coord_shift), np.array(direction),np.array(distance_shift)
    
    def _detect_differ_of_shift(self, coord_sequence):
        collect_shift_differ = []
        for time in range(len(coord_sequence)-1):
            #shift_diff_2 = coord_sequence[time + 1] - coord_sequence[time]
            shift_diff_2 = [x-y for x,y in zip(coord_sequence[time + 1],coord_sequence[time])]
            collect_shift_differ.append(shift_diff_2)
        return collect_shift_differ
    def _detect_differ_of_distance(self, collect_distance):
        collect_distance_differ = []
        for time in range(len(collect_distance)-1):
            distance_diff = [x-y for x,y in zip(collect_distance[time + 1], collect_distance[time])]
            collect_distance_differ.append(distance_diff)
        return collect_distance_differ

    def _detect_direction(self, standard_fiber):
        start = standard_fiber[0]
        end = standard_fiber[1]
        return end - start
        
    def _detect_projection(self, shift, direction):
        # shift (length, x-y 2d)
        # direction (x, y)
        slip_count = []
        # 遍历 shift 中的矢量
        for i in range(1, len(shift) - 1):
            vector = shift[i]
            projection = np.dot(vector, direction) / np.dot(direction, direction) * direction
            projection_distance = np.linalg.norm(projection)
            # 如果投影距离大于1，增加 slip_count,1个1元素
            if projection_distance > 1:
                slip_count.append(1)
            # 如果投影距离小于这个值，增加一个1元素代表没有移动
            else:
                slip_count.append(0)
        return slip_count

    def _detect_disassemble(self, distance):
        count = []
        for i in range(1, len(distance)-1):
            diff = distance[i]
            if diff > 0.5:
                count.append(1)
            elif diff < -0.5:
                count.append(-1)
            else:
                count.append(0)
        return count

def get_strand_node_number(full_path_list):
    count = 0
    for paths in full_path_list:
        count_paths = paths[1:]
        for path in count_paths:
            count += (len(path) - 2)
    return count