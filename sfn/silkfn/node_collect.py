import numpy as np

def get_structure(img):
    update = []
    i = 0
    y_loc, x_loc = np.where(img==1)
    for y,x in zip(y_loc, x_loc):
        i += 1
        update.append([i, y, x, 1])
    return np.array(update)

class network_node_refine:
    def __init__(self,structure,skeleton,shape):
        self.node_index = np.load(structure)[:,0].astype("int")
        self.locations = np.load(structure)[:,1:3].astype("int")
        self.type = np.load(structure)[:,3].astype("int")
        self.image = np.load(skeleton).astype("int")
        self.region = np.pad(self.image, ((1,1),(1,1)), "constant")
        
        self.junction_condition = self.line_junctions()
        
        self.blank_image = np.zeros(shape=shape)
        self.index_image = np.zeros(shape=shape)
        
    def find_junctions_in3x3(self):
        '''
        Search nodes in 3x3 neighbours.
        '''
        update = []
        for index in self.node_index:
            region = self.map_generator(self.locations[index-1])

            if self.is_terminal(region) is True:
                self.type[index-1] = 1
                update.append(1)
            else: 
                if self.is_junction(region) is True:
                    self.type[index-1] = 1
                    update.append(1)
                else:
                    self.type[index-1] = 0
                    update.append(2)
        return update
    
    def map_generator(self, location):
        '''
        Generate the array slice for selected node.
        --------
        location: the coordination [y,x] of the selected node
        '''
        [y,x] = location
        now = self.region[y+1,x+1]
        region_ = self.region[y:y+3, x:x+3]
        return region_
    
    def index_map_generator(self, index_map_expand, loc_y, loc_x):
        '''
        Generate the array slice for selected node index.
        --------
        loc_y, loc_x: coordinates
        '''
        return index_map_expand[loc_y:loc_y+3, loc_x:loc_x+3]
    
    def is_terminal(self, region):
        '''
        Determine whether the selected pixel belongs to terminals.
        --------
        region: 3x3 slice of the selected pixel
        '''
        if np.sum(region) == 2:
            return True
        else:
            return False
    
    def is_junction(self, region):
        '''
        Determine whether the selected pixel belongs to junctions.
        --------
        region: 3x3 slice of the selected pixel
        '''
        cud = region.flatten()
        cud_check = np.array([np.all((cud - x) >=0 ) for x in self.junction_condition])
        cud_check_ = np.any(cud_check==1)
        if cud_check_ == False:
            return False
        else:
            return True
        
    def line_junctions(self):
        '''
        Generate conditions of line junctions.
        '''
        c1 = np.array([
                       [1,0,1,0,1,0,0,1,0],
                       [0,1,0,0,1,1,1,0,0],
                       [0,0,1,1,1,0,0,0,1],
                       [1,0,0,0,1,1,0,1,0],
                       [0,1,0,0,1,0,1,0,1],
                       [0,0,1,1,1,0,0,1,0],
                       [1,0,0,0,1,1,1,0,0],
                       [0,1,0,1,1,0,0,0,1],
                       [1,0,0,0,1,0,1,0,1],
                       [1,0,1,0,1,0,1,0,0],
                       [1,0,1,0,1,0,0,0,1],
                       [0,0,1,0,1,0,1,0,1],
                       [0,1,0,1,1,1,0,0,0],
                       [0,1,0,0,1,1,0,1,0],
                       [0,0,0,1,1,1,0,1,0],
                       [0,1,0,1,1,0,0,1,0]
                      ])
        return c1
    
    def update_image(self):
        '''
        Update the image with new node_type.
        --------
        node_type: the list of nodes types
        '''
        for (idx, loc) in zip(self.type, self.locations):
            self.blank_image[loc[0],loc[1]] = idx+1
    
    def find_node(self):
        '''
        Main function fr finding nodes at the skeleton.
        '''
        node_type = self.find_junctions_in3x3()
        self.update_image()
        return node_type
    
    def node_list_refine(self):
        '''
        Update node information after node finding.
        '''
        node_list = []
        # node first
        loc_y, loc_x = np.where(self.blank_image==2)
        
        count = 0
        
        for y,x in zip(loc_y, loc_x):
            count += 1
            node_list.append([count, y, x, 2])
            self.index_image[y, x] = count
        
        # edge then
        loc_y, loc_x = np.where(self.blank_image==1)
        
        for y,x in zip(loc_y, loc_x):
            count += 1
            node_list.append([count, y, x, 1])
            self.index_image[y, x] = count
        
        return np.array(node_list)
    
    def connection_refine(self, node_list, return_dict = False):
        '''
        Check the adjacent connection in 3x3.
        --------
        node_list: nodes list acquired after refinement
        '''
        update = {}
        con_list = []
        # expand index map
        index_map = np.pad(self.index_image, ((1,1),(1,1)), "constant")
        # loop for nodes
        for index in node_list[:,0]:
            # get the location
            loc_y, loc_x = node_list[index-1, 1:3]
            # get the slice array of seleted node -- to a 1-D array
            index_region = self.index_map_generator(index_map, loc_y, loc_x).flatten()
            # find adjacent index > selected node. the background is excluded for = 1. the duplicated is excluded as well
            adjacent_list = index_region[np.where(index_region > index)]
            adjacent_all = index_region[np.where(index_region > 0)]
            
            for adjacent_index in adjacent_list:
                con_list.append([index, adjacent_index])
            
            update[index] = adjacent_all.tolist()
        
        if return_dict == False:
            return np.array(con_list).astype("int")
        else:
            return np.array(con_list).astype("int"), update