def remove_redundant_nodes(net):
    degree_list = list(nx.degree(net))
    for (i, j) in degree_list:
        if j == 2:
            net.nodes()[i]['node_kind'] = 1
        elif j == 0:
            net.nodes()[i]['node_kind'] = 1
    return net

def zipped_list(net, edge_list):
    zip_list = []
    order = 1
    for (i,j) in edge_list:
        path = nx.shortest_path(net, i, j)
        seq, weight = path_to_seq(path)
        zip_list.append([order, (i,j), tuple(seq), 1])
        order += 1
    return zip_list

def share(zip_list, i, j):
    pair_i = zip_list[i][1]
    pair_j = zip_list[j][1]

    i_1, i_2 = pair_i[0], pair_i[1]
    j_1, j_2 = pair_j[0], pair_j[1]

    if i_1 == j_1:
        return True, zip_list[i][2], zip_list[j][2], 1
    elif i_1 == j_2:
        return True, zip_list[i][2], list(reversed(zip_list[j][2])), 1
    elif i_2 == j_1:
        return True, list(reversed(zip_list[i][2])), zip_list[j][2], 2
    elif i_2 == j_2:
        return True, list(reversed(zip_list[i][2])), list(reversed(zip_list[j][2])), 2
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
            combine_list.append(order_list[a], order_list[b])
    return combine_list

def output_zip_list(zip_list):
    output_list = []
    for zip_seq in zip_list:
        if zip_seq[3] == 1:
            output_list.append(zip_seq)
    return output_list

def edge_connect(npos, zip_list_in):
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
            if (cos_theta<-0.95) or (cos_theta>0.95):
                # Check the number of nodes
                common_node_index = zip_list[i][1][common_node-1]
                set_1 = set([zip_list[i][1][0],
                             zip_list[i][1][-1],
                             zip_list[j][1][0],
                             zip_list[j][1][-1]])
                set_2 = set([zip_list[i][1][common_node-1]])
                new_pair = tuple(set_1 - set_2)
                # The normal condition, two shares one common node.
                if len(new_pair) == 2:
                    net.nodes()[common_node_index]['node_kind'] = 1
                    # Mark the 2 paths in zip_list.
                    zip_list[i][3] = 0
                    zip_list[j][3] = 0
                    # Collect the edge connections.
                    edge_conlist.append(i,j)
                # The two nodes of the two paths are same.
                elif len(new_pair) == 1:
                    len_i = len(zip_list[i][2])
                    len_j = len(zip_list[j][2])
                    if len_i <= len_j:
                        zip_list[j][3] = 0
                    else:
                        zip_list[i][3] = 0
    print(len(edge_conlist),'连接组')
    print(len(output),'孤立线段')
    return zip_list, edge_conlist

def remove_polyline(zip_list):
    check = []
    order = 1
    for zip_seq in zip_list:
        cnt_length = mc.calLen(zip_seq[2])
        e2e_length = mc.e2e(zip_seq[2])
        if (cnt_length - e2e_length * 1.2) > 0:
            pass
        else:
            check.append([order, zip_seq[1], zip_seq[2], zip_seq[3]])
            order += 1
    return check