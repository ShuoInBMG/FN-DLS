import numpy as np
def calculate_path_length(coords):
    # 将坐标转换为 NumPy 数组
    coords_array = np.array(coords)
    
    # 计算相邻点对之间的差值
    diff = np.diff(coords_array, axis=0)
    
    # 计算差值的平方
    diff_squared = diff ** 2
    
    # 计算每对点之间的欧几里得距离的平方
    dist_squared = np.sum(diff_squared, axis=1)
    
    # 开方得到欧几里得距离
    dist = np.sqrt(dist_squared)
    
    # 计算总路径长度
    total_length = np.sum(dist)
    return total_length

def calculate_centroid(coords):
    """计算坐标列表的质心（平均值）。"""
    return np.mean(coords, axis=0)

def calculate_curvature_radius(coords):
    """计算路径的回转半径。"""
    # 将坐标转换为 NumPy 数组
    coords_array = np.array(coords)
    
    # 计算质心
    centroid = calculate_centroid(coords_array)
    
    # 计算每个点到质心的距离平方
    dist_squared = np.sum((coords_array - centroid) ** 2, axis=1)
    
    # 计算距离平方的平均值，即回转半径的平方
    mean_dist_squared = np.mean(dist_squared)
    
    # 返回回转半径，即距离平方的平均值的平方根
    return np.sqrt(mean_dist_squared)
def calculate_endpoint_distance(coords):
    """计算路径首尾两个端点之间的距离。"""
    # 将坐标转换为 NumPy 数组
    coords_array = np.array(coords)
    
    # 获取首尾两个端点
    start_point = coords_array[0]
    end_point = coords_array[-1]
    
    # 计算首尾端点之间的距离
    distance = np.linalg.norm(start_point - end_point)
    
    return distance
def calculate_cosine_similarity(vector1, vector2):
    """计算两个向量的余弦相似度。"""
    dot_product = np.dot(vector1, vector2)
    norm_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    return dot_product / norm_product

def calculate_continuous_length(coords):
    """计算路径的持续长度。"""
    if len(coords) < 5:
        return 0

    cosine_values = []
    length_values = []

    for i in range(len(coords) - 4):
        # 计算向量a和b
        vector_a = np.array(coords[i+2]) - np.array(coords[i])
        vector_b = np.array(coords[i+4]) - np.array(coords[i+2])

        # 计算余弦相似度
        cosine_similarity = calculate_cosine_similarity(vector_a, vector_b)
        cosine_values.append(cosine_similarity)

        # 计算路径长度
        path_length = np.linalg.norm(np.array(coords[i+2]) - np.array(coords[i]))
        length_values.append(path_length)

    # 筛选余弦值为0或1的情况
    filtered_cosine_values = []
    filtered_length_values = []
    for cosine, length in zip(cosine_values, length_values):
        if not (cosine==0 or cosine==1):
            filtered_cosine_values.append(cosine)
            filtered_length_values.append(length)

    if not filtered_cosine_values or not filtered_length_values:
        return 0

    # 计算平均余弦值和平均路径长度
    mean_theta = np.mean(filtered_cosine_values)
    mean_length = np.mean(filtered_length_values)

    # 计算持续长度
    continuous_length = -mean_length / (np.log(mean_theta) * 2)
    return continuous_length