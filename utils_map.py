import numpy as np

class Vertex:
    def __init__(self, x, y, index, lane_token, lane_idx, map_name):
        self.coors = np.array([x, y])
        self.index = index
        self.lane_token = lane_token
        self.lane_idx = lane_idx
        self.next = []
        self.prev = []
        self.direction = None
        self.mode = None
        self.map_name = map_name
    def add_next(self, token):
        self.next.append(token)
    def add_prev(self, token):
        self.prev.append(token)
    def set_mode(self, mode):
        self.mode = mode

class Edge:
    def __init__(self, index_1, index_2, Vertices: dict):
        node_1 = Vertices[index_1]
        node_2 = Vertices[index_2]
        self.token = str(index_1) + '_' + str(index_2)
        self.node_1 = node_1
        self.node_2 = node_2
        self.distence = np.sqrt(np.sum((node_1.coors - node_2.coors) ** 2))

def get_closest_node_from_lane(vertex_xy, discrete_lane):
    dis = np.sum((discrete_lane - vertex_xy) ** 2, axis = 1)
    #print(dis)
    return np.argmin(dis)

def get_direction(vertex, Vertices, lanes_vertices):
    lane_vertices = lanes_vertices[vertex.lane_token]
    if vertex.index == lane_vertices[-1].index and len(lane_vertices) > 1:
        coor = vertex.coors - lane_vertices[-2].coors
    elif len(vertex.next):
        coor = Vertices[vertex.next[0]].coors - vertex.coors
    elif len(vertex.prev):
        coor = vertex.coors - Vertices[vertex.prev[0]].coors
    else:
        coor = np.ones(2)
    return np.arctan2(coor[1], coor[0])

def cal_distance(node_1, node_2):
    return np.sqrt(np.sum((node_1.coors - node_2.coors) ** 2))

def diff_direction(dir_1, dir_2):
    ans = dir_1 - dir_2
    if ans > np.pi:
        ans -= 2 * np.pi
    elif ans < -np.pi:
        ans += 2 * np.pi
    return ans

def diff_direction_abs(dir_1, dir_2):
    return np.pi - np.abs(np.abs(dir_1 - dir_2) - np.pi)

def diff_direction_vector(direction_1, direction_2): #向量，方向差计算
    #return np.pi - np.abs(np.abs(direction_1 - direction_2) - np.pi)
    ans = direction_1 - direction_2
    mask_1 = ans > np.pi
    ans[mask_1] -= 2 * np.pi
    mask_2 = ans < -np.pi
    ans[mask_2] += 2 * np.pi
    return ans

def lower_bound_vertex(location, mode: str, Vertices, Vertices_ordered_x, Vertices_ordered_y):
    assert mode in ['x', 'y'], 'mode {} not found'.format(mode)
    if mode == 'x':
        ordered_list = Vertices_ordered_x
        idx = 0
    else:
        ordered_list = Vertices_ordered_y
        idx = 1
    left = 0
    right = len(ordered_list) - 1
    while left < right:
        mid = left + (right - left) // 2
        if Vertices[ordered_list[mid]].coors[idx] <= location:
            left = mid + 1
        else:
            right = mid
    return mid

def upper_bound_vertex(location, mode: str, Vertices, Vertices_ordered_x, Vertices_ordered_y):
    assert mode in ['x', 'y'], 'mode {} not found'.format(mode)
    if mode == 'x':
        ordered_list = Vertices_ordered_x
        idx = 0
    else:
        ordered_list = Vertices_ordered_y
        idx = 1
    left = 0
    right = len(ordered_list) - 1
    while left < right:
        mid = right - (right - left) // 2
        if Vertices[ordered_list[mid]].coors[idx] >= location:
            right = mid - 1
        else:
            left = mid
    return mid

def get_nodes_in_radius(x, y, radius, Vertices, Vertices_ordered_x, Vertices_ordered_y):
    if type(radius) == list:
        x_1 = lower_bound_vertex(x - radius[0], 'x', Vertices, Vertices_ordered_x, Vertices_ordered_y)
        x_2 = upper_bound_vertex(x + radius[1], 'x', Vertices, Vertices_ordered_x, Vertices_ordered_y)
        y_1 = lower_bound_vertex(y - radius[2], 'y', Vertices, Vertices_ordered_x, Vertices_ordered_y)
        y_2 = upper_bound_vertex(y + radius[3], 'y', Vertices, Vertices_ordered_x, Vertices_ordered_y)
    else:
        x_1 = lower_bound_vertex(x - radius, 'x', Vertices, Vertices_ordered_x, Vertices_ordered_y)
        x_2 = upper_bound_vertex(x + radius, 'x', Vertices, Vertices_ordered_x, Vertices_ordered_y)
        y_1 = lower_bound_vertex(y - radius, 'y', Vertices, Vertices_ordered_x, Vertices_ordered_y)
        y_2 = upper_bound_vertex(y + radius, 'y', Vertices, Vertices_ordered_x, Vertices_ordered_y)
    #assert x_1 <= x_2 and y_1 <= y_2, 'no node in radius {} from ({},{}), xy = {},{},{},{}'.format(radius, x, y, x_1, x_2, y_1, y_2)
    if x_1 > x_2 or y_1 > y_2:
        return []
    node_set_1 = set(Vertices_ordered_x[x_1:x_2 + 1])
    node_set_2 = set(Vertices_ordered_y[y_1:y_2 + 1])
    node_set = node_set_1.intersection(node_set_2)
    return list(node_set)
    
def get_lanes_in_radius(x, y, radius, Vertices, Vertices_ordered_x, Vertices_ordered_y):
    nodes = get_nodes_in_radius(x, y, radius, Vertices, Vertices_ordered_x, Vertices_ordered_y)
    lane_set = set()
    for index in nodes:
        lane_set.add(Vertices[index].lane_token)
    return list(lane_set)

def cal_distance_vector(vector_1, vector_2): #向量，两点距离计算
    diff = (vector_1 - vector_2) ** 2
    return np.sqrt(np.sum(diff, axis = -1))

def cal_direction_vector(vector_1, vector_2): #向量，两点方向计算
    ans = vector_2 - vector_1
    res_shape = ans.shape
    ans = ans.reshape(-1,2)
    ans = np.arctan2(ans[:,1], ans[:,0])
    return ans.reshape(res_shape[:-1])

def diff_direction_vector(direction_1, direction_2): #向量，方向差计算
    #return np.pi - np.abs(np.abs(direction_1 - direction_2) - np.pi)
    ans = direction_1 - direction_2
    mask_1 = ans > np.pi
    ans[mask_1] -= 2 * np.pi
    mask_2 = ans < -np.pi
    ans[mask_2] += 2 * np.pi
    return ans

def generate_maps(maps):
    Vertices_maps = dict() #点集合，根据map分为4组
    Edges_maps = dict() #边集合
    ordered_x_maps = dict() #点根据x坐标排序
    ordered_y_maps = dict() #点根据y坐标排序
    lanes_vertices = dict() #lane-vertex 映射

    discretization_resolution_meters = 1 #超参数:车道分辨率(米)
    radius = 5 #超参数
    direction_threshold = np.pi / 2 #超参数

    unfound1 = []
    unfound2 = []
    for map_name in maps.keys(): #遍历地图
        Vertices = dict() #点集
        map = maps[map_name]
        global_index = 0
        lanes = map.lane + map.lane_connector
        lanes_token = [] #车道线token
        for lane in lanes:
            lanes_token.append(lane['token'])
        
        lanes_dict = map.discretize_lanes(lanes_token, discretization_resolution_meters)
        for key in lanes_dict.keys():
            for i in range(len(lanes_dict[key])):#舍去z坐标
                lanes_dict[key][i] = lanes_dict[key][i][:2]
            temp_xy = []
            num_next = len(map.get_outgoing_lane_ids(key))#后续车道数
            for i in range(len(lanes_dict[key]) - 1):
                if np.sqrt(np.sum((np.array(lanes_dict[key][i]) - np.array(lanes_dict[key][i + 1])) ** 2)) < 0.1:
                    continue #舍去与后一点重合的点
                temp_xy.append(lanes_dict[key][i])
            if num_next == 0: #舍去与后续车道重合的点
                temp_xy.append(lanes_dict[key][-1])
            lanes_dict[key] = np.array(temp_xy)
            lane_coors = lanes_dict[key]
            lanes_vertices[key] = [] #车道-点集映射
            for i in range(len(lane_coors)):
                temp_node = Vertex(*lane_coors[i], global_index, key, i, map_name)
                Vertices[global_index] = temp_node
                lanes_vertices[key].append(temp_node)
                if i != 0:
                    temp_node.add_prev(global_index - 1)
                if i != len(lane_coors) - 1:
                    temp_node.add_next(global_index + 1)
                global_index += 1

        for key in lanes_dict.keys():#将每个车道最后一个点与后续车道第一个点连接
            vertex = lanes_vertices[key][-1]
            target_lanes = map.get_outgoing_lane_ids(vertex.lane_token)
            for token in target_lanes:
                if token not in lanes_vertices:
                    unfound1.append(token)
                    continue
                target_vertex = lanes_vertices[token][0]
                vertex.add_next(target_vertex.index)
                target_vertex.add_prev(vertex.index)

        #按照x、y坐标排序，方便后面查找
        Vertices_ordered_x = sorted([i for i in range(len(Vertices))], key = lambda x : Vertices[x].coors[0])
        Vertices_ordered_y = sorted([i for i in range(len(Vertices))], key = lambda x : Vertices[x].coors[1])
        ordered_x_maps[map_name] = Vertices_ordered_x
        ordered_y_maps[map_name] = Vertices_ordered_y

        for vertex in Vertices.values():
            x_y = vertex.coors
            ver_direction = get_direction(vertex, Vertices, lanes_vertices)
            vertex.direction = ver_direction
            local_lanes = get_lanes_in_radius(*x_y, radius, Vertices, Vertices_ordered_x, Vertices_ordered_y)
            #local_lanes = local_lanes['lane'] + local_lanes['lane_connector']
            outgoing = map.get_outgoing_lane_ids(vertex.lane_token)
            incoming = map.get_incoming_lane_ids(vertex.lane_token)
            out_in = outgoing + incoming
            
            for token in local_lanes:#与周围的车道连接
                if token == vertex.lane_token or token in out_in:   #outgoing == map.get_outgoing_lane_ids(token):
                    continue#不与自己/自己的前后车道连接
                if token not in lanes_dict: #未找到车道
                    unfound2.append(token)
                    continue
                discrete_lane = lanes_dict[token]
                idx = get_closest_node_from_lane(x_y, discrete_lane)
                node = lanes_vertices[token][idx]
                direction = get_direction(node, Vertices, lanes_vertices)
                if abs(diff_direction(ver_direction, direction)) > direction_threshold:
                    continue #舍去方向差大于direction_threshold的车道
                coors = node.coors - vertex.coors
                nodes_direction = np.arctan2(coors[1], coors[0])
                while abs(diff_direction(ver_direction, nodes_direction)) > direction_threshold and idx < discrete_lane.shape[0] - 1:
                    node = Vertices[node.next[0]]
                    idx += 1
                    coors = node.coors - vertex.coors
                    nodes_direction = np.arctan2(coors[1], coors[0])
                if abs(diff_direction(ver_direction, nodes_direction)) < direction_threshold and cal_distance(vertex, node) < radius:
                    vertex.add_next(node.index)
                    node.add_prev(vertex.index)
        
        Edges = dict()
        for key in Vertices.keys(): #建边集
            node = Vertices[key]
            for key_2 in node.next:
                temp_edge = Edge(key, key_2, Vertices)
                Edges[temp_edge.token] = temp_edge
        Vertices_maps[map_name] = Vertices
        Edges_maps[map_name] = Edges
    
    return Vertices_maps, Edges_maps, ordered_x_maps, ordered_y_maps, lanes_vertices

def set_lanes_mode(lanes_vertices):
    lane_dir_diffs = []
    lane_threshold = np.pi / 4
    for key in lanes_vertices.keys():
        nodes = lanes_vertices[key]
        end_idx = min(50, len(nodes) - 1)
        end_node = nodes[end_idx]
        start_node = nodes[0]
        lane_dir = diff_direction(end_node.direction, start_node.direction)
        lane_dir_diffs.append(lane_dir)
        if lane_dir < -lane_threshold:
            mode = 'right'
        elif lane_dir > lane_threshold:
            mode = 'left'
        else:
            mode = 'straight'
        for node in nodes:
            node.set_mode(mode)
    return lane_dir_diffs