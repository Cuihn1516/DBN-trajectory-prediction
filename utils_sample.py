from scipy.special import logsumexp
from pyquaternion import Quaternion
from nuscenes.eval.common.utils import quaternion_yaw
import numpy as np
import scipy.stats as stats
from utils_map import diff_direction, cal_distance, diff_direction_abs
from utils_map import get_nodes_in_radius
from utils_map import Vertex
from utils_map import diff_direction_vector
from utils_social_field import total_field
import copy


def forward(P_trans, p_eq, rv, observed):
    """
    Forward algorithm: computes the log-likelihood of the observed data, 
    given the model. Performs computations in log-space to avoid underflow
    issues. Computes and returns the full forward matrix, and the final 
    sum-of-all-paths probabilities.

    Parameters
    ----------
    theta: transit probs, K*K
    phi: emission probs, K*V
    u: length T array, observed emission

    Returns
    -------
    score: float, log-probability (score) of observed sequence relative to model, 
    alpha: array-like, full forward matrix
    """
    K= P_trans.shape[0]
    T = observed.shape[0]
    #p_eq=np_solve_equilibrium(K,P_trans)
    gamma=np.zeros((T,K))
    eps = 1e-100
    gamma[0] = rv.pdf(observed[0]) * p_eq + eps
    gamma[0] = gamma[0] / gamma[0].sum()

    for t in range(1, T):
        alpha = np.matmul(P_trans.T, gamma[t-1]) * rv.pdf(observed[t]) + eps
        gamma[t] = alpha / alpha.sum()

    return gamma

class Norm_generator():
    def __init__(self, mu, sigma, size):
        self.generator = stats.norm(loc = mu, scale = sigma)
        self.nums = self.generator.rvs(size)
        self.size = size
        self.idx = 0
    def next(self):
        '''
        @summary: 每一次for循环都调用该方法（必须存在）
        '''
        self.idx += 1
        if self.idx == self.size:
            self.idx = 0
        return self.nums[self.idx]
class Choice_generator():
    def __init__(self, prob, n_states, size):
        self.nums = np.random.choice(n_states, size = size, p = prob)
        self.prob = prob
        self.n_states = n_states
        self.size = size
        self.idx = 0
    def next(self):
        '''
        @summary: 每一次for循环都调用该方法（必须存在）
        '''
        self.idx += 1
        if self.idx == self.size:
            self.idx = 0
        return self.nums[self.idx]
    
direction_threshold = np.pi / 2

N_STATES = 21
v_mu = np.array([ 1.55665005e-02,  7.84115276e-01, -1.43591758e-56,  4.74062727e+00,
         8.94278525e+00])
v_sigma = np.array([6.84203175e-02, 7.11866616e-01, 1.44980489e-56, 2.25310390e+00,
        3.38690969e+00])
v_norms = stats.norm(loc = v_mu, scale = v_sigma)
v_w = np.array([0.08311134, 0.10039968, 0.03080547, 0.3774135 , 0.40827])
mu_a = np.linspace(-5,5,N_STATES).reshape(-1,1)
dt = 0.5
def calc_v_weight(v):
    next_v = v + mu_a * dt
    pdf = v_norms.pdf(next_v) * v_w
    pdf = pdf.sum(axis = 1)
    return pdf

def get_target_v():
    idx = np.random.choice(len(v_mu), p = v_w)
    return v_norms.rvs()[idx]

def get_closest_node_in_radius(x, y, map_name, radius, direction, Vertices_maps, ordered_x_maps, ordered_y_maps):
    #print(map_name)
    Vertices = Vertices_maps[map_name]
    Vertices_ordered_x = ordered_x_maps[map_name]
    Vertices_ordered_y = ordered_y_maps[map_name]
    nodes = get_nodes_in_radius(x, y, radius, Vertices, Vertices_ordered_x, Vertices_ordered_y)
    closest_dis = np.Inf
    ans = None
    alter_nodes = []
    distances = []
    for idx in nodes:
        node = Vertices[idx]
        temp_dis = (node.coors[0] - x) ** 2 + (node.coors[1] - y) ** 2
        #if node.next:
        #    next_node = node.next[0]
        #    temp_dis += (next_node.coors[0] - x) ** 2 + (next_node.coors[1] - y) ** 2
        #    temp_dis /= 2
        dir_diff = abs(diff_direction(direction, node.direction))
        if dir_diff < direction_threshold:
            if closest_dis - temp_dis > 0.01:
                alter_nodes.clear()
                distances.clear()
                alter_nodes.append(node)
                distances.append(temp_dis)
                closest_dis = temp_dis
            elif abs(closest_dis - temp_dis) <= 0.01:
                alter_nodes.append(node)
                distances.append(temp_dis)
    if len(alter_nodes) > 1:
        node0 = alter_nodes[0]
        node1 = alter_nodes[1]
        temp_dis = (node1.coors[0] - node0.coors[0]) ** 2 + (node1.coors[1] - node0.coors[1]) ** 2
        while temp_dis <= 0.01:
            flag = False
            for i in range(len(alter_nodes)):
                if len(alter_nodes[i].next):
                    alter_nodes[i] = Vertices[alter_nodes[i].next[0]]
                else:
                    flag = True
                    break
            if flag:
                break
            node0 = alter_nodes[0]
            node1 = alter_nodes[1]
            temp_dis = (node1.coors[0] - node0.coors[0]) ** 2 + (node1.coors[1] - node0.coors[1]) ** 2
        closest_dis = np.Inf
        for i in range(len(alter_nodes)):
            node = alter_nodes[i]
            dis = (node.coors[0] - x) ** 2 + (node.coors[1] - y) ** 2
            if dis < closest_dis:
                closest_dis = dis
                ans = node
    elif len(alter_nodes) > 0:
        ans = alter_nodes[0]
    if ans == None:
        return get_closest_node_in_radius(x, y, map_name, radius * 2, direction)
    return ans
def next_n_nodes(map_name, node, n, max_n, maps, Vertices_maps):
    if n == 0:
        return [[]]
    map = maps[map_name]
    Vertices = Vertices_maps[map_name]
    next = []
    if node.lane_token != -1:
        flag1 = True
        valid_lanes = set(map.get_outgoing_lane_ids(node.lane_token))
        for idx in node.next:
            next_node = Vertices[idx]
            if next_node.lane_token == node.lane_token:
                next.append(next_node)
                flag1 = False
            elif flag1 and (node.mode == 'left' or node.mode == 'right'):
                next.append(next_node)
            elif next_node.lane_token in valid_lanes:
                next.append(next_node)
            elif (next_node.coors[0] - Vertices[node.next[0]].coors[0]) ** 2 + (next_node.coors[1] - Vertices[node.next[0]].coors[1]) ** 2 < 0.01:
                next.append(next_node)
            #elif n == max_n:
            #    dis = np.sqrt(((next_node.coors - node.coors) ** 2).sum())
            #    if dis < 2:
            #        next.append(next_node)

    
    if len(node.next) == 0:
        temp_x = node.coors[0] + np.cos(node.direction)
        temp_y = node.coors[1] + np.sin(node.direction)
        new_node = Vertex(temp_x, temp_y, -1, -1, -1, map_name)
        new_node.direction = node.direction
        next.append(new_node)
    if len(next) == 0:
        return None
    res = []
    for next_node in next:
        temp_nodes = next_n_nodes(map_name, next_node, n - 1, max_n)
        if temp_nodes == None:
            continue
        for nodes in temp_nodes:
            res.append([next_node] + nodes)
    return res

def get_closest_node_idx(nodes_x, nodes_y, idx, x, y):
    min_idx = max(0, idx - 5)
    max_idx = min(idx + 6, nodes_x.shape[0])
    t_x = nodes_x[min_idx:max_idx]
    t_y = nodes_y[min_idx:max_idx]
    dis = (t_x - x) ** 2 + (t_y - y) ** 2
    res = dis.argmin()
    return res + min_idx, dis.min()

dt = 0.5
def predict_func(num, a_init, psi_init, a_generators, psi_generators, init_v, init_yaw, init_x, init_y, nodes_x, nodes_y, nodes_list, psi_rv, N_STATES, 
                 p_trans_a, p_trans_psi):
    res_a = []
    states_a = []
    res_psi = []
    states_psi = []
    v_seq = [init_v]
    x_seq = [init_x]
    y_seq = [init_y]
    yaw_seq = [init_yaw]
    states_a.append(np.random.choice(N_STATES, p = a_init))
    pre_node_idx = 0
    next_node_idx = 10
    eps = 1e-100
    if next_node_idx == 0:
        psi_init_2 = psi_init * psi_rv.pdf(diff_direction(nodes_list[0].direction, init_yaw))
    else:
        psi_init_2 = psi_init * (psi_rv.pdf(diff_direction(np.arctan2(nodes_y[next_node_idx] - init_y, nodes_x[next_node_idx] - init_x), init_yaw)) + eps)#公式，模型写一下
    
    psi_init_2 = psi_init_2 / psi_init_2.sum()
    states_psi.append(np.random.choice(N_STATES, p = psi_init_2))
    #print(psi_init_2)
    #print(diff_direction(np.arctan2(nodes_y[next_node_idx] - init_y, nodes_x[next_node_idx] - init_x), init_yaw))
    for i in range(num):
        a = a_generators[states_a[-1]].next()
        psi = psi_generators[states_psi[-1]].next()
        v = v_seq[-1] + a * dt
        yaw = yaw_seq[-1] + psi
        dx = v * np.cos(yaw) * dt
        dy = v * np.sin(yaw) * dt
        x = x_seq[-1] + dx
        y = y_seq[-1] + dy
        next_node_idx = pre_node_idx + int(v * dt)
        pre_node_idx, min_dis = get_closest_node_idx(nodes_x, nodes_y, next_node_idx, x, y)
        next_node_idx = min(pre_node_idx + int(v * dt + 1), len(nodes_list) - 1)

        states_a.append(np.random.choice(N_STATES, p = p_trans_a[states_a[-1]]))
        if next_node_idx == pre_node_idx:
            temp_psi = diff_direction(nodes_list[next_node_idx].direction, yaw)
        else:
            temp_psi = diff_direction(np.arctan2(nodes_y[next_node_idx] - y, nodes_x[next_node_idx] - x), yaw)
        p_trans = p_trans_psi[states_psi[-1]] * (psi_rv.pdf(temp_psi) + eps)
        #if (p_trans == 0).sum():
        #    print(p_trans, temp_psi, yaw, nodes_list[next_node_idx].direction, np.arctan2(nodes_y[next_node_idx] - y, nodes_x[next_node_idx] - x), v, next_node_idx)
        #    print(min_dis)
        p_trans = p_trans / p_trans.sum()
        #print(p_trans)
        #print(diff_direction(np.arctan2(nodes_y[next_node_idx] - y, nodes_x[next_node_idx] - x), yaw))
        states_psi.append(np.random.choice(N_STATES, p = p_trans))
        res_a.append(a)
        res_psi.append(psi)
        v_seq.append(v)
        yaw_seq.append(yaw)
        x_seq.append(x)
        y_seq.append(y)
    return res_a, res_psi, v_seq[1:], yaw_seq[1:], x_seq[1:], y_seq[1:]

def calc_lane_offset(x, y, node):
    dir = node.direction + np.pi / 2
    dx = x - node.coors[0]
    dy = y - node.coors[1]
    dir2 = np.arctan2(dy, dx)
    dir_diff = diff_direction(dir, dir2)
    return np.sqrt(dx ** 2 + dy ** 2) * np.cos(dir_diff)
def calc_nodes_array(nodes, offset):
    res_x = []
    res_y = []
    for nodes_list in nodes:
        temp_x = []
        temp_y = []
        for node in nodes_list:
            offset_x = offset * np.cos(node.direction + np.pi / 2)
            offset_y = offset * np.sin(node.direction + np.pi / 2)
            temp_x.append(node.coors[0] + offset_x)
            temp_y.append(node.coors[1] + offset_y)
        res_x.append(temp_x)
        res_y.append(temp_y)
    return np.array(res_x), np.array(res_y)

def check_ans(x_seq, y_seq, nodes_x, nodes_y, dis_threshold_sq):
    end_x = x_seq[-1]
    end_y = y_seq[-1]
    min_dis = ((end_x - nodes_x) ** 2 + (end_y - nodes_y) ** 2).min()
    return min_dis > dis_threshold_sq


def get_seq_from_ann_list(anns, helper):
    v_seq = []
    yaw_seq = []
    a_seq = []
    for ann in anns:
        v = helper.get_velocity_for_agent(ann['instance_token'], ann['sample_token'])
        if not np.isnan(v):
            v_seq.append(v)
        a = helper.get_acceleration_for_agent(ann['instance_token'], ann['sample_token'])
        if not np.isnan(a):
            a_seq.append(a)
        yaw_seq.append(quaternion_yaw(Quaternion(ann['rotation'])))

    v_seq = np.array(v_seq)
    #yaw_seq_xy = np.arctan2(y_diff, x_diff)
    yaw_seq = np.array(yaw_seq)
    a_seq = np.array(a_seq)
    #mask = v_seq > 1
    #yaw_seq[mask] = yaw_seq_xy[mask]
    return v_seq,yaw_seq,a_seq

import pytensor.tensor as at
import pymc as pm
import pytensor

def one_step(logp_x, value, loss_, P_trans, N_STATES):
    next_loss = loss_.reshape((N_STATES,1)) + pm.math.log(P_trans)
    
    loss_mean = at.mean(next_loss, axis = 0, keepdims = True)
    next_loss = loss_mean + at.logsumexp(next_loss - loss_mean, axis = 0, keepdims = True)
    next_loss = next_loss.reshape((N_STATES,))
    next_loss = next_loss + logp_x
    mask = at.eq(value, -10)
    res_loss = next_loss * ~mask + loss_ * mask
    return res_loss

def mylogp(value, mu, sigma, P_init, P_trans, N_STATES):
    normal_dist = pm.Normal.dist(mu = mu, sigma = sigma, shape = (N_STATES))
    logp_x = pm.logp(normal_dist, value)

    results, _ = pytensor.scan(fn = one_step,
                            outputs_info=P_init,
                            sequences=[logp_x, value],
                            non_sequences=[P_trans, N_STATES])

    result_loss = results[-1]
    mean_log = at.mean(result_loss, axis = 0, keepdims=True)
    result_loss = mean_log + pm.logsumexp(result_loss - mean_log, axis = 0, keepdims=True)
    return result_loss.reshape((-1,))

    
def calc_init_state(agent_ann, helper, N_STATES, model, p_trans_a, mu_a, true_sigma_a, p_trans_psi,
                    mu_psi, true_sigma_psi, a_rv, psi_rv, p_eq_a, p_eq_psi):
    past_anns = helper.get_past_for_agent(agent_ann['instance_token'], agent_ann['sample_token'], 2, False, False)
    past_anns.reverse()
    past_anns.append(agent_ann)
    v_seq, yaw_seq, a_seq = get_seq_from_ann_list(past_anns, helper)
    if len(v_seq):
        init_v = v_seq[-1]
    else:
        init_v = 0
    if len(yaw_seq):
        init_yaw = yaw_seq[-1]
    else:
        init_yaw = quaternion_yaw(Quaternion(agent_ann['rotation']))
    if len(a_seq) == 0:
        a_init = np.ones(N_STATES) / N_STATES
        psi_init = np.ones(N_STATES) / N_STATES
    else:
        psi_diff = diff_direction_vector(yaw_seq[1:], yaw_seq[:-1])
        psi_diff[np.abs(psi_diff) > 0.4] = 0
        a_seq = np.concatenate([a_seq, np.ones(5-len(a_seq)) * -10])
        psi_seq = np.concatenate([psi_diff, np.ones(4-len(psi_diff)) * -10])
        with model:
            model.set_data('P_trans', p_trans_a)
            model.set_data('mu', mu_a)
            model.set_data('sigma', true_sigma_a)
            model.set_data('obs_seq', a_seq.reshape(-1,1))
            b_map = pm.find_MAP(progressbar=False)
        p_init_a = b_map['P_init'].reshape(-1)
        with model:
            model.set_data('P_trans', p_trans_psi)
            model.set_data('mu', mu_psi)
            model.set_data('sigma', true_sigma_psi)
            model.set_data('obs_seq', psi_seq.reshape(-1,1))
            b_map = pm.find_MAP(progressbar=False)
        p_init_psi = b_map['P_init'].reshape(-1)
        a_gamma = forward(p_trans_a, p_init_a, a_rv, a_seq)
        psi_gamma = forward(p_trans_psi, p_init_psi, psi_rv, psi_diff)
        a_init =a_gamma[-1]
        psi_init = psi_gamma[-1]
    if True in np.isnan(a_init):
        a_init = p_eq_a
    if True in np.isnan(psi_init):
        psi_init = p_eq_psi
    init_x = agent_ann['translation'][0]
    init_y = agent_ann['translation'][1]
    return init_x, init_y, init_v, init_yaw, a_init, psi_init

def calc_init_state_2(agent_ann, helper, N_STATES, model, p_trans_a, mu_a, true_sigma_a, p_trans_psi,
                    mu_psi, true_sigma_psi, a_rv, psi_rv, p_eq_a, p_eq_psi):
    past_anns = helper.get_past_for_agent(agent_ann['instance_token'], agent_ann['sample_token'], 2, False, False)
    past_anns.reverse()
    past_anns.append(agent_ann)
    v_seq, yaw_seq, a_seq = get_seq_from_ann_list(past_anns, helper)
    if len(v_seq):
        init_v = v_seq[-1]
    else:
        init_v = 0
    if len(yaw_seq):
        init_yaw = yaw_seq[-1]
    else:
        init_yaw = quaternion_yaw(Quaternion(agent_ann['rotation']))
    
    a_init = np.ones(N_STATES) / N_STATES
    psi_init = np.ones(N_STATES) / N_STATES
    
    if True in np.isnan(a_init):
        a_init = p_eq_a
    if True in np.isnan(psi_init):
        psi_init = p_eq_psi
    init_x = agent_ann['translation'][0]
    init_y = agent_ann['translation'][1]
    return init_x, init_y, init_v, init_yaw, a_init, psi_init

class Scene:
    def __init__(self,init_all_x=None, init_all_y=None, init_all_v=None, init_all_yaw=None, lane_idx=None, a_states=None, psi_states=None, pa_scene=None):
        self.x = [np.array(init_all_x)]
        self.y = [np.array(init_all_y)]
        self.v = [np.array(init_all_v)]
        self.yaw = [np.array(init_all_yaw)]
        self.lane_idx = [lane_idx]
        if lane_idx:
            self.closest_node = [[0] * len(lane_idx)]
        else:
            self.closest_node = None
        self.a_states = [a_states]
        self.psi_states = [psi_states]
        self.a = [[]]
        self.valid = True
        if pa_scene:
            self.pa_scene = {pa_scene.glob_idx:1}
        else:
            self.pa_scene = {}
        self.sub_scene = []
        self.sub_weight = []
        self.valid_sub_scene = []
        self.valid_sub_weight = []
        self.self_weight = []
    def set_val(self, t_x, t_y, t_v, t_yaw, t_a_state, t_psi_state, t_a):
        self.x = [t_x]
        self.y= [t_y]
        self.v = [t_v]
        self.yaw = [t_yaw]
        self.a_states = [t_a_state]
        self.psi_states = [t_psi_state]
        self.a = [t_a]
    def set_pa(self, all_nodes):
        for key in self.pa_scene.keys():
            pa_scene = all_nodes[key]
            num = self.pa_scene[key]
            pa_scene.sub_scene.append(self.glob_idx)
            pa_scene.sub_weight.append(num)
    def set_self(self):
        for key in self.pa_scene.keys():
            self.self_weight.append(self.pa_scene[key])
        self.self_weight = np.array(self.self_weight)
        self.self_weight = self.self_weight / self.self_weight.sum()
    def set_glob_idx(self, all_nodes):
        self.glob_idx = len(all_nodes)
        all_nodes.append(self)
    def set_candidate_lanes(self, all_lanes):
        self.candidate_lanes = all_lanes
    def set_lane(self, lane_token):
        self.lane_token = lane_token
    def set_offsets(self, offsets):
        self.offsets = offsets
    def set_target_v(self, target_v):
        self.target_v = target_v

min_log = 10 ** 0.1
max_log = 10
risk_threshold = 0.37
def check_scene(scene : Scene, static_input, moving_input, thresholds):
    moving_input['sample_x'] = np.array(scene.x[-1])
    moving_input['sample_y'] = np.array(scene.y[-1])
    moving_input['sample_v_y'] = np.array(scene.v[-1])
    moving_input['sample_direction'] = np.array(scene.yaw[-1])
    moving_input['sample_a'] = np.array(scene.a[-1])
    num_agent = len(scene.x[0])
    init_sigma_x = copy.deepcopy(moving_input['sample_sigma_x'])
    init_sigma_y = copy.deepcopy(moving_input['sample_sigma_y'])
    for i in range(num_agent - 1, num_agent):#只检查主agent？
        t_x = scene.x[-1][i]
        t_y = scene.y[-1][i]

        t_yaw = scene.yaw[-1][i]
        width = moving_input['sample_sigma_x'][i]
        length = moving_input['sample_sigma_y'][i]
        moving_input['sample_sigma_x'] = (init_sigma_x + width)/2
        moving_input['sample_sigma_y'] = (init_sigma_y + length)/2
        static_directions = np.arctan2(static_input['sample_y'] - t_y, static_input['sample_x'] - t_x)
        static_dir_diff = diff_direction_vector(static_directions, t_yaw)
        static_mask = (np.pi - np.abs(static_dir_diff)) / np.pi * (max_log - min_log) + min_log
        static_mask = np.log10(static_mask)
        moving_directions = np.arctan2(moving_input['sample_y'] - t_y, moving_input['sample_x'] - t_x)
        moving_dir_diff = diff_direction_vector(moving_directions, t_yaw)
        moving_mask = (np.pi - np.abs(moving_dir_diff)) / np.pi * (max_log - min_log) + min_log
        moving_mask = np.log10(moving_mask)
        moving_mask[i] = 0
        _, static_risk = total_field(t_x, t_y, **static_input)
        _, moving_risk = total_field(t_x, t_y, **moving_input)
        if type(static_risk) == type(None):
            static_max = 0
        else:
            static_max = (static_risk * static_mask).max()
        moving_max = (moving_risk * moving_mask).max()
        res = max(static_max, moving_max)
        moving_input['sample_sigma_x'] = init_sigma_x
        moving_input['sample_sigma_y'] = init_sigma_y
        if res > thresholds[i]:
            #print(j, i)
            #print(static_mask)
            #print(moving_mask)
            #print(static_risk)
            #print(moving_risk)
            #return True, i, moving_risk, static_risk
            return True
    return False

def init_scene_risk(scene : Scene, static_input, moving_input):
    num_agent = len(scene.x[0])
    init_sigma_x = copy.deepcopy(moving_input['sample_sigma_x'])
    init_sigma_y = copy.deepcopy(moving_input['sample_sigma_y'])
    ans = []
    for i in range(num_agent):
        t_x = scene.x[-1][i]
        t_y = scene.y[-1][i]
        t_yaw = scene.yaw[-1][i]
        width = moving_input['sample_sigma_x'][i]
        length = moving_input['sample_sigma_y'][i]
        moving_input['sample_sigma_x'] = (init_sigma_x + width)/2
        moving_input['sample_sigma_y'] = (init_sigma_y + length)/2
        static_directions = np.arctan2(static_input['sample_y'] - t_y, static_input['sample_x'] - t_x)
        static_dir_diff = diff_direction_vector(static_directions, t_yaw)
        static_mask = (np.pi - np.abs(static_dir_diff)) / np.pi * (max_log - min_log) + min_log
        static_mask = np.log10(static_mask)
        moving_directions = np.arctan2(moving_input['sample_y'] - t_y, moving_input['sample_x'] - t_x)
        moving_dir_diff = diff_direction_vector(moving_directions, t_yaw)
        moving_mask = (np.pi - np.abs(moving_dir_diff)) / np.pi * (max_log - min_log) + min_log
        moving_mask = np.log10(moving_mask)
        moving_mask[i] = 0
        _, static_risk = total_field(t_x, t_y, **static_input)
        _, moving_risk = total_field(t_x, t_y, **moving_input)
        if type(static_risk) == type(None):
            static_max = 0
        else:
            static_max = (static_risk * static_mask).max()
        moving_max = (moving_risk * moving_mask).max()
        res = max(static_max, moving_max)
        moving_input['sample_sigma_x'] = init_sigma_x
        moving_input['sample_sigma_y'] = init_sigma_y
        if res > 0.37:#0.37
            ans.append(res * 1.5)
        else:
            ans.append(0.37)
    return ans

t = np.linspace(0,1,11).reshape(-1,1)
s = 1 - t
t_2 = t ** 2
s_2 = s ** 2
t_1_s_1 = t * s
'''
def bezier_step(xy_0, xy_1, xy_2, xy_3, t0, t1, d0, d1, depth):
    if depth > 4:
        return abs(diff_direction(d0, d1))
    mid = (t0 + t1) / 2
    d_xy_mid = 3 * ((1-mid)**2) * (xy_1 - xy_0) + 6 * (xy_2 - xy_1) * (mid * (1-mid)) + 3 * (xy_3 - xy_2) * (mid ** 2)
    direction_mid = np.arctan2(d_xy_mid[1],d_xy_mid[0])
    if abs(abs(diff_direction(d0,d1)) - abs(diff_direction(d0,direction_mid)) - abs(diff_direction(direction_mid,d1))) > 1e-3:
        ans = bezier_step(xy_0, xy_1, xy_2, xy_3, t0, mid, d0, direction_mid, depth+1) + bezier_step(xy_0, xy_1, xy_2, xy_3, mid, t1, direction_mid, d1, depth+1)
        return ans
    else:
        return abs(diff_direction(d0, d1))
def calc_bezier(xy, v, yaw, node):
    offset = max(5, v)
    yaw2 = node.direction
    xy_0 = xy
    xy_1 = xy + offset * np.array([np.cos(yaw), np.sin(yaw)])
    xy_2 = node.coors - offset * np.array([np.cos(yaw2), np.sin(yaw2)])
    xy_3 = node.coors
    t0 = 0
    t1 = 1
    d_xy_0 = 3 * ((1-t0)**2) * (xy_1 - xy_0) + 6 * (xy_2 - xy_1) * (t0 * (1-t0)) + 3 * (xy_3 - xy_2) * (t0 ** 2)
    d_xy_1 = 3 * ((1-t1)**2) * (xy_1 - xy_0) + 6 * (xy_2 - xy_1) * (t1 * (1-t1)) + 3 * (xy_3 - xy_2) * (t1 ** 2)
    #d_xy = -3 * t_s_1_2 * xy_0 + 3 * t_s_1_2 * xy_1 - 6 * t_s_1 * t * xy_1 + \
    #       6 * t_s_1 * t * xy_2 - 3 * t_2 * xy_2 + 3 * t_2 * xy_3
    #print(d_xy)
    d_x_0 = d_xy_0[0]
    d_y_0 = d_xy_0[1]
    d_x_1 = d_xy_1[0]
    d_y_1 = d_xy_1[1]
    direction_0 = np.arctan2(d_y_0,d_x_0)
    direction_1 = np.arctan2(d_y_1,d_x_1)
    temp_ans = bezier_step(xy_0, xy_1, xy_2, xy_3, t0, t1, direction_0, direction_1, 0)

    return 1 / (1 + np.exp(temp_ans * 10 - 5))
'''


def calc_bezier(xy, v, yaw, node): #计算贝塞尔曲线角度和
    #offset = max(5, v)
    offset = np.sqrt(((xy - node.coors) ** 2).sum()) * 0.45
    yaw2 = node.direction
    xy_0 = xy
    xy_1 = xy + offset * np.array([np.cos(yaw), np.sin(yaw)])
    xy_2 = node.coors - offset * np.array([np.cos(yaw2), np.sin(yaw2)])
    xy_3 = node.coors
    d_xy = 3 * s_2 * (xy_1 - xy_0) + 6 * (xy_2 - xy_1) * t_1_s_1 + 3 * (xy_3 - xy_2) * t_2
    #d_xy = -3 * t_s_1_2 * xy_0 + 3 * t_s_1_2 * xy_1 - 6 * t_s_1 * t * xy_1 + \
    #       6 * t_s_1 * t * xy_2 - 3 * t_2 * xy_2 + 3 * t_2 * xy_3
    #print(d_xy)
    d_x = d_xy[:,0]
    d_y = d_xy[:,1]
    directions = np.arctan2(d_y,d_x)
    dir_diff = np.abs(diff_direction_vector(directions[1:], directions[:-1]))
    #print(dir_diff)
    temp_ans = dir_diff.sum()
    #return 1 / (1e-2 + temp_ans)
    return 1 / (1 + np.exp(temp_ans * 5 - 5))   #车道角度差转概率公式

def generate_bezier(xy, v, yaw, node):
    #offset = max(5, v)
    offset = np.sqrt(((xy - node.coors) ** 2).sum()) * 0.45
    yaw2 = node.direction
    xy_0 = xy
    xy_1 = xy + offset * np.array([np.cos(yaw), np.sin(yaw)])
    xy_2 = node.coors - offset * np.array([np.cos(yaw2), np.sin(yaw2)])
    xy_3 = node.coors
    ans_xy = s ** 3 * xy_0 + 3 * xy_1 * s_2 * t + 3 * xy_2 * t_2 * s + xy_3 * t ** 3
    return ans_xy[:,0], ans_xy[:,1]

def calc_direction_dis(lane_nodes, token, lanes_closest, xy, v, yaw):
    length = len(lane_nodes)
    idx1 = lanes_closest[token][1]
    idx2 = min(idx1 + 10, length-1)
    dis1 = calc_bezier(xy, v, yaw, lane_nodes[idx1])
    dis2 = calc_bezier(xy, v, yaw, lane_nodes[idx2])
    if dis2 > dis1:
        lanes_closest[token][0] = dis2
        lanes_closest[token][1] = idx2
        return dis2
    else:
        lanes_closest[token][0] = dis1
        lanes_closest[token][1] = idx1
        return dis1
    
def update_euclidean_dis(lane_nodes, key, candidate_lanes, temp_xy):
    idx = candidate_lanes[key][3]
    lane_xy = []
    min_idx = max(0, idx -5)
    max_idx = min(idx + 6, len(lane_nodes))
    for i in range(min_idx, max_idx):
        lane_xy.append(lane_nodes[i].coors)
    lane_xy = np.array(lane_xy)
    dis = np.sqrt(((lane_xy - temp_xy) ** 2).sum(axis = 1))
    res = dis.argmin()
    candidate_lanes[key][2] = dis[res]
    candidate_lanes[key][3] = min_idx + res

def update_candidate_lanes(candidate_lanes, x, y, next_x, next_y, next_yaw, next_v, Vertices, Vertices_ordered_x, 
                           Vertices_ordered_y, lanes_vertices):#, use_times):
    dx = next_x - x
    dy = next_y - y
    temp_xy = np.array([next_x, next_y])
    radius_0 = [abs(dx)/2, abs(dx)/2, 20, 20]
    radius_1 = [20, 20, abs(dy)/2, abs(dy)/2]
    ty_0 = next_y
    if dx < 0:
        tx_0 = x - 20 + dx / 2
    else:
        tx_0 = x + 20 + dx / 2
    tx_1 = next_x
    if dy < 0:
        ty_1 = y - 20 + dy / 2
    else:
        ty_1 = y + 20 + dy / 2
    #t1 = time.perf_counter()
    adj_nodes_0 = get_nodes_in_radius(tx_0, ty_0, radius_0, Vertices, Vertices_ordered_x, Vertices_ordered_y)
    adj_nodes_1 = get_nodes_in_radius(tx_1, ty_1, radius_1, Vertices, Vertices_ordered_x, Vertices_ordered_y)
    #t2 = time.perf_counter()
    #use_times[3] += t2 - t1
    for node_idx in adj_nodes_0:
        node = Vertices[node_idx]
        bezier_dis = calc_bezier(temp_xy, next_v, next_yaw, node)
        euclidean_dis = np.sqrt(((temp_xy - node.coors) ** 2).sum())
        if node.lane_token in candidate_lanes:
            if candidate_lanes[node.lane_token][0] < bezier_dis:
                candidate_lanes[node.lane_token][0] = bezier_dis
                candidate_lanes[node.lane_token][1] = node.lane_idx
            if candidate_lanes[node.lane_token][2] > euclidean_dis:
                candidate_lanes[node.lane_token][2] = euclidean_dis
                candidate_lanes[node.lane_token][3] = node.lane_idx
            
        else:
            candidate_lanes[node.lane_token] = [0] * 4
            candidate_lanes[node.lane_token][0] = bezier_dis
            candidate_lanes[node.lane_token][1] = node.lane_idx
            candidate_lanes[node.lane_token][2] = euclidean_dis
            candidate_lanes[node.lane_token][3] = node.lane_idx
    for node_idx in adj_nodes_1:
        node = Vertices[node_idx]
        bezier_dis = calc_bezier(temp_xy, next_v, next_yaw, node)
        euclidean_dis = np.sqrt(((temp_xy - node.coors) ** 2).sum())
        if node.lane_token in candidate_lanes:
            if candidate_lanes[node.lane_token][0] < bezier_dis:
                candidate_lanes[node.lane_token][0] = bezier_dis
                candidate_lanes[node.lane_token][1] = node.lane_idx
            if candidate_lanes[node.lane_token][2] > euclidean_dis:
                candidate_lanes[node.lane_token][2] = euclidean_dis
                candidate_lanes[node.lane_token][3] = node.lane_idx
            
        else:
            candidate_lanes[node.lane_token] = [0] * 4
            candidate_lanes[node.lane_token][0] = bezier_dis
            candidate_lanes[node.lane_token][1] = node.lane_idx
            candidate_lanes[node.lane_token][2] = euclidean_dis
            candidate_lanes[node.lane_token][3] = node.lane_idx

    for key in candidate_lanes.keys():
        lane_nodes = lanes_vertices[key]
        #t1 = time.perf_counter()
        direction_distance = calc_direction_dis(lane_nodes, key, candidate_lanes, temp_xy, next_v, next_yaw)
        #t2 = time.perf_counter()
        #use_times[4] += t2 - t1
        update_euclidean_dis(lane_nodes, key, candidate_lanes, temp_xy)

    return candidate_lanes

eps = 1e-100

def dfs_closest_node(candidate_lanes, lane_token, lanes_vertices, map, add_idx):
    in_coming = map.get_incoming_lane_ids(lane_token)
    res = (1e100, 0, None)
    for key in in_coming:
        if key in candidate_lanes:
            if candidate_lanes[key][2] < candidate_lanes[lane_token][2]:
                temp_res = dfs_closest_node(candidate_lanes, key, lanes_vertices, map, add_idx)
                if temp_res[0] < res[0]:
                    res = temp_res
    #print(res)
    if res[0] == 1e100:
        lane_nodes = lanes_vertices[lane_token]
        temp_dis = candidate_lanes[lane_token]
        rest_idx = max(0, temp_dis[3] + add_idx - len(lane_nodes) + 1)
        node_idx = min(len(lane_nodes) - 1, temp_dis[3] + add_idx)
        res = (temp_dis[2], rest_idx, lane_nodes[node_idx])
    elif res[1] > 0:
        lane_nodes = lanes_vertices[lane_token]
        rest_idx = max(0, res[1] - len(lane_nodes))
        node_idx = min(len(lane_nodes) - 1, res[1] - 1)
        res = (res[0], rest_idx, lane_nodes[node_idx])
    return res

moving_attribute = 'cb5118da1ab342aa947717dc53544259'
def predict_step(scene : Scene, pa_scene, Vertices, Vertices_ordered_x, Vertices_ordered_y, p_trans_a,
                 p_trans_psi, N_STATES, a_rvs, psi_rvs, lanes_vertices, 
                 psi_rv, map, step, print_flag):#, use_times, lane_idxes, ans_nodes, psi_his):
    num_agents = len(scene.x[0])
    t_x = []
    t_y = []
    t_v = []
    t_yaw = []
    t_a_state = []
    t_psi_state = []
    t_a = []
    all_lanes = []
    lane_choosed = []
    if step == 0:
        offsets = []
        target_v = get_target_v()
        #print(target_v)
    else:
        offsets = scene.offsets
        target_v = scene.target_v
    if len(scene.self_weight):
        scene_idx = np.random.choice(a = len(scene.self_weight), p = scene.self_weight)
    else:
        scene_idx = -1
    for i in range(num_agents):
        x = scene.x[-1][i]
        y = scene.y[-1][i]
        v = scene.v[scene_idx][i]
        yaw = scene.yaw[scene_idx][i]
        a_state = scene.a_states[scene_idx][i]
        psi_state = scene.psi_states[scene_idx][i]

        candidate_lanes = copy.deepcopy(scene.candidate_lanes[i])
        lanes_token = list(candidate_lanes.keys())
        #if len(lanes_token) > 0:
        #    temp_values = np.array(list(candidate_lanes.values()))
        #    distance = temp_values[:,0]
        #    init_distance = distance / distance.sum()
        #    d_weight = temp_values[:,2] ** 4 + 1e-3
        #    distance = distance / d_weight
        #    distance = distance / distance.sum()
        
        #lane_nodes = lanes_nodes[i][scene.lane_idx[scene_idx][i]]
        #lane_x = lanes_x[i][scene.lane_idx[scene_idx][i]]
        #lane_y = lanes_y[i][scene.lane_idx[scene_idx][i]]
        #closest_node_idx = scene.closest_node[scene_idx][i]
        
        flag = True
        count = 0
        while flag:
            next_v_array = (v + mu_a * dt).reshape(-1)
            #v_weight = calc_v_weight(v)
            
            v_weight = (target_v - next_v_array + 1e-3) ** (-2) 
            #v_weight = np.clip(v_weight, None, 0.5) #速度修正最大值
            p_trans_a_fix = p_trans_a[a_state] * v_weight
            p_trans_a_fix = p_trans_a_fix / p_trans_a_fix.sum()
            #p_trans_a_fix = p_trans_a[a_state]
            next_a = np.random.choice(N_STATES, p = p_trans_a_fix) 
            a = a_rvs[next_a].rvs()
            next_v = v + a * dt

            count += 1
            if count <= 10 and next_v < -0.3:
                continue
            
            if len(lanes_token) > 0:#周围有车道
                if step == 0:
                    temp_values = np.array(list(candidate_lanes.values()))
                    distance = temp_values[:,0] ** 5
                    #init_distance = distance / distance.sum()
                    min_dis = temp_values[:,2].min()
                    d_weight = (temp_values[:,2] - min_dis) ** 2 + 1e-2
                    distance = distance / d_weight
                    distance = distance / distance.sum()
                    lane_idx = np.random.choice(len(distance), p = distance)
                    #if print_flag[0] and i == num_agents - 1:
                    #    print('0:\n', distance)
                    #    print_flag[0] = False
                    #if i == 5:
                    #    lane_idxes.append(lane_idx)
                    lane_token = lanes_token[lane_idx]
                    #print(lane_token)
                else:
                    lane_token = scene.lane_token[i]
                    if lane_token == None:
                        temp_values = np.array(list(candidate_lanes.values()))
                        distance = temp_values[:,0] ** 5
                        #init_distance = distance / distance.sum()
                        min_dis = temp_values[:,2].min()
                        d_weight = (temp_values[:,2] - min_dis) ** 2 + 1e-2
                        distance = distance / d_weight
                        distance = distance / distance.sum()
                        lane_idx = np.random.choice(len(distance), p = distance)
                        lane_token = lanes_token[lane_idx]

                
                #if lane_token not in candidate_lanes:
                #    print(candidate_lanes, step, i, lane_token, '1')
                #    print(scene.lane_token)
                #    print(len(scene.lane_token))
                
                add_idx = int(abs(next_v) * dt + 1)
                if len(lanes_vertices[lane_token]) > 30:
                    add_idx = add_idx * 4
                eucli_dis, _, node = dfs_closest_node(candidate_lanes, lane_token, lanes_vertices, map, add_idx)
                if node.lane_token == lane_token:
                    lane_nodes = lanes_vertices[lane_token]
                    length = len(lane_nodes)
                    closest_idx = candidate_lanes[lane_token][3]
                    #if closest_idx >= length - 3:
                    if node.lane_idx >= length - 3:
                        last_node = lane_nodes[-1]
                        next_nodes_idx = last_node.next
                        next_lanes = []
                        temp_distance = []
                        nodes_distance = []
                        for node_idx in next_nodes_idx:
                            temp_lane_node = Vertices[node_idx]
                            temp_lane_token = temp_lane_node.lane_token
                            if temp_lane_token in candidate_lanes:
                                next_lanes.append(temp_lane_token)
                                nodes_dis = cal_distance(temp_lane_node, last_node)
                                nodes_distance.append(nodes_dis)
                                temp_distance.append(candidate_lanes[temp_lane_token][0]) #提高相连的车道的概率
                                #print(node_idx, nodes_dis, temp_distance[-1])
                                #if print_flag[1] and i == num_agents - 1:
                                #    print('1:',candidate_lanes[temp_lane_token], nodes_dis)

                        if len(next_lanes) == 0:
                            node = None
                        else:
                            temp_distance = np.array(temp_distance) ** 5
                            nodes_distance = np.array(nodes_distance) - max(1,min(nodes_distance))
                            temp_distance = temp_distance / (nodes_distance ** 2 + 1e-3)
                            
                            temp_distance = temp_distance / temp_distance.sum()
                            lane_idx = np.random.choice(len(temp_distance), p = temp_distance)
                            #if print_flag[1] and i == num_agents - 1:
                            #    print('1:\n', temp_distance)
                            #    print_flag[1] = False
                            lane_token = next_lanes[lane_idx]
                            #if lane_token not in candidate_lanes:
                            #    print(candidate_lanes, step, i, lane_token, '2')
                            eucli_dis, _, node = dfs_closest_node(candidate_lanes, lane_token, lanes_vertices, map, add_idx)
                
            '''
            lane_idx = candidate_lanes[lane_token][1]
            node_idx = node_idxes[lane_idx]
            lane = lanes_vertices[lane_token]
            node = lane[node_idx]
            '''
            if node is None or len(lanes_token) == 0:#没有车道可指引
                p_trans = p_trans_psi[psi_state]
                if step == 0:
                    offsets.append(0)
            else:
                if step == 0:
                    dir_diff = diff_direction(node.direction, yaw)
                    if eucli_dis < 5 and dir_diff < 0.3:
                        offset = calc_lane_offset(x, y, node)
                        rand_off = np.random.rand() * 5
                        if rand_off < abs(offset):
                            offset = 0
                    else:
                        offset = 0
                    offsets.append(offset)
                    #print(offset)
                    
                else:
                    offset = offsets[i]

                if offset != 0:
                    offset_x = offset * np.cos(node.direction + np.pi / 2)
                    offset_y = offset * np.sin(node.direction + np.pi / 2)
                    node_x = node.coors[0] + offset_x
                    node_y = node.coors[1] + offset_y
                else:
                    node_x = node.coors[0]
                    node_y = node.coors[1]

                temp_psi = diff_direction(np.arctan2(node_y - y, node_x - x), yaw)
                #if i == 5:
                #    ans_nodes.append(node)
                #    psi_his.append(temp_psi)
                temp_psi = np.clip(temp_psi, -0.3, 0.3)

                p_trans = p_trans_psi[psi_state] * (psi_rv.pdf(temp_psi) + eps)
                p_trans = p_trans / p_trans.sum()
            next_psi = np.random.choice(N_STATES, p = p_trans)
            psi = psi_rvs[next_psi].rvs()

            next_yaw = yaw + psi
            dx = next_v * np.cos(next_yaw) * dt
            dy = next_v * np.sin(next_yaw) * dt
            next_x = x + dx
            next_y = y + dy

            flag = False
            
            #closest_node_idx, min_dis = get_closest_node_idx(lane_x, lane_y, next_node_idx, next_x, next_y)
            #if next_node_idx == 0:
            #    flag = min_dis > 2
            #else:
            #    flag = min_dis > 5
            #flag = min_dis > 5
            
        #t1 = time.perf_counter()
        update_candidate_lanes(candidate_lanes, x, y, next_x, next_y, next_yaw, next_v, Vertices, Vertices_ordered_x, Vertices_ordered_y,
                               lanes_vertices)#, use_times)
        #t2 = time.perf_counter()
        #use_times[1] += t2 - t1
        if len(lanes_token) == 0:
            lane_choosed.append(None)
        else:
            lane_choosed.append(lane_token)
        t_x.append(next_x)
        t_y.append(next_y)
        t_v.append(next_v)
        t_yaw.append(next_yaw)
        t_a_state.append(next_a)
        t_psi_state.append(next_psi)
        t_a.append(a)
        all_lanes.append(candidate_lanes)
    ans = Scene(t_x, t_y, t_v, t_yaw, scene.lane_idx[scene_idx], t_a_state, t_psi_state, pa_scene)
    ans.valid = count <= 10
    ans.a = [t_a]
    ans.set_lane(lane_choosed) #保存选择的车道
    ans.set_offsets(offsets)
    ans.set_candidate_lanes(all_lanes)
    ans.set_target_v(target_v)
    return ans

from sklearn.cluster import AgglomerativeClustering

def merge_scene(scenes, all_nodes):
    x = []
    y = []
    for scene in scenes:
        x.append(scene.x[-1])
        y.append(scene.y[-1])
    x = np.array(x)
    y = np.array(y)
    #print(x)
    #print(y)
    x = x.mean(axis = 0)
    y = y.mean(axis = 0)
    #print(x.shape)
    #print(scene.pa_scene)
    pa_scene = {}
    v = []
    yaw = []
    a = []
    a_states = []
    psi_states = []
    lane_idx = []
    closest_nodes = []
    for scene in scenes:
        for key in scene.pa_scene.keys():
            if key in pa_scene:
                pa_scene[key] += 1
            else:
                pa_scene[key] = 1
                v.append(scene.v[-1])
                yaw.append(scene.yaw[-1])
                a.append(scene.a[-1])
                a_states.append(scene.a_states[-1])
                psi_states.append(scene.psi_states[-1])
                lane_idx.append(scene.lane_idx[-1])
                closest_nodes.append(scene.closest_node[-1])
    
    ans = Scene()
    ans.x = [x]
    ans.y = [y]
    ans.v = v
    ans.yaw = yaw
    ans.lane_idx = lane_idx
    ans.closest_node = closest_nodes
    ans.a_states = a_states
    ans.psi_states = psi_states
    ans.pa_scene = pa_scene
    ans.set_glob_idx(all_nodes)
    ans.set_pa(all_nodes)
    ans.set_self()
    return ans

def cluster_scene(scenes, all_nodes):
    num_scene = len(scenes)
    num_agents = len(scenes[0].x[-1])
    ans = []
    xy = np.zeros((num_scene, 2, num_agents))
    for i in range(num_scene):
        scene = scenes[i]
        xy[i,0] = scene.x[-1]
        xy[i,1] = scene.y[-1]

    xy = xy.reshape(num_scene, -1)
    clustering = AgglomerativeClustering(n_clusters=None, linkage = 'ward', distance_threshold=0.01 * num_agents).fit(xy)
    #clustering = AgglomerativeClustering(n_clusters= num_scene // 10, linkage = 'ward').fit(xy)
    labels = clustering.labels_
    t_ans = [[] for _ in range(num_scene)]
    for j in range(len(labels)):
        t_ans[labels[j]].append(scenes[j])
    for j in range(num_scene):
        if len(t_ans[j]) == 0:
            break
        ans.append(merge_scene(t_ans[j], all_nodes))
    return ans

import time
repeat = 1000
steps = 12
dt = 0.5
def sample_predict_func(predict_agents, moving_input, static_input, helper, model, p_trans_a, mu_a, true_sigma_a,
                        p_trans_psi, mu_psi, true_sigma_psi, a_rv, psi_rv, p_eq_a, p_eq_psi, maps, Vertices_maps, ordered_x_maps,
                        ordered_y_maps, a_rvs, psi_rvs, lanes_vertices, sample_idx):
    ########return nodes
    print(f'start:{sample_idx}')
    t1 = time.perf_counter()
    init_all_x = []
    init_all_y = []
    init_all_v = []
    init_all_yaw = []
    init_all_a_state = []
    init_all_psi_state = []
    for j in range(len(predict_agents)):
        #init_x, init_y, init_v, init_yaw, a_init, psi_init = calc_init_state(predict_agents[j], helper, N_STATES, model, p_trans_a, 
        #                                                                     mu_a, true_sigma_a, p_trans_psi, mu_psi, true_sigma_psi,
        #                                                                     a_rv, psi_rv, p_eq_a, p_eq_psi)
        init_x, init_y, init_v, init_yaw, a_init, psi_init = calc_init_state_2(predict_agents[j], helper, N_STATES, model, p_trans_a, 
                                                                             mu_a, true_sigma_a, p_trans_psi, mu_psi, true_sigma_psi,
                                                                             a_rv, psi_rv, p_eq_a, p_eq_psi)
        init_all_x.append(init_x)
        init_all_y.append(init_y)
        init_all_v.append(init_v)
        init_all_yaw.append(init_yaw)
        init_all_a_state.append(a_init)
        init_all_psi_state.append(psi_init)

    lanes_nodes = []
    lanes_x = []
    lanes_y = []
    temp_num_lanes = []
    sample_token = predict_agents[-1]['sample_token']
    map_name = helper.get_map_name_from_sample_token(sample_token)
    map = maps[map_name]
    Vertices = Vertices_maps[map_name]
    Vertices_ordered_x = ordered_x_maps[map_name]
    Vertices_ordered_y = ordered_y_maps[map_name]
    
    all_lanes = []
    for j in range(len(predict_agents)):
        temp_xy = np.array([init_all_x[j], init_all_y[j]])
        adj_nodes = get_nodes_in_radius(init_all_x[j], init_all_y[j], 20, Vertices, Vertices_ordered_x, Vertices_ordered_y)
        lanes_closest = dict()
        v = init_all_v[j]
        yaw = init_all_yaw[j]
        for node_idx in adj_nodes:
            node = Vertices[node_idx]
            bezier_dis = calc_bezier(temp_xy, v, yaw, node)
            euclidean_dis = np.sqrt(((temp_xy - node.coors) ** 2).sum())
            if node.lane_token in lanes_closest:
                if lanes_closest[node.lane_token][0] < bezier_dis:
                    lanes_closest[node.lane_token][0] = bezier_dis
                    lanes_closest[node.lane_token][1] = node.lane_idx
                if lanes_closest[node.lane_token][2] > euclidean_dis:
                    lanes_closest[node.lane_token][2] = euclidean_dis
                    lanes_closest[node.lane_token][3] = node.lane_idx
                
            else:
                lanes_closest[node.lane_token] = [0] * 4
                lanes_closest[node.lane_token][0] = bezier_dis
                lanes_closest[node.lane_token][1] = node.lane_idx
                lanes_closest[node.lane_token][2] = euclidean_dis
                lanes_closest[node.lane_token][3] = node.lane_idx
        
        all_lanes.append(lanes_closest)

    ans = []
    all_nodes = []
    root_scene = Scene()
    root_scene.set_glob_idx(all_nodes)
    for j in range(repeat):
        #lane_idx = []
        a_states = []
        psi_states = []
        for k in range(len(predict_agents)):
            #lane_idx.append(np.random.choice(len(lanes_nodes[k])))
            a_states.append(np.random.choice(N_STATES, p = init_all_a_state[k]))
            psi_states.append(np.random.choice(N_STATES, p = init_all_psi_state[k]))
        temp_scene = Scene(init_all_x, init_all_y, init_all_v, init_all_yaw, None, a_states, psi_states, root_scene)
        temp_scene.set_glob_idx(all_nodes)
        temp_scene.set_pa(all_nodes)
        temp_scene.set_candidate_lanes(all_lanes)
        ans.append(temp_scene)
    init_risk_threshold = init_scene_risk(temp_scene, static_input, moving_input)
    t_count = []
    #use_times = [0] * 5
    #lane_idxes = []
    #ans_nodes = []
    #psi_his = []
    print_flag = [True, True]
    for j in range(steps):#steps
        #clear_output()
        #print(j)
        
        num_ans = len(ans)
        temp_ans = []
        count1 = 0
        count2 = 0
        if num_ans == 0:
            print(f'error:{sample_idx}, use time:{time.perf_counter() - t1}')
            return None, None, t_count
        #t_sum = 0
        #check_time = 0
        while(count2 < repeat and count1 < repeat * 10):
            #if count1 % 1000 == 0:
            #    print(count1, t_sum, check_time)
                #print(temp_scene.pa_scene)
            #    t_sum = 0
            #    check_time = 0
            idxes = np.random.permutation(num_ans)
            for idx in idxes:
                if count2 >= repeat:
                    break
                count1 += 1
                #t1 = time.perf_counter()
                temp_scene = predict_step(ans[idx], ans[idx], Vertices, Vertices_ordered_x, Vertices_ordered_y,
                                        p_trans_a, p_trans_psi, N_STATES, a_rvs, psi_rvs,
                                        lanes_vertices, psi_rv, map, j, print_flag)
                #t2 = time.perf_counter()
                #use_times[0] += t2 - t1

                if temp_scene.valid:
                    #t1 = time.perf_counter()
                    flag = check_scene(temp_scene, static_input, moving_input, init_risk_threshold)
                    #t2 = time.perf_counter()
                    #use_times[2] += t2 - t1
                    if not flag:
                        count2 += 1
                        temp_scene.set_glob_idx(all_nodes)
                        temp_scene.set_pa(all_nodes)
                        temp_ans.append(temp_scene)
                        
                #t3 = time.perf_counter()
                #check_time += t3 - t2
        
        ans = temp_ans
        t_count.append((count1, count2, len(ans)))
    print(f'end:{sample_idx}, use time:{time.perf_counter() - t1}')
    return all_nodes, t_count

def dfs_delete_useless(root:Scene, step, all_nodes, visited):
    if root.glob_idx in visited:
        return len(root.valid_sub_weight) > 0
    if len(root.sub_weight) == 0:
        if step == 13:
            return True
        else:
            return False
        
    for i in range(len(root.sub_scene)):
        next_node = all_nodes[root.sub_scene[i]]
        flag = dfs_delete_useless(next_node, step + 1, all_nodes, visited)
        if flag:
            root.valid_sub_scene.append(root.sub_scene[i])
            root.valid_sub_weight.append(root.sub_weight[i])
    visited.add(root.glob_idx)
    return len(root.valid_sub_weight) > 0

def dfs2(root:Scene, step, all_nodes, x, y, all_x, all_y):
    #if step == 11:
    #    print(root.glob_idx)
    if step > 1:
        x.append(root.x[-1])
        y.append(root.y[-1])
    if len(root.valid_sub_scene) == 0:
        all_x.append(copy.deepcopy(x))
        all_y.append(copy.deepcopy(y))
        x.pop()
        y.pop()
        return
        
    for i in range(len(root.valid_sub_scene)):
        next_node = all_nodes[root.valid_sub_scene[i]]
        dfs2(next_node, step + 1, all_nodes, x, y, all_x, all_y)
    if step > 1:
        x.pop()
        y.pop()

def sample_from_DAG(root:Scene, all_nodes, step, x, y):
    if step > 1:
        x.append(root.x[-1])
        y.append(root.y[-1])
    if len(root.valid_sub_weight) == 0:
        return
    if type(root.valid_sub_weight) == list:
        root.valid_sub_weight = np.array(root.valid_sub_weight)
        root.valid_sub_weight = root.valid_sub_weight / root.valid_sub_weight.sum()
    #print(root.sub_scene)
    #print(root.sub_weight)
    #print(root.valid_sub_weight)
    next_idx = np.random.choice(a = root.valid_sub_scene, p = root.valid_sub_weight)
    next_node = all_nodes[next_idx]
    sample_from_DAG(next_node, all_nodes, step + 1, x, y)

def sample_predict_func_2(predict_agents, moving_input, static_input, helper, 
                         model, p_trans_a, mu_a, true_sigma_a, p_trans_psi, 
                         mu_psi, true_sigma_psi, a_rv, psi_rv, p_eq_a, p_eq_psi, 
                         maps, Vertices_maps, ordered_x_maps, ordered_y_maps, a_rvs, 
                         psi_rvs, lanes_vertices, sample_idx):
    #return x,y
    print(f'start:{sample_idx}')
    t1 = time.perf_counter()
    init_all_x = []
    init_all_y = []
    init_all_v = []
    init_all_yaw = []
    init_all_a_state = []
    init_all_psi_state = []
    for j in range(len(predict_agents)):
        #init_x, init_y, init_v, init_yaw, a_init, psi_init = calc_init_state(predict_agents[j], helper, N_STATES, model, p_trans_a, 
        #                                                                     mu_a, true_sigma_a, p_trans_psi, mu_psi, true_sigma_psi,
        #                                                                     a_rv, psi_rv, p_eq_a, p_eq_psi)
        init_x, init_y, init_v, init_yaw, a_init, psi_init = calc_init_state_2(predict_agents[j], helper, N_STATES, model, p_trans_a, 
                                                                             mu_a, true_sigma_a, p_trans_psi, mu_psi, true_sigma_psi,
                                                                             a_rv, psi_rv, p_eq_a, p_eq_psi)
        init_all_x.append(init_x)
        init_all_y.append(init_y)
        init_all_v.append(init_v)
        init_all_yaw.append(init_yaw)
        init_all_a_state.append(a_init)
        init_all_psi_state.append(psi_init)

    lanes_nodes = []
    lanes_x = []
    lanes_y = []
    temp_num_lanes = []
    sample_token = predict_agents[-1]['sample_token']
    map_name = helper.get_map_name_from_sample_token(sample_token)
    map = maps[map_name]
    Vertices = Vertices_maps[map_name]
    Vertices_ordered_x = ordered_x_maps[map_name]
    Vertices_ordered_y = ordered_y_maps[map_name]
    
    all_lanes = []
    for j in range(len(predict_agents)):
        temp_xy = np.array([init_all_x[j], init_all_y[j]])
        adj_nodes = get_nodes_in_radius(init_all_x[j], init_all_y[j], 20, Vertices, Vertices_ordered_x, Vertices_ordered_y)
        lanes_closest = dict()
        v = init_all_v[j]
        yaw = init_all_yaw[j]
        for node_idx in adj_nodes:
            node = Vertices[node_idx]
            bezier_dis = calc_bezier(temp_xy, v, yaw, node)
            euclidean_dis = np.sqrt(((temp_xy - node.coors) ** 2).sum())
            if node.lane_token in lanes_closest:
                if lanes_closest[node.lane_token][0] < bezier_dis:
                    lanes_closest[node.lane_token][0] = bezier_dis
                    lanes_closest[node.lane_token][1] = node.lane_idx
                if lanes_closest[node.lane_token][2] > euclidean_dis:
                    lanes_closest[node.lane_token][2] = euclidean_dis
                    lanes_closest[node.lane_token][3] = node.lane_idx
                
            else:
                lanes_closest[node.lane_token] = [0] * 4
                lanes_closest[node.lane_token][0] = bezier_dis
                lanes_closest[node.lane_token][1] = node.lane_idx
                lanes_closest[node.lane_token][2] = euclidean_dis
                lanes_closest[node.lane_token][3] = node.lane_idx
        
        all_lanes.append(lanes_closest)

    ans = []
    all_nodes = []
    root_scene = Scene()
    root_scene.set_glob_idx(all_nodes)
    for j in range(repeat):
        #lane_idx = []
        a_states = []
        psi_states = []
        for k in range(len(predict_agents)):
            #lane_idx.append(np.random.choice(len(lanes_nodes[k])))
            a_states.append(np.random.choice(N_STATES, p = init_all_a_state[k]))
            psi_states.append(np.random.choice(N_STATES, p = init_all_psi_state[k]))
        temp_scene = Scene(init_all_x, init_all_y, init_all_v, init_all_yaw, None, a_states, psi_states, root_scene)
        temp_scene.set_glob_idx(all_nodes)
        temp_scene.set_pa(all_nodes)
        temp_scene.set_candidate_lanes(all_lanes)
        ans.append(temp_scene)
    init_risk_threshold = init_scene_risk(temp_scene, static_input, moving_input)
    t_count = []
    #use_times = [0] * 5
    #lane_idxes = []
    #ans_nodes = []
    #psi_his = []
    print_flag = [True, True]
    for j in range(steps):#steps
        #clear_output()
        #print(j)
        
        num_ans = len(ans)
        temp_ans = []
        count1 = 0
        count2 = 0
        if num_ans == 0:
            print(f'error:{sample_idx}, use time:{time.perf_counter() - t1}')
            return None, None, t_count
        #t_sum = 0
        #check_time = 0
        while(count2 < repeat and count1 < repeat * 10):
            #if count1 % 1000 == 0:
            #    print(count1, t_sum, check_time)
                #print(temp_scene.pa_scene)
            #    t_sum = 0
            #    check_time = 0
            idxes = np.random.permutation(num_ans)
            for idx in idxes:
                if count2 >= repeat:
                    break
                count1 += 1
                #t1 = time.perf_counter()
                temp_scene = predict_step(ans[idx], ans[idx], Vertices, Vertices_ordered_x, Vertices_ordered_y,
                                        p_trans_a, p_trans_psi, N_STATES, a_rvs, psi_rvs,
                                        lanes_vertices, psi_rv, map, j, print_flag)
                #t2 = time.perf_counter()
                #use_times[0] += t2 - t1

                if temp_scene.valid:
                    #t1 = time.perf_counter()
                    flag = check_scene(temp_scene, static_input, moving_input, init_risk_threshold)
                    #t2 = time.perf_counter()
                    #use_times[2] += t2 - t1
                    if not flag:
                        count2 += 1
                        temp_scene.set_glob_idx(all_nodes)
                        temp_scene.set_pa(all_nodes)
                        temp_ans.append(temp_scene)
                        
                #t3 = time.perf_counter()
                #check_time += t3 - t2
        
        ans = temp_ans
        t_count.append((count1, count2, len(ans)))
    
    visited = set()
    dfs_delete_useless(all_nodes[0], 0, all_nodes, visited)
    num_ans = len(ans)
    temp_x = np.zeros((num_ans,12))
    temp_y = np.zeros((num_ans,12))
    all_x = []
    all_y = []
    t_x = []
    t_y = []
    dfs2(root_scene, 0, all_nodes, t_x, t_y, all_x, all_y)
    for j in range(num_ans):
        for k in range(12):
            temp_x[j][k] = all_x[j][k][-1]
            temp_y[j][k] = all_y[j][k][-1]

    print(f'end:{sample_idx}, use time:{time.perf_counter() - t1}')
    return temp_x, temp_y , t_count
