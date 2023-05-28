from utils_map import *
import numpy as np

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

alpha = 0.1
beta_1 = 0.1
def calculate_field(region, agent_loc, agent_direction, h, sigma_x, sigma_y, velocity_x = 0, velocity_y = 0, a = 0):
    
    agent_region_direction = cal_direction_vector(agent_loc, region)
    #print('agent_region_direction shape:', agent_region_direction.shape)
    distance = cal_distance_vector(agent_loc, region)
    directions_diff = diff_direction_vector(agent_region_direction, agent_direction)
    dx = - distance * np.sin(directions_diff)
    dy = distance * np.cos(directions_diff)
    sigma_x += sigma_x * alpha * velocity_x
    sigma_y += sigma_y * alpha * velocity_y
    h = h * np.exp(beta_1 * a * np.cos(directions_diff))
    result = h * np.exp(-(dx ** 2 / sigma_x ** 2 + dy ** 2 / sigma_y ** 2))
    return result

def calculate_force(agent_loc, agent_direction, target_loc, H, sigma_x, sigma_y, velocity_x = 0, velocity_y = 0, a = 0):
    distance = cal_distance_vector(agent_loc, target_loc)
    agent_to_target_direction = cal_direction_vector(agent_loc, target_loc)
    direction_diff = diff_direction(agent_to_target_direction, agent_direction)
    dx = - distance * np.sin(direction_diff)
    dy = distance * np.cos(direction_diff)
    sigma_x += sigma_x * alpha * velocity_x
    sigma_y += sigma_y * alpha * velocity_y
    h = H * np.exp(beta_1 * a * np.cos(direction_diff))
    F_x = h * dx / sigma_x ** 2 * np.exp(-(dx ** 2 / sigma_x ** 2 + dy ** 2 / sigma_y ** 2))
    F_y = h * dy / sigma_y ** 2 * np.exp(-(dx ** 2 / sigma_x ** 2 + dy ** 2 / sigma_y ** 2))
    #print(F_x, F_y, agent_to_target_direction, dx, dy)
    ans = np.zeros(2)
    ans += (F_x * np.sin(agent_direction), - F_x * np.cos(agent_direction))
    ans += (F_y * np.cos(agent_direction), F_y * np.sin(agent_direction))
    return ans

def cal_direction_total_field(region_x, region_y, sample_x, sample_y):
    dx = region_x - sample_x
    dy = region_y - sample_y
    res_shape = dx.shape
    dx = dx.reshape(-1)
    dy = dy.reshape(-1)
    ans = np.arctan2(dy, dx)
    return ans.reshape(res_shape)
def cal_distance_total_field(region_x, region_y, sample_x, sample_y):
    dx = region_x - sample_x
    dy = region_y - sample_y
    return np.sqrt(dx ** 2 + dy ** 2)
def total_field(region_x, region_y, sample_x, sample_y, sample_direction, sample_v_x, sample_v_y, sample_sigma_x, sample_sigma_y, sample_a, sample_H):
    region_shape = region_x.shape
    
    if len(sample_x) == 0:
        return np.zeros(region_shape), None

    region_x = region_x.reshape(-1,1)
    region_y = region_y.reshape(-1,1)

    sample_region_direction = cal_direction_total_field(region_x, region_y, sample_x, sample_y)
    #print('agent_region_direction shape:', agent_region_direction.shape)
    distance = cal_distance_total_field(region_x, region_y, sample_x, sample_y)
    directions_diff = diff_direction_vector(sample_region_direction, sample_direction)
    dx = - distance * np.sin(directions_diff)
    dy = distance * np.cos(directions_diff)
    sigma_x = sample_sigma_x + sample_v_x
    sigma_y = sample_sigma_y + alpha * (sample_v_y ** 2) * (2 - np.abs(directions_diff) / np.pi)
    h = sample_H #* np.exp(beta_1 * sample_v_y * (np.cos(directions_diff) + 1))
    result = h * np.exp(-(dx ** 2 / sigma_x ** 2 + dy ** 2 / sigma_y ** 2))
    sum_result = result.sum(axis = -1).reshape(region_shape)
    result = result.reshape(region_shape + (-1,))
    return sum_result, result