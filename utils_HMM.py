import sys
import os
from time import time
import pymc as pm
from IPython.core.pylabtools import figsize
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multinomial
from typing import List
import scipy.stats as stats
import pytensor.tensor as at
import pytensor.tensor.slinalg as sla 
import pytensor
import warnings
import arviz as az
print(f"Running on PyMC v{pm.__version__}")
warnings.simplefilter(action="ignore", category=FutureWarning)
#from easydl import clear_output
from IPython.display import clear_output
from nuscenes.nuscenes import NuScenes
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.prediction import PredictHelper
from nuscenes.prediction.input_representation import static_layers
from nuscenes.map_expansion.bitmap import BitMap
from nuscenes.map_expansion import arcline_path_utils


#DATAROOT='/home/hncui/share/Nuscenes/nu_data'
#nusc = NuScenes(version='v1.0-trainval', dataroot=DATAROOT, verbose=True)
#helper = PredictHelper(nusc)
#token_list = get_prediction_challenge_split("train",dataroot=DATAROOT)
#maps = static_layers.load_all_maps(helper)
from utils_map import *

def get_histroy_acc(ins_sample_token, helper):
    instance_token, sample_token = ins_sample_token.split('_')
    past_anns_global = helper.get_past_for_agent(instance_token, sample_token, seconds=2, in_agent_frame=False, just_xy=False)
    past_anns_global = past_anns_global[::-1]
    ann = helper.get_sample_annotation(instance_token, sample_token)
    past_anns_global.append(ann)
    accs = []
    for ann in past_anns_global:
        acc = helper.get_acceleration_for_agent(ann['instance_token'], ann['sample_token'])
        if not np.isnan(acc):
            accs.append(acc)
    return accs
from pyquaternion import Quaternion
from nuscenes.eval.common.utils import quaternion_yaw

def get_histroy_yaw(ins_sample_token, helper, nusc):
    instance_token, sample_token = ins_sample_token.split('_')
    past_anns_global = helper.get_past_for_agent(instance_token, sample_token, seconds=2, in_agent_frame=False, just_xy=False)
    past_anns_global = past_anns_global[::-1]
    ann = helper.get_sample_annotation(instance_token, sample_token)
    past_anns_global.append(ann)
    yaws = []
    for ann in past_anns_global:
        v = helper.get_velocity_for_agent(ann['instance_token'], ann['sample_token'])
        if not np.isnan(v):
            if v > 1:
                ann_pre = nusc.get('sample_annotation', ann['prev'])
                xy1 = np.array(ann_pre['translation'][:2])
                xy2 = np.array(ann['translation'][:2])
                d_xy = xy2 - xy1
                yaw = np.arctan2(d_xy[1], d_xy[0])
                yaws.append(yaw)
            else:
                yaw = quaternion_yaw(Quaternion(ann['rotation']))
                yaws.append(yaw)
    return yaws, yaw

def get_future_acc(ins_sample_token, helper):
    instance_token, sample_token = ins_sample_token.split('_')
    future_anns_global = helper.get_future_for_agent(instance_token, sample_token, seconds=6, in_agent_frame=False, just_xy=False)
    accs = []
    for ann in future_anns_global:
        acc = helper.get_acceleration_for_agent(ann['instance_token'], ann['sample_token'])
        if not np.isnan(acc):
            accs.append(acc)
    return accs

def get_future_yaw(ins_sample_token, current_yaw, helper, nusc):
    instance_token, sample_token = ins_sample_token.split('_')
    future_anns_global = helper.get_future_for_agent(instance_token, sample_token, seconds=6, in_agent_frame=False, just_xy=False)
    yaws = [current_yaw]
    for ann in future_anns_global:
        v = helper.get_velocity_for_agent(ann['instance_token'], ann['sample_token'])
        if not np.isnan(v):
            if v > 1:
                ann_pre = nusc.get('sample_annotation', ann['prev'])
                xy1 = np.array(ann_pre['translation'][:2])
                xy2 = np.array(ann['translation'][:2])
                d_xy = xy2 - xy1
                yaw = np.arctan2(d_xy[1], d_xy[0])
                yaws.append(yaw)
            else:
                yaw = quaternion_yaw(Quaternion(ann['rotation']))
                yaws.append(yaw)
    return yaws

def solve_equilibrium(n_states, p_transition, name):
    A = at.dmatrix('A')
    A = at.eye(n_states) - p_transition + at.ones(shape=(n_states, n_states))
    p_equilibrium = pm.Deterministic("p_equilibrium_" + name, sla.solve(A.T, at.ones(shape=(n_states))))
    return p_equilibrium  

def one_step(logp_x, value, loss_, P_trans, N_STATES):
    next_loss = loss_.reshape((-1,N_STATES,1)) + pm.math.log(P_trans)
    
    loss_mean = at.mean(next_loss, axis = 1, keepdims = True)
    next_loss = loss_mean + at.logsumexp(next_loss - loss_mean, axis = 1, keepdims = True)
    next_loss = next_loss.reshape((-1,N_STATES))
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
                               non_sequences=[P_trans])

    result_loss = results[-1]
    mean_log = at.mean(result_loss, axis = 1, keepdims=True)
    result_loss = mean_log + pm.logsumexp(result_loss - mean_log, axis = 1, keepdims=True)
    return result_loss.reshape((-1,))





def get_HMM(nusc, helper, token_list, hyper_parameters):
    all_past_accs = []
    all_past_yaws = []
    all_future_accs = []
    all_future_yaws = []
    for ins_sample_token in token_list:
        past_acc = get_histroy_acc(ins_sample_token, helper)
        past_yaws, current_yaw = get_histroy_yaw(ins_sample_token, helper, nusc)
        future_accs = get_future_acc(ins_sample_token, helper)
        future_yaws = get_future_yaw(ins_sample_token, current_yaw, helper, nusc)
        all_past_accs.append(past_acc)
        all_past_yaws.append(past_yaws)
        all_future_accs.append(future_accs)
        all_future_yaws.append(future_yaws)
    all_past_psis = []
    all_future_psis = []
    for i in range(len(all_past_yaws)):
        past_yaws = np.array(all_past_yaws[i])
        future_yaws = np.array(all_future_yaws[i])
        if len(past_yaws) > 1:
            all_past_psis.append(list(diff_direction_vector(past_yaws[1:], past_yaws[:-1])))
        else:
            all_past_psis.append([])
        all_future_psis.append(list(diff_direction_vector(future_yaws[1:], future_yaws[:-1])))
    total_accs = []
    total_psis = []
    for i in range(len(all_past_yaws)):
        total_accs.append(all_past_accs[i] + all_future_accs[i])
        total_psis.append(all_past_psis[i] + all_future_psis[i])
    for i in range(len(all_past_yaws)):
        while len(all_past_accs[i]) < 5:
            all_past_accs[i].append(-10)
        while len(all_past_psis[i]) < 4:
            all_past_psis[i].append(-10)
        while len(all_future_accs[i]) < 12:
            all_future_accs[i].append(-10)
        while len(all_future_psis[i]) < 12:
            all_future_psis[i].append(-10)
        while len(total_accs[i]) < 17:
            total_accs[i].append(-10)
        while len(total_psis[i]) < 16:
            total_psis[i].append(-10)
    past_accs_np = np.array(all_past_accs)
    past_psis_np = np.array(all_past_psis)
    future_accs_np = np.array(all_future_accs)
    future_psis_np = np.array(all_future_psis)
    total_accs_np = np.array(total_accs)
    total_psis_np = np.array(total_psis)
    mask_past_accs = np.expand_dims(~(past_accs_np == -10).T, -1)
    mask_past_psis = np.expand_dims(~(past_psis_np == -10).T, -1)
    mask_future_accs = np.expand_dims(~(future_accs_np == -10).T, -1)
    mask_future_psis = np.expand_dims(~(future_psis_np == -10).T, -1)
    mask_total_accs = np.expand_dims(~(total_accs_np == -10).T, -1)
    mask_total_psis = np.expand_dims(~(total_psis_np == -10).T, -1)
    past_accs_np.shape, future_accs_np.shape
    N_STATES = hyper_parameters['num_states']
    past_t = past_psis_np.shape[1]
    future_t = future_psis_np.shape[1]
    T = total_psis_np.shape[1]
    num_seq = past_psis_np.shape[0]
    input_psi_past = past_psis_np.T.reshape((past_t,num_seq,1))
    input_psi_future = future_psis_np.T.reshape((future_t,num_seq,1))
    input_psi_total = total_psis_np.T.reshape((T,num_seq,1))
    dt = 0.5
    trans_init =  np.ones((N_STATES, N_STATES))

    with pm.Model() as a_model:

        P_trans_psi = pm.Dirichlet('P_trans_psi', a=trans_init, shape=(N_STATES,N_STATES))

        mu_psi = at.linspace(-0.3, 0.3, N_STATES)

        sigma_psi = at.ones((N_STATES,)) * 0.01
        P_init = pm.MutableData('P_init_psi', np.ones((num_seq, N_STATES)) / N_STATES)
        #P_equilibrium_a = solve_equilibrium(N_STATES, P_trans_a, '')

        a = pm.DensityDist('a', mu_psi, sigma_psi, P_init, P_trans_psi, N_STATES, logp=mylogp, observed = input_psi_total)
    with a_model:
        a_map = pm.find_MAP()
    with pm.Model() as b_model:

        P_trans_psi = pm.MutableData('P_trans_psi', a_map['P_trans_psi'])

        mu_psi = at.linspace(-0.3, 0.3, N_STATES)

        sigma_psi = at.ones((N_STATES,)) * 0.01
        P_init = pm.Dirichlet('P_init_psi', a=np.ones((num_seq, N_STATES)), shape=(num_seq,N_STATES))
        #P_equilibrium_a = solve_equilibrium(N_STATES, P_trans_a, '')

        a = pm.DensityDist('a', mu_psi, sigma_psi, P_init, P_trans_psi, N_STATES, logp=mylogp, observed = input_psi_past) 
    with b_model:
        b_map = pm.find_MAP()
    P_trans_psi_history = [a_map['P_trans_psi']]
    P_init_psi_history = [b_map['P_init_psi']]
    for i in range(100):
        print(i)
        P_trans_psi_pre = P_trans_psi_history[-1]
        P_init_psi_pre = P_init_psi_history[-1]
        with a_model:
            a_model.set_data('P_init_psi', P_init_psi_pre)
            temp_a_map = pm.find_MAP(start = {'P_trans_psi':P_trans_psi_pre})
        with b_model:
            b_model.set_data('P_trans_psi', P_trans_psi_pre)
            temp_b_map = pm.find_MAP()
            #temp_b_map = pm.find_MAP(start = {'P_init_a':P_init_a_pre})
        P_trans_psi_history.append(temp_a_map['P_trans_psi'])
        P_init_psi_history.append(temp_b_map['P_init_psi'])
        P_trans_psi_current = P_trans_psi_history[-1]
        P_init_psi_current = P_init_psi_history[-1]
        diff1 = np.abs(P_trans_psi_current - P_trans_psi_pre).sum()
        diff2 = np.abs(P_init_psi_current - P_init_psi_pre).sum()
        print(diff1, diff2)
        if diff1 < 1e-20 and diff2 < 1e-21:
            print(i)
            break
    P_trans_psi_history_np = np.array(P_trans_psi_history)
    P_init_psi_history_np = np.array(P_init_psi_history)

    past_t = past_accs_np.shape[1]
    future_t = future_accs_np.shape[1]
    T = total_accs_np.shape[1]
    num_seq = past_accs_np.shape[0]
    input_a_past = past_accs_np.T.reshape((past_t,num_seq,1))
    input_a_future = future_accs_np.T.reshape((future_t,num_seq,1))
    input_a_total = total_accs_np.T.reshape((T,num_seq,1))
    dt = 0.5
    trans_init =  np.ones((N_STATES, N_STATES))

    with pm.Model() as a_model:

        P_trans_a = pm.Dirichlet('P_trans_a', a=trans_init, shape=(N_STATES,N_STATES))

        mu_a = at.linspace(-5,5,N_STATES)

        sigma_a = at.ones((N_STATES,)) * 0.15
        P_init = pm.MutableData('P_init_a', np.ones((num_seq, N_STATES)) / N_STATES)
        #P_equilibrium_a = solve_equilibrium(N_STATES, P_trans_a, '')

        a = pm.DensityDist('a', mu_a, sigma_a, P_init, P_trans_a, N_STATES, logp=mylogp, observed = input_a_total) 

    with a_model:
        a_map = pm.find_MAP()

    with pm.Model() as b_model:

        P_trans_a = pm.MutableData('P_trans_a', a_map['P_trans_a'])

        mu_a = at.linspace(-5,5,N_STATES)

        sigma_a = at.ones((N_STATES,)) * 0.15
        P_init = pm.Dirichlet('P_init_a', a=np.ones((num_seq, N_STATES)), shape=(num_seq,N_STATES))
        #P_equilibrium_a = solve_equilibrium(N_STATES, P_trans_a, '')

        a = pm.DensityDist('a', mu_a, sigma_a, P_init, P_trans_a, N_STATES, logp=mylogp, observed = input_a_past) 

    with b_model:
        b_map = pm.find_MAP()

    P_trans_a_history = [a_map['P_trans_a']]
    P_init_a_history = [b_map['P_init_a']]

    for i in range(100):
        print(i)
        P_trans_a_pre = P_trans_a_history[-1]
        P_init_a_pre = P_init_a_history[-1]
        with a_model:
            a_model.set_data('P_init_a', P_init_a_pre)
            temp_a_map = pm.find_MAP(start = {'P_trans_a':P_trans_a_pre})
        with b_model:
            b_model.set_data('P_trans_a', P_trans_a_pre)
            temp_b_map = pm.find_MAP()
            #temp_b_map = pm.find_MAP(start = {'P_init_a':P_init_a_pre})
        P_trans_a_history.append(temp_a_map['P_trans_a'])
        P_init_a_history.append(temp_b_map['P_init_a'])
        P_trans_a_current = P_trans_a_history[-1]
        P_init_a_current = P_init_a_history[-1]
        diff1 = np.abs(P_trans_a_current - P_trans_a_pre).sum()
        diff2 = np.abs(P_init_a_current - P_init_a_pre).sum()
        if diff1 < 1e-20 and diff2 < 1e-21:
            print(i)
            break
    P_trans_a_history_np = np.array(P_trans_a_history)
    P_init_a_history_np = np.array(P_init_a_history)

    return P_trans_a_history_np, P_trans_psi_history_np