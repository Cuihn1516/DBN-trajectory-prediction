from pyquaternion import Quaternion
from nuscenes.eval.common.utils import quaternion_yaw
import numpy as np
from utils_map import diff_direction

def get_field_input(ann, main_agent, field_input, moving = False, helper = None):
    agent_x = np.array(ann['translation'][0])
    agent_y = np.array(ann['translation'][1])
    agent_direction = quaternion_yaw(Quaternion(ann['rotation']))
    if 'vehicle' not in ann['category_name']:
        sigma_x = ann['size'][0] + 1
        sigma_y = ann['size'][1] + 1
    else:
        sigma_x = ann['size'][0]
        sigma_y = ann['size'][1]
    v_x = 0
    v_y = 0
    if moving:
        v_y = helper.get_velocity_for_agent(ann['instance_token'], ann['sample_token'])
        if np.isnan(v_y): v_y = 0
    a = 0
    if moving:
        a = helper.get_acceleration_for_agent(ann['instance_token'], ann['sample_token'])
        if np.isnan(a): a = 0
    h = 1 #可以修改，根据不同情况
    if type(field_input['sample_x']) == np.ndarray:
        field_input['sample_x'] = np.append(field_input['sample_x'], agent_x)
        field_input['sample_y'] = np.append(field_input['sample_y'], agent_y)
        field_input['sample_direction'] = np.append(field_input['sample_direction'], agent_direction)
        field_input['sample_v_x'] = np.append(field_input['sample_v_x'], v_x)
        field_input['sample_v_y'] = np.append(field_input['sample_v_y'], v_y)
        field_input['sample_sigma_x'] = np.append(field_input['sample_sigma_x'], sigma_x)
        field_input['sample_sigma_y'] = np.append(field_input['sample_sigma_y'], sigma_y)
        field_input['sample_a'] = np.append(field_input['sample_a'], a)
        field_input['sample_H'] = np.append(field_input['sample_H'], h)
    else:
        field_input['sample_x'].append(agent_x)
        field_input['sample_y'].append(agent_y)
        field_input['sample_direction'].append(agent_direction)
        field_input['sample_v_x'].append(v_x)
        field_input['sample_v_y'].append(v_y)
        field_input['sample_sigma_x'].append(sigma_x)
        field_input['sample_sigma_y'].append(sigma_y)
        field_input['sample_a'].append(a)
        field_input['sample_H'].append(h)
    return len(field_input['sample_x'])

def get_field_input_lane(node, main_agent, field_input):
    node_x = node.coors[0]
    node_y = node.coors[1]
    node_direction = node.direction
    sigma_x = 0.5
    sigma_y = 0.5
    v_x = 0
    v_y = 0
    a = 0
    h = -1 * np.cos(diff_direction(main_agent['direction'], node_direction))

    field_input['sample_x'].append(node_x)
    field_input['sample_y'].append(node_y)
    field_input['sample_direction'].append(node_direction)
    field_input['sample_v_x'].append(v_x)
    field_input['sample_v_y'].append(v_y)
    field_input['sample_sigma_x'].append(sigma_x)
    field_input['sample_sigma_y'].append(sigma_y)
    field_input['sample_a'].append(a)
    field_input['sample_H'].append(h)
    return len(field_input['sample_x'])

def field_input_init():
    return {'sample_x' : [],
            'sample_y' : [],
            'sample_direction' : [],
            'sample_v_x' : [],
            'sample_v_y' : [],
            'sample_sigma_x' : [],
            'sample_sigma_y' : [],
            'sample_a' : [],
            'sample_H' : []}

def get_label(helper, X):
    subsequences_X = []
    gt_all = []
    for i in range(len(X)):
        ann_list = helper.get_future_for_agent(X[i]['instance_token'], X[i]['sample_token'], seconds = 6, in_agent_frame= False, just_xy= False)
        samples = []
        gt = []
        for ann in ann_list:
            samples.append(ann['sample_token'])
            gt.append(ann['translation'][:2])
        subsequences_X.append(samples)
        gt_all.append(gt)
    return gt_all, subsequences_X

def get_sample_input(helper, nusc, X):
    sample_input = {
        'animal':[],
        'human':[],
        'object':[],
        'cycle':[],
        'vehicle':[]
    }
    for x in X:
        sample_annotation = helper.get_annotations_for_sample(x['sample_token'])
        count = 0
        animal = []
        human = {'sit':[], 'moving':[]}
        object = []
        cycle = {'with_rider':[], 'without_rider':[]}
        vehicle = {'parked':[], 'moving':[], 'stopped':[]}
        for ann in sample_annotation:
            #if ann['token'] == x['token']:
            #    continue
            cate_name = ann['category_name']
            if cate_name == 'animal':
                animal.append(ann)
            elif 'object' in cate_name:
                object.append(ann)
            elif 'human' in cate_name:
                if len(ann['attribute_tokens']):#没有attribute的情况
                    attribute = nusc.get('attribute', ann['attribute_tokens'][0])
                else:
                    attribute = {'name':'moving'}
                if 'sitting' in attribute['name']:
                    human['sit'].append(ann)
                else:
                    human['moving'].append(ann)
            elif 'cycle' in cate_name:
                if len(ann['attribute_tokens']):#没有attribute的情况
                    attribute = nusc.get('attribute', ann['attribute_tokens'][0])
                else:
                    attribute = {'name':'without_rider'}
                if attribute['name'] == 'cycle.with_rider':
                    cycle['with_rider'].append(ann)
                else:
                    cycle['without_rider'].append(ann)
            else:
                if len(ann['attribute_tokens']):#没有attribute的情况
                    attribute = nusc.get('attribute', ann['attribute_tokens'][0])
                else:
                    attribute = {'name':'vehicle.moving'}
                if attribute['name'] == 'vehicle.parked':
                    vehicle['parked'].append(ann)
                elif attribute['name'] == 'vehicle.stopped':
                    vehicle['stopped'].append(ann)
                else:
                    vehicle['moving'].append(ann)
        sample_input['animal'].append(animal)
        sample_input['human'].append(human)
        sample_input['object'].append(object)
        sample_input['cycle'].append(cycle)
        sample_input['vehicle'].append(vehicle)
        '''
        x_coor = np.array(x['translation'][:2])
        for ann in sample_annotation: #是否只收集
            object_coor = np.array(ann['translation'][:2])
            if np.sqrt(np.sum((x_coor - object_coor) ** 2)) < 20:
                count += 1
        print(count)
        '''
    return sample_input

def calc_ann_dis(ann1, ann2):
    xy1 = np.array(ann1['translation'][:2])
    xy2 = np.array(ann2['translation'][:2])
    return np.sqrt(((xy1 - xy2)**2).sum())

def get_moving_static_inputs(X, sample_input, helper):
    dt = 0.5
    radius = 20
    static_inputs = {}
    moving_inputs = {}
    moving_anns = {}

    for i, main_agent in enumerate(X):#
        sample_token = main_agent['sample_token']
        instance_token = main_agent['instance_token']
        ins_sample = instance_token + '_' + sample_token
        static_input = field_input_init()
        moving_vehicles = field_input_init()
        moving_anns[ins_sample] = []
        #周围agent输入
        for key in sample_input.keys():
            if key == 'human':
                sample_anns = sample_input[key][i]['sit']
                for ann in sample_anns:
                    get_field_input(ann, main_agent, static_input, moving = False, helper = helper)
                sample_anns = sample_input[key][i]['moving']
                for ann in sample_anns:
                    get_field_input(ann, main_agent, static_input, moving = False, helper = helper)
                    pass
            elif key == 'cycle':
                sample_anns = sample_input[key][i]['without_rider']
                for ann in sample_anns:
                    get_field_input(ann, main_agent, static_input, moving = False, helper = helper)
                sample_anns = sample_input[key][i]['with_rider']
                for ann in sample_anns:
                    pass
                    #get_field_input(ann, main_agent, field_input[key], moving = True, helper = helper)
            elif key == 'vehicle':
                sample_anns = sample_input[key][i]['parked']
                for ann in sample_anns:
                    if ann['instance_token'] == main_agent['instance_token']:
                        continue
                    get_field_input(ann, main_agent, static_input, moving = False, helper = helper)
                sample_anns = sample_input[key][i]['stopped']
                for ann in sample_anns:
                    if ann['instance_token'] == main_agent['instance_token']:
                        continue
                    dis = calc_ann_dis(ann, main_agent)
                    if dis < radius:
                        get_field_input(ann, main_agent, moving_vehicles, moving = False, helper = helper)
                        moving_anns[ins_sample].append(ann)
                sample_anns = sample_input[key][i]['moving']
                for ann in sample_anns:
                    if ann['instance_token'] == main_agent['instance_token']:
                        continue
                    dis = calc_ann_dis(ann, main_agent)
                    if dis < radius:
                        get_field_input(ann, main_agent, moving_vehicles, moving = True, helper = helper)
                        moving_anns[ins_sample].append(ann)
            elif key == 'animal':
                pass
                #sample_anns = sample_input[key][i]
                #for ann in sample_anns:
                #    get_field_input(ann, main_agent, field_input[key], moving = True, helper = helper)
            else: #object
                sample_anns = sample_input[key][i]
                for ann in sample_anns:
                    get_field_input(ann, main_agent, static_input, moving = False, helper = helper)
        for key in static_input.keys():
            static_input[key] = np.array(static_input[key])
        for key in moving_vehicles.keys():
            moving_vehicles[key] = np.array(moving_vehicles[key])
        static_inputs[ins_sample] = static_input
        moving_inputs[ins_sample] = moving_vehicles
    return static_inputs, moving_inputs, moving_anns