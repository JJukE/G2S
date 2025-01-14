import json
import yaml
import numpy as np
import torch


#============================================================
# for graph visualization
#============================================================

def load_semantic_scene_graphs_custom(yml_relationships, color_palette, rel_label_to_id, with_manipuation=False):
    scene_graphs = {}

    with open(yml_relationships, "r") as read_file:
        graphs = yaml.load(read_file)
    for scene_id, scene in graphs['Scenes'].items():

        scene_graphs[str(scene_id)] = {}
        scene_graphs[str(scene_id)]['objects'] = []
        scene_graphs[str(scene_id)]['relationships'] = []
        scene_graphs[str(scene_id)]['node_mask'] = [1] * len(scene['nodes'])
        scene_graphs[str(scene_id)]['edge_mask'] = [1] * len(scene['relships'])

        for (i, n) in enumerate(scene['nodes']):
            obj_item = {'ply_color': color_palette[i%len(color_palette)],
                        'id': str(i),
                        'label': n}
            scene_graphs[str(scene_id)]['objects'].append(obj_item)
        for r in scene['relships']:
            rel_4 = [r[0], r[1], rel_label_to_id[r[2]], r[2]]
            scene_graphs[str(scene_id)]['relationships'].append(rel_4)
        counter = len(scene['nodes'])
        if with_manipuation:
            for m in scene['manipulations']:
                if m[1] == 'add':
                    # visualize an addition
                    # ['chair', 'add', [[2, 'standing on'], [1, 'left']]]
                    obj_item = {'ply_color': color_palette[counter%len(color_palette)],
                                'id': str(counter),
                                'label': m[0]}
                    scene_graphs[str(scene_id)]['objects'].append(obj_item)

                    scene_graphs[str(scene_id)]['node_mask'].append(0)
                    for mani_rel in m[2]:
                        rel_4 = [counter, mani_rel[0], rel_label_to_id[mani_rel[1]], mani_rel[1]]
                        scene_graphs[str(scene_id)]['relationships'].append(rel_4)
                        scene_graphs[str(scene_id)]['edge_mask'].append(0)
                    counter += 1
                if m[1] == 'rel':
                    # visualize changes in the relationship
                    for (rid, r) in enumerate(scene_graphs[str(scene_id)]['relationships']):
                        s, o, p, l = r
                        if isinstance(m[2][3], list):
                            # ['', 'rel', [0, 1, 'right', [0, 1, 'left']]]
                            if s == m[2][0] and o == m[2][1] and l == m[2][2] and s == m[2][3][0] and o == m[2][3][1]:
                                # a change on the SAME (s, o) pair, indicate the change
                                scene_graphs[str(scene_id)]['edge_mask'][rid] = 0
                                scene_graphs[str(scene_id)]['relationships'][rid][3] = m[2][2] + '->' + m[2][3][2]
                                scene_graphs[str(scene_id)]['relationships'][rid][2] = rel_label_to_id[m[2][3][2]]
                                break
                            elif s == m[2][0] and o == m[2][1] and l == m[2][2]:
                                # overwrite this edge with a new pair (s,o)
                                del scene_graphs[str(scene_id)]['edge_mask'][rid]
                                del scene_graphs[str(scene_id)]['relationships'][rid]
                                scene_graphs[str(scene_id)]['edge_mask'].append(0)
                                new_edge = [m[2][3][0], m[2][3][1], rel_label_to_id[m[2][3][2]], m[2][3][2]]
                                scene_graphs[str(scene_id)]['relationships'].append(new_edge)
                        else:
                            # ['', 'rel', [0, 1, 'right', 'left']]
                            if s == m[2][0] and o == m[2][1] and l == m[2][2]:
                                scene_graphs[str(scene_id)]['edge_mask'][rid] = 0
                                scene_graphs[str(scene_id)]['relationships'][rid][3] = m[2][2] + '->' + m[2][3]
                                scene_graphs[str(scene_id)]['relationships'][rid][2] = rel_label_to_id[m[2][3]]
                                break

    return scene_graphs


def load_semantic_scene_graphs(json_relationships, json_objects):
    scene_graphs_obj = {}

    with open(json_objects, "r") as read_file:
        data = json.load(read_file)
        for s in data["scans"]:
            scan = s["scan"]
            objs = s['objects']
            scene_graphs_obj[scan] = {}
            scene_graphs_obj[scan]['scan'] = scan
            scene_graphs_obj[scan]['objects'] = []
            for obj in objs:
                scene_graphs_obj[scan]['objects'].append(obj)
    scene_graphs = {}
    with open(json_relationships, "r") as read_file:
        data = json.load(read_file)
        for s in data["scans"]:
            scan = s["scan"]
            split = str(s["split"])
            if scan + "_" + split not in scene_graphs:
                scene_graphs[scan + "_" + split] = {}
                scene_graphs[scan + "_" + split]['objects'] = []
                # print("WARNING: no objects for this scene")
            scene_graphs[scan + "_" + split]['relationships'] = []
            for k in s["objects"].keys():
                ob = s['objects'][k]
                for i,o in enumerate(scene_graphs_obj[scan]['objects']):
                    if o['id'] == k:
                        inst = i
                        break
                scene_graphs[scan + "_" + split]['objects'].append(scene_graphs_obj[scan]['objects'][inst])
            for rel in s["relationships"]:
                scene_graphs[scan + "_" + split]['relationships'].append(rel)
    return scene_graphs


def read_relationships(read_file):
    relationships = [] 
    with open(read_file, 'r') as f: 
        for line in f: 
            relationship = line.rstrip().lower() 
            relationships.append(relationship) 
    return relationships


#============================================================
# for box and shape visualization
#============================================================

def get_cross_prod_mat(pVec_Arr):
    """ Convert pVec_Arr of shape (3) to its cross product matrix
    """
    qCross_prod_mat = np.array([
        [0, -pVec_Arr[2], pVec_Arr[1]],
        [pVec_Arr[2], 0, -pVec_Arr[0]],
        [-pVec_Arr[1], pVec_Arr[0], 0],
    ])
    return qCross_prod_mat


def params_to_8points(box, degrees=False):
    """ Given bounding box as 7 parameters: w, l, h, cx, cy, cz, z, compute the 8 corners of the box
    """
    w, l, h, cx, cy, cz, z = box
    points = []
    for i in [-1, 1]:
        for j in [-1, 1]:
            for k in [-1, 1]:
                points.append([w.item()/2 * i, l.item()/2 * j, h.item()/2 * k])
    points = np.asarray(points)
    points = (get_rotation(z.item(), degree=degrees) @ points.T).T
    points += np.expand_dims(np.array([cx.item(), cy.item(), cz.item()]), 0)
    return points


def params_to_8points_no_rot(box):
    """ Given bounding box as 6 parameters (without rotation): w, l, h, cx, cy, cz, compute the 8 corners of the box.
        Works when the box is axis aligned
    """
    w, l, h, cx, cy, cz = box
    points = []
    for i in [-1, 1]:
        for j in [-1, 1]:
            for k in [-1, 1]:
                points.append([w.item()/2 * i, l.item()/2 * j, h.item()/2 * k])
    points = np.asarray(points)
    points += np.expand_dims(np.array([cx.item(), cy.item(), cz.item()]), 0)
    return points


def fit_shapes_to_box(box, shape, withangle=True):
    """ Given normalized shape, transform it to fit the input bounding box.
        Expects denormalized bounding box with optional angle channel in degrees
        :param box: tensor
        :param shape: tensor
        :param withangle: boolean
        :return: transformed shape
    """
    if isinstance(box, torch.Tensor):
        box = box.detach().cpu().numpy()
    if isinstance(shape, torch.Tensor):
        shape = shape.detach().cpu().numpy()
    if withangle:
        w, l, h, cx, cy, cz, z = box
    else:
        w, l, h, cx, cy, cz = box
    # scale
    shape_size = np.max(shape, axis=0) - np.min(shape, axis=0)
    shape = shape / shape_size
    shape *= box[:3]
    if withangle:
        # rotate
        shape = (get_rotation(z, degree=False).astype("float32") @ shape.T).T
    # translate
    shape += [cx, cy, cz]

    return shape


def refineBoxes(boxes, objs, triples, relationships, vocab):
    for idx in range(len(boxes)):
      child_box = boxes[idx]
      w, l, h, cx, cy, cz = child_box
      for t in triples:
         if idx == t[0] and relationships[t[1]] in ["supported by", "lying on", "standing on"]:
            parent_idx = t[2]
            cat = vocab['object_idx_to_name'][objs[parent_idx]].replace('\n', '')
            if cat != 'floor':
                continue
            parent_box = boxes[parent_idx]
            base = parent_box[5] + 0.0125

            new_bottom = base
            # new_h = cz + h / 2 - new_bottom
            new_cz = new_bottom + h / 2
            shift = new_cz - cz
            boxes[idx][:] = [w, l, h, cx, cy, new_cz]

            # fix adjusmets
            for t_ in triples:
                if t_[2] == t[0] and relationships[t_[1]] in ["supported by", "lying on", "standing on"]:
                    cat = vocab['object_idx_to_name'][t_[2]].replace('\n', '')
                    if cat != 'floor':
                        continue

                    w_, l_, h_, cx_, cy_, cz_ = boxes[t_[0]]
                    boxes[t_[0]][:] = [w_, l_, h_, cx_, cy_, cz_ + shift]
    return boxes


def get_rotation(z, degree=True):
    """ Get rotation matrix given rotation angle along the z axis.
    :param z: angle of z axos rotation
    :param degree: boolean, if true angle is given in degrees, else in radians
    :return: rotation matrix as np array of shape[3,3]
    """
    if degree:
        z = np.deg2rad(z)
    rot = np.array([[np.cos(z), -np.sin(z),  0],
                    [np.sin(z),  np.cos(z),  0],
                    [        0,          0,  1]])
    return rot


def normalize_box_params(box_params, scale=3):
    """ Normalize the box parameters for more stable learning utilizing the accumulated dataset statistics

    :param box_params: float array of shape [7] containing the box parameters
    :param scale: float scalar that scales the parameter distribution
    :return: normalized box parameters array of shape [7]
    """
    mean = np.array([1.3827214, 1.309359, 0.9488993, -0.12464812, 0.6188591, -0.54847, 0.73127955])
    std = np.array([1.7797655, 1.657638, 0.8501885, 1.9160025, 2.0038228, 0.70099753, 0.50347435])

    return scale * ((box_params - mean) / std)


def denormalize_box_params(box_params, scale=3, params=7):
    """ Denormalize the box parameters utilizing the accumulated dataset statistics

    :param box_params: float array of shape [params] containing the box parameters
    :param scale: float scalar that scales the parameter distribution
    :param params: number of bounding box parameters. Expects values of either 6 or 7. 6 omits the angle
    :return: denormalized box parameters array of shape [params]
    """
    if params == 6:
        mean = np.array([1.3827214, 1.309359, 0.9488993, -0.12464812, 0.6188591, -0.54847])
        std = np.array([1.7797655, 1.657638, 0.8501885, 1.9160025, 2.0038228, 0.70099753])
    elif params == 7:
        mean = np.array([1.3827214, 1.309359, 0.9488993, -0.12464812, 0.6188591, -0.54847, 0.73127955])
        std = np.array([1.7797655, 1.657638, 0.8501885, 1.9160025, 2.0038228, 0.70099753, 0.50347435])
    else:
        raise NotImplementedError
    return (box_params * std) / scale + mean


def batch_torch_denormalize_box_params(box_params, scale=3):
    """ Denormalize the box parameters utilizing the accumulated dateaset statistics

    :param box_params: float tensor of shape [N, 6] containing the 6 box parameters, where N is the number of boxes
    :param scale: float scalar that scales the parameter distribution
    :return: float tensor of shape [N, 6], the denormalized box parameters
    """

    mean = torch.tensor([1.3827214, 1.309359, 0.9488993, -0.12464812, 0.6188591, -0.54847]).reshape(1,-1).float().cuda()
    std = torch.tensor([1.7797655, 1.657638, 0.8501885, 1.9160025, 2.0038228, 0.70099753]).reshape(1,-1).float().cuda()

    return (box_params * std) / scale + mean

