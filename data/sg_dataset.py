import os
import random
import json
import copy

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.util import seed_all
from data.util import normalize_box_params, denormalize_box_params, get_rotation

# with ShapeNet metadata
id_to_cate = {
    '02691156': 'airplane', '02773838': 'bag', '02801938': 'basket',
    '02808440': 'bathtub', '02818832': 'bed', '02828884': 'bench',
    '02876657': 'bottle', '02880940': 'bowl', '02924116': 'bus',
    '02933112': 'cabinet', '02747177': 'can', '02942699': 'camera',
    '02954340': 'cap', '02958343': 'car', '03001627': 'chair',
    '03046257': 'clock', '03207941': 'dishwasher', '03211117': 'monitor',
    '04379243': 'table', '04401088': 'telephone', '02946921': 'tin_can',
    '04460130': 'tower', '04468005': 'train', '03085013': 'keyboard',
    '03261776': 'earphone', '03325088': 'faucet', '03337140': 'file',
    '03467517': 'guitar', '03513137': 'helmet', '03593526': 'jar',
    '03624134': 'knife', '03636649': 'lamp', '03642806': 'laptop',
    '03691459': 'speaker', '03710193': 'mailbox', '03759954': 'microphone',
    '03761084': 'microwave', '03790512': 'motorcycle', '03797390': 'mug',
    '03928116': 'piano', '03938244': 'pillow', '03948459': 'pistol',
    '03991062': 'pot', '04004475': 'printer', '04074963': 'remote_control',
    '04090263': 'rifle', '04099429': 'rocket', '04225987': 'skateboard',
    '04256520': 'sofa', '04330267': 'stove', '04530566': 'vessel',
    '04554684': 'washer', '02992529': 'cellphone',
    '02843684': 'birdhouse', '02871439': 'bookshelf',
    # '02858304': 'boat', no boat in our dataset, merged into vessels
    # '02834778': 'bicycle', not in our taxonomy
}
id_to_cate_overlapped_with_3rscan = { # 22 out of 55 objects overlapped with 3RScan
    '02773838': 'bag', '02801938': 'basket', '02808440': 'bathtub',
    '02818832': 'bed', '02828884': 'bench', '02876657': 'bottle',
    '02933112': 'cabinet', '03001627': 'chair', '03046257': 'clock',
    '03211117': 'monitor', '04379243': 'table', '04401088': 'telephone',
    '03593526': 'jar', '03636649': 'lamp', '03642806': 'laptop',
    '03761084': 'microwave', '03938244': 'pillow', '03991062': 'pot',
    '04004475': 'printer', '04256520': 'sofa', '04330267': 'stove',
    '02871439': 'bookshelf'
}
cate_to_id = {v: k for k, v in id_to_cate_overlapped_with_3rscan.items()}


class SceneGraphDataset(Dataset):
    def __init__(self, args, categories, use_seed=True, split='train'):
        """Properties of the dataset
        vocab (): information about classes, relationships
        filelist (): file names containing 'train' or 'test'
        rel_json_file, box_json_file, floor_json_file (): 
        
        """
        # args에 추가할 내용: args.crop_useless, args.center_scene_to_floor,
        # args.use_large_dataset, args.use_canonical, args.use_scene_rels,
        # args.use_scene_splits, args.label_fname
        # args.shuffle_objs, args.use_rio27
        
        # dataset arguments
        self.args = args
        args.center_scene_to_floor
        
        if eval and use_seed:
            if args.use_randomseed:
                self.seed = args.seed
            else:
                self.seed = 47
        seed_all(self.seed)
        
        self.sgdata_dir = os.path.join(args.data_dir, "Graph")
        self.scandata_dir = os.path.join(args.data_dir, "3RScan")
        
        # class categories
        if 'all' in categories:
            self.categories = cate_to_id.keys() # TODO: cate_to_id 정의 -> mapping 이후
        self.catfile = os.path.join(self.sgdata_dir, "classes.txt")
        self.cates = {}
        self.scans = []
        
        # define vocab
        self.vocab = {}
        with open(os.path.join(self.sgdata_dir, 'classes.txt'), "r") as f:
            self.vocab['object_idx_to_name'] = f.readlines()
        with open(os.path.join(self.sgdata_dir, 'relationships.txt'), "r") as f:
            self.vocab['pred_idx_to_name'] = f.readlines()
        
        # list of the name of txt files according to the split(train or test)
        with open(os.path.join(self.sgdata_dir, "{}.txt".format(split)), "r") as read_file:
            self.filelist = [file.rstrip() for file in read_file.read().splitlines()]
        
        # list of relationship categories
        self.relationships = self.read_relationships(os.path.join(self.sgdata_dir, "relationships.txt"))
        
        # load relationship, objects, tight boxes from json files
        # use scene sections of up to 9 objects if true, or full scenes otherwise
        self.use_scene_splits = args.use_scene_splits
        if split == 'train': # training set
            splits_fname = 'relationships_train_clean' if self.use_splits else 'relationships_merged_train_clean'
            self.rel_json_file = os.path.join(self.sgdata_dir, '{}.json'.format(splits_fname))
            self.box_json_file = os.path.join(self.sgdata_dir, 'obj_boxes_train_refined.json')
            self.floor_json_file = os.path.join(self.sgdata_dir, 'floor_boxes_split_train.json')
        elif split == 'test': # test set
            splits_fname = 'relationships_test_clean' if self.use_splits else 'relationships_merged_test_clean'
            self.rel_json_file = os.path.join(self.sgdata_dir, '{}.json'.format(splits_fname))
            self.box_json_file = os.path.join(self.sgdata_dir, 'obj_boxes_test_refined.json')
            self.floor_json_file = os.path.join(self.sgdata_dir, 'floor_boxes_split_test.json')
        else:
            raise ValueError("Invalid argument '{}' for split. It should be either 'train' or 'test'.")
        
        if args.crop_useless: # TODO: floor 말고도 wall 등 다른 것도 자를 수 없나?
            with open(self.floor_json_file, "r") as read_file:
                self.cropped_data = json.load(read_file)
        
        self.relationship_json, self.objs_json, self.tight_boxes_json = \
            self.read_relatinoship_json(self.rel_json_file, self.box_json_file)
        
        # TODO: use_rio27 (또는 다른 방법으로) mapping, classes_rio27.json 직접 만들어야 할 듯
        if args.use_rio27:
            with open(os.path.join(self.sgdata_dir, "classes_rio27.json"), "r") as read_file:
                self.vocab_rio27 = json.load(read_file)
            self.vocab['object_idx_to_name'] = self.vocab_rio27['rio27_idx_to_name']
            self.vocab['object_name_to_idx'] = self.vocab_rio27['rio27_name_to_idx']
        with open(os.path.join(self.sgdata_dir, "mapping_full2rio27.json"), "r") as read_file:
            self.mapping_full2rio27 = json.load(read_file)
        
        # define categories, classes
        with open(self.catfile, "r") as f:
            for line in f:
                category = line.rstrip()
                if category in categories:
                    self.cates[category] = category
        self.cates.sort()
        self.classes = dict(zip(self.cates), range(len(self.cates))) # TODO: print해보기
        
        # TODO: shape 정상적인 layout만 학습하도록 handling 필요

    
    def read_relationship_json(self, json_file, box_json_file):
        """ Reads from json files the relationship labels, object labels and bounding boxes

        Args:
            json_file: file that stores the objects and relationships
            box_json_file: file that stores the oriented 3D bounding box parameters
        
        Returns:
            three dicts, relationships, object labels and boxes
        """
        rel = {}
        objs = {}
        tight_boxes = {}

        with open(box_json_file, "r") as read_file:
            box_data = json.load(read_file)

        with open(json_file, "r") as read_file:
            data = json.load(read_file)
            for scan in data['scans']:

                relationships = []
                for realationship in scan["relationships"]:
                    realationship[2] -= 1
                    relationships.append(realationship)

                # for every scan in rel json, we append the scan id
                rel[scan["scan"] + "_" + str(scan["split"])] = relationships
                self.scans.append(scan["scan"] + "_" + str(scan["split"]))

                objects = {}
                boxes = {}
                for k, v in scan["objects"].items():
                    objects[int(k)] = v
                    try:
                        boxes[int(k)] = {}
                        boxes[int(k)]['param7'] = box_data[scan["scan"]][k]["param7"]
                        boxes[int(k)]['param7'][6] = np.deg2rad(boxes[int(k)]['param7'][6])
                        if self.args.use_canonical:
                            if "direction" in box_data[scan["scan"]][k].keys():
                                boxes[int(k)]['direction'] = box_data[scan["scan"]][k]["direction"]
                            else:
                                boxes[int(k)]['direction'] = 0
                    except:
                        # probably box was not saved because there were 0 points in the instance!
                        continue
                objs[scan["scan"] + "_" + str(scan["split"])] = objects
                tight_boxes[scan["scan"] + "_" + str(scan["split"])] = boxes
        return rel, objs, tight_boxes

    def read_relationships(self, read_file):
        """ Loads list of relationship labels

        Args:
            read_file: path of relationship list txt file
        
        Returns:
            relationships: relationship labels list
        """
        relationships = []
        with open(read_file, 'r') as f:
            for line in f:
                relationship = line.rstrip().lower()
                relationships.append(relationship)
        return relationships
    
    def load_semseg(self, json_file):
        """ Loads semantic segmentation from json file

        Args:
            json_file: path to file
        
        Returns:
            instance2label: dict that maps instance label to text semantic label
        """
        instance2label = {}
        with open(json_file, "r") as read_file:
            data = json.load(read_file)
            for segGroups in data['segGroups']:
                instance2label[segGroups["id"]] = segGroups["label"].lower()

        return instance2label
    
    def __getitem__(self, idx):
        """
        Args:
            idx

        Returns:
            output: scene graph data of each scan
                N: number of instances in the scene (added numbers if represents 'scene')
                M: number of relationships in the scene (containing scene relationships)
                
                output['objs'] (N,): global object ids which is in the scene
                output['triples'] (M, 3): triplets
                output['boxes'] (N, 7): informations of modified tight bounding boxes (7 params)
                output['scan_id'] (str): id of the scan
                output['split_id'] (str): split of the scan (not train, test) # TODO: 전부 다 1인데?
                output['instance_id'] : ids of the instances in the scan
        """
        scan_id_with_split = self.scans[idx]
        scan_id = scan_id_with_split.split('_')[0]
        split = scan_id_with_split.split('_')[1]
        
        if self.args.crop_useless:
            scene_floor = self.cropped_data[scan_id][split]
            floor_idx = list(scene_floor.keys())[0]
            if self.args.center_scene_to_floor:
                scene_center = np.asarray(scene_floor[floor_idx]['params7'][3:6])
            else:
                scene_center = np.array([0, 0, 0])

            min_box = np.asarray(scene_floor[floor_idx]['min_box']) - scene_center
            max_box = np.asarray(scene_floor[floor_idx]['max_box']) - scene_center
        
        file = os.path.join(self.scandata_dir, scan_id, self.args.label_fname)
        if os.path.exists(os.path.join(self.scandata_dir, scan_id, "semseg.v2.json")):
            semseg_file = os.path.join(self.scandata_dir, scan_id, "semseg.v2.json")
        elif os.path.exists(os.path.join(self.scandata_dir, scan_id, "semseg.json")):
            semseg_file = os.path.join(self.scandata_dir, scan_id, "semseg.json")
        else:
            raise FileNotFoundError("Cannot find semseg.json file.")

        # instance2label, e.g. {1: 'floor', 2: 'wall', 3: 'picture', 4: 'picture'}
        instance2label = self.load_semseg(semseg_file)
        selected_instances = list(self.objs_json[scan_id].keys()) # ids of instances only used for training
        keys = list(instance2label.keys())
        
        if self.args.shuffle_objs:
            random.shuffle(keys)
        
        # instance2mask : for the instances in the scan, determine whether it is in the scan.
        # if the value of instance2mask is 0, it is not existed in that scan.
        instance2mask = {} # key: object id in 3RScan, value: index of the class which is in the file
        instance2mask[0] = 0
        cat = [] # list of the global object ids
        tight_boxes = []
        instances_order = [] # list of the object ids in the scan
        
        counter = 0
        
        for key in keys:
            scene_instance_id = key
            scene_instance_class = instance2label[key]
            scene_class_id = -1
            if scene_instance_class in self.classes and \
                    (not self.args.use_rio27 or self.mapping_full2rio27[scene_instance_class] != '-'):
                if self.args.use_rio27:
                    scene_instance_class = self.mapping_full2rio27[scene_instance_class]
                    scene_class_id = int(self.vocab_rio27['rio27_name_to_idx'][scene_instance_class])
                else:
                    scene_class_id = self.classes[scene_instance_class]
            if scene_class_id != -1 and key in selected_instances:
                instance2mask[scene_instance_id] = counter + 1
                counter += 1
            else:
                instance2mask[scene_instance_id] = 0

            # mask to cat:
            if (scene_class_id >= 0) and (scene_instance_id > 0) and (key in selected_instances):
                cat.append(scene_class_id)
                bbox = self.tight_boxes_json[scan_id_with_split][key]['param7'].copy()
                if self.args.crop_useless and key in self.cropped_data[scan_id][split].keys():
                    bbox = self.cropped_data[scan_id][split][key]['params7'].copy()
                    bbox[6] = np.deg2rad(bbox[6])
                    direction = self.cropped_data[scan_id][split][key]['direction']

                if self.args.crop_useless and self.args.center_scene_to_floor:
                    bbox[3:6] -= scene_center

                if self.args.use_canonical:
                    if direction > 1 and direction < 5:
                        # update direction-less angle with direction data (shifts it by 90 degree
                        # for every added direction value
                        bbox[6] += (direction - 1) * np.deg2rad(90)
                        if direction == 2 or direction == 4:
                            temp = bbox[0]
                            bbox[0] = bbox[1]
                            bbox[1] = temp
                    # for other options, do not change the box
                instances_order.append(key)
                bins = np.linspace(0, np.deg2rad(360), 24)
                angle = np.digitize(bbox[6], bins)
                bbox = normalize_box_params(bbox)
                bbox[6] = angle
                tight_boxes.append(bbox)
                
        # define scene graph data
        triples = []
        rel_json = self.relationship_json[scan_id_with_split]
        
        for r in rel_json:
            if r[0] in instance2mask.keys() and r[1] in instance2mask.keys():
                subject = instance2mask[r[0]] - 1 # new object id (in the scan) should start from 0
                object = instance2mask[r[1]] - 1 # new object id (in the scan) should start from 0
                predicate = r[2] + 1 # index of relationship
                if subject >= 0 and object >= 0:
                    triples.append([subject, predicate, object])
            else:
                continue
        
        if self.args.use_scene_rels:
            # add _scene_ object and _in_scene_ connections
            scene_idx = len(cat) # scene index is assigned as an index of new category
            for i, ob in enumerate(cat):
                triples.append([i, 0, scene_idx])
            cat.append(0)
            # dummy scene box
            tight_boxes.append([-1, -1, -1, -1, -1, -1, -1])
        
        output = {}
        output['objs'] = torch.from_numpy(np.array(cat, dtype=np.int64))
        output['triples'] = torch.from_numpy(np.array(triples, dtype=np.int64))
        output['boxes'] = torch.from_numpy(np.array(tight_boxes, dtype=np.float32))
        
        output['scan_id'] = scan_id
        output['split_id'] = split
        output['instance_id'] = instances_order
        
        return output
    
    def __len__(self):
        return len(self.scans)
