import copy

import torch
import numpy as np

import point_cloud_utils as pcu
import trimesh
import open3d as o3d

class BaseVisualizer():
    """3D visualizer
    3D visualizer contains object visualizer and scene visualizer.
    It can be used to either save 3d point cloud files or visualize on window.
    
    TODO: visualize mesh
    """
    def __init__(self):
        self.objs = None
        self.colors = None

class ObjectVisualizer(BaseVisualizer):
    """3D object visualizer
    You can visualize on window or save the point clouds.
    """
    
    def __init__(self):
        super().__init__()
        
    def visualize(self, objs_pc, num_in_row=10, color=[0, 0, 0]):
        """
        visualize point cloud objects on the window.
        If multiple objects proveded, it will be arranged by the grid
        with given num_in_row and distance between objects.
        
        Args:
            objs_pc (num_objs, num_points, 3) or (num_points, 3): point cloud objects (torch or numpy)
            num_in_row (int): number of objects in row of 2d grid to arrange objects
            color (r, g, b): rgb color with range of each value is 0 ~ 255
        
        Returns:
            orig_objs (list): trimesh.PointCloud objects corresponding to objs_pc (before arranged)
        """
        
        if objs_pc.ndim == 2: # one object
            self.objs = trimesh.PointCloud(objs_pc, colors=color)
            self.objs.show()
        
        elif objs_pc.ndim == 3: # multiple objects
            self.objs = [trimesh.PointCloud(objs_pc[i],colors=color) for i in range(len(objs_pc))]
            
            # compute distance between objects with the max length of the object
            max_length = max(obj.extents.max() for obj in self.objs)
            dist = max_length * 1.5
            
            # translate each object along the x-axis and y-axis to arrange them
            arranged_objs = copy.deepcopy(self.objs)
            for i, obj in enumerate(arranged_objs):
                row = i // num_in_row
                col = i % num_in_row
                mat_trans = trimesh.transformations.translation_matrix((col*dist, -row*dist, 0))
                obj.apply_transform(mat_trans)
            
            scene = trimesh.Scene(arranged_objs)
            scene.show()
        
        else:
            raise ValueError("Unsupported dimension: {}. objs_pc should have 2 or 3 dimensions.".format(objs_pc.ndim))
        
    
    def save(self, objs_pc, path, num_in_row=10, color=[0, 0, 0]):
        """
        save a 3d point cloud object. Arguments are same except path to save.
        
        args:
            path (str): path to save
        """
        if objs_pc.ndim == 2: # one object
            self.objs = trimesh.PointCloud(objs_pc, colors=color)
            self.objs.export(path)
        
        elif objs_pc.ndim == 3: # multiple objects
            self.objs = [trimesh.PointCloud(objs_pc[i],colors=color) for i in range(len(objs_pc))]
            
            # compute distance between objects with the max length of the object
            max_length = max(obj.extents.max() for obj in self.objs)
            dist = max_length * 1.5
            
            # translate each object along the x-axis and y-axis to arrange them
            arranged_objs = copy.deepcopy(self.objs)
            for i, obj in enumerate(arranged_objs):
                row = i // num_in_row
                col = i % num_in_row
                mat_trans = trimesh.transformations.translation_matrix((col*dist, -row*dist, 0))
                obj.apply_transform(mat_trans)
            
            vertices = np.concatenate([obj.vertices for obj in arranged_objs])
            colors = np.concatenate([obj.colors for obj in arranged_objs])
            merged_objs = trimesh.PointCloud(vertices, colors=colors)
            merged_objs.export(path)
        
        else:
            raise ValueError("Unsupported dimension: {}. objs_pc should have 2 or 3 dimensions.".format(objs_pc.ndim))


class SceneVisualizer(BaseVisualizer):
    """3D scene visualizer
    In the case of scene, there can be objects and bounding boxes.
    
    If you want to visualize both objects and bounding boxes, you can use <>.
    Because of the difference between 3d file properties of objects and
    bounding boxes, you should use open3d and overlap two different 3d
    structures. (It can not be saved simultaneously.)
    
    Otherwise, you should visualize or save each structure.
    """
    def __init__(self):
        super().__init__()
    
    def visualize(self):
        """
        visualize objects and bounding boxes.
        """
        pass
    
    def save(self):
        """
        save objects and bounding boxes separately. You can also save
        either objects or bounding boxes.
        """
        pass