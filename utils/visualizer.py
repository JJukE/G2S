import point_cloud_utils as pcu

class BaseVisualizer():
    """3D visualizer
    3D visualizer contains object visualizer and scene visualizer.
    It can be used to either save 3d point cloud files or visualize on window.
    """

class ObjectVisualizer(BaseVisualizer):
    """3D object visualizer
    You can visualize on window or save the point clouds.
    """
    
    def __init__():
        super().__init__()
        
    def visualize(self):
        """
        visualize a 3d point cloud object.
        """
        pass
    
    def save(self):
        """
        save a 3d point cloud object.
        """
        pass

class SceneVisualizer(BaseVisualizer):
    """3D scene visualizer
    In the case of scene, there can be objects and bounding boxes.
    
    If you want to visualize both objects and bounding boxes, you can use <>.
    Because of the difference between 3d file properties of objects and
    bounding boxes, you should use open3d and overlap two different 3d
    structures. (It can not be saved simultaneously.)
    
    Otherwise, you should visualize or save each structure.
    """
    def __init__():
        super().__init__()
    
    def visualize(self):
        """
        visualize objects and bounding boxes. You can also visualize
        either objects or bounding boxes.
        """
        pass
    
    def save(self):
        """
        save objects and bounding boxes separately. You can also save
        either objects or bounding boxes.
        """
        pass