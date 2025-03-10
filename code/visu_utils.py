import os
import os.path as osp
from typing import Dict, List, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from matplotlib import cm
from pyquaternion import Quaternion
from truckscenes.utils import colormap
from truckscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
from truckscenes.utils.geometry_utils import view_points, transform_matrix, \
    BoxVisibility


class TruckScenesExplorer:
    """ Helper class to list and visualize truckscenes data. These are meant to serve
    as tutorials and templates for working with the data.
    """
    def __init__(self, trucksc):
        self.trucksc = trucksc               

    def render_pointcloud(self,
                          sample_rec: Dict,
                          chans: Union[str, List[str]],
                          ref_chan: str,
                          with_anns: bool = True,
                          box_vis_level: BoxVisibility = BoxVisibility.ANY,
                          nsweeps: int = 5,
                          min_distance: float = 1.0,
                          cmap: str = 'viridis',
                          out_path: str = None) -> None:
        """Renders a 3D representation of all given point clouds.

        Arguments:
            sample_rec: Sample record.
            chans: Sensor channels to render.
            ref_chan: Reference sensor channel.
            with_anns: Whether to draw box annotations.
            box_vis_level: If sample_data is an image, this sets required visibility for boxes.
            nsweeps: Number of sweeps for lidar and radar.
            min_distance: Minimum distance to include points.
            cmap: Colormap or colormap name.
            out_path: Optional path to write a image file of the rendered point clouds.
        """
        print('Executing render pointcloud from TruckScenesExplorer')
        # Convert chans to list
        if not isinstance(chans, list):
            chans = [chans]

        # Initialize point clouds and intensities
        point_clouds = []
        intensities = []

        for chan in chans:
            # Get sensor modality
            sd_record = self.trucksc.get('sample_data', sample_rec['data'][chan])
            sensor_modality = sd_record['sensor_modality']

            # Load point cloud
            if sensor_modality in {'lidar'}:
                point_obj, _ = LidarPointCloud.from_file_multisweep(self.trucksc, sample_rec,
                                                                    chan, ref_chan, nsweeps,
                                                                    min_distance)
                pc = point_obj.points.T
                intens = pc[:, 3]

            elif sensor_modality in {'radar'}:
                point_obj, _ = RadarPointCloud.from_file_multisweep(self.trucksc, sample_rec,
                                                                    chan, ref_chan, nsweeps,
                                                                    min_distance)
                pc = point_obj.points.T
                intens = pc[:, 6]

            # Add channel data to channels collection
            point_clouds.append(pc[:, :3])
            intensities.append(intens)

        # Concatenate all channels
        point_clouds = np.concatenate(point_clouds, axis=0)
        intensities = np.concatenate(intensities, axis=0)

        # Convert intensities to colors
        rgb = cm.get_cmap(cmap)(intensities)[..., :3]

        # Initialize vizualization objets
        vis_obj = []

        # Define point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_clouds)
        pcd.colors = o3d.utility.Vector3dVector(rgb)
        vis_obj.append(pcd)

        # Save point cloud to file
        if out_path is not None:
            os.makedirs(os.path.dirname(out_path),exist_ok=True)
            o3d.io.write_point_cloud(out_path,pcd)
            print(f'Point cloud saved to {out_path}')
        else:
            o3d.io.write_point_cloud(f"{sample_rec['token']}.pcd",pcd)
            print(f"Point cloud saved to {sample_rec['token']}.pcd (default location)")
  
        if with_anns:
            # Get boxes in reference sensor frame
            ref_sd_token = sample_rec['data'][ref_chan]
            _, boxes, _ = self.trucksc.get_sample_data(ref_sd_token, box_vis_level=box_vis_level)

            # Define bounding boxes
            for box in boxes:
                bbox = o3d.geometry.OrientedBoundingBox()
                bbox.center = box.center
                bbox.extent = box.wlh[[1, 0, 2]]
                bbox.R = Quaternion(box.orientation).rotation_matrix
                bbox.color = np.asarray(colormap.get_colormap()[box.name]) / 255
                vis_obj.append(bbox)

        # Visualize point cloud
        rend = o3d.visualization.RenderOption()
        rend.line_width = 8.0
        vis = o3d.visualization.Visualizer()
        vis.update_renderer()
        vis.create_window()
        for obj in vis_obj:
            vis.add_geometry(obj)
            vis.poll_events()
            vis.update_geometry(obj)

        # Save visualization
        if out_path is not None:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            vis.capture_screen_image(filename=out_path)
            vis.destroy_window()
        else:
            vis.run()