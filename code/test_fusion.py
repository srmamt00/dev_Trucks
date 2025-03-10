from truckscenes import TruckScenes #import Truckscenes
#from truckscenes.utils.visualization_utils import TruckScenesExplorer
from visu_utils import TruckScenesExplorer
trucksc = TruckScenes('v1.0-mini', '/home/sriz/Documents/Datasets/man-truckscenes/data/', True)
my_scene = trucksc.scene[0]

first_sample_token = my_scene['first_sample_token']
my_sample = trucksc.get('sample', first_sample_token)
#print(my_sample)
lidar_channels=['LIDAR_LEFT', 'LIDAR_REAR', 'LIDAR_RIGHT', 'LIDAR_TOP_FRONT', 'LIDAR_TOP_LEFT', 'LIDAR_TOP_RIGHT']

#test
explorer = TruckScenesExplorer(trucksc)
explorer.render_pointcloud(my_sample, chans=lidar_channels, ref_chan='LIDAR_LEFT', with_anns=True)

