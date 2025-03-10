import json
import os
from truckscenes import TruckScenes 
from visu_utils import TruckScenesExplorer
trucksc = TruckScenes('v1.0-mini', '/home/sriz/Documents/Datasets/man-truckscenes/data/', True)
my_scene = trucksc.scene[0]
#print(my_scene)
sample_json = '/home/sriz/Documents/Datasets/man-truckscenes/data/v1.0-mini/sample.json'

with open (sample_json,'r') as f:
    samples = json.load(f)

sample_dict = {sample['token']: sample for sample in samples}
first_sample_token = my_scene['first_sample_token']
last_sample_token = my_scene['last_sample_token']

current_token = first_sample_token
intermediate_tokens = []

while current_token != last_sample_token:
    intermediate_tokens.append(current_token)
    current_sample = sample_dict.get(current_token)
    if not current_sample:
        print(f'Warning: Missing Sample entry for token {current_token}')
        break
    current_token = current_sample['next']
intermediate_tokens.append(last_sample_token)

#Initializing the explorer
explorer = TruckScenesExplorer(trucksc)
output_dir = 'output_pointclouds_pcd'
os.makedirs(output_dir, exist_ok=True)

for idx,token in enumerate(intermediate_tokens):
    print(f'Processing sample {idx+1}/{len(intermediate_tokens)}: {token}')
    my_sample = trucksc.get('sample', token)
    pcd_path = os.path.join(output_dir, f'{token}.pcd')
    lidar_channels=['LIDAR_LEFT', 'LIDAR_REAR', 'LIDAR_RIGHT', 'LIDAR_TOP_FRONT', 'LIDAR_TOP_LEFT', 'LIDAR_TOP_RIGHT']
    explorer.render_pointcloud(sample_rec=my_sample, chans=lidar_channels, ref_chan='LIDAR_LEFT', with_anns=True, out_path=pcd_path)
    print(f'Saved: {pcd_path}')
print('Complete')