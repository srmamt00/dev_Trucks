import os 
import json
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

# Load the json files
with open('sample.json','r') as f:
    samples = json.load(f)
with open('sample_data.json','r') as f:
    sample_data = json.load(f)
with open('calibrated_sensor.json','r') as f:
    calibrated_sensor={calib['token']: calib for calib in json.load(f)}

# creating dictionaries to map sample tokens to their corresponding entries
sample_dict={sample['token']: sample for sample in samples}

first_sample_token='32d2bcf46e734dffb14fe2e0a823d059'
last_sample_token='94640f753b284a1c9c0e8694243f60cd'

#Extract all the sample tokens in the scene
current_token = first_sample_token
sample_tokens = []

while current_token != last_sample_token:
    sample_tokens.append(current_token)
    current_sample = sample_dict.get(current_token)

    if not current_sample:
        print(f'Missing sample entries for token {current_token}')

    current_token = current_sample['next']

sample_tokens.append(last_sample_token)
print(f'Total samples in the scene : {len(sample_tokens)}')

# Function to load the PCD file
#file_path = '/home/sriz/Documents/Datasets/man-truckscenes/data/sweeps'
def load_pcd(file_path):
    if os.path.exists(file_path):
        pcd=o3d.io.read_point_cloud(file_path)
        return np.asarray(pcd.points)
    else:
        print(f'Warning: PCD file not found: {file_path}')
        return np.array([])

# Function to transform point clouds
def transform_point_cloud(points, translation, rotation):
    R_matrix= R.from_quat([rotation[1], rotation[2], rotation[3], rotation[0]]).as_matrix()
    transformed_points = (R_matrix @ points.T).T + translation #Applying rotation and translation
    return transformed_points   

output_dir = 'output_pcds'
os.makedirs(output_dir, exist_ok=True)

for sample_token in sample_tokens:
    print(f'Processing sample token: {sample_token}')
    lidar_samples = [sample for sample in sample_data if sample['sample_token']==sample_token]

    merged_points = []

    for sample in lidar_samples:
        calib_token = sample['calibrated_sensor_token']
        if calib_token in calibrated_sensor:
            calib=calibrated_sensor[calib_token]
            translation=np.array(calib['translation'])
            rotation=np.array(calib['rotation'])
            
            pcd_file = sample['filename']
            points = load_pcd(pcd_file) 

            if points.shape[0] > 0:
                # Transform point clouds to ego frame
                transformed_points = transform_point_cloud(points, translation, rotation)
                # Append Transformed points
                merged_points.append(transformed_points)
    
    if merged_points:
        merged_points = np.vstack(merged_points)
        # Save merged point cloud
        merged_pcd = o3d.geometry.PointCloud()
        merged_pcd.points = o3d.utility.Vector3dVector(merged_points)
        output_pcd_file = os.path.join(output_dir, f'{sample_token}.pcd')
        o3d.io.write_point_cloud(output_pcd_file, merged_pcd)

        print(f'saved: {output_pcd_file}')
    else:
        print(f'No LIDAR data found for sample token: {sample_token}')

print('Processing Complete!')