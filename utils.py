import numpy as np
import binvox_rw as brw
import tensorflow as tf
import subprocess

def save2binvox(reconstructed_volume, data_name1, data_name2):
    with open(data_name1,"rb") as f:
        bvx = brw.read_as_3d_array(f)
    bvx.dims=[256, 256, 256]
    bvx.data=reconstructed_volume
    with open(data_name2,"wb") as f:
        brw.write(bvx, f)

def extract_surface(volume_solid, ksize=3):
    volume_min = -tf.nn.max_pool3d(-volume_solid, [1,ksize,ksize,ksize,1], [1,1,1,1,1], 'SAME')
    volume_surface = tf.cast(tf.logical_and(volume_solid>=0.5, volume_min<0.5), tf.float32)

    return volume_surface

def volume2pc(volume, vox_sz, data_name):
    sz = np.max(np.shape(volume))
    if sz*vox_sz != 384:
        print ("wrong voxel size!")
        exit()
    volume_idx = np.stack(np.meshgrid(range(sz), range(sz), range(sz)))
    volume_idx = np.reshape(volume_idx, [3,-1])
    volume_yxz = volume_idx * vox_sz
    volume_yxz = np.swapaxes(volume_yxz, 0, 1)
    valid_idx = np.flatnonzero(volume)
    volume_yxz = volume_yxz[valid_idx]
    volume_xyz = np.zeros_like(volume_yxz)
    volume_xyz[:,0] = volume_yxz[:,1]
    volume_xyz[:,1] = volume_yxz[:,0]
    volume_xyz[:,2] = volume_yxz[:,2]
    with open(data_name, 'wb') as f:
        for v in volume_xyz:
            f.write('v {:2} {:2} {:2}\n'.format(v[0], v[1], v[2]))

def load_X_from_txt(path, out_vox_res=64):
    volume_compressed = np.loadtxt(path, int)
    voxel_data = np.zeros(out_vox_res*out_vox_res*out_vox_res, np.bool)
    length = len(volume_compressed)
    idx = 0
    for i in range(length):
        voxel_data[idx:idx+volume_compressed[i][1]] = volume_compressed[i][0]
        idx += volume_compressed[i][1]
    voxel_data = np.reshape(voxel_data, [out_vox_res,out_vox_res,out_vox_res,1])
    # flip along y and z
    volume_tmp = voxel_data[:,::-1,::-1,:]
    return volume_tmp

def poisson_reconstruction(in_file, out_file, in_path, out_path):
    # Add input mesh
    command = "meshlabserver -i " + in_path + in_file
    #command = "meshlabserver -i ./pc.obj"
    # Add the filter script
    command += " -s poisson_reconstruction_256.mlx"
    # Add the output filename and output flags
    command += " -o " + out_path + out_file
    #command += " -o ./reconstruction.obj"
    # Execute command
    #print "Going to execute: "
    output = subprocess.check_output(command, shell=True)
    last_line = output.splitlines()[-1]
    #print
    #print "Done:"
    print in_file + " > " + out_file