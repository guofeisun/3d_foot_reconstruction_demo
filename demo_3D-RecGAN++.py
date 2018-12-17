import os
import numpy as np
import scipy.io
import tensorflow as tf
from data_process import Foot_Data
import binvox_rw as brw
import time
from model_config import load_config

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
GPU0 = '0'

# load sample data
# save output to binvox
def save2binvox(reconstructed_volume, data_name1, data_name2):
    with open(data_name1,"rb") as f:
        bvx = brw.read_as_3d_array(f)
    bvx.dims=[256, 256, 256]
    bvx.data=reconstructed_volume
    with open(data_name2,"wb") as f:
        brw.write(bvx, f)

# extract surface voxels with tensorflow
def extract_surface(volume_solid, ksize=3):
    volume_min = -tf.nn.max_pool3d(-volume_solid, [1,ksize,ksize,ksize,1], [1,1,1,1,1], 'SAME')
    volume_surface = tf.cast(tf.logical_and(volume_solid>=0.5, volume_min<0.5), tf.float32)

    return volume_surface

# volume to point cloud
def volume2pc(reconstructed_volume, scale, translate, data_name):
    volume_xyz = Foot_Data.volume2pc(reconstructed_volume, scale)
    with open(data_name, 'wb') as f:
        for v in volume_xyz:
            f.write('v {:2} {:2} {:2}\n'.format(v[0], v[1], v[2]))

    # w,h,d = np.shape(reconstructed_volume)
    # with open(data_name+'.obj','wb') as f:
    #     for i in range(w):
    #         for j in range(h):
    #             for k in range(d):
    #                 if reconstructed_volume[i,j,k] == 1:
    #                     f.write('v {:2} {:2} {:2}\n'.format((i)*scale+translate[0],(j)*scale+translate[1],(k)*scale+translate[2]))

# load real data list
def load_datas(config):
    data_files = {}
    idx = range(config['sample_number'])
    for n in config['data_list']:
        idx_f = np.random.permutation(idx)
        idx_r = np.random.permutation(idx)
        idx_b = np.random.permutation(idx)
        idx_l = np.random.permutation(idx)
        views = []
        for i in range(config['sample_number']):        
            view_f = config['data_path'] + n + '_l/' + n + '_l_1_' + '{:03}.txt.txt'.format(idx_f[i]%4 + 1)
            view_r = config['data_path'] + n + '_l/' + n + '_l_2_' + '{:03}.txt.txt'.format(idx_r[i]%4 + 1)
            view_b = config['data_path'] + n + '_l/' + n + '_l_3_' + '{:03}.txt.txt'.format(idx_b[i]%4 + 1)
            view_l = config['data_path'] + n + '_l/' + n + '_l_4_' + '{:03}.txt.txt'.format(idx_l[i]%4 + 1)
            views.append([view_f, view_r, view_b, view_l])
        data_files[n] = views
    return data_files

def inference_demo():
    # load config
    config_para = load_config("real_data.json")
    data_path = config_para['data_path']
    save_path = config_para['save_path']
    model_path = config_para['model_path']
    model_id = 'model_{}'.format(config_para['model_id'])

    # load real datas
    data_files = load_datas(config_para)

    # load model
    if not os.path.isfile(model_path + model_id + '.cptk.data-00000-of-00001'):
        print ('no pretrained model found!')
        return
    if not os.path.exists(save_path + model_id):
        os.makedirs(save_path+model_id)

    config = tf.ConfigProto(gpu_options = tf.GPUOptions(allow_growth = True), allow_soft_placement=True)
    config.gpu_options.visible_device_list = GPU0
    with tf.Session(config=config) as sess:
        saver = tf.train.import_meta_graph( model_path + model_id + '.cptk.meta', clear_devices=True)
        saver.restore(sess, model_path + model_id + '.cptk')
        print ('model restored!')

        X_f = tf.get_default_graph().get_tensor_by_name("Placeholder:0")
        X_r = tf.get_default_graph().get_tensor_by_name("Placeholder_1:0")
        X_b = tf.get_default_graph().get_tensor_by_name("Placeholder_2:0")
        X_l = tf.get_default_graph().get_tensor_by_name("Placeholder_3:0")
        Y_pred = tf.get_default_graph().get_tensor_by_name("aeu_multiview/decoder/Sigmoid:0")
        Y_surface = extract_surface(Y_pred, 3)
        for n in config_para['data_list']:
            if not os.path.exists(save_path + model_id +'/' + n):
                os.mkdir(save_path + model_id +'/' + n)
            for i in range(config_para['sample_number']):
                x_f = Foot_Data.load_X_from_txt(data_files[n][i][0])
                x_f = np.reshape(x_f, [1,64,64,64,1])
                x_r = Foot_Data.load_X_from_txt(data_files[n][i][1])
                x_r = np.reshape(x_r, [1,64,64,64,1])
                x_b = Foot_Data.load_X_from_txt(data_files[n][i][2])
                x_b = np.reshape(x_b, [1,64,64,64,1])
                x_l = Foot_Data.load_X_from_txt(data_files[n][i][3])
                x_l = np.reshape(x_l, [1,64,64,64,1])

                t0 = time.clock()
                [y_surface, y_sample] = sess.run([Y_surface, Y_pred], feed_dict={X_f:x_f, X_r:x_r, X_b:x_b, X_l:x_l})
                #y_surface = sess.run(Y_surface, feed_dict={X_f:x_f, X_r:x_r, X_b:x_b, X_l:x_l})

                ###### save binvox
                y_sample = y_sample.reshape([256,256,256])
                y_sample = (y_sample>=0.5)
                save2binvox(y_sample, 'template.binvox', save_path+model_id+'/'+n+'/'+str(i)+'.binvox')
                ###### save obj
                y_surface = y_surface.reshape([256,256,256])
                volume2pc(y_surface, 1.5, [0,0,0], save_path+model_id+'/'+n+'/'+str(i)+'.obj')
                print("time spend: {} seconds.".format(time.clock()-t0))

#########################
if __name__ == '__main__':
    inference_demo()
    #visualize()