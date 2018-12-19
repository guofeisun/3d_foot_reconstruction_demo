#################
# start
# load pre-trained model
# get current time-stamp as folder name xxx
# collect four-view data and save them in folder xxx
# read data and feed to graph, run reconstruction
# visualize reconstructed models

import os
import numpy as np
import tensorflow as tf
from utils import *
import binvox_rw as brw

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
cwd = os.getcwd()

class demo:
    def __init__(self):
        GPU0 = '0'
        config = tf.ConfigProto(gpu_options = tf.GPUOptions(allow_growth = True), allow_soft_placement=True)
        config.gpu_options.visible_device_list = GPU0
        
        if not os.path.isfile('model/train_mod/model_9.cptk.data-00000-of-00001'):
            print ('no pretrained model found!')
            return

        self.sess = tf.Session(config=config)
        saver = tf.train.import_meta_graph('model/train_mod/model_9.cptk.meta', clear_devices=True)
        saver.restore(self.sess, 'model/train_mod/model_9.cptk')
        print ('model restored!')

        self.X_f = tf.get_default_graph().get_tensor_by_name("Placeholder:0")
        self.X_r = tf.get_default_graph().get_tensor_by_name("Placeholder_1:0")
        self.X_b = tf.get_default_graph().get_tensor_by_name("Placeholder_2:0")
        self.X_l = tf.get_default_graph().get_tensor_by_name("Placeholder_3:0")
        self.Y_pred = tf.get_default_graph().get_tensor_by_name("aeu_multiview/decoder/Sigmoid:0")
        self.Y_surface = extract_surface(self.Y_pred, 3)

    def run(self, folder_name):
        print('loading data ...')
        x_f = load_X_from_txt(folder_name + '/001.txt')
        x_f = np.reshape(x_f, [1,64,64,64,1])
        x_r = load_X_from_txt(folder_name + '/002.txt')
        x_r = np.reshape(x_r, [1,64,64,64,1])
        x_b = load_X_from_txt(folder_name + '/003.txt')
        x_b = np.reshape(x_b, [1,64,64,64,1])
        x_l = load_X_from_txt(folder_name + '/004.txt')
        x_l = np.reshape(x_l, [1,64,64,64,1])
        print('starting reconstruction ...')
        [y_surface, y_sample] = self.sess.run([self.Y_surface, self.Y_pred], \
            feed_dict={self.X_f:x_f, self.X_r:x_r, self.X_b:x_b, self.X_l:x_l})

        ###### save binvox
        y_sample = y_sample.reshape([256,256,256])
        y_sample = (y_sample>=0.5)
        #save2binvox(y_sample, 'template.binvox', folder_name + '/volume.binvox')
        ###### save obj
        y_surface = y_surface.reshape([256,256,256])
        volume2pc(y_surface, 1.5, folder_name + '/pc.obj')
        print('point cloud reconstruction complete.')
        ###### reconstruct and save mesh
        poisson_reconstruction('/pc.obj', '/mesh.obj', folder_name, folder_name)
        print('mesh reconstruction complete.')
        os.system('meshlab ' + cwd + '/' + folder_name + '/mesh.obj')