import numpy as np
import os
import re
from random import shuffle
import shutil
import tensorflow as tf
import scipy.io
import scipy.misc
import sklearn.metrics
import copy
import math
import random
from skimage.measure import block_reduce
import threading
import sys
import binvox_rw as brw
import time

if sys.version_info >= (2,0):
    print (sys.version)
    import Queue as queue
if sys.version_info >= (3,0):
    print (sys.version)
    import queue

class Foot_Data(threading.Thread):
    # load binvox gt data as Y
    # load txt data represents 3d volume of depth map
    # transfer txt data to 3d volume as X
    # or load depth image and convert to 3d volume as X
    def __init__(self,config):
        super(Foot_Data,self).__init__()
        self.config = config

        self.batch_size = config['batch_size']
        self.vox_res_x = config['vox_res_x']
        self.vox_res_y = config['vox_res_y']
        self.data_path_x = str(config['data_path_x'])
        self.data_path_y = str(config['data_path_y'])
        self.subfix_x = str(config['subfix_x'])
        self.subfix_y = str(config['subfix_y'])
        self.subfix_train = str(config['subfix_train'])
        self.subfix_test = str(config['subfix_test'])

        self.train_names = self.read_names_from_list(config['train_list'])
        self.test_names = self.read_names_from_list(config['test_list'])
        self.rotation_matrixs = self.read_rotation_matrixs(config['rotation_list'])
        print("training list: {}".format(len(self.train_names)))
        print("test list: {}".format(len(self.test_names)))

        self.queue_train = queue.Queue(4)
        self.stop_queue = False

        if self.config['view_flag'] == 'single_view':
            # single-view
            self.view_list_all = self.load_singleview_list()
            # use all training samples
            self.train_sample_idxs = range(len(self.train_names)*len(self.view_list_all))
            self.test_sample_idxs = range(len(self.test_names)*len(self.view_list_all))
            self.train_sample_idx = 0
            self.train_list_idx =0
            self.view_all_idx = 0
            print("view numbers:\n all: {}".format(len(self.view_list_all)))


        else:
            # multi-view
            self.view_list_front, self.view_list_right, self.view_list_back, self.view_list_left \
                = self.load_multiview_list()
            self.train_list_idx = 0
            self.view_front_idx = 0
            self.view_right_idx = 0
            self.view_back_idx = 0
            self.view_left_idx = 0
            print("view numbers:\nfront: {}\nright: {}\nback: {}\nleft: {}".format( \
                len(self.view_list_front), len(self.view_list_right), len(self.view_list_back), len(self.view_list_left)))
        self.shuffle_X_Y_files(label='all')

    @staticmethod
    def save_binvox(templates, x, y_pd, y_gt, epoch, iters, path):
        x = np.reshape(x, np.shape(x)[:-1])
        y_pd = np.reshape(y_pd, np.shape(y_pd)[:-1])
        y_gt = np.reshape(y_gt, np.shape(y_gt)[:-1])
        x_name = path + '{:02}_{:08}_X.binvox'.format(epoch, iters)
        y_pd_name = path + '{:02}_{:08}_Y_pd.binvox'.format(epoch, iters)
        y_gt_name = path + '{:02}_{:08}_Y_gt.binvox'.format(epoch, iters)
        with open(x_name, 'wb') as f:
            templates.data = x
            templates.dims = np.shape(x)
            brw.write(templates, f)
        with open(y_pd_name, 'wb') as f:
            templates.data = y_pd
            templates.dims = np.shape(y_pd)
            brw.write(templates, f)
        with open(y_gt_name, 'wb') as f:
            templates.data = y_gt
            templates.dims = np.shape(y_gt)
            brw.write(templates, f)
        print("epoch:{} iters:{} saved!".format(epoch,iters))
    
    @staticmethod
    def volume2pc(volume, vox_sz = 1.5):
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
        return volume_xyz

    @staticmethod
    def read_rotation_matrixs(data_list):
        with open(data_list) as f:
            lines = f.readlines()
            rotations = {}
            for l in lines:
                l = l.replace('\n', '').replace('\r', '')
                m = l.split(' ')
                for i in range(1,len(m)):
                    m[i] = float(m[i])
                rt = np.reshape(np.asarray(m[1:]), [4,4])
                rotations[m[0]] = rt[:3,:3]
        return rotations

    @staticmethod
    def read_names_from_list(data_list):
        # read file names
        names = []
        with open(data_list) as f:
            lines = f.readlines()
            for l in lines:
                l = l.replace('\r','').replace('\n','')
                if len(l) == 0:
                    continue
                names.append(l)
        return names

    @staticmethod
    def vox_down_single(vox, to_res):
        from_res = vox.shape[0]
        step = int(from_res / to_res)
        vox = np.reshape(vox,[from_res,from_res,from_res])
        new_vox = block_reduce(vox,(step,step,step),func=np.max)
        new_vox = np.reshape(new_vox,[to_res,to_res,to_res,1])
        return new_vox

    @staticmethod
    def vox_down_batch(vox_bat, to_res):
        from_res = vox_bat.shape[1]
        step = int(from_res / to_res)
        new_vox_bat = []
        for b in range(vox_bat.shape[0]):
            tp = np.reshape(vox_bat[b,:,:,:,:], [from_res,from_res,from_res])
            tp = block_reduce(tp,(step,step,step),func=np.max)
            tp = np.reshape(tp,[to_res,to_res,to_res,1])
            new_vox_bat.append(tp)
        new_vox_bat = np.asarray(new_vox_bat)
        return new_vox_bat

    @staticmethod
    def voxel_grid_padding(a):
        x_d = a.shape[0]
        y_d = a.shape[1]
        z_d = a.shape[2]
        channel = a.shape[3]
        ori_vox_res = 256
        size = [ori_vox_res, ori_vox_res, ori_vox_res,channel]
        b = np.zeros(size,dtype=np.float32)

        bx_s = 0;bx_e = size[0];by_s = 0;by_e = size[1];bz_s = 0; bz_e = size[2]
        ax_s = 0;ax_e = x_d;ay_s = 0;ay_e = y_d;az_s = 0;az_e = z_d
        if x_d > size[0]:
            ax_s = int((x_d - size[0]) / 2)
            ax_e = int((x_d - size[0]) / 2) + size[0]
        else:
            bx_s = int((size[0] - x_d) / 2)
            bx_e = int((size[0] - x_d) / 2) + x_d

        if y_d > size[1]:
            ay_s = int((y_d - size[1]) / 2)
            ay_e = int((y_d - size[1]) / 2) + size[1]
        else:
            by_s = int((size[1] - y_d) / 2)
            by_e = int((size[1] - y_d) / 2) + y_d

        if z_d > size[2]:
            az_s = int((z_d - size[2]) / 2)
            az_e = int((z_d - size[2]) / 2) + size[2]
        else:
            bz_s = int((size[2] - z_d) / 2)
            bz_e = int((size[2] - z_d) / 2) + z_d
        b[bx_s:bx_e, by_s:by_e, bz_s:bz_e,:] = a[ax_s:ax_e, ay_s:ay_e, az_s:az_e, :]

        #Data.plotFromVoxels(b)
        return b

    @staticmethod
    def load_single_voxel_grid(path, out_vox_res=256):
        with open(path, 'rb') as f:
            voxel_grid = brw.read_as_3d_array(f)
        #voxel_data=voxel_grid.data.astype(float)
        voxel_data=voxel_grid.data
        if len(voxel_data)<=0:
            print (" load_single_voxel_grid error: ", path)
            exit()
        ## downsample
        if out_vox_res < 256:
            voxel_data = Foot_Data.vox_down_single(voxel_data, to_res=out_vox_res)
        voxel_data = np.reshape(voxel_data, [out_vox_res, out_vox_res, out_vox_res, 1])
        return voxel_data

    @staticmethod
    def transform_volume_simple(input_volume, flag):
        # code verified by sgf 2018.11.30

        # rotate y and z first
        input_volume = np.swapaxes(input_volume, 1, 2)
        input_volume = input_volume[:,::-1,:,:]
        if flag == 'front':
            return input_volume
        res = np.zeros_like(input_volume)
        volume_flip = np.swapaxes(input_volume, 0, 1)
        data_shape = np.shape(res)
        # res = input_volume[::-1,::-1,::-1,:]
        if flag == 'back':
            res = input_volume[::-1,::-1,:,:]
        elif flag == 'left':
            res = volume_flip[::-1,:,:,:]
        elif flag == 'right':
            res = volume_flip[:,::-1,:,:]
        else:
            print("wrong volume flags!")
            exit()
        # for i in range(data_shape[0]):
        #     for j in range(data_shape[1]):
        #         for k in range(data_shape[2]):
        #             if flag == 'left':
        #                 res[i][j][k][0] = input_volume[j][-i][k][0]
        #             elif flag == 'right':
        #                 res[i][j][k][0] = input_volume[-j][i][k][0]
        #             elif flag == 'back':
        #                 res[i][j][k][0] = input_volume[-i][-j][k][0]
        #             else:
        #                 print("wrong volume flags!")
        #                 exit()
        return res

    @staticmethod
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
        volume_tmp = voxel_data[:,::-1,::-1,:] # a faster way to flip
        # volume_tmp = np.zeros((64,64,64,1),np.bool)
        # for i in range(64):
        #     for j in range(64):
        #         for k in range(64):
        #             volume_tmp[i][j][k][0]=voxel_data[i][63-j][63-k][0]
        return volume_tmp

    def load_X_Y_files_paths_all(self, obj_names, label='train'):
        if label =='train':
            subfix = self.subfix_train
        elif label == 'test':
            subfix = self.subfix_test
        else:
            print ("label error!!")
            exit()

        X_data_files_all = []
        Y_data_files_all = []
        for name in obj_names:
            X_folder = self.data_path_x + name + subfix + '/'
            Y_file = self.data_path_y + name + subfix
            with open(self.data_path_x + 'view_list_all.txt', 'rb') as f:
                lst = f.readlines()
                for l in lst:
                    l = l.replace('\n','').replace('\r', '')
                    if len(l)>0:
                        X_data_files_all.append(X_folder + l + self.subfix_x + '.txt')
                        Y_data_files_all.append(Y_file + self.subfix_y + '.binvox')
                        if not os.path.exists(X_data_files_all[-1]):
                            print (X_data_files_all[-1] + "not exist!")
                            exit()
                        if not os.path.exists(Y_data_files_all[-1]):
                            print (Y_data_files_all[-1] + "not exist!")
                            exit()
        return X_data_files_all, Y_data_files_all
    def load_multiview_list(self):
        with open(self.data_path_x + 'view_list_front.txt', 'rb') as f:
            lst = f.readlines()
            view_list_front = []
            for l in lst:
                l = l.replace('\n','').replace('\r', '')
                if len(l)>0:
                    view_list_front.append(l)
        with open(self.data_path_x + 'view_list_right.txt', 'rb') as f:
            lst = f.readlines()
            view_list_right = []
            for l in lst:
                l = l.replace('\n','').replace('\r', '')
                if len(l)>0:
                    view_list_right.append(l)
        with open(self.data_path_x + 'view_list_back.txt', 'rb') as f:
            lst = f.readlines()
            view_list_back = []
            for l in lst:
                l = l.replace('\n','').replace('\r', '')
                if len(l)>0:
                    view_list_back.append(l)
        with open(self.data_path_x + 'view_list_left.txt', 'rb') as f:
            lst = f.readlines()
            view_list_left = []
            for l in lst:
                l = l.replace('\n','').replace('\r', '')
                if len(l)>0:
                    view_list_left.append(l)
        
        return view_list_front, view_list_right, view_list_back, view_list_left

    def load_singleview_list(self):
        with open(self.data_path_x + 'view_list_all.txt', 'rb') as f:
            lst = f.readlines()
            view_list_all = []
            for l in lst:
                l = l.replace('\n','').replace('\r', '')
                if len(l)>0:
                    view_list_all.append(l)
        return view_list_all

    def load_X_Y_voxel_grids_singleview(self, X_data_files, Y_data_files, view_files=None):
        X_voxel_grids = []
        Y_voxel_grids = []
        R_matrixs = []
        for Y_f in Y_data_files:
            Y_voxel_grid = self.load_single_voxel_grid(Y_f, out_vox_res=256)
            Y_voxel_grids.append(Y_voxel_grid)
        for X_f in X_data_files:
            X_voxel_grid = self.load_X_from_txt(X_f, out_vox_res=64)
            X_voxel_grids.append(X_voxel_grid)
        if not view_files == None:
            for v_f in view_files:
                R_matrixs.append(self.rotation_matrixs[v_f])

        return np.asarray(X_voxel_grids), np.asarray(Y_voxel_grids), np.asarray(R_matrixs)

    def load_X_Y_voxel_grids_multiview(self, X_f_files, X_r_files, X_b_files, X_l_files, Y_files):
        X_f_voxel_grids = []
        X_r_voxel_grids = []
        X_b_voxel_grids = []
        X_l_voxel_grids = []
        Y_voxel_grids = []
        for Y_f in Y_files:
            Y_voxel_grid = self.load_single_voxel_grid(Y_f, out_vox_res=256)
            Y_voxel_grids.append(Y_voxel_grid)
        for X_f in X_f_files:
            X_voxel_grid = self.load_X_from_txt(X_f, out_vox_res=64)
            if self.config['transform_views']:
                X_voxel_grid = self.transform_volume_simple(X_voxel_grid, 'front')
            X_f_voxel_grids.append(X_voxel_grid)
        for X_f in X_r_files:
            X_voxel_grid = self.load_X_from_txt(X_f, out_vox_res=64)
            if self.config['transform_views']:
                X_voxel_grid = self.transform_volume_simple(X_voxel_grid, 'right')
            X_r_voxel_grids.append(X_voxel_grid)
        for X_f in X_b_files:
            X_voxel_grid = self.load_X_from_txt(X_f, out_vox_res=64)
            if self.config['transform_views']:
                X_voxel_grid = self.transform_volume_simple(X_voxel_grid, 'back')
            X_b_voxel_grids.append(X_voxel_grid)
        for X_f in X_l_files:
            X_voxel_grid = self.load_X_from_txt(X_f, out_vox_res=64)
            if self.config['transform_views']:
                X_voxel_grid = self.transform_volume_simple(X_voxel_grid, 'left')
            X_l_voxel_grids.append(X_voxel_grid)
        return np.asarray(X_f_voxel_grids), np.asarray(X_r_voxel_grids), np.asarray(X_b_voxel_grids),\
             np.asarray(X_l_voxel_grids), np.asarray(Y_voxel_grids)
    
    def shuffle_X_Y_files(self, label='obj'):
        if self.config['view_flag'] == 'single_view':
            self.train_sample_idxs = np.random.permutation(self.train_sample_idxs)
            self.train_sample_idx = 0
            if label == 'obj':
                self.train_names = np.random.permutation(self.train_names)
                self.train_list_idx = 0
            elif label == 'view_all':
                self.view_list_all = np.random.permutation(self.view_list_all)
                self.view_all_idx = 0
            else:
                self.train_names = np.random.permutation(self.train_names)
                self.train_list_idx = 0
                self.view_list_all = np.random.permutation(self.view_list_all)
                self.view_all_idx = 0
        else:
            if label == 'obj':
                self.train_names = np.random.permutation(self.train_names)
                self.train_list_idx = 0
            elif label == 'front':
                self.view_list_front = np.random.permutation(self.view_list_front)
                self.view_front_idx = 0
            elif label == 'right':
                self.view_list_right = np.random.permutation(self.view_list_right)
                self.view_right_idx = 0
            elif label == 'back':
                self.view_list_back = np.random.permutation(self.view_list_back)
                self.view_back_idx = 0
            elif label == 'left':
                self.view_list_left = np.random.permutation(self.view_list_left)
                self.view_left_idx = 0
            else:
                self.train_names = np.random.permutation(self.train_names)
                self.train_list_idx = 0
                self.view_list_front = np.random.permutation(self.view_list_front)
                self.view_front_idx = 0
                self.view_list_right = np.random.permutation(self.view_list_right)
                self.view_right_idx = 0
                self.view_list_back = np.random.permutation(self.view_list_back)
                self.view_back_idx = 0
                self.view_list_left = np.random.permutation(self.view_list_left)
                self.view_left_idx = 0

    
    def acquire_data_files_single_view(self, label='train'):
        X_files = []
        Y_files = []
        view_files = []
        if label == 'train':
            for i in range(self.batch_size):
                if self.config['sampling_strategy'] == "full":
                    # full traing samples are used
                    obj_idx = int(self.train_sample_idx)/len(self.view_list_all)
                    view_idx = int(self.train_sample_idx)%len(self.view_list_all)
                    x_f = self.data_path_x + self.train_names[obj_idx] +'/'\
                        + self.view_list_all[view_idx] + self.subfix_x + '.txt'
                    y_f = self.data_path_y + self.train_names[obj_idx] + '.binvox'
                    X_files.append(x_f)
                    Y_files.append(y_f)
                    view_files.append(self.view_list_all[view_idx])
                    self.train_sample_idx += 1
                    if self.train_sample_idx >= len(self.train_sample_idxs):
                        self.shuffle_X_Y_files()
                else:
                    # random permutation of both training objs and views
                    x_f = self.data_path_x + self.train_names[self.train_list_idx] +'/'\
                        + self.view_list_all[self.view_all_idx] + self.subfix_x + '.txt'
                    y_f = self.data_path_y + self.train_names[self.view_all_idx] + '.binvox'
                    X_files.append(x_f)
                    Y_files.append(y_f)
                    view_files.append(self.view_list_all[self.view_all_idx])
                    self.train_list_idx += 1
                    self.view_all_idx += 1
                    if self.train_list_idx >= len(self.train_names):
                        self.shuffle_X_Y_files(label='obj')
                    if self.view_all_idx >= len(self.view_list_all):
                        self.shuffle_X_Y_files(label='view_all')
        else:
            idxs = random.sample(self.test_sample_idxs, self.batch_size)
            for i in idxs:
                obj_idx = int(i)/len(self.view_list_all)
                view_idx = int(i)%len(self.view_list_all)
                x_f = self.data_path_x + self.train_names[obj_idx] +'/'\
                    + self.view_list_all[view_idx] + self.subfix_x + '.txt'
                y_f = self.data_path_y + self.train_names[obj_idx] + '.binvox'
                X_files.append(x_f)
                Y_files.append(y_f)
                view_files.append(self.view_list_all[view_idx])
        return X_files, Y_files, view_files

    def acquire_data_files_multi_view(self, label='train'):
        X_f_files = []
        X_r_files = []
        X_b_files = []
        X_l_files = []
        Y_files = []
        if label == 'train':
            for i in range(self.batch_size):
                x_f = self.data_path_x + self.train_names[self.train_list_idx] +'/'\
                    + self.view_list_front[self.view_front_idx] + self.subfix_x + '.txt'
                x_r = self.data_path_x + self.train_names[self.train_list_idx] +'/'\
                    + self.view_list_right[self.view_right_idx] + self.subfix_x + '.txt'
                x_b = self.data_path_x + self.train_names[self.train_list_idx] +'/'\
                    + self.view_list_back[self.view_back_idx] + self.subfix_x + '.txt'
                x_l = self.data_path_x + self.train_names[self.train_list_idx] +'/'\
                    + self.view_list_back[self.view_back_idx] + self.subfix_x + '.txt'
                y_f = self.data_path_y + self.train_names[self.train_list_idx] + '.binvox'
                X_f_files.append(x_f)
                X_r_files.append(x_r)
                X_b_files.append(x_b)
                X_l_files.append(x_l)
                Y_files.append(y_f)
                self.train_list_idx += 1
                self.view_front_idx += 1
                self.view_right_idx += 1
                self.view_back_idx += 1
                self.view_left_idx += 1
                if self.train_list_idx >= len(self.train_names):
                    self.shuffle_X_Y_files(label='obj')
                    self.train_list_idx = 0
                if self.view_front_idx >= len(self.view_list_front):
                    self.shuffle_X_Y_files(label='front')
                    self.train_list_idx = 0
                if self.view_right_idx >= len(self.view_list_right):
                    self.shuffle_X_Y_files(label='right')
                    self.view_right_idx = 0
                if self.view_back_idx >= len(self.view_list_back):
                    self.shuffle_X_Y_files(label='back')
                    self.view_back_idx = 0
                if self.view_left_idx >= len(self.view_list_left):
                    self.shuffle_X_Y_files(label='left')
                    self.view_left_idx = 0
        else:
            test_name_lists = random.sample(self.test_names, self.batch_size)
            test_view_f_lists = random.sample(self.view_list_front, self.batch_size)
            test_view_r_lists = random.sample(self.view_list_right, self.batch_size)
            test_view_b_lists = random.sample(self.view_list_back, self.batch_size)
            test_view_l_lists = random.sample(self.view_list_left, self.batch_size)
            for i in range(self.batch_size):
                y_f = self.data_path_y + test_name_lists[i] + '.binvox'
                x_f = self.data_path_x + test_name_lists[i] + '/' + test_view_f_lists[i] + self.subfix_x + '.txt'
                x_r = self.data_path_x + test_name_lists[i] + '/' + test_view_r_lists[i] + self.subfix_x + '.txt'
                x_b = self.data_path_x + test_name_lists[i] + '/' + test_view_b_lists[i] + self.subfix_x + '.txt'
                x_l = self.data_path_x + test_name_lists[i] + '/' + test_view_l_lists[i] + self.subfix_x + '.txt'
                Y_files.append(y_f)
                X_f_files.append(x_f)
                X_r_files.append(x_r)
                X_b_files.append(x_b)
                X_l_files.append(x_l)
        return X_f_files, X_r_files, X_b_files, X_l_files, Y_files

    ###################### voxel grids
    def load_X_Y_voxel_grids_train_next_batch(self):
        # X_data_files_batch = self.X_train_files[self.batch_size * self.train_batch_index:self.batch_size * (self.train_batch_index + 1)]
        # Y_data_files_batch = self.Y_train_files[self.batch_size * self.train_batch_index:self.batch_size * (self.train_batch_index + 1)]
        # self.train_batch_index += 1
        if self.config['view_flag'] == 'single_view':
            X_data_files_batch, Y_data_files_batch, V_files_batch = self.acquire_data_files_single_view(label='train')
            X_voxel_grids, Y_voxel_grids, R_matrixs = self.load_X_Y_voxel_grids_singleview(X_data_files_batch, Y_data_files_batch, V_files_batch)
            return X_voxel_grids, Y_voxel_grids, R_matrixs
        else:
            X_f_files_batch, X_r_files_batch, X_b_files_batch, X_l_files_batch, Y_data_files_batch \
                = self.acquire_data_files_multi_view(label='train')
            X_f_voxel_grids, X_r_voxel_grids, X_b_voxel_grids, X_l_voxel_grids, Y_voxel_grids \
                = self.load_X_Y_voxel_grids_multiview(X_f_files_batch, X_r_files_batch, X_b_files_batch, X_l_files_batch, Y_data_files_batch)
            return X_f_voxel_grids, X_r_voxel_grids, X_b_voxel_grids, X_l_voxel_grids, Y_voxel_grids

    def load_X_Y_voxel_grids_test_next_batch(self,fix_sample=False):
        if fix_sample:
            random.seed(42)
        # idx = random.sample(range(len(self.Y_test_files)), self.batch_size)
        # X_test_files_batch = []
        # Y_test_files_batch = []
        # for i in idx:
        #     X_test_files_batch.append(self.X_test_files[i])
        #     Y_test_files_batch.append(self.Y_test_files[i])

        # X_test_batch, Y_test_batch = self.load_X_Y_voxel_grids(X_test_files_batch, Y_test_files_batch)
        # return X_test_batch, Y_test_batch
        if self.config['view_flag'] == 'single_view':
            X_data_files_batch, Y_data_files_batch, V_files_batch = self.acquire_data_files_single_view(label='test')
            X_voxel_grids, Y_voxel_grids, R_matrixs = self.load_X_Y_voxel_grids_singleview(X_data_files_batch, Y_data_files_batch, V_files_batch)
            return X_voxel_grids, Y_voxel_grids, R_matrixs
        else:
            X_f_files_batch, X_r_files_batch, X_b_files_batch, X_l_files_batch, Y_data_files_batch \
                = self.acquire_data_files_multi_view(label='test')
            X_f_voxel_grids, X_r_voxel_grids, X_b_voxel_grids, X_l_voxel_grids, Y_voxel_grids \
                = self.load_X_Y_voxel_grids_multiview(X_f_files_batch, X_r_files_batch, X_b_files_batch, X_l_files_batch, Y_data_files_batch)
            return X_f_voxel_grids, X_r_voxel_grids, X_b_voxel_grids, X_l_voxel_grids, Y_voxel_grids

    def run(self):
        while not self.stop_queue:
            ## train
            if not self.queue_train.full():
                # if self.train_batch_index>=self.total_train_batch_num:
                #     self.shuffle_X_Y_files(label='train')
                #     print ('shuffle')
                if self.config['view_flag'] == 'single_view':
                    X_b, Y_b, V_b = self.load_X_Y_voxel_grids_train_next_batch()
                    self.queue_train.put((X_b, Y_b, V_b))
                else:
                    Xf_b, Xr_b, Xb_b, Xl_b, Y_b = self.load_X_Y_voxel_grids_train_next_batch()
                    self.queue_train.put((Xf_b, Xr_b, Xb_b, Xl_b, Y_b))

class Data(threading.Thread):
    def __init__(self,config):
        super(Data,self).__init__()
        self.config = config
        self.train_batch_index = 0

        self.batch_size = config['batch_size']
        self.vox_res_x = config['vox_res_x']
        self.vox_res_y = config['vox_res_y']
        self.data_path_x = str(config['data_path_x'])
        self.data_path_y = str(config['data_path_y'])
        self.subfix_x = str(config['subfix_x'])
        self.subfix_y = str(config['subfix_y'])
        self.subfix_train = str(config['subfix_train'])
        self.subfix_test = str(config['subfix_test'])

        self.train_names = self.read_names_from_list(config['train_list'])
        self.test_names = self.read_names_from_list(config['test_list'])

        self.queue_train = queue.Queue(3)
        self.stop_queue = False

        self.X_train_files, self.Y_train_files = self.load_X_Y_files_paths_all( self.train_names,label='train')
        self.X_test_files, self.Y_test_files = self.load_X_Y_files_paths_all(self.test_names, label='test')

        print ('X_train_files:',len(self.X_train_files))
        print ('X_test_files:',len(self.X_test_files))

        self.total_train_batch_num = int(len(self.X_train_files) // self.batch_size)
        self.total_test_seq_batch = int(len(self.X_test_files) // self.batch_size)

    @staticmethod
    def read_names_from_list(data_list):
        # read file names
        names = []
        with open(data_list) as f:
            lines = f.readlines()
            for l in lines:
                l = l.replace('\r','').replace('\n','')
                if len(l) == 0:
                    continue
                names.append(l)
        return names

    @staticmethod
    def vox_down_single(vox, to_res):
        from_res = vox.shape
        step = int(from_res[0] / to_res[0])
        vox = np.reshape(vox,[from_res[0],from_res[1],from_res[2]])
        new_vox = block_reduce(vox,(step,step,step),func=np.max)
        new_vox = np.reshape(new_vox,[to_res[0],to_res[1],to_res[2],1])
        return new_vox

    @staticmethod
    def vox_down_batch(vox_bat, to_res):
        from_res = vox_bat.shape[1]
        step = int(from_res / to_res)
        new_vox_bat = []
        for b in range(vox_bat.shape[0]):
            tp = np.reshape(vox_bat[b,:,:,:,:], [from_res,from_res,from_res])
            tp = block_reduce(tp,(step,step,step),func=np.max)
            tp = np.reshape(tp,[to_res,to_res,to_res,1])
            new_vox_bat.append(tp)
        new_vox_bat = np.asarray(new_vox_bat)
        return new_vox_bat

    @staticmethod
    def voxel_grid_padding(a):
        x_d = a.shape[0]
        y_d = a.shape[1]
        z_d = a.shape[2]
        channel = a.shape[3]
        ori_vox_res = 256
        size = [ori_vox_res, ori_vox_res, ori_vox_res,channel]
        b = np.zeros(size,dtype=np.float32)

        bx_s = 0;bx_e = size[0];by_s = 0;by_e = size[1];bz_s = 0; bz_e = size[2]
        ax_s = 0;ax_e = x_d;ay_s = 0;ay_e = y_d;az_s = 0;az_e = z_d
        if x_d > size[0]:
            ax_s = int((x_d - size[0]) / 2)
            ax_e = int((x_d - size[0]) / 2) + size[0]
        else:
            bx_s = int((size[0] - x_d) / 2)
            bx_e = int((size[0] - x_d) / 2) + x_d

        if y_d > size[1]:
            ay_s = int((y_d - size[1]) / 2)
            ay_e = int((y_d - size[1]) / 2) + size[1]
        else:
            by_s = int((size[1] - y_d) / 2)
            by_e = int((size[1] - y_d) / 2) + y_d

        if z_d > size[2]:
            az_s = int((z_d - size[2]) / 2)
            az_e = int((z_d - size[2]) / 2) + size[2]
        else:
            bz_s = int((size[2] - z_d) / 2)
            bz_e = int((size[2] - z_d) / 2) + z_d
        b[bx_s:bx_e, by_s:by_e, bz_s:bz_e,:] = a[ax_s:ax_e, ay_s:ay_e, az_s:az_e, :]

        #Data.plotFromVoxels(b)
        return b

    @staticmethod
    def load_single_voxel_grid(path, out_vox_res=[256, 256, 256]):
        with np.load(path) as da:
            voxel_grid = da['arr_0']
        if len(voxel_grid)<=0:
            print (" load_single_voxel_grid error: ", path)
            exit()

        #Data.plotFromVoxels(voxel_grid)
        voxel_grid = Data.voxel_grid_padding(voxel_grid)

        ## downsample
        if max(out_vox_res) < 256:
            voxel_grid = Data.vox_down_single(voxel_grid, to_res=out_vox_res)
        return voxel_grid

    def load_X_Y_files_paths_all(self, obj_names, label='train'):
        if label =='train':
            subfix = self.subfix_train
        elif label == 'test':
            subfix = self.subfix_test
        else:
            print ("label error!!")
            exit()

        X_data_files_all = []
        Y_data_files_all = []
        for name in obj_names:
            X_folder = self.data_path_x + name + '/' + subfix + self.subfix_x
            Y_folder = self.data_path_y + name + '/' + subfix + self.subfix_y
            X_data_files = [X_f for X_f in sorted(os.listdir(X_folder))]
            Y_data_files = [Y_f for Y_f in sorted(os.listdir(Y_folder))]

            for X_f, Y_f in zip(X_data_files, Y_data_files):
                if X_f[0:15] != Y_f[0:15]:
                    print ("index inconsistent!!")
                    exit()
                X_data_files_all.append(X_folder + X_f)
                Y_data_files_all.append(Y_folder + Y_f)
        return X_data_files_all, Y_data_files_all

    def load_X_Y_voxel_grids(self,X_data_files, Y_data_files):
        if len(X_data_files) !=self.batch_size or len(Y_data_files)!=self.batch_size:
            print ("load_X_Y_voxel_grids error:", X_data_files, Y_data_files)
            exit()

        X_voxel_grids = []
        Y_voxel_grids = []
        index = -1
        for X_f, Y_f in zip(X_data_files, Y_data_files):
            index += 1
            X_voxel_grid = self.load_single_voxel_grid(X_f, out_vox_res=self.vox_res_x)
            X_voxel_grids.append(X_voxel_grid)

            Y_voxel_grid = self.load_single_voxel_grid(Y_f, out_vox_res=self.vox_res_y)
            Y_voxel_grids.append(Y_voxel_grid)

        X_voxel_grids = np.asarray(X_voxel_grids)
        Y_voxel_grids = np.asarray(Y_voxel_grids)
        return X_voxel_grids, Y_voxel_grids

    def shuffle_X_Y_files(self, label='train'):
        X_new = []; Y_new = []
        if label == 'train':
            X = self.X_train_files; Y = self.Y_train_files
            self.train_batch_index = 0
            index = list(range(len(X)))
            shuffle(index)
            for i in index:
                X_new.append(X[i])
                Y_new.append(Y[i])
            self.X_train_files = X_new
            self.Y_train_files = Y_new
        else:
            print ("shuffle_X_Y_files error!\n")
            exit()

    ###################### voxel grids
    def load_X_Y_voxel_grids_train_next_batch(self):
        X_data_files = self.X_train_files[self.batch_size * self.train_batch_index:self.batch_size * (self.train_batch_index + 1)]
        Y_data_files = self.Y_train_files[self.batch_size * self.train_batch_index:self.batch_size * (self.train_batch_index + 1)]
        self.train_batch_index += 1

        X_voxel_grids, Y_voxel_grids = self.load_X_Y_voxel_grids(X_data_files, Y_data_files)
        return X_voxel_grids, Y_voxel_grids

    def load_X_Y_voxel_grids_test_next_batch(self,fix_sample=False):
        if fix_sample:
            random.seed(42)
        idx = random.sample(range(len(self.X_test_files)), self.batch_size)
        X_test_files_batch = []
        Y_test_files_batch = []
        for i in idx:
            X_test_files_batch.append(self.X_test_files[i])
            Y_test_files_batch.append(self.Y_test_files[i])

        X_test_batch, Y_test_batch = self.load_X_Y_voxel_grids(X_test_files_batch, Y_test_files_batch)
        return X_test_batch, Y_test_batch

    def run(self):
        while not self.stop_queue:
            ## train
            if not self.queue_train.full():
                if self.train_batch_index>=self.total_train_batch_num:
                    self.shuffle_X_Y_files(label='train')
                    print ('shuffle')
                X_b, Y_b = self.load_X_Y_voxel_grids_train_next_batch()
                self.queue_train.put((X_b, Y_b))