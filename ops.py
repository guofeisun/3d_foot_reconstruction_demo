import tensorflow as tf
import numpy as np

class Ops:

    @staticmethod
    def lrelu(x, leak=0.2):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

    @staticmethod
    def relu(x):
        return tf.nn.relu(x)

    @staticmethod
    def xxlu(x,label,name=None):
        if label =='relu':
            return  Ops.relu(x)
        if label =='lrelu':
            return  Ops.lrelu(x,leak=0.2)

    @staticmethod
    def variable_sum(var, name):
        with tf.name_scope(name):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    @staticmethod
    def variable_count():
        total_para = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_para = 1
            for dim in shape:
                variable_para *= dim.value
            total_para += variable_para
        return total_para

    @staticmethod
    def fc(x, out_d, name):
        xavier_init = tf.contrib.layers.xavier_initializer()
        zero_init = tf.zeros_initializer()
        in_d = x.get_shape()[1]
        w = tf.get_variable(name + '_w', [in_d, out_d], initializer=xavier_init)
        b = tf.get_variable(name + '_b', [out_d], initializer=zero_init)
        y = tf.nn.bias_add(tf.matmul(x, w), b)
        Ops.variable_sum(w, name)
        return y

    @staticmethod
    def maxpool3d(x,k,s,pad='SAME'):
        ker =[1,k,k,k,1]
        str =[1,s,s,s,1]
        y = tf.nn.max_pool3d(x,ksize=ker,strides=str,padding=pad)
        return y

    @staticmethod
    def conv3d(x, k, out_c, str, name,pad='SAME'):
        xavier_init = tf.contrib.layers.xavier_initializer()
        zero_init = tf.zeros_initializer()
        in_c = x.get_shape()[4]
        w = tf.get_variable(name + '_w', [k, k, k, in_c, out_c], initializer=xavier_init)
        b = tf.get_variable(name + '_b', [out_c], initializer=zero_init)
        stride = [1, str, str, str, 1]
        y = tf.nn.bias_add(tf.nn.conv3d(x, w, stride, pad), b)
        Ops.variable_sum(w, name)
        return y

    @staticmethod
    def deconv3d(x, k, out_c, str, name,pad='SAME'):
        xavier_init = tf.contrib.layers.xavier_initializer()
        zero_init = tf.zeros_initializer()
        [_, in_d1, in_d2, in_d3, in_c] = x.get_shape()
        in_d1 = int(in_d1); in_d2 = int(in_d2); in_d3 = int(in_d3); in_c = int(in_c)
        bat = tf.shape(x)[0]
        w = tf.get_variable(name + '_w', [k, k, k, out_c, in_c], initializer=xavier_init)
        b = tf.get_variable(name + '_b', [out_c], initializer=zero_init)
        out_shape = [bat, in_d1 * str, in_d2 * str, in_d3 * str, out_c]
        stride = [1, str, str, str, 1]
        y = tf.nn.conv3d_transpose(x, w, output_shape=out_shape, strides=stride, padding=pad)
        y = tf.nn.bias_add(y, b)
        Ops.variable_sum(w, name)
        return y

    @staticmethod
    def rotate_volume(X, R, sz=256):
        sp = X.get_shape()
        X = tf.reshape(X, sp[:-1])
        hf_sz = sz/2
        v_shape = [sz, sz, sz]
        v_range = range(sz)
        
        # generate index array
        idx_grid = tf.stack(tf.meshgrid(v_range, v_range, v_range))
        idx_grid = tf.cast(idx_grid, tf.float32)
        idx_vector = tf.reshape(idx_grid, [3, -1])
        idx_vector_r = idx_vector - np.array([[hf_sz],[hf_sz],[hf_sz]], float)

        # compute rotated index array
        idx_vector_r = tf.matmul(R, idx_vector_r)
        idx_vector_r = idx_vector_r + np.array([[hf_sz],[hf_sz],[hf_sz]], float)

        idx_vector_rx = tf.clip_by_value(idx_vector_r[0], 0.0, float(sz-1))
        idx_vector_ry = tf.clip_by_value(idx_vector_r[1], 0.0, float(sz-1))
        idx_vector_rz = tf.clip_by_value(idx_vector_r[2], 0.0, float(sz-1))
        idx_vector_r = tf.stack([idx_vector_rx, idx_vector_ry, idx_vector_rz], axis=1)
        idx_vector_r = tf.cast(tf.round(idx_vector_r), tf.int32)

        # assign values 
        X_r = tf.gather_nd(X, idx_vector_r)
        X_r = tf.reshape(X_r, X.shape)
        X_r = tf.transpose(X_r, [1,0,2]) # # the x y z axis here needs small modification, reverse x and y

        return tf.reshape(X_r, sp)

    @staticmethod
    def IoU(X, Y):
        union = tf.reduce_sum(tf.cast(tf.logical_or(X>=0.5, Y>=0.5), tf.float32), axis=[1,2,3,4])
        intersection = tf.reduce_sum(tf.cast(tf.logical_and(X>=0.5, Y>=0.5), tf.float32), axis=[1,2,3,4])
        iou = tf.reduce_mean(tf.div(intersection, union))

        return iou
        
    @staticmethod
    def get_foot_size(V, voxel_size=1.5, batch_size=4):
        '''V, dtype must be bool'''
        v_sz = []
        for i in range(batch_size):
            valid_idx = tf.where(V[i])
            max_loc = tf.reduce_max(valid_idx, axis=0)
            min_loc = tf.reduce_min(valid_idx, axis=0)
            v_sz.append(tf.cast((max_loc-min_loc), tf.float32)*voxel_size)
        return tf.stack(v_sz, axis=0)