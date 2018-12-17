from ops import Ops
import os
import shutil
import scipy
import tensorflow as tf
import numpy as np
from cnn_models import aeu, dis, aeu_multiview
from model_config import save_config
import binvox_rw as brw

class Network_Four_View:
    def __init__(self, config, sess, demo_only=False):
        if demo_only:
            return  # no need to creat folders
        self.config = config
        self.sess = sess
        self.train_mod_dir = str(config['save_path']) + 'train_mod/'
        self.train_sum_dir = str(config['save_path']) + 'train_sum/'
        self.test_res_dir = str(config['save_path']) + 'test_res/'
        self.test_sum_dir = str(config['save_path']) + 'test_sum/'
        self.retrain_mod_dir = str(config['retrain_path']) + 'train_mod/'
        self.retrain_epoch = str(config['retrain_epoch'])
        self.retrain = config['retrain']


        print ("re_train:", config['retrain'])
        if self.retrain:
            if not os.path.exists(self.retrain_mod_dir + 'model_' + str(self.retrain_epoch) + '.cptk.data-00000-of-00001'):
                print ('retrain model not found!')
                exit()
            else:
                print ('retrain model found!')
        
        if os.path.exists(self.test_res_dir):
            shutil.rmtree(self.test_res_dir)
            os.makedirs(self.test_res_dir)
            print ('test_res_dir: deleted and then created!')
        else:
            os.makedirs(self.test_res_dir)
            print ('test_res_dir: created!')

        if os.path.exists(self.train_mod_dir):
            shutil.rmtree(self.train_mod_dir)
            os.makedirs(self.train_mod_dir)
            print ('train_mod_dir: deleted and then created!')
        else:
            os.makedirs(self.train_mod_dir)
            print ('train_mod_dir: created!')

        if os.path.exists(self.train_sum_dir):
            shutil.rmtree(self.train_sum_dir)
            os.makedirs(self.train_sum_dir)
            print ('train_sum_dir: deleted and then created!')
        else:
            os.makedirs(self.train_sum_dir)
            print ('train_sum_dir: created!')

        if os.path.exists(self.test_sum_dir):
            shutil.rmtree(self.test_sum_dir)
            os.makedirs(self.test_sum_dir)
            print ('test_sum_dir: deleted and then created!')
        else:
            os.makedirs(self.test_sum_dir)
            print ('test_sum_dir: created!')

        save_config(self.config['save_path'] + 'config.json' ,self.config)

    def build_graph(self):
        self.X_f_bool = tf.placeholder(shape=[None, self.config['vox_res_x'][0], self.config['vox_res_x'][1], self.config['vox_res_x'][2], 1], dtype=tf.bool)
        self.X_r_bool = tf.placeholder(shape=[None, self.config['vox_res_x'][0], self.config['vox_res_x'][1], self.config['vox_res_x'][2], 1], dtype=tf.bool)
        self.X_b_bool = tf.placeholder(shape=[None, self.config['vox_res_x'][0], self.config['vox_res_x'][1], self.config['vox_res_x'][2], 1], dtype=tf.bool)
        self.X_l_bool = tf.placeholder(shape=[None, self.config['vox_res_x'][0], self.config['vox_res_x'][1], self.config['vox_res_x'][2], 1], dtype=tf.bool)
        self.Y_bool = tf.placeholder(shape=[None, self.config['vox_res_y'][0], self.config['vox_res_y'][1], self.config['vox_res_y'][2], 1], dtype=tf.bool)
        self.X_f = tf.cast(self.X_f_bool, tf.float32)
        self.X_r = tf.cast(self.X_r_bool, tf.float32)
        self.X_b = tf.cast(self.X_b_bool, tf.float32)
        self.X_l = tf.cast(self.X_l_bool, tf.float32)
        self.Y = tf.cast(self.Y_bool, tf.float32)

        with tf.variable_scope('aeu_multiview'):
            self.Y_pred, self.Y_pred_modi = aeu_multiview(self.X_f, self.X_r, self.X_b, self.X_l, self.config['vox_res_x'], flag=self.config['model_flag'])
        with tf.variable_scope('dis'):
            self.XY_real_pair = dis(self.X_f, self.Y, self.config['vox_res_x'], self.config['vox_res_y'], is_conditional=False)
        with tf.variable_scope('dis',reuse=True):
            self.XY_fake_pair = dis(self.X_f, self.Y_pred, self.config['vox_res_x'], self.config['vox_res_y'], is_conditional=False)

        ################################ IoU
        self.IoU = Ops.IoU(self.Y, self.Y_pred_modi)
        sum_IoU = tf.summary.scalar('IoU', self.IoU)

        ################################ ae loss
        Y_ = tf.reshape(self.Y, shape=[-1, self.config['vox_res_y'][0], self.config['vox_res_y'][1], self.config['vox_res_y'][2]])
        Y_pred_modi_ = tf.reshape(self.Y_pred_modi, shape=[-1, self.config['vox_res_y'][0], self.config['vox_res_y'][1], self.config['vox_res_y'][2]])
        w = 0.85
        self.aeu_loss = tf.reduce_mean(-tf.reduce_mean(w * Y_ * tf.log(Y_pred_modi_ + 1e-8), reduction_indices=[1]) -
                                    tf.reduce_mean((1 - w) * (1 - Y_) * tf.log(1 - Y_pred_modi_ + 1e-8), reduction_indices=[1]))
        sum_aeu_loss = tf.summary.scalar('aeu_loss', self.aeu_loss)

        ################################ wgan loss
        self.gan_g_loss = -tf.reduce_mean(self.XY_fake_pair)
        self.gan_d_loss_no_gp = tf.reduce_mean(self.XY_fake_pair) - tf.reduce_mean(self.XY_real_pair)
        sum_gan_g_loss = tf.summary.scalar('gan_g_loss', self.gan_g_loss)
        sum_gan_d_loss_no_gp = tf.summary.scalar('gan_d_loss_no_gp', self.gan_d_loss_no_gp)
        alpha = tf.random_uniform(shape=[tf.shape(self.Y)[0], self.config['vox_res_y'][0], self.config['vox_res_y'][1], self.config['vox_res_y'][2]], minval=0.0, maxval=1.0)

        Y_pred_ = tf.reshape(self.Y_pred, shape=[-1, self.config['vox_res_y'][0], self.config['vox_res_y'][1], self.config['vox_res_y'][2]])
        differences_ = Y_pred_ - Y_
        interpolates = Y_ + alpha*differences_
        with tf.variable_scope('dis',reuse=True):
            XY_fake_intep = dis(self.X_f, interpolates, self.config['vox_res_x'], self.config['vox_res_y'])
        gradients = tf.gradients(XY_fake_intep, [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes - 1.0) ** 2)
        self.gan_d_loss_gp = self.gan_d_loss_no_gp + 10 * gradient_penalty
        sum_gan_d_loss_gp = tf.summary.scalar('gan_d_loss_gp', self.gan_d_loss_gp)

        #################################  ae + gan loss
        gan_g_w = 20
        aeu_w = 100 - gan_g_w
        self.aeu_gan_g_loss = aeu_w*self.aeu_loss + gan_g_w*self.gan_g_loss

        aeu_var = [var for var in tf.trainable_variables() if var.name.startswith('aeu_multiview')]
        dis_var = [var for var in tf.trainable_variables() if var.name.startswith('dis')]
        self.aeu_g_optim = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999, epsilon=1e-8).\
                        minimize(self.aeu_gan_g_loss, var_list=aeu_var)
        self.dis_optim = tf.train.AdamOptimizer(learning_rate=0.00005, beta1=0.9, beta2=0.999, epsilon=1e-8).\
                        minimize(self.gan_d_loss_gp,var_list=dis_var)

        print (Ops.variable_count())
        self.sum_merged = tf.summary.merge_all()
        self.saver = tf.train.Saver(max_to_keep=10)

        self.sum_writer_train = tf.summary.FileWriter(self.train_sum_dir, self.sess.graph)
        self.sum_write_test = tf.summary.FileWriter(self.test_sum_dir)

        # to retrain
        if self.config['retrain']:
            print ('restoring saved model')
            self.saver.restore(self.sess, self.retrain_mod_dir + 'model_' + str(self.retrain_epoch) + '.cptk')
        else:
            print ('initilizing model')
            self.sess.run(tf.global_variables_initializer())

        return 0

    def train(self, data):
        for epoch in range(10):
            total_train_batch_num = 10000
            for i in range(total_train_batch_num):

                #################### training
                X_f_train_batch, X_r_train_batch, X_b_train_batch, X_l_train_batch, Y_train_batch = data.queue_train.get()
                self.sess.run(self.dis_optim, feed_dict={self.X_f:X_f_train_batch, self.X_r:X_r_train_batch, self.X_b:X_b_train_batch, self.X_l:X_l_train_batch, self.Y:Y_train_batch})
                self.sess.run(self.aeu_g_optim, feed_dict={self.X_f:X_f_train_batch, self.X_r:X_r_train_batch, self.X_b:X_b_train_batch, self.X_l:X_l_train_batch, self.Y:Y_train_batch})

                #aeu_loss_c, gan_g_loss_c, gan_d_loss_no_gp_c, gan_d_loss_gp_c, sum_train = self.sess.run(
                #[self.aeu_loss, self.gan_g_loss, self.gan_d_loss_no_gp, self.gan_d_loss_gp, self.sum_merged],
                #feed_dict={self.X:X_train_batch, self.Y:Y_train_batch})

                if i%50==0:
                    iou_c, aeu_loss_c, gan_g_loss_c, gan_d_loss_no_gp_c, gan_d_loss_gp_c, sum_train = self.sess.run(
                        [self.IoU, self.aeu_loss, self.gan_g_loss, self.gan_d_loss_no_gp, self.gan_d_loss_gp, self.sum_merged],
                        feed_dict={self.X_f_bool:X_f_train_batch, self.X_r_bool:X_r_train_batch, self.X_b_bool:X_b_train_batch, self.X_l_bool:X_l_train_batch, self.Y_bool:Y_train_batch})

                    self.sum_writer_train.add_summary(sum_train, epoch * total_train_batch_num + i)
                    print ('ep:',epoch,'i:',i, 'train IoU:', iou_c, 'train aeu loss:',aeu_loss_c, 'gan g loss:',gan_g_loss_c,
                       'gan d loss no gp:',gan_d_loss_no_gp_c,'gan d loss gp:', gan_d_loss_gp_c)

                #################### testing
                if i%500==0:
                    X_f_test_batch, X_r_test_batch, X_b_test_batch, X_l_test_batch, Y_test_batch = data.load_X_Y_voxel_grids_test_next_batch()

                    iou_t, aeu_loss_t, gan_g_loss_t, gan_d_loss_no_gp_t, gan_d_loss_gp_t, Y_pred_t, sum_test = self.sess.run(
                        [self.IoU, self.aeu_loss, self.gan_g_loss, self.gan_d_loss_no_gp, self.gan_d_loss_gp, self.Y_pred, self.sum_merged],
                        feed_dict={self.X_f_bool:X_f_test_batch, self.X_r_bool:X_r_test_batch, self.X_b_bool:X_b_test_batch, self.X_l_bool:X_l_test_batch, self.Y_bool:Y_test_batch})
                    
                    if i%2000==0:
                        X_f_test_batch=X_f_test_batch.astype(np.int8)
                        X_r_test_batch=X_r_test_batch.astype(np.int8)
                        X_b_test_batch=X_b_test_batch.astype(np.int8)
                        X_l_test_batch=X_l_test_batch.astype(np.int8)
                        Y_pred_t=Y_pred_t.astype(np.float16)
                        Y_test_batch=Y_test_batch.astype(np.int8)
                        to_save = {'X_f_test':X_f_test_batch, 'X_r_test':X_r_test_batch, 'X_b_test':X_b_test_batch, 'X_l_test':X_l_test_batch, 'Y_test_pred':Y_pred_t, 'Y_test_true':Y_test_batch}

                        scipy.io.savemat(self.test_res_dir+'X_Y_pred_'+str(epoch).zfill(2)+'_'+str(i).zfill(5)+'.mat',
                        to_save, do_compression=True)

                    self.sum_write_test.add_summary(sum_test, epoch*total_train_batch_num+i)
                    print ('ep:',epoch, 'i:', i, "test IoU:", iou_t, 'test aeu loss:', aeu_loss_t,'gan g loss:', gan_g_loss_t,
                           'gan d loss no gp:',gan_d_loss_no_gp_t,'gan d loss gp:',gan_d_loss_gp_t)

            #### model saving
            if epoch%1 == 0:
                self.saver.save(self.sess, save_path=self.train_mod_dir + 'model_{}.cptk'.format(epoch))
                print ('ep:', epoch, 'model saved!')

        data.stop_queue=True
