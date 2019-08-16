import os
import time
import inspect #The built-in lib of Python, inspecting the live objects 
import numpy as np
import tensorflow as tf
import struct


class VAENetwork(object):
    def __init__(self, features, labels, model_fn, batch_size, latent_dim, 
                 spectra_fc_filters=(5, 10, 15), decoder_fc_filters=(5,10,15),
                 encoder_fc_filters=(5, 10, 15), reg_scale=.001, learn_rate=1e-4, decay_step=200, decay_rate=0.1,
                 ckpt_dir=os.path.join(os.path.abspath(''), 'models'), make_folder=True, geoboundary = [-1 , 1, -1, 1]):
        """
        Initialize a Network class
        :param features: input features
        :param labels: input labels
        :param model_fn: model definition function, can be customized by user
        :param batch_size: batch size
        :param XXX_fc_filters: #neurons in each fully connected layers in module XXX
        :param learn_rate: learning rate
        :param decay_step: decay learning rate at this number of steps
        :param decay_rate: decay learn rate by multiplying this factor
        :param ckpt_dir: checkpoint directory, default to ./models
        :param make_folder: if True, create the directory if not exists
        """
        self.features = features
        self.labels = labels
        self.model_fn = model_fn
        self.batch_size = batch_size
        #self.clip = clip
        self.spectra_fc_filters = spectra_fc_filters
        #assert len(tconv_dims) == len(tconv_filters)
        #assert len(tconv_Fnums) == len(tconv_filters)
        #self.tconv_Fnums = tconv_Fnums
        #self.tconv_dims = tconv_dims
        #self.tconv_filters = tconv_filters
        #self.n_filter = n_filter
        #self.n_branch = n_branch
        self.encoder_fc_filters = encoder_fc_filters
        self.decoder_fc_filters = decoder_fc_filters
        self.reg_scale = reg_scale
        self.latent_dim = latent_dim
        self.boundary = geoboundary
        
        self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name='global_step')
        self.learn_rate = tf.train.exponential_decay(learn_rate, self.global_step,
                                                     decay_step, decay_rate, staircase=True)

        self.ckpt_dir = os.path.join(ckpt_dir, time.strftime('%Y%m%d_%H%M%S', time.localtime()))
        if not os.path.exists(self.ckpt_dir) and make_folder:
            os.makedirs(self.ckpt_dir)
            self.write_record()

        self.logits, self.z_mean, self.z_log_var, self.boundary_loss = self.create_graph()
        #self.model = tf.keras.Model(self.features, self.logits,name = 'Backward')
        if self.labels==[]:
            print('labels list is empty')
        else:
            self.loss = self.make_loss()
            self.optm = self.make_optimizer()
  
    def create_graph(self):
        """
        Create model graph
        :return: outputs of the last layer
        """
        return self.model_fn(self.features,self.labels, self.latent_dim, self.batch_size, self.reg_scale,
                                self.spectra_fc_filters, self.encoder_fc_filters, self.decoder_fc_filters, self.boundary)
     

    def write_record(self):
        """
        Write records, including model_fn, parameters into the checkpoint folder
        These records can be used to reconstruct & repeat experiments
        :return:
        """
        #insepect.getsource = return the text of the source code for an object
        model_fn_str = inspect.getsource(self.model_fn)  #Get the text of the source code of the object
        params = inspect.getmembers(self, lambda a: not inspect.isroutine(a)) #get all the members that are not a routine (function)
        params = [a for a in params if not (a[0].startswith('__') and a[0].endswith('__'))]
        with open(os.path.join(self.ckpt_dir, 'model_meta.txt'), 'w+') as f:
            f.write('model_fn:\n')
            f.writelines(model_fn_str)
            f.write('\nparams:\n')
            for key, val in params:
                f.write('{}: {}\n'.format(key, val))

    def make_loss(self):
        """
        Make cross entropy loss for forward part of the model
        :return: mean cross entropy loss of the batch
        """
        with tf.variable_scope('loss'):
            loss = tf.losses.mean_squared_error(self.labels, self.logits)
            kl_loss = 1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var)
            kl_loss = K.sum(kl_loss, axis = -1)
            kl_loss *= -0.5
            loss += kl_loss
            loss += tf.losses.get_regularization_loss()
            #loss += self.Boundary_loss
            return loss
            
    def make_optimizer(self):
        """
        Make an Adam optimizer with the learning rate defined when the class is initialized
        :return: an AdamOptimizer
        """
        return tf.train.AdamOptimizer(learning_rate=self.learn_rate).minimize(self.loss, self.global_step)
    
    def save(self, sess):
        """
        Save the model to the checkpoint directory
        :param sess: current running session
        :return:
        """
        saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)
        saver.save(sess, os.path.join(self.ckpt_dir, 'model.ckpt'))

    def load(self, sess, ckpt_dir):
        """
        Load the model from the checkpoint directory
        :param sess: current running session
        :param ckpt_dir: checkpoint directory
        :return:
        """
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        saver = tf.train.Saver(var_list=tf.global_variables())
        latest_check_point = tf.train.latest_checkpoint(ckpt_dir)
        saver.restore(sess, latest_check_point)
        print('loaded {}'.format(latest_check_point))

    def train(self, train_init_op, step_num,backward_step_num, forward_hooks, 
              write_summary=False,load_forward_ckpt = None):
        """
        Train the model with step_num steps
        First train the forward model and then the tandem part
        :param train_init_op: training dataset init operation
        :param step_num: number of steps to train
        :param hooks: hooks for monitoring the training process !!!ALWASYS PUT VALIDATION HOOK THE LAST ONE
        :param write_summary: write summary into tensorboard or not
        :return:
        """
        with tf.Session() as sess:
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            if write_summary:
                summary_writer = tf.summary.FileWriter(self.ckpt_dir, sess.graph)
            else:
                summary_writer = None
            
            print("Training forward model now:")
            
            #assign_true_op = self.train_Forward.assign(True)
            
            ##Train the forward model
            for i in range(int(step_num)):
                sess.run([train_init_op])#, assign_true_op])
                sess.run(self.optm)
                for hook in forward_hooks:
                    hook.run(sess, writer=summary_writer)
                if forward_hooks[-1].stop:     #if it either trains to the threshold or have NAN value, stop here
                    break
            self.save(sess)

    def evaluate(self, valid_init_op, ckpt_dir, save_file=os.path.join(os.path.abspath(''), 'data'),
                 model_name='', write_summary=False, eval_forward = False):
        """
        Evaluate the model, and save predictions to save_file
        :param valid_init_op: validation dataset init operation
        :param checkpoint directory
        :param save_file: full path to pred file
        :param model_name: name of the model
        :param eval_forward
        :return:
        """
        assign_eval_forward_op = self.train_Forward.assign(eval_forward) #Change the graph accordingly
        
        with tf.Session() as sess:
            self.load(sess, ckpt_dir)

            if write_summary:
                writer_path = os.path.join(ckpt_dir, 'evalSummary')
                print("summary_writer directory is {}".format(writer_path))
                activation_summary_writer = tf.summary.FileWriter(writer_path, sess.graph)
            else:
                activation_summary_writer = None
            
            sess.run([valid_init_op,assign_eval_forward_op])
            pred_file = os.path.join(save_file, 'test_Ypred_{}.csv'.format(model_name))
            feature_file = os.path.join(save_file, 'test_Xtruth_{}.csv'.format(model_name))
            truth_file = os.path.join(save_file, 'test_Ytruth_{}.csv'.format(model_name))
            feat_file = os.path.join(save_file, 'test_Xpred_{}.csv'.format(model_name))
            
            eval_cnt = 0
            start_pred = time.time()
            print("Backward_out:",sess.run(self.backward_out)[0,:])
            print("Forward_in",sess.run(self.forward_in)[0,:])
            print("Feature:",sess.run(self.features)[0,:])
            print("Train_bool:",sess.run(self.train_Forward))
            try:
                while True:
                    with open(feature_file, 'a') as f0, open(pred_file, 'a') as f1,\
                          open(truth_file, 'a') as f2, open(feat_file, 'a') as f3:
                        Xtruth, Ypred, Ytruth, Xpred, summary = sess.run([self.features,
                                                                         self.logits,
                                                                         self.labels,
                                                                         self.forward_in,
                                                                         self.merged_summary_op])
                        np.savetxt(f0, Xtruth, fmt='%.3f')
                        np.savetxt(f1, Ypred, fmt='%.3f')
                        np.savetxt(f2, Ytruth, fmt='%.3f')
                        np.savetxt(f3, Xpred, fmt='%.3f')
                        if write_summary:
                            activation_summary_writer.add_summary(summary, eval_cnt)
                    eval_cnt += 1
            except tf.errors.OutOfRangeError:
                print("evaluation took {} seconds".format(time.time() - start_pred))
                activation_summary_writer.flush()
                activation_summary_writer.close()
                
                print("Xpred",Xpred[0,:])
                print("Xtruth",Xtruth[0,:])
                print("Ypred",Ypred[0,:])
                print("Ytruth",Ytruth[0,:])
                
                return pred_file, truth_file
    """
    def predict(self, pred_init_op, ckpt_dir, save_file=os.path.join(os.path.abspath(''), 'dataGrid'),
                model_name=''):
        """"""
        Evaluate the model, and save predictions to save_file
        :param ckpt_dir directory
        :param save_file: full path to pred file
        :param model_name: name of the model
        :return:
        """"""
        with tf.Session() as sess:
            self.load(sess, ckpt_dir)
            sess.run(pred_init_op)
            pred_file = os.path.join(save_file, 'test_pred_{}.csv'.format(model_name))
            feat_file = os.path.join(save_file, 'test_feat_{}'.format(model_name) + '.csv')
            with open(pred_file, 'w'):
                pass
            try:
                start = time.time()
                cnt = 1
                while True:
                    with open(pred_file, 'a') as f1: #, open(feat_file, 'a') as f2
                        pred_batch, features_batch = sess.run([self.logits, self.features])
                        for pred, features in zip(pred_batch, features_batch):
                            pred_str = [str(el) for el in pred]
                            features_str = [ str(el) for el in features]
                            f1.write(','.join(pred_str)+'\n')
                            # f2.write(','.join(features_str)+'\n')
                    if (cnt % 100) == 0:
                        print('cnt is {}, time elapsed is {}, features are {} '.format(cnt,
                                                                                       np.round(time.time()-start),
                                                                                       features_batch))
                    cnt += 1
            except tf.errors.OutOfRangeError:
                return pred_file, feat_file
                pass
    """
