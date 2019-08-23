import os
import time
import inspect #The built-in lib of Python, inspecting the live objects 
import numpy as np
import tensorflow as tf
import struct
import pandas as pd
import model_maker
class BackPropCnnNetwork(object):
    def __init__(self, features, labels, model_fn, batch_size,
                 clip=0,forward_fc_filters=(5, 10, 15),tconv_Fnums=(4,4), tconv_dims=(60, 120, 240), 
                 tconv_filters=(1, 1, 1),n_filter=5, n_branch=3,
                 reg_scale=.001, learn_rate=1e-4, decay_step=200, decay_rate=0.1,
                 ckpt_dir=os.path.join(os.path.abspath(''), 'models'),
                 make_folder=True, geoboundary = [30, 55, 42, 52]):
                 
        """
        Initialize a Network class
        :param features: input features
        :param labels: input labels
        :param model_fn: model definition function, can be customized by user
        :param batch_size: batch size
        :param fc_filters: #neurons in each fully connected layers
        :param tconv_dims: dimensionality of data after each transpose convolution
        :param tconv_filters: #filters at each transpose convolution
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
        self.clip = clip
        assert len(tconv_dims) == len(tconv_filters)
        assert len(tconv_Fnums) == len(tconv_filters)
        self.tconv_Fnums = tconv_Fnums
        self.tconv_dims = tconv_dims
        self.tconv_filters = tconv_filters
        self.best_validation_loss = float("inf")
        self.n_filter = n_filter
        self.n_branch = n_branch
        self.forward_fc_filters = forward_fc_filters
        self.reg_scale = reg_scale
        self.geoboundary = geoboundary
        self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name='global_step')
        self.learn_rate = tf.train.exponential_decay(learn_rate, self.global_step,
                                                     decay_step, decay_rate, staircase=True)

        self.ckpt_dir = os.path.join(ckpt_dir, time.strftime('%Y%m%d_%H%M%S', time.localtime()))
        if not os.path.exists(self.ckpt_dir) and make_folder:
            os.makedirs(self.ckpt_dir)
            self.write_record()

        self.forward_in, self.logits, self.merged_summary_op, self.geometry_variable,\
                                      self.train_Forward, self.Boundary_loss  = self.create_graph()
        #self.model = tf.keras.Model(self.features, self.logits,name = 'Backward')
        if self.labels==[]:
            print('labels list is empty')
        else:
            self.loss, self.mse_loss, self.reg_loss, self.bdy_loss = self.make_loss()
            self.optm = self.make_optimizer()
            self.backprop_optm = self.make_backprop_optimizer()
  
    def create_graph(self):
        """
        Create model graph
        :return: outputs of the last layer
        """
        return self.model_fn(self.features,  self.batch_size, 
                             self.clip, self.forward_fc_filters, self.tconv_Fnums,
                             self.tconv_dims, self.tconv_filters,
                             self.n_filter, self.n_branch, self.reg_scale, self.geoboundary)
     

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
            mse_loss = tf.losses.mean_squared_error(self.labels, self.logits) #reconstruction loss
            reg_loss = tf.losses.get_regularization_loss()      #regularizaiton loss
            bdy_loss = self.Boundary_loss                       #boundary loss
            total_loss = mse_loss + reg_loss + bdy_loss         #Total loss
            return total_loss, mse_loss, reg_loss, bdy_loss
            
    def make_optimizer(self):
        """
        Make an Adam optimizer with the learning rate defined when the class is initialized
        :return: an AdamOptimizer
        """
        return tf.train.AdamOptimizer(learning_rate=self.learn_rate).minimize(self.loss, self.global_step)
    
    def make_backprop_optimizer(self):
        """
        Make an Backproping optimizer with the learning rate defined when the class is initialized
        :return: an AdamOptimizer
        """
        return tf.train.AdamOptimizer(learning_rate=self.learn_rate * 5000).minimize(self.loss, 
                                                                              self.global_step,
                                                                              var_list = [self.geometry_variable])

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

    def train(self, train_init_op, step_num, forward_hooks,\
              write_summary=False,load_forward_ckpt = None):
        """
        Train the model with step_num steps
        First train the forward model and then the tandem part
        :param train_init_op: training dataset init operation
        :param step_num: number of steps to train
        :param hooks: hooks for monitoring the training process
        :param write_summary: write summary into tensorboard or not
        :return:
        """
        with tf.Session() as sess:
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            if load_forward_ckpt != None:
              self.load(sess, load_forward_ckpt)
            if write_summary:
                summary_writer = tf.summary.FileWriter(self.ckpt_dir, sess.graph)
            else:
                summary_writer = None
            
            print("Training forward model now:")
            
            assign_true_op = self.train_Forward.assign(True)
            sess.run([train_init_op, assign_true_op])
            
            ##Train the forward model
            for i in range(int(step_num)):
                sess.run([train_init_op, assign_true_op])
                [feature, optm_out] = sess.run([self.features ,self.optm])
                if (i % 500 == 0):
                    print("Feature now is:", feature[0,:])
                for hook in forward_hooks:
                    hook.run(sess, writer=summary_writer)
                if forward_hooks[-1].save:                       #If the hook tells to save the model, then save it
                    self.save(sess)
                    self.best_validation_loss = forward_hooks[-1].best_validation_loss
                if forward_hooks[-1].stop:
                    break
    
    def evaluate_one(self, target_spectra, back_prop_epoch, sess, verb_step, stop_thres, point_index):
        """
        The function that evaluate one single given target spectra and return the results
        :param target_spectra: The target spectra to back prop towards. Should be only 1 row
        :param back_prop_epoch: #epochs to do the gradient descend
        :param sess: The current session to do the back prop
        """

        #Set up target output
        print("shape before repeat",np.shape(target_spectra.values))
        target_spectra_repeat = np.repeat(np.reshape(target_spectra.values,(1,-1)), self.batch_size, axis = 0)
        print("Size of the target spectra repeat", np.shape(target_spectra_repeat))
        #target_spectra_dataset = tf.data.Dataset.from_tensor_slices(target_spectra_repeat)
        #target_spectra_dataset = target_spectra_dataset.repeat()
        for i in range(back_prop_epoch):
            loss_back_prop, optm_out, inferred_spectra = sess.run([self.loss, self.backprop_optm, self.logits], 
                                                                feed_dict={self.labels: target_spectra_repeat})  
            if (i % verb_step == 0):
                print("Loss at inference step{} : {}".format(i,loss_back_prop))
                if (loss_back_prop < stop_thres):
                    print("Loss is lower than the threshold{}, inference stop".format(stop_thres))
                    break
        #Then it is time to get the best performing one
        Xpred, Ypred, loss = sess.run([self.forward_in, self.logits, self.loss], feed_dict={self.labels: target_spectra_repeat})
        loss_list = np.sum(np.square(Ypred - target_spectra_repeat), axis = 1) / self.batch_size
        best_estimate_index = np.argmin(loss_list)
        print('best error is {}, in best estimate-indx is {}, squared loss is {}'.format(min(loss_list), 
                                                                                              loss_list[best_estimate_index],
                                                                                              loss))
        print('Best error for point {} is having absolute error of {}'.format(point_index, loss_list[best_estimate_index]))
        Xpred_best = Xpred[best_estimate_index,:]
        Ypred_best = Ypred[best_estimate_index,:]
        return Xpred_best, Ypred_best
        

    def evaluate(self, valid_init_op, train_init_op, ckpt_dir,verb_step = 500, 
                back_prop_epoch = 10000, stop_thres = 1e-3,
                 save_file=os.path.join(os.path.abspath(''), 'data'),
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
            
            sess.run([valid_init_op, assign_eval_forward_op])
            pred_file = os.path.join(save_file, 'test_Ypred_{}.csv'.format(model_name))
            feature_file = os.path.join(save_file, 'test_Xtruth_{}.csv'.format(model_name))
            truth_file = os.path.join(save_file, 'test_Ytruth_{}.csv'.format(model_name))
            feat_file = os.path.join(save_file, 'test_Xpred_{}.csv'.format(model_name))
            
            eval_cnt = 0
            start_pred = time.time()
            print("Train_bool:",sess.run(self.train_Forward))
            try:
                while True:
                    with open(feature_file, 'a') as f0, open(truth_file, 'a') as f2: 
                        Xtruth, Ytruth = sess.run([self.features, self.labels])
                        np.savetxt(f0, Xtruth, fmt='%.3f')
                        np.savetxt(f2, Ytruth, fmt='%.3f')
            except tf.errors.OutOfRangeError:
                Ytruth = pd.read_csv(truth_file,header= None, delimiter= ' ')
                h ,w = Ytruth.values.shape
                print(h)
            
            #inference time
            with open(feat_file, 'a') as f1, open(pred_file, 'a') as f3: 
                #First initialize the starting points
                RN = model_maker.initializeInBoundary(self.geometry_variable.shape, self.geoboundary)               
                print("Random number within range", self.geoboundary)
                assign_var_op = self.geometry_variable.assign(RN) #Assign the variable
                sess.run([assign_var_op, train_init_op])
                for i in range(h):
                    Xpred, Ypred = self.evaluate_one(Ytruth.iloc[i,:], back_prop_epoch, sess, verb_step, stop_thres, i)
                    np.savetxt(f1, Xpred, fmt='%.3f')
                    np.savetxt(f3, Ypred, fmt='%.3f')

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
