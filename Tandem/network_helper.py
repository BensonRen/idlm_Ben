import os
import time
import tfplot
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('agg') 		#To make it silent and dont output image and thus cause error
import matplotlib.pyplot as plt
import math
class Hook(object):
    """
    Parent class of all hooks
    Hooks are used in network.train() to monitor training progress
    """
    def __init__(self):
        self.step = -1

    def run(self, sess, writer=None):
        raise NotImplementedError

class TrainValueHook(Hook):
    """
    This hook monitors performance on the training set
    """
    def __init__(self, verb_step, loss, ckpt_dir=None, value_name='mean_squared_error', write_summary=False,
                 verb=True):
        """
        Initialize the hook
        :param verb_step: # steps between every print message
        :param loss: value to log at every verbose step
        :param ckpt_dir: checkpoint directory, only use it if write_summary is True
        :param value_name: name of this summary in tensorboard
        :param write_summary: log summary or not
        :param verb: if True, print out message every verb_step
        """
        super(TrainValueHook, self).__init__()
        self.verb_step = verb_step
        self.loss = loss
        self.write_summary = write_summary
        self.name = value_name
        if self.write_summary:
            assert ckpt_dir is not None
            self.train_mse_summary = HookValueSummary(value_name)
        self.verb = verb

    def run(self, sess, writer=None):
        """
        Run the hook at each step
        :param sess: current session
        :param writer: summary writer used to write variables into tensorboard, default to None
        :return:
        """
        self.step += 1
        if self.step % self.verb_step == 0:
            loss_val = sess.run(self.loss)
            if self.verb:
                print('Step {},{}  loss: {:.2E}'.format(self.step, self.name, loss_val))
            if self.write_summary:
                self.train_mse_summary.log(loss_val, self.step, sess, writer)

class SummaryWritingHook(Hook):
    """
    This is a hook for writing summary periodically into the tensorboard
    """
    def __init__(self, summary_op, write_step = 3000):
        """
        :param write_step: The #steps to write to summary
        :param summary_op: The summary operation to run
        :param writer: write to 
        """
        super(SummaryWritingHook, self).__init__()
        self.write_step = write_step
        self.summary_op = summary_op
    def run(self, sess, writer):
        self.step +=1
        if (self.step % self.write_step == 0):
            summary = sess.run(self.summary_op)
            writer.add_summary(summary, self.step)
            writer.flush()
            print("writing histograms")


class ValidationHook(Hook):
    """
    This hook monitors performance on the validation set
    """
    def __init__(self, valid_step, valid_init_op, truth, pred, loss, stop_threshold, value_name = 'valid_mse', ckpt_dir=None, 
                 write_summary=False, curve_num=6, low_validation_loss = 0.03):
        """
        Initialize the hook
        :param valid_step: # steps between evaluations
        :param valid_init_op: validation dataset init operation
        :param truth: truth
        :param pred: prediction
        :param loss: loss to log at every eval_step
        :param ckpt_dir: ckpt_dir: checkpoint directory, only use it if write_summary is True
        :param write_summary: log summary or not
        :param curve_num: #curve plots in validation images
	:param stop_threshold: The Loss threshold to stop the algorithm
	:param stop: Boolean value to stop the training
        """
        super(ValidationHook, self).__init__()
        self.valid_step = valid_step
        self.valid_init_op = valid_init_op
        self.truth = truth
        self.pred = pred
        self.loss = loss
        self.write_summary = write_summary
        self.curve_num = curve_num
        self.stop_threshold = stop_threshold
        self.best_validation_loss = low_validation_loss #initialize the best validation to a low validation loss
        self.stop = False
        self.save = False               #Whether save the session in this epoch
        if self.write_summary:
            assert ckpt_dir is not None
            self.valid_mse_summary = HookValueSummary(value_name)
            #self.valid_curve_summary = HookCurvePlotSummary('pred_plot')
            #self.valid_preconv_summary = HookCurvePlotSummary('preconv_plot')
            #self.valid_preTconv_summary = HookCurvePlotSummary('preTconv_plot')
        self.time_cnt = time.time()

    def run(self, sess, writer=None):
        """
        Run the hook at each step
        :param sess: current session
        :param writer: summary writer used to write variables into tensorboard, default to None
        :return:
        """
        self.step += 1
        self.save = False
        if self.step % self.valid_step == 0 and self.step != 0:
            sess.run(self.valid_init_op)
            loss_val = []
            truth = None
            try:
                while True:
                    loss, truth, pred= sess.run([self.loss,self.truth,self.pred,])
                    loss_val.append(loss)
            except tf.errors.OutOfRangeError:
                pass
            loss_mean = np.mean(loss_val)
            print('Eval @ Step {}, loss: {:.2E}, duration {:.3f}s'.\
                  format(self.step, loss_mean, time.time()-self.time_cnt))
            if math.isnan(loss_mean):   #If the loss is NAN, then Stop
                print("The validation loss is NAN, please adjust (Lower) your learning rate and retrain. Aborting now")
                self.stop  = True
            else:                       #If the loss is not NAN
                if loss_mean < self.stop_threshold:
                    print('Validation loss is lower than threshold{}, training is stopped'.format(self.stop_threshold))
                    self.stop = True
                if loss_mean < self.best_validation_loss:       #If the loss is smaller than the best, then save this one now
                    self.best_validation_loss = loss_mean
                    self.save = True
            self.time_cnt = time.time()
            if self.write_summary:
                self.valid_mse_summary.log(loss_mean, self.step, sess, writer)
                #self.valid_curve_summary.log(pred=pred,
                #                             step=self.step,
                #                             writer=writer,
                #                             curve_num=self.curve_num,
                #                             truth=truth)
               # self.valid_preconv_summary.log(pred=preconv,
               #                                step=self.step,
               #    writer=writer,
               #                                curve_num=self.curve_num)
               # self.valid_preTconv_summary.log(pred=preTconv,
               #                                step=self.step,
               #                                writer=writer,
               #                                curve_num=self.curve_num)


class HookValueSummary(object):
    """
    Write summary inside hooks
    """
    def __init__(self, summary_name, d_type=tf.float32):
        """
        Initialize the summaries
        :param summary_name: name of this summary
        :param d_type: data type to write into the summary
        """
        self.val = tf.placeholder(d_type, [])
        self.val_summary_op = tf.summary.scalar(summary_name, self.val)

    def log(self, val, step, sess, writer):
        """
        log the value into summary
        :param val: value to log
        :param step: step num
        :param sess: current session
        :param writer: summary writer used to write variables into tensorboard, default to None
        :return:
        """
        summary = sess.run(self.val_summary_op, feed_dict={self.val: val})
        writer.add_summary(summary, step)
        writer.flush()


class HookCurvePlotSummary(object):
    """
    Write summary inside hooks
    """
    def __init__(self, summary_name):
        """
        Initialize the summaries
        :param summary_name: name of this summary
        :param d_type: data type to write into the summary
        """
        self.summary_name = summary_name

    def log(self, pred, step, writer, curve_num, truth=None):
        """
        log the value into summary
        :param val: value to log
        :param step: step num
        :param writer: summary writer used to write variables into tensorboard, default to None
        :param curve_num: #curve plots in validation images
        :return:
        """
        fig_num = pred.shape[0]
        fig_idx = np.random.permutation(fig_num)
		
        fig = plt.figure(figsize=(14, 6))
        for i in range(curve_num):
            ax = fig.add_subplot(2, 3, i+1)
            if truth is not None:
                ax.plot(truth[fig_idx[i], :], label='truth')
            ax.plot(pred[fig_idx[i], :], label='pred', alpha=0.7, linewidth=1)
            ax.legend()
            if truth is not None:
                mse = np.mean(np.square(truth[fig_idx[i], :] - pred[fig_idx[i], :]))
                plt.title('Step {}, MSE={:.3f}'.format(step, mse))
            else:
                plt.title('Step {}'.format(step))
            fig.tight_layout()

        summary = tfplot.figure.to_summary(fig, tag=self.summary_name)
        writer.add_summary(summary, step)
        writer.flush()
        plt.close(fig)


def get_parameters(model_dir):
    def replace_str(s):
        for char in [',', '(', ')', '[', ']']:
            s = s.replace(char, ' ')
        return s

    file = os.path.join(model_dir, 'model_meta.txt')
    with open(file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        #print(line)
        if line[:4] =='clip':
            clip = [int(s) for s in line.split() if s.isdigit()]
        if line[:10] =='batch_size':
            batch_size = [int(s) for s in line.split() if s.isdigit()]
            print("Batch_size read is:",batch_size)
        elif line[:18] == 'forward_fc_filters':
            line = replace_str(line)
            forward_fc_filters = tuple([int(s) for s in line.split() if s.isdigit()])
        elif line[:19] == 'backward_fc_filters':
            line = replace_str(line)
            backward_fc_filters = tuple([int(s) for s in line.split() if s.isdigit()])
        elif line[:14] == 'conv1d_filters':
            line = replace_str(line)
            conv1d_filters =  tuple([int(s) for s in line.split() if s.isdigit()])
        elif line[:17] == 'conv_channel_list':
            line = replace_str(line)
            conv_channel_list = tuple([int(s) for s in line.split() if s.isdigit()])
        
        elif line[:11] == 'tconv_Fnums':
            line =replace_str(line)
            tconv_Fnums = tuple([int(s) for s in line.split() if s.isdigit()])
        elif line[:10] == 'tconv_dims':
            line = replace_str(line)
            tconv_dims = tuple([int(s) for s in line.split() if s.isdigit()])
        elif line[:13] == 'tconv_filters':
            line = replace_str(line)
            tconv_filters = tuple([int(s) for s in line.split() if s.isdigit()])
        elif line[:8] =='n_filter':
            line = replace_str(line)
            n_filter = [int(s) for s in line.split() if s.isdigit()]
        elif line[:8] =='n_branch':
            line = replace_str(line)
            n_branch = [int(s) for s in line.split() if s.isdigit()]
        elif line[:9] =='reg_scale':
            line = replace_str(line)
            reg_scale = float(line[11:])
    return clip[0], forward_fc_filters,  tconv_Fnums, tconv_dims, tconv_filters, n_filter, n_branch[0], reg_scale, backward_fc_filters, conv1d_filters, conv_channel_list, batch_size[0]
