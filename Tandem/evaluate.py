import argparse
import tensorflow as tf
import data_reader
import model_maker
import Tandem_network_maker
import network_helper
import plotsAnalysis
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import train
import flag_reader
import time_recorder
def compare_truth_pred(pred_file, truth_file):
    """
    Read truth and pred from csv files, compute their mean-absolute-error and the mean-squared-error
    :param pred_file: full path to pred file
    :param truth_file: full path to truth file
    :return: mae and mse
    """
    pred = np.loadtxt(pred_file, delimiter=' ')
    truth = np.loadtxt(truth_file, delimiter=' ')

    mae = np.mean(np.abs(pred-truth), axis=1)
    mse = np.mean(np.square(pred-truth), axis=1)

    return mae, mse
def evaluatemain(flags, eval_forward, test_ratio,plot_histo = True):
    #Clear the default graph first for resolving potential name conflicts
    print("Start Evaluating now...")
    TK = time_recorder.time_keeper(time_keeping_file = "data/time_keeper.txt")

    tf.reset_default_graph()
    
    ckpt_dir = os.path.join(os.path.abspath(''), 'models', flags.model_name)
    clip, forward_fc_filters, tconv_Fnums, tconv_dims, tconv_filters,  n_filter, n_branch, \
    reg_scale, backward_fc_filters, conv1d_filters, conv_channel_list, batch_size = network_helper.get_parameters(ckpt_dir)
    print(ckpt_dir)
    # initialize data reader
    if len(tconv_dims) == 0:
        output_size = fc_filters[-1]
    else:
        output_size = tconv_dims[-1]
    features, labels, train_init_op, valid_init_op = data_reader.read_data(input_size=flags.input_size,
                                                               output_size=output_size-2*clip,
                                                               x_range=flags.x_range,
                                                               y_range=flags.y_range,
							        geoboundary=flags.geoboundary,
                                                               cross_val=flags.cross_val,
                                                               val_fold=flags.val_fold,
                                                               batch_size=batch_size,
                                                               shuffle_size=flags.shuffle_size,
								data_dir = flags.data_dir,
							        normalize_input = flags.normalize_input,
                                                                test_ratio = test_ratio) #Test ratio < 0 means manual pick of test set

    #if the input is normalized
    if flags.normalize_input:
		    flags.boundary = [-1, 1, -1, 1]
		
		# make network
    ntwk = Tandem_network_maker.TandemCnnNetwork(features, labels, model_maker.tandem_model, batch_size,
                                clip=clip, forward_fc_filters=forward_fc_filters,
                                backward_fc_filters=backward_fc_filters,reg_scale=reg_scale,
	                        learn_rate=flags.learn_rate,tconv_Fnums=tconv_Fnums,
				tconv_dims=tconv_dims,n_branch=n_branch,
			        tconv_filters=tconv_filters, n_filter=n_filter,
				decay_step=flags.decay_step, decay_rate=flags.decay_rate, geoboundary = geoboundary,
                                conv1d_filters = conv1d_filters, conv_channel_list = conv_channel_list)
    # evaluate the results if the results do not exist or user force to re-run evaluation
    save_file = os.path.join(os.path.abspath(''), 'data', 'test_pred_{}.csv'.format(flags.model_name))
    if flags.force_run or (not os.path.exists(save_file)):
        print('Evaluating the model ...')
        pred_file, truth_file = ntwk.evaluate(valid_init_op, ckpt_dir=ckpt_dir,
                                              model_name=flags.model_name, write_summary=True,
                                              eval_forward = eval_forward)
    else:
        pred_file = save_file
        truth_file = os.path.join(os.path.abspath(''), 'data', 'test_truth.csv')
    
    TK.record(write_number = int(22000 * test_ratio))
    mae, mse = compare_truth_pred(pred_file, truth_file)
    
    if (plot_histo):
        plt.figure(figsize=(12, 6))
        plt.hist(mse, bins=100)
        plt.xlabel('Mean Squared Error')
        plt.ylabel('cnt')
        plt.suptitle('Tandem (Avg MSE={:.4e})'.format(np.mean(mse)))
        plt.savefig(os.path.join(os.path.abspath(''), 'data',
                             'tandem_{}.png'.format(flags.model_name)))
        plt.show()
    print('Tandem (Avg MSE={:.4e})'.format(np.mean(mse)))
    
def evaluate_with_ratio(test_ratio):
    flags = flag_reader.read_flag()
    evaluatemain(flags, eval_forward = False, test_ratio = test_ratio,plot_histo = False)
    plotsAnalysis.SpectrumComparisonNGeometryComparison(3,2, (13,8), flags.model_name,flags.boundary)	
    

if __name__ == '__main__':
    flags = flag_reader.read_flag()
    evaluatemain(flags, eval_forward = False, test_ratio = 0.9999)
    plotsAnalysis.SpectrumComparisonNGeometryComparison(3,2, (13,8), flags.model_name,flags.boundary)	



