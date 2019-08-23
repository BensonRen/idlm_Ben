import argparse
import tensorflow as tf
import data_reader
import model_maker
import VAE_network_maker
import network_helper
import plotsAnalysis
import os
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import train

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
def evaluatemain(flags, eval_forward):
    #Clear the default graph first for resolving potential name conflicts
    tf.reset_default_graph()
    
    ckpt_dir = os.path.join(os.path.abspath(''), 'models', flags.model_name)
    clip, forward_fc_filters, tconv_Fnums, tconv_dims, tconv_filters, \
    n_filter, n_branch, reg_scale = network_helper.get_parameters(ckpt_dir)
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
                                                               batch_size=flags.batch_size,
                                                               shuffle_size=flags.shuffle_size,
								data_dir = flags.data_dir,
							        normalize_input = flags.normalize_input,
                                                                test_ratio = -1)
    #if the input is normalized
    if flags.normalize_input:
		    flags.boundary = [-1, 1, -1, 1]
	
    # make network
    ntwk = VAE_network_maker.VAENetwork(geometry, spectra, model_maker.VAE, flags.batch_size, flags.latent_dim,
                            spectra_fc_filters=flags.spectra_fc_filters, decoder_fc_filters=flags.decoder_fc_filters,
                            encoder_fc_filters=flags.encoder_fc_filters,reg_scale=flags.reg_scale,
                            learn_rate=flags.learn_rate, decay_step=flags.decay_step, decay_rate=flags.decay_rate,
                            geoboundary = flags.geoboundary)
    
    # evaluate the results if the results do not exist or user force to re-run evaluation
    save_file = os.path.join(os.path.abspath(''), 'data', 'test_pred_{}.csv'.format(flags.model_name))
    
    if flags.force_run or (not os.path.exists(save_file)):
        print('Evaluating the model ...')
        #pred_file, truth_file = ntwk.evaluate(valid_init_op, ckpt_dir=ckpt_dir,
        Xpred_file = ntwk.evaluate(valid_init_op, ckpt_dir=ckpt_dir, model_name=flags.model_name, write_summary=True,
                                                                        eval_forward = eval_forward)
        pred_file, truth_file = get_spectra_from_geometry(Xpred_file)
    else:
        pred_file = save_file
        truth_file = os.path.join(os.path.abspath(''), 'data', 'test_truth.csv')

    mae, mse = compare_truth_pred(pred_file, truth_file)

    plt.figure(figsize=(12, 6))
    plt.hist(mse, bins=100)
    plt.xlabel('Mean Squared Error')
    plt.ylabel('cnt')
    plt.suptitle('VAE (Avg MSE={:.4e})'.format(np.mean(mse)))
    plt.savefig(os.path.join(os.path.abspath(''), 'data',
                             'VAE_{}.png'.format(flags.model_name)))
    plt.show()
    print('VAE (Avg MSE={:.4e})'.format(np.mean(mse)))

if __name__ == '__main__':
	flags = train.read_flag()
	evaluatemain(flags, eval_forward = False)
	plotsAnalysis.SpectrumComparisonNGeometryComparison(3,2, (13,8), flags.model_name,flags.boundary)	

def get_spectra_from_geometry(Xpred_file):
    """
    This function is to call the prediction function for the forward model and returns the Yprediction file
    :param Xpred_file: The prediction X given that used to infer Y values
    """





