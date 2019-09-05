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
import flag_reader
import time_recorder
import get_pred_truth_file

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
    #Set the environment variable for if this is a cpu only script
    if flags.use_cpu_only:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    print("Start Evaluating now...")
    TK = time_recorder.time_keeper(time_keeping_file = "data/time_keeper.txt")

    tf.reset_default_graph()
    ckpt_dir = os.path.join(os.path.abspath(''), 'models', flags.model_name)
    
    decoder_fc_filters, encoder_fc_filters, spectra_fc_filters, conv1d_filters, \
    filter_channel_list, geoboundary, latent_dim, batch_size = network_helper.get_parameters(ckpt_dir)  
    batch_size = batch_size[0] #Get rid of the list 
    geometry, spectra, train_init_op, valid_init_op = data_reader.read_data(input_size=flags.input_size,
                                                               output_size=300,
                                                               x_range=flags.x_range,
                                                               y_range=flags.y_range,
							        geoboundary=flags.geoboundary,
                                                               cross_val=flags.cross_val,
                                                               val_fold=flags.val_fold,
                                                               batch_size=flags.batch_size,
                                                               shuffle_size=flags.shuffle_size,
								data_dir = flags.data_dir,
							        normalize_input = flags.normalize_input,
                                                                test_ratio = 0.9999)
    #if the input is normalized
    if flags.normalize_input:
		    flags.boundary = [-1, 1, -1, 1]
    print("Boundary read from meta_file is ", geoboundary)
    print("batch_size read from meta_file is ", batch_size)
    print("latent_dim read from meta_file is ", latent_dim)
    # make network
    ntwk = VAE_network_maker.VAENetwork(geometry, spectra, model_maker.VAE, batch_size, latent_dim,
                            spectra_fc_filters=spectra_fc_filters, decoder_fc_filters=decoder_fc_filters,
                            encoder_fc_filters=encoder_fc_filters,reg_scale=flags.reg_scale,
                            learn_rate=flags.learn_rate, decay_step=flags.decay_step, decay_rate=flags.decay_rate,
                            geoboundary = flags.geoboundary, conv1d_filters = conv1d_filters, filter_channel_list = filter_channel_list)
    
    # evaluate the results if the results do not exist or user force to re-run evaluation
    save_file = os.path.join(os.path.abspath(''), 'data', 'test_pred_{}.csv'.format(flags.model_name))
    
    if flags.force_run or (not os.path.exists(save_file)):
        print('Evaluating the model ...')
        #pred_file, truth_file = ntwk.evaluate(valid_init_op, ckpt_dir=ckpt_dir,
        Xpred_file = ntwk.evaluate(valid_init_op, train_init_op, ckpt_dir=ckpt_dir, 
                                        model_name=flags.model_name, write_summary=True,
                                        eval_forward = eval_forward, time_keeper = TK)

        print("Prediction File output at:", Xpred_file)
        unpack_Xpred(Xpred_file,batch_size)
        #pred_file, truth_file = get_spectra_from_geometry(Xpred_file)
    """
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
    """

def unpack_Xpred(Xpred_file, batch_size):
    """
    THis is the function which unpacks the Xpred file from VAE evaluation to a long file
    Since VAE prediction gives #batch_size of Geometry each time, unpack them into a long list for Tandem inference
    """
    Xpred = np.loadtxt(Xpred_file, delimiter=' ')
    h,w = np.shape(Xpred)
    with open("data/Unpackinformation.txt",'w') as f1:
        f1.write('The number of data point is {}, each with {} predicted geometries'.format(h, w/8))
    Xpred_reshaped = np.transpose(np.reshape(Xpred, (8, -1)))
    h,w = np.shape(Xpred_reshaped)
    assert w == 8, "Your unpack function didn't work, check again what was wrong in the evaluateion output and unpack"
    with open(Xpred_file,'w') as f:
        np.savetxt(f, Xpred_reshaped, fmt='%.3f')

def after_Tandem_pred():
    """
    This function handles the rest of the evaluation after the Ypred has been generated by the Tandem model (forward model)
    """
    data_dir = 'data'
    Ypred_file = get_pred_truth_file.get_Ypred(data_dir)
    Xpred_file = get_pred_truth_file.get_Xpred(data_dir)
    Ytruth_file = get_pred_truth_file.get_Ytruth(data_dir)
    Ypred = np.loadtxt(Ypred_file, delimiter = ' ')
    Ytruth = np.loadtxt(Ytruth_file, delimiter = ' ')
    Xpred = np.loadtxt(Xpred_file, delimiter = ' ')
    
    l_Ypred = len(Ypred)
    l_Ytruth = len(Ytruth)
    k =  l_Ypred / l_Ytruth
    print("l_Ypred",l_Ypred)
    print("l_Ytruth",l_Ytruth)
    print("k",k)
    assert k - int(k) < 0.001,"Check you length, the divide result k is not an int!!"
    print('For each data point in your truth file, the VAE generated {} data points'.format(k))
    k = int(k) 
    #best_index_list = np.zeros([1,l_Ytruth])
    Xpred_new = np.zeros([l_Ytruth,8])
    Ypred_new = np.zeros(np.shape(Ytruth))
    for i in range(l_Ytruth):
        diff_mat = Ypred[i*k:(i+1)*k,:] - Ytruth[i,:] 
        distance_mat = np.linalg.norm(diff_mat, axis = 1)
        best_index = np.argmax(distance_mat)
        #best_index_list[i] = best_index
        Xpred_new[i,:] = Xpred[i*k + best_index,:]
        Ypred_new[i,:] = Ypred[i*k + best_index,:]
    with open(Xpred_file, 'w') as f1:
        np.savetxt(f1, Xpred_new, fmt='%.3f')
    with open(Ypred_file, 'w') as f2:
        np.savetxt(f2, Ypred_new, fmt='%.3f')
    
    mae, mse = compare_truth_pred(Ypred_file, Ytruth_file)

    plt.figure(figsize=(12, 6))
    plt.hist(mse, bins=100)
    plt.xlabel('Mean Squared Error')
    plt.ylabel('cnt')
    plt.suptitle('VAE (Avg MSE={:.4e})'.format(np.mean(mse)))
    plt.savefig(os.path.join(os.path.abspath(''), 'data',
                             'VAE.png'))
    plt.show()
    print('VAE (Avg MSE={:.4e})'.format(np.mean(mse)))

if __name__ == '__main__':
	flags = flag_reader.read_flag()
	evaluatemain(flags, eval_forward = False)
	#plotsAnalysis.SpectrumComparisonNGeometryComparison(3,2, (13,8), flags.model_name,flags.boundary)	





