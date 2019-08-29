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
import os
import pandas as pd
import train
import flag_reader


def read_tensor_from_test_data(data_path, batch_size):
    """
    The function to read the data tensors for the test data for prediction
    :param data_path: data path of the source, specific to name
    return data_tensor
    """
    print("Getting data tensor for test case...")
    data = pd.read_csv(data_path, header = None, delimiter = ' ')
    data = np.array(data, dtype = 'float32')
    data_tensor_slice = tf.data.Dataset.from_tensor_slices(data)
    print("data tensor before batch", data_tensor_slice)
    print("batch_size:", batch_size)
    data_tensor_slice = data_tensor_slice.batch(batch_size, drop_remainder=True)
    #iterator = data_tensor_slice.make_one_shot_iterator()
    iterator = tf.data.Iterator.from_structure(data_tensor_slice.output_types, data_tensor_slice.output_shapes)
    data_tensor = iterator.get_next()
    pred_init_op = iterator.make_initializer(data_tensor_slice)
    print("Data_tensor:",data_tensor)
    return data_tensor, pred_init_op


def predict(flags, geo2spec, data_path, save_path):
    #Clear the default graph first for resolving potential name conflicts
    tf.reset_default_graph()
    spec2geo_flag = not geo2spec #Get geo2spec from spec2geo flagg
    ckpt_dir = os.path.join(os.path.abspath(''), 'models', flags.model_name)
    
    clip, forward_fc_filters, tconv_Fnums, tconv_dims, tconv_filters, n_filter, n_branch, \
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
                                                                test_ratio = 0.2)

    #if the input is normalized
    if flags.normalize_input:
		    flags.boundary = [-1, 1, -1, 1]

    #Adjust the input of geometry and spectra given the flag
    if (spec2geo_flag):
        geometry = features;
        spectra, pred_init_op = read_tensor_from_test_data(data_path, batch_size)
        print("Your are inferring from spectra to geometry")
    else:
        geometry, pred_init_op = read_tensor_from_test_data(data_path, batch_size)
        spectra = labels
        print("Your are inferring from geometry to spectra")

    # make network
    ntwk = Tandem_network_maker.TandemCnnNetwork(geometry, spectra, model_maker.tandem_model, batch_size,
                                clip=clip, forward_fc_filters=forward_fc_filters,
                                backward_fc_filters = backward_fc_filters,reg_scale=reg_scale,
	                        learn_rate=flags.learn_rate,tconv_Fnums=tconv_Fnums,
				tconv_dims=tconv_dims,n_branch=n_branch,
			        tconv_filters=tconv_filters, n_filter=n_filter,
				decay_step=flags.decay_step, decay_rate=flags.decay_rate, geoboundary = flags.geoboundary,
                                conv1d_filters = conv1d_filters, conv_channel_list = conv_channel_list)

    if (spec2geo_flag):
        ntwk.predict_spec2geo([train_init_op, pred_init_op], ckpt_dir = ckpt_dir, 
                                model_name = flags.model_name, save_file = save_path)
    else:
        ntwk.predict_geo2spec([train_init_op, pred_init_op], ckpt_dir = ckpt_dir, 
                                model_name = flags.model_name, save_file = save_path)
        
if __name__ == '__main__':
    flags = flag_reader.read_flag()
    #Set up the model
    predict(flags, geo2spec = flags.predict_geo2spec, data_path = flags.predict_file_path, save_path = flags.predict_save_path)
