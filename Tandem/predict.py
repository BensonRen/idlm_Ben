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


def read_tensor_from_test_data(data_path, flags):
    """
    The function to read the data tensors for the test data for prediction
    :param data_path: data path of the source, specific to name
    return data_tensor
    """
    print("Getting data tensor for test case...")
    data = pd.read_csv(data_path, header = None)
    data_tensor_slice = tf.data.Dataset.from_tensor_slice(data)
    data_tensor_slice = data_tensor.batch(flags.batch_size, drop_remainder=False)
    iterator = data_tensor_slice.make_one_shot_iterator()
    #iterator = tf.data.Iterator.from_strcture(data_tensor.output_types, data_tensor.output_shapes)
    data_tensor = iterator.get_next()
    return data_tensor


def predict(flags, spec2geo_flag, data_path):
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
                                                                test_ratio = 0.2)

    #if the input is normalized
    if flags.normalize_input:
		    flags.boundary = [-1, 1, -1, 1]

    #Adjust the input of geometry and spectra given the flag
    if (spec2geo_flag):
        geometry = features;
        spectra = read_tensor_from_test_data(data_path, flags)
    else:
        geometry = read_tensor_from_test_data(data_path, flags)
        spectra = labels

    # make network
    ntwk = Tandem_network_maker.TandemCnnNetwork(geometry, spectra, model_maker.tandem_model, flags.batch_size,
                                clip=flags.clip, forward_fc_filters=flags.forward_fc_filters,
                                backward_fc_filters=flags.backward_fc_filters,reg_scale=flags.reg_scale,
	                        learn_rate=flags.learn_rate,tconv_Fnums=flags.tconv_Fnums,
				tconv_dims=flags.tconv_dims,n_branch=flags.n_branch,
			        tconv_filters=flags.tconv_filters, n_filter=flags.n_filter,
				decay_step=flags.decay_step, decay_rate=flags.decay_rate, geoboundary = flags.geoboundary,
                                conv1d_filters = flags.conv1d_filters, conv_channel_list = flags.conv_channel_list)

    if (spec2geo_flag):
        ntwk.predict_spec2geo(train_init_op, ckpt_dir = ckpt_dir, model_name = flags.model_name)
    else:
        ntwk.predict_geo2spec(train_init_op, ckpt_dir = ckpt_dir, model_name = flags.model_name)
        
if __name__ == '__main__':
	flags = flag_reader.read_flag()
        #Set up the model
        assert(len(sys.argv) == 2 ) , "Make sure you provide a argument that is either : --spec2geo or --geo2spec"
        assert((sys.argv[1] == 'spec2geo') or (sys.argv[1] == 'geo2spec')), \
                                        "Make sure you provide a argument that is either : spec2geo or geo2spec"
	print("You are doing prediction using", sys.argv[1])

        if (sys.argv[1] == 'spec2geo'):
            predict(flags, spec2geo = True, flags.predict_file_path)
        else:
            predict(flags, spec2geo = False,flags.predict_file_path)
