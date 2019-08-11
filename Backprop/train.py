import argparse
import tensorflow as tf
import data_reader
import network_helper
import Backprop_network_maker
import model_maker
import flag_reader
def backpropmain(flags):
    # initialize data reader

    spectra, geometry, train_init_op, valid_init_op = data_reader.read_data(input_size=0,
                                                               output_size=0,
                                                               x_range=flags.x_range,
                                                               y_range=flags.y_range,
                                                               cross_val=flags.cross_val,
                                                               val_fold=flags.val_fold,
                                                               batch_size=flags.batch_size,
                                                               shuffle_size=flags.shuffle_size,
                                                               forward = False
																															 data_dir = flags.data_dir)
    
    # make network
    ntwk = Backprop_network_maker.BackPropCnnNetwork(geometry, spectra, model_maker.back_prop_model, flags.batch_size,
                            clip=flags.clip, forward_fc_filters=flags.forward_fc_filters, reg_scale=flags.reg_scale,
                            learn_rate=flags.learn_rate,tconv_Fnums=flags.tconv_Fnums,
                            tconv_dims=flags.tconv_dims,n_branch=flags.n_branch,
                            tconv_filters=flags.tconv_filters, n_filter=flags.n_filter,
                            decay_step=flags.decay_step, decay_rate=flags.decay_rate)
    
    
    # define hooks for monitoring training
    train_forward_hook = network_helper.TrainValueHook(flags.verb_step, ntwk.loss, value_name = 'forward_train_loss',
                               ckpt_dir=ntwk.ckpt_dir, write_summary=True)
    forward_Boundary_hook = network_helper.TrainValueHook(flags.verb_step, ntwk.Boundary_loss, value_name = 'forward_Boundary_loss',
                               ckpt_dir=ntwk.ckpt_dir, write_summary=True)
    #lr_hook = TrainValueHook(flags.verb_step, ntwk.learn_rate, ckpt_dir=ntwk.ckpt_dir,
    #                                        write_summary=True, value_name='learning_rate')
    valid_forward_hook = network_helper.ValidationHook(flags.eval_step, valid_init_op, ntwk.labels, ntwk.logits,ntwk.loss,
                                        value_name = 'forward_test_loss', ckpt_dir=ntwk.ckpt_dir, write_summary=True)
    
    
    # train the network
    #ntwk.train(train_init_op, flags.train_step, [train_hook, valid_hook, lr_hook], write_summary=True)
    ntwk.train(train_init_op, flags.train_step, [train_forward_hook,forward_Boundary_hook, valid_forward_hook], write_summary=True)
                #,load_forward_ckpt = flags.forward_model_ckpt)


if __name__ == '__main__':
	flags = flag_reader.read_flag()
	tf.reset_default_graph()
	backpropmain(flags)
