import argparse
import tensorflow as tf
import data_reader
import network_helper
import Tandem_network_maker
import model_maker
import flag_reader
import os
import glob
import shutil
def tandemmain(flags):
    # initialize data reader

    geometry, spectra, train_init_op, valid_init_op = data_reader.read_data(input_size=0,
                                                               output_size=0,
                                                               x_range=flags.x_range,
                                                               y_range=flags.y_range,
								geoboundary = flags.geoboundary,
                                                               cross_val=flags.cross_val,
                                                               val_fold=flags.val_fold,
                                                               batch_size=flags.batch_size,
                                                               shuffle_size=flags.shuffle_size,
							        data_dir = flags.data_dir,
								normalize_input = flags.normalize_input,
                                                                test_ratio = 0.2)
  	#If the input is normalized, then make the boundary useless
    if flags.normalize_input:
        flags.geoboundary = [-1, 1, -1, 1]

    print("making network now")
    # make network
    ntwk = Tandem_network_maker.TandemCnnNetwork(geometry, spectra, model_maker.tandem_model, flags.batch_size,
                            clip=flags.clip, forward_fc_filters=flags.forward_fc_filters,
                            backward_fc_filters=flags.backward_fc_filters,reg_scale=flags.reg_scale,
                            learn_rate=flags.learn_rate,tconv_Fnums=flags.tconv_Fnums,
                            tconv_dims=flags.tconv_dims,n_branch=flags.n_branch,
                            tconv_filters=flags.tconv_filters, n_filter=flags.n_filter,
                            decay_step=flags.decay_step, decay_rate=flags.decay_rate,
                            boundary = flags.geoboundary)
    
    print("Setting the hooks now")
    # define hooks for monitoring training
    train_loss_hook_list = []
    losses = [ntwk.loss, ntwk.mse_loss, ntwk.reg_loss, ntwk.bdy_loss]
    loss_names = ['train_loss','mse_loss','regularizaiton_loss','boundary_loss']
    #Forward detailed loss hooks, the training detail depend on input flag
    forward_hooks = get_hook_list(flags, ntwk, losses, loss_names, 'forward', flags.detail_train_loss_forward) 
    #Assume Tandem one always show the training detailed loss
    tandem_hooks = get_hook_list(flags, ntwk, losses, loss_names, 'tandem', detail_train_loss = True)
    
    # train the network
    print("Start the training now")
    #ntwk.train(train_init_op, flags.train_step, [train_hook, valid_hook, lr_hook], write_summary=True)
    ntwk.train(train_init_op, flags.train_step, flags.backward_train_step, forward_hooks, backward_hooks,
                write_summary=True, load_forward_ckpt = flags.forward_model_ckpt)

    #Put the parameter.txt file into the latest folder from model
    put_param_into_folder()

def get_hook_list(flags, ntwk, losses, loss_name,  forward_or_backward_str, detail_train_loss=True):
    hook_list = []
    if (detail_train_loss):
        for (loss, name) in enumerate(zip(losses, loss_names)):
            hook_list.append(network_helper.TrainValueHook(flags.verb_step, loss, value_name = forward_or_backward_str + name,
                                                            ckpt_dir=ntwk.ckpt_dir, write_summary=True))
    valid_hook = network_helper.ValidationHook(flags.eval_step, valid_init_op, ntwk.labels, ntwk.logits, ntwk.mse_loss,
                                   stop_threshold = flags.stop_threshold,value_name = forward_or_backward_str + 'test_loss',
                                   ckpt_dir=ntwk.ckpt_dir, write_summary=True)
    hook_list.append(valid_hook)                     #The validation hook is always in the list
    return hook_list

def put_param_into_folder():
    list_of_files = glob.glob('models/*')
    latest_file = max(list_of_files, key = os.path.getctime)
    print("The parameter.txt is put into folder " + latest_file)
    destination = os.path.join(latest_file, "parameters.txt");
    shutil.move("parameters.txt",destination)
    
def train_from_flag(flags): 
    flag_reader.write_flags(flags)
    tf.reset_default_graph()
    tandemmain(flags)
    
if __name__ == '__main__':
    flags = flag_reader.read_flag()
    flag_reader.write_flags(flags)
    tf.reset_default_graph()
    tandemmain(flags)
