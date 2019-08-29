import argparse
import tensorflow as tf
import data_reader
import network_helper
import VAE_network_maker
import model_maker
import flag_reader
import os
import glob
import shutil
def VAEtrainmain(flags):
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

    # make network
    ntwk = VAE_network_maker.VAENetwork(geometry, spectra, model_maker.VAE, flags.batch_size, flags.latent_dim,
                            spectra_fc_filters=flags.spectra_fc_filters, decoder_fc_filters=flags.decoder_fc_filters,
                            encoder_fc_filters=flags.encoder_fc_filters,reg_scale=flags.reg_scale,
                            learn_rate=flags.learn_rate, decay_step=flags.decay_step, decay_rate=flags.decay_rate,
                            geoboundary = flags.geoboundary, conv1d_filters = flags.conv1d_filters,
                            filter_channel_list = flags.filter_channel_list)
    
    
    print("Setting the hooks now")
    # define hooks for monitoring training
    train_loss_hook_list = []
    losses = [ntwk.loss, ntwk.mse_loss, ntwk.reg_loss, ntwk.bdy_loss,ntwk.kl_loss, ntwk.learn_rate]
    loss_names = ["train_loss", "mse_loss", "regularizaiton_loss", "boundary_loss","KL_loss","Learning_rate"]
    #Forward detailed loss hooks, the training detail depend on input flag
    VAE_hooks = get_hook_list(flags, ntwk, valid_init_op, losses, loss_names, "VAE_") 
    
    print("Starting training now")
    ntwk.train(train_init_op, flags.train_step, VAE_hooks,
                write_summary=True)
    
    #Write the flag into the current folder and move it to the models/ folder along with the best validation error
    flag_reader.write_flags_and_BVE(flags, ntwk.best_validation_loss)
    
    #Put the parameter.txt file into the latest folder from model
    put_param_into_folder()


def get_hook_list(flags, ntwk, valid_init_op, losses, loss_names,  forward_or_backward_str, detail_train_loss=True, summary_op=None):
    hook_list = []
    if (detail_train_loss):
        print("Losses:", losses)
        print("loss_name", loss_names)
        for cnt, (loss, name) in enumerate(zip(losses, loss_names)):
            print("forward_or_backward_str:",forward_or_backward_str)
            print("name:", name)
            print("loss:", loss)
            hook_list.append(network_helper.TrainValueHook(flags.verb_step, loss, value_name = forward_or_backward_str + name,
                                                            ckpt_dir=ntwk.ckpt_dir, write_summary=True))
    #add a summary op hook for histograms
    print("Merged Summary op:", ntwk.merged_summary_op)
    summary_op_hook = network_helper.SummaryWritingHook(ntwk.merged_summary_op, flags.write_weight_step)
    hook_list.append(summary_op_hook)

    #Add a validation hook at the END!! (THE end controls the stopping of the training
    valid_hook = network_helper.ValidationHook(flags.eval_step, valid_init_op, ntwk.labels, ntwk.logits, ntwk.mse_loss,
                                   stop_threshold = flags.stop_threshold,value_name = forward_or_backward_str + "test_loss",
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
    tf.reset_default_graph()
    VAEtrainmain(flags)
    
if __name__ == '__main__':
    flags = flag_reader.read_flag()
    tf.reset_default_graph()
    VAEtrainmain(flags)
