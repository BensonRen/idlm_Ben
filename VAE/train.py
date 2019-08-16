import argparse
import tensorflow as tf
import data_reader
import network_helper
import VAE_network_maker
import model_maker
import flag_reader
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
								normalize_input = flags.normalize_input)
  	#If the input is normalized, then make the boundary useless
    if flags.normalize_input:
        flags.geoboundary = [-1, 1, -1, 1]

    # make network
    ntwk = VAE_network_maker.VAENetwork(geometry, spectra, model_maker.VAE, flags.batch_size, flags.latent_dim,
                            spectra_fc_filters=flags.spectra_fc_filters, decoder_fc_filters=flags.decoder_fc_filters,
                            encoder_fc_filters=flags.encoder_fc_filters,reg_scale=flags.reg_scale,
                            learn_rate=flags.learn_rate, decay_step=flags.decay_step, decay_rate=flags.decay_rate,
                            geoboundary = flags.geoboundary)
    
    
    # define hooks for monitoring training
    train_VAE_hook = network_helper.TrainValueHook(flags.verb_step, ntwk.loss, value_name = 'VAE_train_loss',
                               ckpt_dir=ntwk.ckpt_dir, write_summary=True)
    VAE_Boundary_hook = network_helper.TrainValueHook(flags.verb_step, ntwk.Boundary_loss, value_name = 'VAE_Boundary_loss',
                               ckpt_dir=ntwk.ckpt_dir, write_summary=True)
    #lr_hook = TrainValueHook(flags.verb_step, ntwk.learn_rate, ckpt_dir=ntwk.ckpt_dir,
    #                                        write_summary=True, value_name='learning_rate')
    valid_VAE_hook = network_helper.ValidationHook(flags.eval_step, valid_init_op, ntwk.labels, ntwk.logits,ntwk.loss,
                                        stop_threshold = flags.stop_threshold,value_name = 'VAE_test_loss', 
                                        ckpt_dir=ntwk.ckpt_dir, write_summary=True)
    
def train_from_flag(flags): 
    flag_reader.write_flags(flags)
    tf.reset_default_graph()
    VAEtrainmain(flags)
    
if __name__ == '__main__':
    flags = flag_reader.read_flag()
    flag_reader.write_flags(flags)
    tf.reset_default_graph()
    VAEtrainmain(flags)
