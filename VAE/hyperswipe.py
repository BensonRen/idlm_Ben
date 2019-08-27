"""
This .py file is to run train.py for hyper-parameter swipping in a linear fashion.
"""
import train
import flag_reader
if __name__ == '__main__':
    #encoder_fc_filters_front = ( 100,  )
    #encoder_fc_filters_back = ( 300, 100, 20)
    #spectra_fc_filters_front = ( 100,  )
    #spectra_fc_filters_back = ( 300, 100, 20)
    #added_layer_size = 500
    #Setting the loop for setting the parameter
    for i in range(5,41,5):
        flags = flag_reader.read_flag()  	#setting the base case
        #encoder_fc_filters = encoder_fc_filters_front
        #for j in range(i):
        #    encoder_fc_filters += (added_layer_size,)
        #encoder_fc_filters += encoder_fc_filters_back
        #flags.encoder_fc_filters = encoder_fc_filters
        #print(flags.encoder_fc_filters)
        flags.latent_dim = i
        train.train_from_flag(flags)


