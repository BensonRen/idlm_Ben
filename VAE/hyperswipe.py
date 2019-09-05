"""
This .py file is to run train.py for hyper-parameter swipping in a linear fashion.
"""
import train
import flag_reader
if __name__ == '__main__':
    encoder_fc_filters_front = ( 100,  )
    encoder_fc_filters_back = (  100, 30)
    decoder_fc_filters_front = ( 500,  )
    decoder_fc_filters_back = (  500, 8)
    #spectra_fc_filters_front = ( 100,  )
    #spectra_fc_filters_back = ( 300, 100, 20)
    added_layer_size = 500
    #Setting the loop for setting the parameter
    #laten_dim_list = [5,10,15,20,25,30,35,40]
    for i in range(5):
        flags = flag_reader.read_flag()
        #flags.latent_dim = latent_dim
        encoder_fc_filters = encoder_fc_filters_front
        for kk in range(i):
            encoder_fc_filters += (added_layer_size,)
            encoder_fc_filters += encoder_fc_filters_back
        for j in range(5):
            flags = flag_reader.read_flag()  	#setting the base case
            decoder_fc_filters = decoder_fc_filters_front
            for ll in range(j):
                decoder_fc_filters += (added_layer_size,)
            decoder_fc_filters += decoder_fc_filters_back
            flags.encoder_fc_filters = encoder_fc_filters
            flags.decoder_fc_filters = decoder_fc_filters
            print(flags.encoder_fc_filters)
        #flags.latent_dim = i
            train.train_from_flag(flags)


