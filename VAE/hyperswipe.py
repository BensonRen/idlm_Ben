"""
This .py file is to run train.py for hyper-parameter swipping in a linear fashion.
"""
import train
import flag_reader
if __name__ == '__main__':
    spectra_fc_filters_front = ( 100,  )
    spectra_fc_filters_back = ( 300, 100, 20)
    added_layer_size = 500
    #Setting the loop for setting the parameter
    for i in range(10):
        flags = flag_reader.read_flag()  	#setting the base case
        spectra_fc_filters = spectra_fc_filters_front
        for j in range(i):
            spectra_fc_filters += (added_layer_size,)
        spectra_fc_filters += spectra_fc_filters_back
        flags.spectra_fc_filters = spectra_fc_filters
        print(flags.spectra_fc_filters)
        train.train_from_flag(flags)


