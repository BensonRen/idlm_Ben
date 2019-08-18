"""
This .py file is to run train.py for hyper-parameter swipping in a linear fashion.
"""
import train
#os.environ["CUDA_VISIBLE_DEVICE"] = "-1"               #Uncomment this line if you want to use CPU only
import tensorflow as tf
import flag_reader
if __name__ == '__main__':
    backward_fc_filters_front = ( 500, )
    backward_fc_filters_back = (500, 300, 100, 8)
    added_layer_size = 1000
    #Setting the loop for setting the parameter
    for i in range(4):
        flags = flag_reader.read_flag()  	#setting the base case
        backward_fc_filters = backward_fc_filters_front
        for j in range(i):
            backward_fc_filters += (added_layer_size,)
        backward_fc_filters += backward_fc_filters_back
        flags.backward_fc_filters = backward_fc_filters
        print(flags.backward_fc_filters)
        train.train_from_flag(flags)


