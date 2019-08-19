"""
This .py file is to run train.py for hyper-parameter swipping in a linear fashion.
"""
import train
#os.environ["CUDA_VISIBLE_DEVICE"] = "-1"               #Uncomment this line if you want to use CPU only
import tensorflow as tf
import flag_reader
if __name__ == '__main__':
    #backward_fc_filters_front = ( 500, )
    #backward_fc_filters_back = (500, 300, 100, 8)
    #added_layer_size = 1000
    conv_channel_list = []
    conv1d_filters_list = []
    conv1d_filters_list.append((160,))
    conv1d_filters_list.append((160,20))
    conv1d_filters_list.append((160,20,5))
    conv_channel_list.append((1,))
    conv_channel_list.append((2,1))
    conv_channel_list.append((4,2,1))
    #Setting the loop for setting the parameter
    for cnt, (conv_channels, conv1d_filters) in enumerate(zip(conv_channel_list, conv1d_filters_list)):
        flags = flag_reader.read_flag()  	#setting the base case
        flags.conv_channel_list = conv_channels
        flags.conv1d_filters = conv1d_filters
        train.train_from_flag(flags)


