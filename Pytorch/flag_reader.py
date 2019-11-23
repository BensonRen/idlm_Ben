import argparse
import pprint
import pandas as pd
from parameters import *


def read_flag():
    """
    This function is to write the read the flags from a parameter file and put them in formats
    :return: flags: a struct where all the input params are stored
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--linear', type=list, default=LINEAR, help='The fc layers units')
    parser.add_argument('--conv-out-channel', type=list, default=CONV_OUT_CHANNEL, help='The output channel of your 1d conv')
    parser.add_argument('--conv-kernel-size', type=list, default=CONV_KERNEL_SIZE, help='The kernel size of your 1d conv')
    parser.add_argument('--conv-stride', type=list, default=CONV_STRIDE, help='The strides of your 1d conv')
    parser.add_argument('--reg-scale', type=float, default=REG_SCALE, help='#scale for regularization of dense layers')
    parser.add_argument('--x-range', type=list, default=X_RANGE, help='columns of input parameters')
    parser.add_argument('--y-range', type=list, default=Y_RANGE, help='columns of output parameters')
    parser.add_argument('--batch-size', default=BATCH_SIZE, type=int, help='batch size (100)')
    parser.add_argument('--eval-step', default=EVAL_STEP, type=int, help='# steps between evaluations')
    parser.add_argument('--train-step', default=TRAIN_STEP, type=int, help='# steps to train on the dataSet')
    parser.add_argument('--lr', default=LEARN_RATE, type=float, help='learning rate')
    parser.add_argument('--decay-step', default=DECAY_STEP, type=int,
                        help='decay learning rate at this number of steps')
    parser.add_argument('--decay-rate', default=DECAY_RATE, type=float,
                        help='decay learn rate by multiplying this factor')
    parser.add_argument('--model-name', default=MODEL_NAME, type=str, help='name of the model')
    parser.add_argument('--data-dir', default=DATA_DIR, type=str, help='data directory')
    parser.add_argument('--normalize-input', default=NORMALIZE_INPUT, type=bool,
                        help='whether we should normalize the input or not')
    parser.add_argument('--stop_threshold', default=STOP_THRESHOLD, type=float,
                        help='The threshold below which training should stop')
    parser.add_argument('--optim', default=OPTIM, type=str, help='the type of optimizer that you want to use')
    parser.add_argument('--geoboundary', default=GEOBOUNDARY, type=tuple, help='the boundary of the geometric data')
    flags = parser.parse_args()  # This is for command line version of the code
    # flags = parser.parse_args(args = [])#This is for jupyter notebook version of the code
    # flagsVar = vars(flags)
    return flags


def write_flags_and_BVE(flags, best_validation_loss):
    """
    The function that is usually executed at the end of the training where the flags and the best validation loss are recorded
    They are put in the folder that called this function and save as "parameters.txt"
    This parameter.txt is also attached to the generated email
    :param flags: The flags struct containing all the parameters
    :param best_validation_loss: The best_validation_loss recorded in a training
    :return: None
    """
    #To avoid terrible looking shape of y_range
    yrange = flags.y_range
    yrange_str = str(yrange[0]) + ' to ' + str(yrange[-1])
    yrange_str = [yrange[0], yrange[-1]]
    flags_dict = vars(flags)
    flags_dict_copy = flags_dict.copy()                 # in order to not corrupt the original data strucutre
    flags_dict_copy['y_range'] = yrange_str             # Change the y range to be acceptable long string
    flags_dict_copy['best_validation_loss'] = best_validation_loss #Append the bvl
    #Convert the dictionary into pandas data frame which is easier to handle with and write read
    print(flags_dict_copy)
    with open('parameters.txt','w') as f:
        print(flags_dict_copy, file=f)

