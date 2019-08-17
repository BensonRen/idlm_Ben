import argparse
import tensorflow as tf
import data_reader
import network_helper
import model_maker
import pprint
from parameters import *
def read_flag():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-size', type=int, default=INPUT_SIZE, help='input size')
    parser.add_argument('--clip', type=int, default=CLIP, help='points clipped from each end of output after final conv')
    parser.add_argument('--forward-fc-filters', type=tuple, default=FORWARD_FC_FILTERS, help='#neurons in each fully connected layers')
    parser.add_argument('--backward-fc-filters', type=tuple, default=BACKWARD_FC_FILTERS, help='#neurons in each fully connected layers')
    parser.add_argument('--tconv-Fnums', type=tuple, default=TCONV_FNUMS, help='#0th shape dim of each tconv layer')
    parser.add_argument('--tconv-dims', type=tuple, default=TCONV_DIMS,
                        help='dimensionality of data after each transpose convolution')
    parser.add_argument('--tconv-filters', type=tuple, default=TCONV_FILTERS,
                        help='#filters at each transpose convolution')
    parser.add_argument('--conv1d-filters', type=tuple, default=CONV1D_FILTERS, help='#0 shape dim of each conv layer in backward model')
    parser.add_argument('--conv-channel-list', type=tuple, default=CONV_CHANNEL_LIST ,help='number of channels in each conv layers')
    parser.add_argument('--n-filter', type=int, default=N_FILTER, help='#neurons in the tensor module'),
    parser.add_argument('--n-branch', type=int, default=N_BRANCH, help='#parallel branches in the tensor module')
    parser.add_argument('--reg-scale', type=float, default=REG_SCALE, help='#scale for regularization of dense layers')
    parser.add_argument('--x-range', type=list, default=X_RANGE, help='columns of input parameters')
    parser.add_argument('--y-range', type=list, default=Y_RANGE, help='columns of output parameters')
    parser.add_argument('--cross-val', type=int, default=CROSS_VAL, help='# cross validation folds')
    parser.add_argument('--val-fold', type=int, default=VAL_FOLD, help='fold to be used for validation')
    parser.add_argument('--batch-size', default=BATCH_SIZE, type=int, help='batch size (100)')
    parser.add_argument('--shuffle-size', default=SHUFFLE_SIZE, type=int, help='shuffle size (100)')
    parser.add_argument('--verb-step', default=VERB_STEP, type=int, help='# steps between every print message')
    parser.add_argument('--eval-step', default=EVAL_STEP, type=int, help='# steps between evaluations')
    parser.add_argument('--train-step', default=TRAIN_STEP, type=int, help='# steps to train on the dataset')
    parser.add_argument('--backward-train-step', default=BACKWARD_TRAIN_STEP, type=int, help='# steps to train on the backward model of the dataset')
    parser.add_argument('--learn-rate', default=LEARN_RATE, type=float, help='learning rate')
    parser.add_argument('--decay-step', default=DECAY_STEP, type=int,
                        help='decay learning rate at this number of steps')
    parser.add_argument('--decay-rate', default=DECAY_RATE, type=float,
                        help='decay learn rate by multiplying this factor')
    parser.add_argument('--forward-model-ckpt', default=FORWARDMODEL_CKPT, type=str,
                        help='name of the forward ckpt file')
    parser.add_argument('--force-run', default=FORCE_RUN, type=bool, help='force it to rerun')
    parser.add_argument('--model-name', default=MODEL_NAME, type=str, help='name of the model')
    parser.add_argument('--data-dir', default=DATA_DIR, type=str, help='data directory')
    parser.add_argument('--normalize-input', default=NORMALIZE_INPUT, type=bool, help='whether we should normalize the input or not')
    parser.add_argument('--stop_threshold', default=STOP_THRESHOLD, type=float, help='The threshold below which training should stop')
    parser.add_argument('--geoboundary', default=GEOBOUNDARY, type=tuple, help='the boundary of the geometric data')
    parser.add_argument('--detail-train-loss-forward', default=DETAIL_TRAIN_LOSS_FORWARD, type=bool, help='whether make hook for detailed training process')
    parser.add_argument('--write-weight-step', default=WRITE_WEIGHT_STEP, type=int, help='#steps to write the weight summary histogram into tensorboard')
    # parser.add_argument('--train-file', default=TRAIN_FILE, type=str, help='name of the training file')
    # parser.add_argument('--valid-file', default=VALID_FILE, type=str, help='name of the validation file')
    
    flags = parser.parse_args()  #This is for command line version of the code
    #flags = parser.parse_args(args = [])#This is for jupyter notebook version of the code
    
	#flagsVar = vars(flags)

    return flags

def write_flags_and_BVE(flags, best_validation_error):
    #To avoid terrible looking shape of y_range
    yrange = flags.y_range
    yrange_str = str(yrange[0]) + ' to ' + str(yrange[-1])
    flags_dict = vars(flags)
    flags_dict_copy = flags_dict.copy() #in order to not corrupt the original data strucutre
    flags_dict_copy['y_range'] = yrange_str
    flags_dict_copy['best_validation_loss'] = best_validation_loss
    #Convert the dictionary into pandas data frame which is easier to handle with and write read
    flags_df = pd.DataFrame.from_dict(flags_dict_copy)
    flags_df.to_csv("parameters.txt")

    #dict_str = pprint.pformat(flags_dict_copy)
    ##with open("parameters.txt","w") as log_file:
    #    log_file.write(dict_str)
    #pprint(flags_dict)
    #return dict_str

